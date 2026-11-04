"""Amputation and the imputation models must only fill the values that are missing."""

import logging

import gin
import numpy as np
import pandas as pd
import pytest
import torch

from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.constants import DataSegment as Segment, DataSplit as Split
from icu_benchmarks.data.loader import ImputationPandasDataset
from icu_benchmarks.imputation.amputations import ampute_data
from icu_benchmarks.imputation.mlp import MLPImputation
from icu_benchmarks.imputation.rnn import BRNNImputation, RNNImputation
from icu_benchmarks.models.constants import ImputationInit
from icu_benchmarks.models.train import train_common
from icu_benchmarks.run_utils import load_pretrained_imputation_model

VARS = {"GROUP": "stay_id", "SEQUENCE": "time", "DYNAMIC": ["x", "y", "z"], "STATIC": []}


def frame(rows=400, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(rows, 3)), columns=["x", "y", "z"])


@pytest.mark.parametrize("mechanism", ["MCAR", "MAR", "MNAR", "BO"])
def test_ampute_data_masks_requested_share(mechanism):
    torch.manual_seed(0)
    data = frame()
    amputed, mask = ampute_data(data, mechanism, p_miss=0.3, p_obs=0.5)
    mask = np.asarray(mask, dtype=bool)
    assert mask.shape == data.shape
    np.testing.assert_array_equal(amputed.isna().to_numpy(), mask)
    np.testing.assert_array_equal(amputed.to_numpy()[~mask], data.to_numpy()[~mask])
    if mechanism == "BO":  # Blacks out whole rows
        assert mask.mean() == pytest.approx(0.3) and set(mask.sum(axis=1)) <= {0, 3}
    elif mechanism == "MAR":  # Some variables stay fully observed and drive the others' missingness
        assert mask.any(axis=0).sum() < 3 and 0.05 < mask.mean() < 0.4
    else:
        assert 0.1 < mask.mean() < 0.4


def test_ampute_data_rejects_unknown_mechanism():
    with pytest.raises(ValueError, match="Invalid amputation mechanism"):
        ampute_data(frame(), "XYZ", 0.3)


def test_imputation_dataset_masks_amputated_and_missing_values():
    torch.manual_seed(0)
    features = pd.DataFrame({"stay_id": np.repeat([1, 2], 4), "time": np.tile(range(4), 2), **frame(8).to_dict(orient="list")})
    features.loc[0, "x"] = np.nan
    data = {Split.train: {Segment.features: features, Segment.static: pd.DataFrame({"stay_id": [1, 2]})}}
    with pytest.warns(DeprecationWarning):
        dataset = ImputationPandasDataset(data, vars=VARS, mask_proportion=0.5, ram_cache=False, name="demo")
    assert len(dataset) == 2 and dataset.name == "demo"
    amputated, amputation_mask, target, target_missing = dataset[0]
    assert amputated.shape == amputation_mask.shape == target.shape == (4, 3)
    # The originally missing value is masked for the model and filled with zero in the target
    assert amputation_mask[0, 0] == 1 and target_missing[0, 0] == 1 and target[0, 0] == 0
    assert torch.isnan(amputated[amputation_mask.bool() & (target_missing == 0)]).all()


@pytest.fixture
def clean_gin():
    gin.clear_config()
    yield
    gin.clear_config()


def imputation_splits(stays=8, steps=4):
    rng = np.random.default_rng(0)
    data = {}
    for offset, split in enumerate([Split.train, Split.val, Split.test]):
        stay_ids = np.arange(stays) + 100 * offset
        features = pd.DataFrame({"stay_id": np.repeat(stay_ids, steps), "time": np.tile(range(steps), stays)})
        features[VARS["DYNAMIC"]] = rng.normal(size=(stays * steps, 3))
        data[split] = {Segment.features: features, Segment.static: pd.DataFrame({"stay_id": stay_ids})}
    return data


def test_train_common_trains_and_reloads_imputation_model(tmp_path, clean_gin, caplog):
    # #261: train_common names its datasets, and tests on a best checkpoint that holds the ImputationInit enum
    torch.manual_seed(0)
    gin.bind_parameter("ImputationPandasDataset.vars", VARS)
    names = {"train": "train-set", "val": "val-set", "test": "test-set"}
    kwargs = dict(model=MLPImputation, mode=RunMode.imputation, epochs=1, batch_size=4, cpu=True, num_workers=0)
    with caplog.at_level(logging.INFO):
        loss = train_common(imputation_splits(), log_dir=tmp_path, dataset_names=names, **kwargs)
    assert np.isfinite(loss)
    assert "Training on train-set" in caplog.text and "validating on val-set" in caplog.text
    assert "Testing on test-set" in caplog.text
    loaded = load_pretrained_imputation_model(str(tmp_path / "model.ckpt"))
    assert isinstance(loaded, MLPImputation) and loaded.trained_columns == VARS["DYNAMIC"]


@pytest.mark.parametrize(
    "model",
    [
        lambda: MLPImputation(input_size=(2, 4, 3), num_hidden_layers=1, hidden_layer_size=5),
        lambda: RNNImputation(input_size=(2, 4, 3), hidden_size=5),
        lambda: RNNImputation(input_size=(2, 4, 3), hidden_size=5, cell="lstm", state_init="noise"),
        lambda: BRNNImputation(input_size=(2, 4, 3), hidden_size=5),
    ],
    ids=["mlp", "gru", "lstm", "brnn"],
)
def test_predict_fills_only_missing_values(model):
    torch.manual_seed(0)
    model = model()
    data = torch.rand(2, 4, 3)
    data[0, 1, 2] = data[1, 3, 0] = float("nan")
    observed = data.clone()
    imputed = model.predict(data)
    assert not torch.isnan(imputed).any()
    keep = ~torch.isnan(observed)
    torch.testing.assert_close(imputed[keep], observed[keep])


def test_rnn_imputation_rejects_unknown_cell():
    with pytest.raises(NotImplementedError):
        RNNImputation(input_size=(2, 4, 3), cell="transformer")


@pytest.mark.parametrize(
    "init", [ImputationInit.NORMAL, ImputationInit.XAVIER, ImputationInit.KAIMING, ImputationInit.ORTHOGONAL]
)
def test_init_weights_resets_linear_layers(init):
    model = MLPImputation(input_size=(2, 4, 3), num_hidden_layers=1, hidden_layer_size=5)
    linear = model.model[1]
    torch.nn.init.constant_(linear.bias, 1.0)
    model.init_weights(init)
    assert torch.all(linear.bias == 0) and linear.weight.std() > 0
