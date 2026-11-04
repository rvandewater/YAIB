"""Models on ObservationMaskWrapper must read the observation mask from the recipe's missingness indicator columns,
predict causally, ignore padding, and train, test and reload through train_common."""

import json

import gin
import numpy as np
import polars as pl
import pytest
import torch
from torch import nn

from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.constants import DataSegment as Segment, DataSplit as Split
from icu_benchmarks.data.loader import PredictionPolarsDataset
from icu_benchmarks.data.target_transform import transform_outcomes
from icu_benchmarks.models.train import load_model, train_common
from icu_benchmarks.models.wrappers import ObservationMaskWrapper

VARS = {"GROUP": "stay_id", "SEQUENCE": "time", "LABEL": "label"}
COLUMNS = ["age", "hr", "MissingIndicator_age", "MissingIndicator_hr", "counter"]


@gin.configurable
class ObservationMaskProbe(ObservationMaskWrapper):
    """Smallest model on the wrapper: a linear layer on the inputs and their observation mask."""

    def __init__(self, input_size, num_classes, *args, **kwargs):
        super().__init__(*args, input_size=input_size, num_classes=num_classes, **kwargs)
        self.logit = nn.Linear(2 * input_size[2], num_classes)

    def forward(self, x):
        return self.logit(torch.cat([x, self.observed_mask(x)], dim=-1))


# Model class and the hyperparameters of a small instance
MODELS = {
    "probe": (ObservationMaskProbe, {}),
}


@pytest.fixture(autouse=True)
def clean_gin():
    gin.clear_config()
    yield
    gin.clear_config()


def recipe_splits(stays=12, max_hours=6, hourly=False, regression=False):
    """Splits shaped like the recipe output: standardised values that are forward-filled (0 before the first measurement)
    next to their MissingIndicator columns, a static feature, stays of different lengths so that batches are padded, and
    labels at every hour or at the last hour of each stay."""
    rng = np.random.default_rng(0)
    data = {}
    for split in (Split.train, Split.val, Split.test):
        features, outcomes = [], []
        for stay, length in enumerate(rng.integers(2, max_hours + 1, size=stays)):
            measured = rng.random((length, 2)) < 0.6
            values = np.zeros((length, 2))
            for t in range(length):
                previous = values[t - 1] if t else 0.0
                values[t] = np.where(measured[t], rng.normal(size=2), previous)
            features.append(
                pl.DataFrame(
                    {
                        "stay_id": [stay] * length,
                        "time": np.arange(length),
                        "hr": values[:, 0],
                        "map": values[:, 1],
                        "age": [rng.normal()] * length,
                        "MissingIndicator_hr": (~measured[:, 0]).astype(float),
                        "MissingIndicator_map": (~measured[:, 1]).astype(float),
                        "MissingIndicator_age": [0.0] * length,
                    }
                )
            )
            target = values[:, 0] + values[:, 1] if regression else (values[:, 0] > 0).astype(int)
            if hourly:
                outcomes.append(pl.DataFrame({"stay_id": [stay] * length, "time": np.arange(length), "label": target}))
            else:
                outcomes.append(pl.DataFrame({"stay_id": [stay], "label": target[-1:]}))
        outcome = pl.concat(outcomes)
        if not regression:
            labels = outcome["label"].to_numpy().copy()
            labels[:2] = [0, 1]  # Both classes in every split
            outcome = outcome.with_columns(label=pl.Series(labels))
        data[split] = {Segment.features: pl.concat(features), Segment.outcome: outcome}
    return data


def train(model_cls, params, data, log_dir, mode=RunMode.classification, **kwargs):
    gin.bind_parameter("PredictionPolarsDataset.vars", VARS)
    loss = torch.nn.functional.mse_loss if mode == RunMode.regression else torch.nn.functional.cross_entropy
    gin.bind_parameter("DLPredictionWrapper.loss", loss)
    num_classes = 1 if mode == RunMode.regression else 2
    for key, value in {**params, "num_classes": num_classes}.items():
        gin.bind_parameter(f"{model_cls.__name__}.{key}", value)
    kwargs = dict(model=model_cls, mode=mode, epochs=1, batch_size=4, cpu=True, num_workers=0, **kwargs)
    return train_common(data, log_dir=log_dir, **kwargs)


def make(name, columns=None, input_size=(2, 6, len(COLUMNS)), num_classes=2):
    torch.manual_seed(0)
    model_cls, params = MODELS[name]
    model = model_cls(input_size, num_classes=num_classes, **params).eval()
    model.set_trained_columns(columns or COLUMNS)
    return model


def recipe_batch(batch=2, hours=6):
    """Random inputs for COLUMNS with 0/1 indicators."""
    x = torch.randn(batch, hours, len(COLUMNS))
    x[..., 2:4] = (torch.rand(batch, hours, 2) < 0.5).float()
    return x


def test_pairs_columns_with_their_indicators():
    model = ObservationMaskProbe((1, 2, len(COLUMNS)), num_classes=2)
    x = torch.tensor([[[0.5, -1.0, 0.0, 1.0, 3.0], [0.5, 2.0, 1.0, 0.0, 4.0]]])
    # Until the names are known, every column counts as observed
    torch.testing.assert_close(model.observed_mask(x), torch.ones_like(x))
    model.set_trained_columns(COLUMNS)
    assert model.trained_columns == COLUMNS
    assert model.indicator_index.tolist() == [2, 3, -1, -1, -1]
    assert model.is_indicator.tolist() == [False, False, True, True, False]
    expected = torch.tensor([[[1.0, 0.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0, 1.0]]])
    torch.testing.assert_close(model.observed_mask(x), expected)


def test_rejects_names_that_do_not_match_the_input():
    with pytest.raises(ValueError, match="4 names for 5 input columns"):
        ObservationMaskProbe((1, 2, len(COLUMNS)), num_classes=2).set_trained_columns(COLUMNS[:-1])


def test_pairing_is_saved_with_the_model(tmp_path):
    model = make("probe")
    reloaded = ObservationMaskProbe((2, 6, len(COLUMNS)), num_classes=2)
    reloaded.load_state_dict(model.state_dict())
    assert reloaded.indicator_index.tolist() == [2, 3, -1, -1, -1]
    model.save_model(tmp_path, "probe")
    pickled = torch.load(tmp_path / "probe.ckpt", weights_only=False)
    assert pickled.is_indicator.tolist() == [False, False, True, True, False]


def test_mask_follows_the_dataset_column_order():
    # The dataset sorts the indicators behind the other columns; the pairing must follow the names, not the positions
    data = recipe_splits()
    dataset = PredictionPolarsDataset(data, split=Split.train, vars=VARS, ram_cache=False)
    names = dataset.get_feature_names()
    assert names[-1].startswith("MissingIndicator_") and not names[0].startswith("MissingIndicator_")
    x, _, _ = dataset[0]
    model = ObservationMaskProbe((1, *x.shape), num_classes=2)
    model.set_trained_columns(names)
    observed = model.observed_mask(x[None])[0]
    features = data[Split.train][Segment.features].filter(pl.col("stay_id") == 0)
    for name in ("hr", "map", "age"):
        measured = 1 - features[f"MissingIndicator_{name}"].to_numpy()
        torch.testing.assert_close(observed[: len(features), names.index(name)], torch.tensor(measured, dtype=torch.float))
    assert observed[:, model.is_indicator].eq(1).all()


@pytest.mark.parametrize("name", MODELS)
def test_forward_is_causal_in_values_and_mask(name):
    model = make(name)
    x = recipe_batch()
    out = model(x)
    assert out.shape == (2, 6, 2) and torch.isfinite(out).all()
    future = x.clone()
    future[:, 3:, :2] += 10
    future[:, 3:, 2:4] = 1 - future[:, 3:, 2:4]  # Flip which future values were measured
    changed = model(future)
    # Predictions up to hour 2 must not change when the values or the mask of later hours change
    torch.testing.assert_close(changed[:, :3], out[:, :3])
    assert not torch.allclose(changed[:, 3:], out[:, 3:])


@pytest.mark.parametrize("name", MODELS)
def test_padding_does_not_change_predictions(name):
    # YAIB pads every stay with zero rows up to the longest stay of the split
    model = make(name)
    x = recipe_batch(batch=1, hours=4)
    padded = torch.cat([x, torch.zeros(1, 3, len(COLUMNS))], dim=1)
    other = recipe_batch(batch=1, hours=7)
    torch.testing.assert_close(model(torch.cat([padded, other]))[:1, :4], model(x))


@pytest.mark.parametrize("hourly", [False, True], ids=["per_stay", "hourly"])
@pytest.mark.parametrize("name", MODELS)
def test_train_test_and_reload(name, hourly, tmp_path):
    model_cls, params = MODELS[name]
    data = recipe_splits(hourly=hourly)
    loss = train(model_cls, params, data, tmp_path, weight="balanced")
    metrics = json.loads((tmp_path / "test_metrics.json").read_text())
    assert np.isfinite(loss) and metrics["loss"] == pytest.approx(loss) and 0 <= metrics["AUC"] <= 1
    loaded = load_model(model_cls, tmp_path, mode=RunMode.classification)
    assert isinstance(loaded, model_cls)
    names = loaded.trained_columns
    assert [names[i] for i in loaded.indicator_index if i >= 0] == [
        "MissingIndicator_age",
        "MissingIndicator_hr",
        "MissingIndicator_map",
    ]
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    train(model_cls, params, data, eval_dir, source_dir=tmp_path, eval_only=True, load_weights=True)
    evaluated = json.loads((eval_dir / "test_metrics.json").read_text())
    assert evaluated["AUC"] == pytest.approx(metrics["AUC"])


@pytest.mark.parametrize("name", MODELS)
def test_regression(name, tmp_path):
    model_cls, params = MODELS[name]
    data = recipe_splits(hourly=True, regression=True)
    state = transform_outcomes(data, "label", "minmax", None, None)
    kwargs = dict(mode=RunMode.regression, target_transform=state)
    score = train(model_cls, params, data, tmp_path, **kwargs)
    assert np.isfinite(score) and score == pytest.approx(
        json.loads((tmp_path / "test_metrics.json").read_text())["MSE_native"]
    )
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    replay = train(model_cls, params, data, replay_dir, source_dir=tmp_path, eval_only=True, load_weights=True, **kwargs)
    assert replay == pytest.approx(score, rel=1e-5, abs=1e-5)
