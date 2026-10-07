"""Regression target units must survive splits, caches, metrics and model reloads."""

import json
from dataclasses import replace
from types import SimpleNamespace

import gin
import numpy as np
import pandas as pd
import polars as pl
import pytest
import torch
from numpy.testing import assert_allclose
from sklearn.metrics import mean_squared_error

from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.constants import DataSegment as Segment, DataSplit as Split
from icu_benchmarks.data.loader import PredictionPandasDataset, PredictionPolarsDataset
from icu_benchmarks.data.preprocessor import PandasRegressionPreprocessor, PolarsRegressionPreprocessor
from icu_benchmarks.data.split_process_data import preprocess_data
from icu_benchmarks.data.target_transform import TargetTransform, transform_outcomes
from icu_benchmarks.models.dl_models.rnn import GRUNet
from icu_benchmarks.models.ml_models.sklearn import LinearRegression
from icu_benchmarks.models.train import load_model, persist_shap_data, train_common
from icu_benchmarks.models.wrappers import DLPredictionWrapper, MLWrapper

VARS = {"GROUP": "stay_id", "SEQUENCE": "time", "LABEL": "label", "DYNAMIC": ["x"], "STATIC": []}


def outcome_splits(frame=pl.DataFrame):
    return {
        split: {Segment.outcome: frame({"stay_id": [1, 2], "label": values})}
        for split, values in zip((Split.train, Split.val, Split.test), ([0.0, 10.0], [10.0, 20.0], [20.0, 30.0]))
    }


@pytest.mark.parametrize("frame", [pl.DataFrame, pd.DataFrame])
def test_fit_training_only_and_reuse_source(frame, monkeypatch):
    data = outcome_splits(frame)
    state = transform_outcomes(data, "label", "minmax", None, None)
    for split, expected in zip(data, ([0, 1], [1, 2], [2, 3])):
        assert_allclose(data[split][Segment.outcome]["label"].to_numpy(), expected)
    assert state.minimum == 0 and state.maximum == 10
    monkeypatch.setattr(TargetTransform, "fit", lambda *a, **k: pytest.fail("Refitted source transform"))
    shifted = outcome_splits(frame)
    transform_outcomes(shifted, "label", "minmax", None, None, state)
    assert_allclose(shifted[Split.test][Segment.outcome]["label"].to_numpy(), [2, 3])
    with pytest.raises(ValueError, match="does not match"):
        transform_outcomes(shifted, "label", "minmax", None, None, replace(state, label="different"))


@pytest.mark.parametrize("values", [[-5.0, 10.0], [7.0, 7.0], [0.0, np.nan, 10.0]])
@pytest.mark.parametrize("policy,bounds", [("minmax", (None, None)), ("none", (None, None)), ("fixed", (0, 168))])
def test_affine_round_trip(values, policy, bounds, tmp_path):
    state = TargetTransform.fit(values, "label", policy, *bounds)
    values = np.array([-20.0, 0.0, 3.0, 200.0, np.nan])
    assert_allclose(state.inverse_transform(state.transform(values)), values)
    tensor = torch.tensor(values, dtype=torch.float64)
    torch.testing.assert_close(state.inverse_transform(state.transform(tensor)), tensor, equal_nan=True)
    state.save(tmp_path)
    assert TargetTransform.load(tmp_path) == state


@pytest.mark.parametrize(
    "policy,lower,upper",
    [("fixed", None, 2), ("fixed", 2, 2), ("fixed", 3, 2), ("fixed", 0, np.inf), ("none", 0, 2), ("other", None, None)],
)
def test_invalid_policy(policy, lower, upper):
    for preprocessor in (PolarsRegressionPreprocessor, PandasRegressionPreprocessor):
        with pytest.raises(ValueError):
            preprocessor(target_scaling=policy, outcome_min=lower, outcome_max=upper)


def test_missing_and_infinite_targets():
    for values in ([np.nan], [1, np.inf], []):
        with pytest.raises(ValueError):
            TargetTransform.fit(values, "label")
    data = outcome_splits()
    data[Split.test][Segment.outcome] = pl.DataFrame({"stay_id": [1, 2], "label": [np.inf, 2.0]})
    with pytest.raises(ValueError, match="Infinite"):
        transform_outcomes(data, "label", "minmax", None, None)


@pytest.mark.parametrize(
    "frame,preprocessor", [(pl.DataFrame, PolarsRegressionPreprocessor), (pd.DataFrame, PandasRegressionPreprocessor)]
)
def test_preprocessor_and_masks(frame, preprocessor, monkeypatch):
    # Isolate target preprocessing from the unrelated feature recipes.
    monkeypatch.setattr(preprocessor.__bases__[0], "apply", lambda self, data, vars: data)
    data = outcome_splits(frame)
    for split in data.values():
        split[Segment.dynamic] = frame({"stay_id": [1, 2], "time": [0, 0], "x": [1.0, np.nan]})
    processor = preprocessor(scaling=False, generate_features=False, use_static_features=False)
    processed = processor.apply(data, VARS)
    assert_allclose(processed[Split.test][Segment.outcome]["label"].to_numpy(), [2, 3])
    assert processor.target_transform.maximum == 10
    first = preprocessor(target_scaling="fixed", outcome_min=0, outcome_max=15)
    second = preprocessor(target_scaling="fixed", outcome_min=0, outcome_max=168)
    assert first.to_cache_string() != second.to_cache_string()

    # Missing labels and padding must be excluded identically by ML and DL loaders.
    data = {
        Split.train: {
            Segment.features: frame({"stay_id": [1, 1, 2], "time": [0, 1, 0], "x": [1.0, 2.0, 3.0]}),
            Segment.outcome: frame({"stay_id": [1, 1, 2], "label": [0.0, np.nan, 10.0]}),
        }
    }
    dataset_cls = PredictionPolarsDataset if frame is pl.DataFrame else PredictionPandasDataset
    dataset = dataset_cls(data, vars=VARS, ram_cache=False)
    _, labels, _ = dataset.get_data_and_labels()
    assert_allclose(labels, [0, 10])
    assert sum(int(dataset[i][2].sum()) for i in range(len(dataset))) == 2


@pytest.mark.parametrize("policy,bounds", [("minmax", (None, None)), ("none", (None, None)), ("fixed", (0, 168))])
def test_ml_dl_native_metrics(policy, bounds):
    state = TargetTransform.fit([0, 10], "label", policy, *bounds)
    labels = state.transform(np.array([2.0, 10.0]))
    predictions = state.transform(np.array([4.0, 12.0]))
    ml = MLWrapper(run_mode=RunMode.regression, loss=mean_squared_error)
    ml.target_transform = state
    ml.set_metrics(labels)
    assert ml.metrics["MAE_native"](ml.label_transform(labels), ml.output_transform(predictions)) == pytest.approx(2)
    assert ml.metrics["MSE_native"](ml.label_transform(labels), ml.output_transform(predictions)) == pytest.approx(4)
    dl = DLPredictionWrapper(run_mode=RunMode.regression, loss=torch.nn.functional.mse_loss)
    dl.target_transform = state
    metrics = dl.set_metrics()
    output = dl.output_transform((torch.tensor(predictions).reshape(-1, 1), torch.tensor(labels)))
    for name, expected in (("MAE_native", 2), ("MSE_native", 4)):
        metric = metrics[name]()
        metric.update(output)
        assert metric.compute() == pytest.approx(expected)
    assert mean_squared_error(labels, predictions) == pytest.approx(4 * state.scale**2)


def test_dl_masked_metrics(monkeypatch):
    model = DLPredictionWrapper(run_mode=RunMode.regression, loss=torch.nn.functional.mse_loss)
    model.target_transform = TargetTransform.fit([0, 10], "label")
    model.metrics = {"test": {name: metric() for name, metric in model.set_metrics().items()}}
    monkeypatch.setattr(model, "forward", lambda x: x)
    monkeypatch.setattr(model, "log", lambda *a, **k: None)
    batch = (torch.tensor([[[0.2], [999.0]]]), torch.tensor([[0.0, -1.0]]), torch.tensor([[True, False]]))
    assert model.step_fn(batch, "test").item() == pytest.approx(0.04)
    assert model.metrics["test"]["MAE_native"].compute() == pytest.approx(2)
    assert model.metrics["test"]["MSE_native"].compute() == pytest.approx(4)
    assert model.step_fn((*batch[:2], torch.zeros_like(batch[2])), "test").item() == 0
    assert model.metrics["test"]["MSE_native"].compute() == pytest.approx(4)


def test_cache_and_missing_labels(tmp_path):
    gin.clear_config()
    raw = {
        Segment.dynamic: pl.DataFrame({"stay_id": range(24), "time": [0] * 24, "x": range(24)}),
        Segment.outcome: pl.DataFrame({"stay_id": range(24), "label": [None] + list(range(1, 24))}),
    }
    files = {Segment.dynamic: "dyn.parquet", Segment.outcome: "outc.parquet"}
    for segment, filename in files.items():
        raw[segment].write_parquet(tmp_path / filename)
    kwargs = dict(
        data_dir=tmp_path,
        file_names=files,
        vars=VARS,
        preprocessor=PolarsRegressionPreprocessor,
        required_segments=list(files),
        cv_repetitions=2,
        cv_folds=2,
        runmode=RunMode.regression,
    )
    first, state = preprocess_data(**kwargs, generate_cache=True)
    cached, cached_state = preprocess_data(**kwargs, load_cache=True)
    assert cached_state == state
    for split in first:
        for segment in first[split]:
            assert first[split][segment].equals(cached[split][segment])
    assert sum(first[s][Segment.outcome]["label"].null_count() for s in first) == 1
    state.save(tmp_path)
    changed = TargetTransform.fit([0, 168], "label", "fixed", 0, 168)
    _, loaded = preprocess_data(**kwargs, load_cache=True, generate_cache=True, target_transform=changed)
    assert loaded == changed
    assert len(list((tmp_path / "cache").iterdir())) == 2
    assert len(list((tmp_path / "preproc").iterdir())) == 2
    preprocess_data(**{**kwargs, "cv_folds": 3}, load_cache=True, generate_cache=True)
    assert len(list((tmp_path / "cache").iterdir())) == 3


@pytest.mark.parametrize("model_cls", [LinearRegression, GRUNet])
@pytest.mark.parametrize("upper", [15.0, 168.0])
def test_training_and_reload(model_cls, upper, tmp_path, monkeypatch):
    gin.clear_config()
    gin.bind_parameter("PredictionPolarsDataset.vars", VARS)
    gin.bind_parameter("GRUNet.hidden_dim", 4)
    gin.bind_parameter("GRUNet.layer_dim", 1)
    gin.bind_parameter("GRUNet.num_classes", 1)
    gin.bind_parameter("DLPredictionWrapper.loss", torch.nn.functional.mse_loss)
    gin.bind_parameter("MLWrapper.loss", mean_squared_error)
    data = {}
    for split in (Split.train, Split.val, Split.test):
        x = np.linspace(0, 1, 8)
        data[split] = {
            Segment.features: pl.DataFrame({"stay_id": range(8), "time": [0] * 8, "x": x}),
            Segment.outcome: pl.DataFrame({"stay_id": range(8), "label": x * upper}),
        }
    state = transform_outcomes(data, "label", "minmax", None, None)
    kwargs = dict(
        model=model_cls,
        mode=RunMode.regression,
        target_transform=state,
        epochs=1,
        batch_size=2,
        cpu=True,
        num_workers=0,
        weight="",
        verbose=False,
    )
    try:
        score = train_common(data, log_dir=tmp_path, **kwargs)
        metrics = json.loads((tmp_path / "test_metrics.json").read_text())
        assert score == pytest.approx(metrics["MSE_native"])
        assert np.isfinite(score)
        loaded = load_model(model_cls, tmp_path)
        assert loaded.target_transform == state
        replay_dir = tmp_path / "replay"
        replay_dir.mkdir()
        monkeypatch.setattr(TargetTransform, "fit", lambda *a, **k: pytest.fail("Refitted during evaluation"))
        replay = train_common(data, log_dir=replay_dir, source_dir=tmp_path, eval_only=True, load_weights=True, **kwargs)
        assert replay == pytest.approx(score, rel=1e-5, abs=1e-5)
        replace(state, offset=state.offset + 1).save(tmp_path)
        with pytest.raises(ValueError, match="do not match"):
            load_model(model_cls, tmp_path)
        (tmp_path / "target_transform.json").unlink()
        with pytest.raises(ValueError, match="Missing"):
            load_model(model_cls, tmp_path)
    finally:
        gin.clear_config()


def test_cv_reuses_source_state_and_averages_native_scores(tmp_path, monkeypatch):
    import icu_benchmarks.cross_validation as cv

    state = TargetTransform.fit([0, 168], "label")
    state.save(tmp_path)
    seen = []

    def preprocess(*args, **kwargs):
        assert kwargs["target_transform"] == state
        return kwargs["fold_index"], state

    def train(fold, **kwargs):
        assert kwargs["target_transform"] == state
        assert kwargs["test_on"] == "val"
        seen.append(fold)
        return [4.0, 16.0][fold]

    monkeypatch.setattr(cv, "preprocess_data", preprocess)
    monkeypatch.setattr(cv, "train_common", train)
    score = cv.execute_repeated_cv(
        tmp_path,
        tmp_path / "run",
        seed=42,
        mode=RunMode.regression,
        load_weights=True,
        source_dir=tmp_path,
        cv_repetitions_to_train=1,
        cv_folds_to_train=2,
        test_on="val",
    )
    assert seen == [0, 1]
    assert score == 10


def test_native_prediction_and_shap_exports(tmp_path):
    state = TargetTransform.fit([10, 20], "label")
    model = MLWrapper(run_mode=RunMode.regression)
    model.target_transform = state
    explanation = SimpleNamespace(values=np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]), base_values=np.ones(3))
    model.test_shap_values = explanation
    model.trained_columns = ["x", "y"]
    persist_shap_data(SimpleNamespace(lightning_module=model), tmp_path)
    assert_allclose(pl.read_parquet(tmp_path / "test_shap_values.parquet").to_numpy(), explanation.values / state.scale)
    assert_allclose(np.load(tmp_path / "test_shap_base_values.npy"), state.inverse_transform(explanation.base_values))
    model._trainer = SimpleNamespace(logger=SimpleNamespace(save_dir=tmp_path))
    model._save_model_outputs(np.array([[1, 0], [2, 0]]), np.array([0.2, 0.4]), np.array([0.1, 0.3]))
    assert_allclose(np.loadtxt(tmp_path / "pred_indicators.csv", delimiter=","), [[1, 0, 11, 12], [2, 0, 13, 14]])
