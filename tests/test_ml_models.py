"""The ML wrappers must train, score and save through train_common on an easy synthetic task."""

import json
import pickle

import gin
import numpy as np
import polars as pl
import pytest
from sklearn.metrics import mean_squared_error

from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.constants import DataSegment as Segment, DataSplit as Split
from icu_benchmarks.data.loader import PredictionPolarsDataset
from icu_benchmarks.data.target_transform import transform_outcomes
from icu_benchmarks.models.ml_models.catboost import CBClassifier
from icu_benchmarks.models.ml_models.imblearn import BRFClassifier, RUSBClassifier
from icu_benchmarks.models.ml_models.lgbm import LGBMClassifier, LGBMRegressor
from icu_benchmarks.models.ml_models.sklearn import (
    ElasticNet,
    LogisticRegression,
    MLPClassifier,
    MLPRegressor,
    PerceptronClassifier,
    RFClassifier,
)
from icu_benchmarks.models.ml_models.xgboost import XGBClassifier
from icu_benchmarks.models.train import train_common
from icu_benchmarks.models.wrappers import MLWrapper

VARS = {"GROUP": "stay_id", "SEQUENCE": "time", "LABEL": "label", "DYNAMIC": ["x", "noise"], "STATIC": []}
STAYS = 60


@pytest.fixture(autouse=True)
def clean_gin():
    gin.clear_config()
    gin.bind_parameter("PredictionPolarsDataset.vars", VARS)
    yield
    gin.clear_config()


def splits(regression: bool) -> dict:
    rng = np.random.default_rng(0)
    data = {}
    for split in (Split.train, Split.val, Split.test):
        x = rng.normal(size=STAYS)
        label = 5 * x + 20 + rng.normal(scale=2, size=STAYS) if regression else (x > 0).astype(int)
        data[split] = {
            Segment.features: pl.DataFrame(
                {"stay_id": range(STAYS), "time": [0] * STAYS, "x": x, "noise": rng.normal(size=STAYS)}
            ),
            Segment.outcome: pl.DataFrame({"stay_id": range(STAYS), "label": label}),
        }
    return data


def train(model_cls, tmp_path, mode, hyperparams, data, **kwargs):
    for key, value in hyperparams.items():
        gin.bind_parameter(f"{model_cls.__name__}.{key}", value)
    loss = train_common(data, log_dir=tmp_path, model=model_cls, mode=mode, cpu=True, num_workers=0, **kwargs)
    return loss, json.loads((tmp_path / "test_metrics.json").read_text())


CLASSIFIERS = [
    (LogisticRegression, {"max_iter": 200}),
    (RFClassifier, {"n_estimators": 10, "random_state": 0}),
    (BRFClassifier, {"n_estimators": 10, "random_state": 0}),
    (RUSBClassifier, {"n_estimators": 5, "random_state": 0}),
    (LGBMClassifier, {"n_estimators": 10, "min_child_samples": 2}),
    (CBClassifier, {"iterations": 10, "verbose": False, "random_seed": 0, "allow_writing_files": False}),
    (MLPClassifier, {"hidden_layer_sizes": (4,), "max_iter": 300, "random_state": 0}),
    (PerceptronClassifier, {"hidden_layer_sizes": (2,), "max_iter": 300, "random_state": 0}),
    (XGBClassifier, {"n_estimators": 10, "max_depth": 2}),
]


@pytest.mark.parametrize("model_cls,hyperparams", CLASSIFIERS, ids=[c.__name__ for c, _ in CLASSIFIERS])
def test_classifiers_learn_separable_task(model_cls, hyperparams, tmp_path):
    loss, metrics = train(model_cls, tmp_path, RunMode.classification, hyperparams, splits(regression=False))
    assert metrics["loss"] == pytest.approx(loss) and np.isfinite(loss)
    assert metrics["AUC"] > 0.85 and metrics["PR"] > 0.75
    assert (tmp_path / "last.joblib").exists()


REGRESSORS = [
    (ElasticNet, {"alpha": 0.001}, 0.6),
    (LGBMRegressor, {"n_estimators": 20, "min_child_samples": 2}, 0.6),
    # Not gin-configurable, so it trains with the sklearn defaults and an unseeded init; its fit varies across platforms
    (MLPRegressor, {}, 0.3),
]


@pytest.mark.parametrize("model_cls,hyperparams,min_r2", REGRESSORS, ids=[c.__name__ for c, _, _ in REGRESSORS])
def test_regressors_report_native_units(model_cls, hyperparams, min_r2, tmp_path):
    gin.bind_parameter("MLWrapper.loss", mean_squared_error)
    data = splits(regression=True)
    target_transform = transform_outcomes(data, "label", "minmax", None, None)
    loss, metrics = train(model_cls, tmp_path, RunMode.regression, hyperparams, data, target_transform=target_transform)
    assert loss == pytest.approx(metrics["MSE_native"])
    # The label noise has a standard deviation of 2: the errors are in label units, not in the scaled [0, 1] ones
    assert metrics["R2"] > min_r2
    assert metrics["RMSE_native"] == pytest.approx(np.sqrt(metrics["MSE_native"]), rel=1e-4)
    assert 1 < metrics["MAE_native"] < 4


def test_set_model_args_drops_unknown_hyperparameters():
    model = LogisticRegression(C=0.5, n_estimators=3)
    assert model.model.C == 0.5 and "n_estimators" not in model.model.get_params()


def test_wrappers_reject_unsupported_run_modes():
    with pytest.raises(ValueError, match="not supported"):
        LogisticRegression(run_mode=RunMode.regression)
    with pytest.raises(ValueError, match="not supported"):
        LGBMRegressor(run_mode=RunMode.classification)


def test_train_only_skips_testing(tmp_path):
    loss = train_common(
        splits(regression=False), log_dir=tmp_path, model=LogisticRegression, train_only=True, cpu=True, num_workers=0
    )
    assert loss == 0 and (tmp_path / "last.joblib").exists() and not (tmp_path / "test_metrics.json").exists()


def test_regression_requires_target_transform(tmp_path):
    with pytest.raises(ValueError, match="target transform"):
        train_common(splits(regression=True), log_dir=tmp_path, model=ElasticNet, mode=RunMode.regression)


def hourly_splits(stays=20, hours=3) -> dict:
    """Hourly labels that depend on the current value of x."""
    rng = np.random.default_rng(0)
    data = {}
    for split in (Split.train, Split.val, Split.test):
        x = rng.normal(size=stays * hours)
        ids, time = np.repeat(np.arange(stays), hours), np.tile(np.arange(hours), stays)
        data[split] = {
            Segment.features: pl.DataFrame({"stay_id": ids, "time": time, "x": x, "noise": rng.normal(size=stays * hours)}),
            Segment.outcome: pl.DataFrame({"stay_id": ids, "time": time, "label": (x > 0).astype(int)}),
        }
    return data


def test_debug_saves_hourly_class_probabilities(tmp_path, monkeypatch):
    monkeypatch.setattr(MLWrapper, "debug", True)
    data = hourly_splits()
    train(LogisticRegression, tmp_path, RunMode.classification, {}, data)
    saved = np.loadtxt(tmp_path / "pred_indicators.csv", delimiter=",")
    # One row per hour: stay id, hour, label and the probability of each class
    test = data[Split.test]
    assert saved.shape == (len(test[Segment.outcome]), 5)
    np.testing.assert_array_equal(saved[:, :2], test[Segment.features].select("stay_id", "time").to_numpy())
    np.testing.assert_array_equal(saved[:, 2], test[Segment.outcome]["label"].to_numpy())
    np.testing.assert_allclose(saved[:, 3] + saved[:, 4], 1)
    assert ((saved[:, 4] > 0.5) == saved[:, 2]).mean() > 0.85


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#275: The probabilities of one-row-per-stay tasks are not saved: their indicators have 1 column",
)
def test_debug_saves_per_stay_class_probabilities(tmp_path, monkeypatch):
    monkeypatch.setattr(MLWrapper, "debug", True)
    train(LogisticRegression, tmp_path, RunMode.classification, {}, splits(regression=False))
    assert (tmp_path / "pred_indicators.csv").exists()


def test_debug_saves_regression_predictions_in_label_units(tmp_path, monkeypatch):
    monkeypatch.setattr(MLWrapper, "debug", True)
    gin.bind_parameter("MLWrapper.loss", mean_squared_error)
    data = splits(regression=True)
    labels = data[Split.test][Segment.outcome]["label"].to_numpy()
    target_transform = transform_outcomes(data, "label", "minmax", None, None)
    _, metrics = train(ElasticNet, tmp_path, RunMode.regression, {"alpha": 0.001}, data, target_transform=target_transform)
    saved = np.loadtxt(tmp_path / "pred_indicators.csv", delimiter=",")
    # Stay id, label and prediction, in label units
    np.testing.assert_array_equal(saved[:, 0], np.arange(STAYS))
    np.testing.assert_allclose(saved[:, 1], labels, rtol=1e-5)
    assert mean_squared_error(saved[:, 1], saved[:, 2]) == pytest.approx(metrics["MSE_native"], rel=1e-4)


def datasets(data) -> tuple:
    return tuple(PredictionPolarsDataset(data, split=split, vars=VARS) for split in (Split.train, Split.val, Split.test))


def test_fitted_wrapper_pickles_without_transforms():
    train_set, val_set, test_set = datasets(splits(regression=False))
    model = LogisticRegression(run_mode=RunMode.classification)
    model.fit(train_set, val_set)
    features = test_set.get_data_and_labels()[0]
    # The fitted transforms are lambdas, which cannot be pickled
    restored = pickle.loads(pickle.dumps(model))
    assert not hasattr(restored, "output_transform") and hasattr(model, "output_transform")
    np.testing.assert_array_equal(restored.predict(features), model.predict(features))


@pytest.mark.xfail(
    strict=True, raises=TypeError, reason="#275: validation_step passes the val loss to log_metrics instead of log"
)
def test_validation_step_logs_loss_and_metrics(monkeypatch):
    train_set, val_set, _ = datasets(splits(regression=False))
    model = LogisticRegression(run_mode=RunMode.classification)
    model.fit(train_set, val_set)
    logged = {}
    monkeypatch.setattr(model, "log", lambda name, value, **kwargs: logged.update({name: value}))
    monkeypatch.setattr(model, "log_dict", lambda values, **kwargs: logged.update(values))
    model.validation_step(val_set, 0)
    assert {"val/loss", "val/AUC"} <= set(logged)
