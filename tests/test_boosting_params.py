"""XGBoost and LightGBM wrappers must pass their configured parameters to the estimator (#234)."""

import gin
import numpy as np
import pytest
import torch

from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.ml_models.lgbm import LGBMClassifier, LGBMRegressor
from icu_benchmarks.models.ml_models.xgboost import XGBClassifier


@pytest.fixture(autouse=True)
def clean_gin():
    gin.clear_config()
    yield
    gin.clear_config()


def test_xgb_hyperparameters_reach_estimator():
    # XGBClassifier.__init__ only declares **kwargs, which used to drop every hyperparameter.
    gin.bind_parameter("XGBClassifier.max_depth", 3)
    gin.bind_parameter("XGBClassifier.eval_metric", "aucpr")
    params = XGBClassifier(run_mode=RunMode.classification, n_estimators=50).model.get_params()
    assert (params["max_depth"], params["eval_metric"], params["n_estimators"]) == (3, "aucpr", 50)
    assert (params["device"], params["verbosity"]) == ("cpu", 0)


def test_xgb_device_bindable_through_gin():
    gin.bind_parameter("XGBClassifier.device", "cuda")
    assert XGBClassifier(run_mode=RunMode.classification).model.get_params()["device"] == "cuda"


@pytest.mark.parametrize(
    ("device", "cpu", "cuda_available", "expected"),
    [
        ("auto", False, True, "cuda"),
        ("auto", False, False, "cpu"),
        ("auto", True, True, "cpu"),
        ("cuda", True, True, "cpu"),
        ("cuda:1", False, False, "cuda:1"),
    ],
)
def test_xgb_resolve_device(monkeypatch, device, cpu, cuda_available, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    assert XGBClassifier.resolve_device(device, cpu=cpu) == expected


@pytest.mark.parametrize(
    ("model_cls", "mode"), [(LGBMClassifier, RunMode.classification), (LGBMRegressor, RunMode.regression)]
)
def test_lgbm_quiet_by_default(model_cls, mode):
    assert model_cls(run_mode=mode).model.get_params()["verbosity"] == -1
    gin.bind_parameter(f"{model_cls.__name__}.verbosity", 1)
    assert model_cls(run_mode=mode).model.get_params()["verbosity"] == 1


def test_xgb_early_stopping_uses_patience():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3))
    y = (x[:, 0] > 0).astype(int)
    gin.bind_parameter("MLWrapper.patience", 2)
    model = XGBClassifier(run_mode=RunMode.classification, n_estimators=500, learning_rate=0.5)
    model.fit_model(x[:100], y[:100], x[100:], y[100:])
    assert model.model.best_iteration + 1 < 500
