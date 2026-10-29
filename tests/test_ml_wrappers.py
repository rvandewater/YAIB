"""Every registered ML wrapper must be constructible and fit through train_common (#239)."""

import gin
import numpy as np
import polars as pl
import pytest

import icu_benchmarks.models.ml_models.catboost  # noqa: F401
import icu_benchmarks.models.ml_models.imblearn  # noqa: F401
import icu_benchmarks.models.ml_models.lgbm  # noqa: F401
import icu_benchmarks.models.ml_models.sklearn  # noqa: F401
import icu_benchmarks.models.ml_models.xgboost  # noqa: F401
from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.constants import DataSegment as Segment, DataSplit as Split
from icu_benchmarks.models.ml_models.sklearn import SVMClassifier, SVMRegressor
from icu_benchmarks.models.train import train_common
from icu_benchmarks.models.wrappers import MLWrapper

VARS = {"GROUP": "stay_id", "SEQUENCE": "time", "LABEL": "label", "DYNAMIC": ["x"], "STATIC": []}


def all_subclasses(cls):
    return {sub for direct in cls.__subclasses__() for sub in (direct, *all_subclasses(direct))}


# Concrete models only; shared bases such as LGBMWrapper have no estimator
ML_MODELS = sorted(
    (cls for cls in all_subclasses(MLWrapper) if not cls.__name__.endswith("Wrapper")), key=lambda cls: cls.__name__
)


@pytest.mark.parametrize("model_cls", ML_MODELS, ids=lambda cls: cls.__name__)
def test_ml_wrapper_constructs(model_cls):
    model = model_cls(run_mode=model_cls._supported_run_modes[0])
    assert model.model is not None


def classification_splits():
    rng = np.random.default_rng(0)
    data = {}
    for split in (Split.train, Split.val, Split.test):
        x = rng.normal(size=40)
        data[split] = {
            Segment.features: pl.DataFrame({"stay_id": range(40), "time": [0] * 40, "x": x}),
            Segment.outcome: pl.DataFrame({"stay_id": range(40), "label": (x > 0).astype(int)}),
        }
    return data


def test_svm_classifier_trains(tmp_path):
    gin.clear_config()
    gin.bind_parameter("PredictionPolarsDataset.vars", VARS)
    try:
        loss = train_common(classification_splits(), log_dir=tmp_path, model=SVMClassifier, cpu=True, num_workers=0)
        assert np.isfinite(loss)
    finally:
        gin.clear_config()


def test_svm_regressor_predicts():
    model = SVMRegressor(run_mode=RunMode.regression)
    x = np.linspace(0, 1, 20).reshape(-1, 1)
    model.model.fit(x, 2 * x[:, 0])
    assert model.predict(x).shape == (20,)
