"""SHAP explanations must be opt-in, and runs without them must not log aggregation errors (#226)."""

import json
import logging

import gin
import numpy as np
import polars as pl
import pytest

from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.constants import DataSegment as Segment, DataSplit as Split
from icu_benchmarks.models.ml_models.catboost import CBClassifier
from icu_benchmarks.models.ml_models.imblearn import BRFClassifier, RUSBClassifier
from icu_benchmarks.models.ml_models.lgbm import LGBMClassifier
from icu_benchmarks.models.ml_models.sklearn import LogisticRegression, RFClassifier
from icu_benchmarks.models.ml_models.xgboost import XGBClassifier
from icu_benchmarks.models.train import train_common
from icu_benchmarks.run_utils import aggregate_results, build_parser

VARS = {"GROUP": "stay_id", "SEQUENCE": "time", "LABEL": "label", "DYNAMIC": ["x", "noise"], "STATIC": []}


def classification_splits():
    rng = np.random.default_rng(0)
    data = {}
    for split in (Split.train, Split.val, Split.test):
        x = rng.normal(size=60)
        features = pl.DataFrame({"stay_id": range(60), "time": [0] * 60, "x": x, "noise": rng.normal(size=60)})
        data[split] = {
            Segment.features: features,
            Segment.outcome: pl.DataFrame({"stay_id": range(60), "label": (x > 0).astype(int)}),
        }
    return data


@pytest.fixture(autouse=True)
def clean_gin(tmp_path, monkeypatch):
    # CatBoost writes catboost_info/ to the working directory
    monkeypatch.chdir(tmp_path)
    gin.clear_config()
    gin.bind_parameter("PredictionPolarsDataset.vars", VARS)
    yield
    gin.clear_config()


def train(model_cls, log_dir, explain_features):
    kwargs = dict(model=model_cls, mode=RunMode.classification, cpu=True, num_workers=0, verbose=False)
    return train_common(classification_splits(), log_dir=log_dir, explain_features=explain_features, **kwargs)


@pytest.mark.parametrize("model_cls", [XGBClassifier, LGBMClassifier, CBClassifier, RFClassifier, BRFClassifier])
def test_explain_features_saves_test_shap_values(model_cls, tmp_path):
    train(model_cls, tmp_path, explain_features=True)
    shaps = pl.read_parquet(tmp_path / "test_shap_values.parquet")
    assert sorted(shaps.columns) == ["noise", "x"]
    assert shaps.height == 60
    # The label depends on x only
    importance = shaps.select(pl.all().abs().mean()).row(0, named=True)
    assert importance["x"] > importance["noise"]


def test_no_shap_values_by_default(tmp_path):
    train(XGBClassifier, tmp_path, explain_features=False)
    assert (tmp_path / "test_metrics.json").exists()
    assert not (tmp_path / "test_shap_values.parquet").exists()


@pytest.mark.parametrize("model_cls", [LogisticRegression, RUSBClassifier])
def test_unsupported_model_fails_before_training(model_cls, tmp_path):
    with pytest.raises(NotImplementedError, match="does not support SHAP"):
        train(model_cls, tmp_path, explain_features=True)
    assert not (tmp_path / "last.joblib").exists()


def test_cli_flag():
    required = ["-d", "data", "-t", "Mortality24"]
    assert build_parser().parse_args([*required, "--explain-features"]).explain_features
    assert not build_parser().parse_args(required).explain_features


@pytest.mark.parametrize("with_shap", [False, True])
def test_aggregate_results_shap(with_shap, tmp_path, caplog):
    for fold in ("fold_0", "fold_1"):
        fold_dir = tmp_path / "repetition_0" / fold
        fold_dir.mkdir(parents=True)
        (fold_dir / "test_metrics.json").write_text(json.dumps({"AUC": 0.5}))
        if with_shap:
            pl.DataFrame({"x": [1.0, 2.0]}).write_parquet(fold_dir / "test_shap_values.parquet")
    with caplog.at_level(logging.ERROR):
        aggregate_results(tmp_path)
    assert not [r for r in caplog.records if "SHAP" in r.getMessage()]
    aggregated = tmp_path / "aggregated_shap_values.parquet"
    assert aggregated.exists() == with_shap
    if with_shap:
        assert pl.read_parquet(aggregated).height == 4
