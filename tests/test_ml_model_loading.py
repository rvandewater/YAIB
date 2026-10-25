"""Saved ML models must be re-loadable for evaluation (#204, #205, #206)."""

import json

import gin
import numpy as np
import polars as pl
import pytest
from joblib import dump
from sklearn.linear_model import LogisticRegression as SkLogisticRegression

from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.constants import DataSegment as Segment, DataSplit as Split
from icu_benchmarks.models.ml_models.sklearn import LogisticRegression
from icu_benchmarks.models.ml_models.xgboost import XGBClassifier
from icu_benchmarks.models.train import load_model, train_common
from icu_benchmarks.models.wrappers import MLWrapper

VARS = {"GROUP": "stay_id", "SEQUENCE": "time", "LABEL": "label", "DYNAMIC": ["x"], "STATIC": []}


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


@pytest.mark.parametrize("model_cls", [XGBClassifier, LogisticRegression])
def test_evaluate_saved_ml_classifier(model_cls, tmp_path):
    gin.clear_config()
    gin.bind_parameter("PredictionPolarsDataset.vars", VARS)
    data = classification_splits()
    kwargs = dict(model=model_cls, mode=RunMode.classification, cpu=True, num_workers=0, verbose=False)
    try:
        score = train_common(data, log_dir=tmp_path, **kwargs)
        # Training writes last.joblib; evaluation must find it (#204, #205).
        assert (tmp_path / "last.joblib").exists() and not (tmp_path / "model.joblib").exists()
        loaded = load_model(model_cls, tmp_path, mode=RunMode.classification)
        # The bare estimator is wrapped again, so wrapper methods such as set_weight exist (#206).
        assert isinstance(loaded, model_cls)
        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        replay = train_common(data, log_dir=eval_dir, source_dir=tmp_path, eval_only=True, load_weights=True, **kwargs)
        assert replay == pytest.approx(score)
        original = json.loads((tmp_path / "test_metrics.json").read_text())
        evaluated = json.loads((eval_dir / "test_metrics.json").read_text())
        assert evaluated["AUC"] == pytest.approx(original["AUC"])
    finally:
        gin.clear_config()


def test_model_joblib_preferred_over_last_joblib(tmp_path):
    dump(SkLogisticRegression(C=1.0), tmp_path / "last.joblib")
    dump(SkLogisticRegression(C=2.0), tmp_path / "model.joblib")
    loaded = load_model(LogisticRegression, tmp_path, mode=RunMode.classification)
    assert isinstance(loaded, MLWrapper) and loaded.model.C == 2.0
