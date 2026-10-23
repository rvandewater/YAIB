"""SHAP values of the ML models must be exported under the names of the features the model was trained on."""

from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import shap
import xgboost as xgb

from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.constants import DataSegment as Segment, DataSplit as Split
from icu_benchmarks.data.loader import PredictionPolarsDataset
from icu_benchmarks.models.train import persist_shap_data
from icu_benchmarks.models.wrappers import MLWrapper

VARS = {"GROUP": "stay_id", "SEQUENCE": "time", "LABEL": "label", "DYNAMIC": ["x", "noise"], "STATIC": []}
STAYS, HOURS = 40, 3


def synthetic_data(hourly: bool) -> dict:
    """Features as the preprocessor leaves them for the ML models (with the hour index), label driven by x only."""
    rng = np.random.default_rng(0)
    stay_id = np.repeat(np.arange(STAYS), HOURS)
    time = np.tile(np.arange(HOURS), STAYS)
    x = rng.normal(size=STAYS * HOURS)
    features = pl.DataFrame(
        {"stay_id": stay_id, "time": time, "x": x, "noise": rng.normal(size=STAYS * HOURS), "counter": time + 1.0}
    )
    label = (x > 0).astype(float)
    if hourly:
        outcome = pl.DataFrame({"stay_id": stay_id, "time": time, "label": label})
    else:
        # One label per stay, aligned with the last hour that get_data_and_labels keeps
        outcome = pl.DataFrame({"stay_id": np.arange(STAYS), "label": label[HOURS - 1 :: HOURS]})
    return {Split.train: {Segment.features: features, Segment.outcome: outcome}}


@pytest.mark.parametrize("hourly", [False, True], ids=["per_stay", "hourly"])
def test_shap_export_uses_model_feature_names(hourly, tmp_path):
    dataset = PredictionPolarsDataset(synthetic_data(hourly), vars=VARS, ram_cache=False)
    rep, labels, _ = dataset.get_data_and_labels()
    names = dataset.get_feature_names()
    assert len(names) == rep.shape[1]
    assert "stay_id" not in names
    # The hour index is a feature for the hourly tasks only
    assert ("counter" in names) == hourly

    classifier = xgb.XGBClassifier(n_estimators=10, max_depth=2).fit(rep, labels)
    model = MLWrapper(run_mode=RunMode.classification)
    model.trained_columns = names
    model.train_shap_values = shap.TreeExplainer(classifier)(rep)
    persist_shap_data(SimpleNamespace(lightning_module=model), tmp_path)

    shaps = pl.read_parquet(tmp_path / "train_shap_values.parquet")
    assert shaps.columns == names
    assert shaps.height == rep.shape[0]
    # The label depends on x only, so x has the largest mean absolute SHAP value
    importance = shaps.select(pl.all().abs().mean()).row(0, named=True)
    assert max(importance, key=importance.get) == "x"
