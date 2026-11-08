"""Every task config must parse with the modules the CLI imports, and preprocess the demo data (#254)."""

from pathlib import Path

import gin
import pandas as pd
import polars as pl
import pytest

# The CLI imports these modules before it parses the configs, which bind their configurables
import icu_benchmarks.cross_validation  # noqa: F401
import icu_benchmarks.imputation  # noqa: F401
import icu_benchmarks.models.train  # noqa: F401
from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.constants import DataSegment, DataSplit
from icu_benchmarks.data.split_process_data import preprocess_data
from icu_benchmarks.run import get_mode

ROOT = Path(__file__).parent.parent
TASKS = sorted((ROOT / "configs" / "tasks").glob("*.gin"))
# The demo cohort that suits a task, by default the mortality cohort
DEMO_DATA = {"Regression": "los/mimic_demo"}
STATIC = ["age", "sex", "height", "weight"]
DYNAMIC = ["hr", "map", "temp"]  # Some of the dynamic variables


@pytest.fixture(autouse=True)
def clean_gin():
    gin.clear_config()
    yield
    gin.clear_config()


def preprocess(task: str, monkeypatch, **kwargs):
    # The configs include each other relative to the repository
    monkeypatch.chdir(ROOT)
    gin.parse_config_file(f"configs/tasks/{task}.gin")
    mode = get_mode()
    # Not debug: it keeps 1% of the stays, which leaves none of the demo's
    data, _ = preprocess_data(ROOT / "demo_data" / DEMO_DATA.get(task, "mortality24/mimic_demo"), runmode=mode, **kwargs)
    return data, mode


def test_task_configs_are_found():
    assert {"BinaryClassificationNoDynamic", "BinaryClassificationNoStatic", "DatasetImputation"} <= {t.stem for t in TASKS}


@pytest.mark.parametrize("task", [task.stem for task in TASKS])
def test_task_config_preprocesses_demo_data(task, monkeypatch):
    data, mode = preprocess(task, monkeypatch)
    assert set(data) == {DataSplit.train, DataSplit.val, DataSplit.test}
    for split in data.values():
        features = split[DataSegment.features]
        assert len(features) > 0
        if mode == RunMode.imputation:
            # The imputation datasets read Pandas frames, with the missing values that they learn to fill
            assert isinstance(features, pd.DataFrame) and DataSegment.outcome not in split
            assert set(gin.query_parameter("vars/gin.macro.value")[DataSegment.dynamic]) < set(features.columns)
            assert set(split[DataSegment.static]["stay_id"]) == set(features["stay_id"])
        else:
            assert isinstance(features, pl.DataFrame) and not features.null_count().sum_horizontal().item()
            assert len(split[DataSegment.outcome]) > 0


@pytest.mark.parametrize(
    ("task", "kept", "dropped"),
    [("BinaryClassificationNoStatic", DYNAMIC, STATIC), ("BinaryClassificationNoDynamic", STATIC, DYNAMIC)],
)
def test_modality_variants_drop_a_modality(task, kept, dropped, monkeypatch):
    data, _ = preprocess(task, monkeypatch)
    variables = {column.removeprefix("MissingIndicator_") for column in data[DataSplit.train][DataSegment.features].columns}
    assert set(kept) <= variables and not set(dropped) & variables


def test_label_cannot_be_requested_for_task_without_labels(monkeypatch):
    with pytest.raises(ValueError, match="configures no labels"):
        preprocess("DatasetImputation", monkeypatch, label="mortality")
