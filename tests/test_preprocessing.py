"""The preprocessing pipeline must split stays without leakage and leave model-ready features (on the public demo data)."""

import shutil
from pathlib import Path

import gin
import numpy as np
import pandas as pd
import polars as pl
import pytest

from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.constants import DataSegment as Segment, DataSplit as Split
from icu_benchmarks.data.preprocessor import PolarsClassificationPreprocessor, PolarsRegressionPreprocessor
from icu_benchmarks.data.split_process_data import (
    check_required_keys,
    check_sanitize_data,
    make_single_split,
    make_train_val,
    modality_selection,
    preprocess_data,
)

DEMO = Path(__file__).parent.parent / "demo_data"
FILE_NAMES = {Segment.dynamic: "dyn.parquet", Segment.outcome: "outc.parquet", Segment.static: "sta.parquet"}
SPLITS = (Split.train, Split.val, Split.test)


@pytest.fixture(autouse=True)
def clean_gin():
    gin.clear_config()
    yield
    gin.clear_config()


def demo_vars(data_dir: Path) -> dict:
    """The task variables, read from the columns of the demo files."""
    dynamic = pl.read_parquet_schema(data_dir / "dyn.parquet")
    static = pl.read_parquet_schema(data_dir / "sta.parquet")
    return {
        "GROUP": "stay_id",
        "SEQUENCE": "time",
        "LABEL": "label",
        "DYNAMIC": [c for c in dynamic if c not in ("stay_id", "time")],
        "STATIC": [c for c in static if c != "stay_id"],
    }


def run_preprocess(data_dir: Path, **kwargs):
    kwargs.setdefault("cv_repetitions", 2)
    kwargs.setdefault("cv_folds", 2)
    return preprocess_data(data_dir, file_names=dict(FILE_NAMES), vars=demo_vars(data_dir), **kwargs)


def stays(data, split, segment=Segment.outcome) -> set:
    return set(data[split][segment]["stay_id"].to_list())


@pytest.mark.parametrize("task", ["mortality24", "aki"], ids=["per_stay", "hourly"])
def test_demo_splits_are_disjoint_and_model_ready(task):
    data_dir = DEMO / task / "mimic_demo"
    data, target_transform = run_preprocess(data_dir)
    assert target_transform is None
    all_stays = set(pl.read_parquet(data_dir / "outc.parquet")["stay_id"].to_list())
    split_stays = [stays(data, split) for split in SPLITS]
    assert set.union(*split_stays) == all_stays
    assert sum(len(s) for s in split_stays) == len(all_stays)
    for split in SPLITS:
        assert set(data[split]) == {Segment.features, Segment.outcome}
        features = data[split][Segment.features]
        assert stays(data, split, Segment.features) == stays(data, split)
        # Static features are joined to the dynamic ones, with missing indicators and no remaining gaps
        assert {"age", "sex", "hr", "MissingIndicator_hr", "MissingIndicator_age"} <= set(features.columns)
        assert features.null_count().sum_horizontal().item() == 0
        assert features["sex"].dtype.is_numeric()
        if task == "aki":
            assert len(features) == len(data[split][Segment.outcome])


def test_feature_generation_and_no_static():
    gin.bind_parameter("base_classification_preprocessor.generate_features", True)
    data, _ = run_preprocess(DEMO / "mortality24" / "mimic_demo", use_static=False)
    columns = data[Split.train][Segment.features].columns
    assert {"hrmin_hist", "hrmax_hist", "hrmean_hist", "hrcount_hist", "counter"} <= set(columns)
    assert "age" not in columns and "sex" not in columns


def test_splits_are_reproducible_and_folds_differ():
    data_dir = DEMO / "mortality24" / "mimic_demo"
    first, _ = run_preprocess(data_dir, seed=1)
    again, _ = run_preprocess(data_dir, seed=1)
    other_fold, _ = run_preprocess(data_dir, seed=1, fold_index=1)
    assert stays(first, Split.test) == stays(again, Split.test)
    assert stays(first, Split.val) == stays(other_fold, Split.train)
    assert stays(first, Split.test) == stays(other_fold, Split.test)


def test_complete_train_tests_on_validation_split():
    data, _ = run_preprocess(DEMO / "mortality24" / "mimic_demo", complete_train=True, train_size=0.8)
    assert stays(data, Split.test) == stays(data, Split.val)
    assert not stays(data, Split.train) & stays(data, Split.val)
    assert len(stays(data, Split.train)) == pytest.approx(0.8 * 99, abs=1)


def test_regression_fits_target_transform_on_train():
    data, target_transform = run_preprocess(
        DEMO / "kidney_function" / "mimic_demo", preprocessor=PolarsRegressionPreprocessor, runmode=RunMode.regression
    )
    labels = data[Split.train][Segment.outcome]["label"].drop_nulls().drop_nans()
    assert target_transform is not None and target_transform.label == "label"
    assert labels.min() == pytest.approx(0) and labels.max() == pytest.approx(1)


def test_cache_round_trip(tmp_path, monkeypatch):
    data_dir = tmp_path / "mimic_demo"
    shutil.copytree(DEMO / "mortality24" / "mimic_demo", data_dir)
    data, _ = run_preprocess(data_dir, generate_cache=True)
    assert len(list((data_dir / "cache").iterdir())) == 1
    assert {p.name.rsplit("_", 1)[-1] for p in (data_dir / "preproc").iterdir()} == {"STATIC", "DYNAMIC"}

    monkeypatch.setattr(pl, "read_parquet", lambda *a, **k: pytest.fail("Cache was not used"))
    cached, _ = run_preprocess(data_dir, load_cache=True)
    for split in SPLITS:
        assert cached[split][Segment.features].equals(data[split][Segment.features])


def test_preprocessor_cache_string_tracks_settings():
    assert (
        PolarsClassificationPreprocessor(scaling=True).to_cache_string()
        != PolarsClassificationPreprocessor(scaling=False).to_cache_string()
    )


def test_check_required_keys():
    check_required_keys({"GROUP": 1, "LABEL": 2}, ["GROUP"])
    with pytest.raises(KeyError, match="LABEL"):
        check_required_keys({"GROUP": 1}, ["GROUP", "LABEL"])


def test_check_sanitize_data_keeps_last_duplicate():
    vars = {"GROUP": "stay_id", "SEQUENCE": "time"}
    data = {
        Segment.static: pl.DataFrame({"stay_id": [1, 1, 2], "age": [50, 51, 60]}),
        Segment.dynamic: pl.DataFrame({"stay_id": [1, 1, 1], "time": [0, 0, 1], "hr": [80, 81, 90]}),
        Segment.outcome: pl.DataFrame({"stay_id": [1, 2, 2], "label": [0, 1, 0]}),
    }
    clean = check_sanitize_data(data, vars)
    assert clean[Segment.static]["age"].to_list() == [51, 60]
    assert clean[Segment.dynamic]["hr"].to_list() == [81, 90]
    assert clean[Segment.outcome]["label"].to_list() == [0, 0]


def test_modality_selection():
    vars = {"GROUP": "stay_id", "SEQUENCE": "time", "LABEL": "label", "DYNAMIC": ["hr", "map"], "STATIC": ["age"]}
    data = {Segment.dynamic: pl.DataFrame({"stay_id": [1], "time": [0], "hr": [80], "map": [70]})}
    mapping = {"vitals": ["hr"], "demographics": ["age"]}
    selected, new_vars = modality_selection(data, mapping, ["vitals"], dict(vars))
    assert selected[Segment.dynamic].columns == ["stay_id", "time", "hr"]
    assert new_vars["DYNAMIC"] == ["hr"] and new_vars["STATIC"] == []
    with pytest.raises(ValueError, match="None of the selected modalities"):
        modality_selection(data, mapping, ["labs"], dict(vars))


def synthetic_raw(n=40):
    rng = np.random.default_rng(0)
    ids = np.arange(n)
    return {
        Segment.outcome: pl.DataFrame({"stay_id": ids, "label": (ids % 4 == 0).astype(int)}),
        Segment.dynamic: pl.DataFrame(
            {"stay_id": np.repeat(ids, 2), "time": np.tile([0, 1], n), "hr": rng.normal(size=2 * n)}
        ),
    }


VARS = {"GROUP": "stay_id", "SEQUENCE": "time", "LABEL": "label", "DYNAMIC": ["hr"]}


def split_ids(data) -> list[set]:
    return [set(np.asarray(data[split][Segment.outcome]["stay_id"]).tolist()) for split in SPLITS]


@pytest.mark.parametrize("runmode", [RunMode.classification, RunMode.regression])
def test_single_split_backends_agree_and_folds_partition(runmode):
    raw = synthetic_raw()
    folds = [make_single_split(dict(raw), VARS, 2, 0, 3, fold, runmode=runmode) for fold in range(3)]
    pandas = make_single_split({k: v.to_pandas() for k, v in raw.items()}, VARS, 2, 0, 3, 0, runmode=runmode, polars=False)
    assert split_ids(pandas) == split_ids(folds[0])
    assert isinstance(pandas[Split.train][Segment.dynamic], pd.DataFrame)
    # Every development stay is validated on exactly once; the test split is fixed by the repetition
    val_sets = [split_ids(f)[1] for f in folds]
    assert sum(len(s) for s in val_sets) == len(set.union(*val_sets)) == 20
    assert all(split_ids(f)[2] == split_ids(folds[0])[2] for f in folds)
    if runmode == RunMode.classification:
        positives = [folds[0][split][Segment.outcome]["label"].sum() for split in SPLITS]
        assert min(positives) >= 1


def test_single_split_requires_enough_samples_per_class():
    raw = synthetic_raw(8)  # two positive stays
    with pytest.raises(Exception, match="smallest amount of samples"):
        make_single_split(raw, VARS, 2, 0, 3, 0)


@pytest.mark.parametrize("polars", [True, False])
def test_train_val_split(polars):
    raw = synthetic_raw()
    if not polars:
        raw = {k: v.to_pandas() for k, v in raw.items()}
    data = make_train_val(raw, VARS, train_size=0.75, polars=polars)
    train, val, test = split_ids(data)
    assert len(train) == 30 and len(val) == 10 and val == test and not train & val


# Selecting one of several labels with label="other" crashes today (#253); its fix adds that case
@pytest.mark.parametrize("label,expected", [(None, "label")])
def test_label_selection_with_multiple_labels(label, expected, tmp_path):
    shutil.copytree(DEMO / "mortality24" / "mimic_demo", tmp_path, dirs_exist_ok=True)
    outcome = pl.read_parquet(tmp_path / "outc.parquet")
    outcome.with_columns(other=1 - pl.col("label")).write_parquet(tmp_path / "outc.parquet")
    vars = {**demo_vars(tmp_path), "LABEL": ["label", "other"]}
    preprocess_data(tmp_path, file_names=dict(FILE_NAMES), vars=vars, label=label, cv_repetitions=2, cv_folds=2)
    assert vars["LABEL"] == expected
