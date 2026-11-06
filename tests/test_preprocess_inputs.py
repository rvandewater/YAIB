"""preprocess_data must handle --label with multiple labels, --modalities STATIC and non-integer stay IDs (#253)."""

import gin
import polars as pl
import pytest

from icu_benchmarks.data.constants import DataSegment as Segment, DataSplit as Split
from icu_benchmarks.data.loader import PredictionPolarsDataset
from icu_benchmarks.data.split_process_data import preprocess_data
from icu_benchmarks.run_utils import bind_selected_label

FILES = {Segment.dynamic: "dyn.parquet", Segment.static: "sta.parquet", Segment.outcome: "outc.parquet"}


def write_data(path, stays):
    n = len(stays)
    dynamic = pl.DataFrame({"stay_id": stays, "time": [0] * n, "x": [float(i) for i in range(n)]})
    static = pl.DataFrame({"stay_id": stays, "age": [50.0 + i for i in range(n)], "sex": ["F", "M"] * (n // 2)})
    outcome = pl.DataFrame({"stay_id": stays, "label": [i % 2 for i in range(n)], "label2": [i % 3 == 0 for i in range(n)]})
    for segment, frame in ((Segment.dynamic, dynamic), (Segment.static, static), (Segment.outcome, outcome)):
        frame.write_parquet(path / FILES[segment])


def preprocess(path, **kwargs):
    # preprocess_data edits vars and file_names in place, so every call gets fresh copies
    vars = {"GROUP": "stay_id", "SEQUENCE": "time", "LABEL": ["label", "label2"], "DYNAMIC": ["x"], "STATIC": ["age", "sex"]}
    gin.clear_config()
    try:
        data, _ = preprocess_data(data_dir=path, file_names=dict(FILES), vars=vars, cv_repetitions=2, cv_folds=2, **kwargs)
    finally:
        gin.clear_config()
    return data


def test_label_selects_one_of_multiple_labels(tmp_path):
    write_data(tmp_path, list(range(24)))
    data = preprocess(tmp_path, label="label2")
    assert data[Split.train][Segment.outcome]["label2"].dtype == pl.Boolean
    with pytest.raises(ValueError, match="not one of the configured labels"):
        preprocess(tmp_path, label="unknown")


def test_static_modality_only(tmp_path):
    write_data(tmp_path, list(range(24)))
    data = preprocess(tmp_path, modality_mapping={"DYNAMIC": ["x"], "STATIC": ["age", "sex"]}, selected_modalities=["STATIC"])
    features = data[Split.train][Segment.features]
    assert {"age", "sex"} <= set(features.columns) and "x" not in features.columns


@pytest.mark.parametrize("complete_train", [False, True])
def test_string_stay_ids(tmp_path, complete_train):
    stays = [f"s{i}" for i in range(24)]
    write_data(tmp_path, stays)
    data = preprocess(tmp_path, complete_train=complete_train)
    features = pl.concat([data[split][Segment.features] for split in data]).unique("stay_id")
    assert features.schema["stay_id"] == pl.String
    # Every stay keeps its own features; the ID itself is not encoded as a static categorical
    assert sorted(features["stay_id"].to_list()) == sorted(stays)
    assert features.filter(pl.col("stay_id") == "s1")["sex"].item() == 1


@pytest.mark.parametrize(("label", "expected"), [("label2", "label2"), (None, "label")])
def test_selected_label_reaches_the_dataset(tmp_path, label, expected):
    # The datasets get their own copy of %vars, so run.py binds the selected label into the macro
    write_data(tmp_path, list(range(24)))
    gin.clear_config()
    try:
        gin.parse_config(
            'vars = {"GROUP": "stay_id", "SEQUENCE": "time", "LABEL": ["label", "label2"], "DYNAMIC": ["x"], '
            '"STATIC": ["age", "sex"]}\npreprocess.vars = %vars\nPredictionPolarsDataset.vars = %vars'
        )
        bind_selected_label(label)
        data, _ = preprocess_data(data_dir=tmp_path, file_names=dict(FILES), cv_repetitions=2, cv_folds=2, label=label)
        dataset = PredictionPolarsDataset(data, split=Split.train)
    finally:
        gin.clear_config()
    _, labels, _ = dataset[0]
    assert dataset.vars["LABEL"] == expected and labels.ndim == 1


@pytest.mark.parametrize("binding", ['preprocess.label = "label2"', 'selected = "label2"\npreprocess.label = %selected'])
def test_label_bound_in_gin_reaches_the_dataset(tmp_path, binding):
    # Without --label, a config's preprocess.label binding is the label preprocessing uses, so %vars must hold it too
    write_data(tmp_path, list(range(24)))
    gin.clear_config()
    try:
        gin.parse_config(
            'vars = {"GROUP": "stay_id", "SEQUENCE": "time", "LABEL": ["label", "label2"], "DYNAMIC": ["x"], '
            f'"STATIC": ["age", "sex"]}}\npreprocess.vars = %vars\n{binding}\nPredictionPolarsDataset.vars = %vars'
        )
        bind_selected_label(None)
        data, _ = preprocess_data(data_dir=tmp_path, file_names=dict(FILES), cv_repetitions=2, cv_folds=2)
        dataset = PredictionPolarsDataset(data, split=Split.train)
    finally:
        gin.clear_config()
    assert dataset.vars["LABEL"] == "label2"


def test_unconfigured_string_static_column_is_encoded(tmp_path):
    # Static columns not listed in vars stay in the features, so string ones must be encoded; a string ID must not be
    stays = [f"s{i}" for i in range(24)]
    write_data(tmp_path, stays)
    static = pl.read_parquet(tmp_path / FILES[Segment.static])
    static.with_columns(unit=pl.Series(["icu", "ccu", "micu"] * 8)).write_parquet(tmp_path / FILES[Segment.static])
    features = preprocess(tmp_path)[Split.train][Segment.features]
    assert features.schema["unit"].is_integer() and features.schema["stay_id"] == pl.String


@pytest.mark.parametrize(
    ("outcome_id", "feature_id"),
    [
        # String outcome IDs would not match integer feature IDs after the cast ("001" is not "1")
        (pl.col("stay_id").cast(pl.String).str.zfill(3), pl.col("stay_id")),
        # Distinct string feature IDs can cast to the same float ("1" and "01" are both 1.0)
        (pl.col("stay_id").cast(pl.Float64), pl.col("stay_id").cast(pl.String)),
        # Float feature IDs would not match string outcome IDs after the cast ("1.0" is not "1")
        (pl.col("stay_id").cast(pl.String), pl.col("stay_id").cast(pl.Float64)),
    ],
    ids=["string-integer", "float-string", "string-float"],
)
def test_mixed_id_types_are_rejected(tmp_path, outcome_id, feature_id):
    write_data(tmp_path, list(range(24)))
    for segment in FILES:
        frame = pl.read_parquet(tmp_path / FILES[segment])
        frame.with_columns(outcome_id if segment == Segment.outcome else feature_id).write_parquet(tmp_path / FILES[segment])
    with pytest.raises(ValueError, match="same ID type"):
        preprocess(tmp_path)
