"""preprocess_data must handle --label with multiple labels, --modalities STATIC and non-integer stay IDs (#253)."""

import gin
import polars as pl
import pytest

from icu_benchmarks.data.constants import DataSegment as Segment, DataSplit as Split
from icu_benchmarks.data.split_process_data import preprocess_data

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
