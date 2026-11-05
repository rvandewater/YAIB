"""Missing classification labels must stay missing and be masked, not become negatives (#252)."""

import gin
import numpy as np
import polars as pl

from icu_benchmarks.data.constants import DataSegment as Segment, DataSplit as Split
from icu_benchmarks.data.loader import PredictionPolarsDataset
from icu_benchmarks.data.split_process_data import preprocess_data

VARS = {"GROUP": "stay_id", "SEQUENCE": "time", "LABEL": "label", "DYNAMIC": ["x"]}
FILES = {Segment.dynamic: "dyn.parquet", Segment.outcome: "outc.parquet"}


def hourly_data(path):
    # 24 stays of 4 hours, every fifth hourly label unknown
    rows = pl.DataFrame({"stay_id": np.repeat(np.arange(24), 4), "time": np.tile(np.arange(4), 24)})
    rows = rows.with_row_index().with_columns(x=pl.col("index") * 1.0)
    label = pl.when(pl.col("index") % 5 == 0).then(None).otherwise(pl.col("stay_id") % 2 == 0)
    rows.select("stay_id", "time", "x").write_parquet(path / FILES[Segment.dynamic])
    rows.select("stay_id", "time", label.alias("label")).write_parquet(path / FILES[Segment.outcome])


def test_hourly_missing_labels_are_kept_and_masked(tmp_path):
    gin.clear_config()
    hourly_data(tmp_path)
    try:
        data, _ = preprocess_data(
            data_dir=tmp_path,
            file_names=dict(FILES),
            vars=dict(VARS),
            required_segments=list(FILES),
            cv_repetitions=2,
            cv_folds=2,
        )
    finally:
        gin.clear_config()
    assert sum(data[split][Segment.outcome]["label"].null_count() for split in data) == 96 // 5 + 1
    dataset = PredictionPolarsDataset(data, split=Split.train, vars=VARS, ram_cache=False)
    labels = dataset.outcome_df["label"]
    # The ML path drops the unlabeled hours
    _, ml_labels, _ = dataset.get_data_and_labels()
    assert len(ml_labels) == labels.len() - labels.null_count()
    # The DL path masks them
    masked = sum(int((~dataset[i][2]).sum()) for i in range(len(dataset)))
    assert masked == labels.null_count()


def test_per_stay_missing_label_is_masked():
    stays = range(6)
    split = {
        Segment.features: pl.DataFrame({"stay_id": stays, "time": [0] * 6, "x": [float(i) for i in stays]}),
        Segment.outcome: pl.DataFrame({"stay_id": stays, "label": [True, None, False, True, False, None]}),
    }
    dataset = PredictionPolarsDataset({Split.train: split}, split=Split.train, vars=VARS, ram_cache=False)
    _, labels, row_indicators = dataset.get_data_and_labels()
    assert row_indicators.ravel().tolist() == [0, 2, 3, 4] and labels.tolist() == [1, 0, 1, 0]
    assert [bool(dataset[i][2].any()) for i in range(len(dataset))] == [True, False, True, True, True, False]
