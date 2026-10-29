"""The preprocessing cache must not be reused across modality selections (#249)."""

import gin
import polars as pl

from icu_benchmarks.data.constants import DataSegment as Segment, DataSplit as Split
from icu_benchmarks.data.split_process_data import preprocess_data

FILES = {Segment.dynamic: "dyn.parquet", Segment.static: "sta.parquet", Segment.outcome: "outc.parquet"}
MAPPING = {"DYNAMIC": ["x"], "STATIC": ["age"]}


def write_data(path):
    stays = range(24)
    pl.DataFrame({"stay_id": stays, "time": [0] * 24, "x": [float(i) for i in stays]}).write_parquet(
        path / FILES[Segment.dynamic]
    )
    pl.DataFrame({"stay_id": stays, "age": [50.0 + i for i in stays]}).write_parquet(path / FILES[Segment.static])
    pl.DataFrame({"stay_id": stays, "label": [i % 2 for i in stays]}).write_parquet(path / FILES[Segment.outcome])


def feature_columns(data_dir, modalities, **kwargs):
    # preprocess_data edits vars and file_names in place, so every call gets fresh copies
    vars = {"GROUP": "stay_id", "SEQUENCE": "time", "LABEL": "label", "DYNAMIC": ["x"], "STATIC": ["age"]}
    data, _ = preprocess_data(
        data_dir=data_dir,
        file_names=dict(FILES),
        vars=vars,
        modality_mapping=MAPPING,
        selected_modalities=modalities,
        cv_repetitions=2,
        cv_folds=2,
        **kwargs,
    )
    return set(data[Split.train][Segment.features].columns)


def test_cache_key_includes_modality_selection(tmp_path):
    gin.clear_config()
    write_data(tmp_path)
    try:
        dynamic_only = feature_columns(tmp_path, ["DYNAMIC"], generate_cache=True)
        assert "age" not in dynamic_only
        # A later run with all modalities must not load the DYNAMIC-only cache
        assert feature_columns(tmp_path, ["all"], load_cache=True, generate_cache=True) == feature_columns(tmp_path, ["all"])
        assert "age" in feature_columns(tmp_path, ["all"], load_cache=True)
        assert feature_columns(tmp_path, ["DYNAMIC"], load_cache=True) == dynamic_only
        assert len(list((tmp_path / "cache").iterdir())) == 2
    finally:
        gin.clear_config()
