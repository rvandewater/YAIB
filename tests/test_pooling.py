"""PooledData must sample stays from each dataset, keep their rows together and hold out the remaining stays."""

import numpy as np
import pandas as pd

from icu_benchmarks.data.constants import DataSegment as Segment
from icu_benchmarks.data.pooling import PooledData

VARS = {"GROUP": "stay_id", "SEQUENCE": "time", "LABEL": "label"}
FILES = {Segment.static: "sta.parquet", Segment.dynamic: "dyn.parquet", Segment.outcome: "outc.parquet"}
STAYS, SAMPLES = 20, 6


def write_dataset(folder, seed):
    rng = np.random.default_rng(seed)
    stay_ids = np.arange(1, STAYS + 1)  # Every dataset uses the same stay IDs
    segments = {
        Segment.outcome: pd.DataFrame({"stay_id": stay_ids, "label": np.tile([0, 1], STAYS // 2)}),
        Segment.static: pd.DataFrame({"stay_id": stay_ids, "age": rng.normal(size=STAYS)}),
        Segment.dynamic: pd.DataFrame(
            {"stay_id": np.repeat(stay_ids, 2), "time": np.tile([0, 1], STAYS), "hr": rng.normal(size=2 * STAYS)}
        ),
    }
    folder.mkdir()
    for segment, file_name in FILES.items():
        segments[segment].to_parquet(folder / file_name)


def read_segments(folder):
    return {segment: pd.read_parquet(folder / file_name) for segment, file_name in FILES.items()}


def test_generate_pools_stratified_samples_and_saves_held_out_stays(tmp_path):
    # #262: generate passed the builtin vars, and _select_stays indexed the frames with the builtin id
    for seed, name in enumerate(["a", "b"]):
        write_dataset(tmp_path / name, seed)
    PooledData(tmp_path, VARS, ["a", "b"], FILES, shuffle=True).generate(["a", "b"], samples=SAMPLES)

    pooled = read_segments(tmp_path / f"a_b_{SAMPLES}")
    stays = set(pooled[Segment.outcome]["stay_id"])
    # The datasets' clashing stay IDs are made unique, and every segment holds the same stays
    assert len(stays) == len(pooled[Segment.outcome]) == 2 * SAMPLES
    assert set(pooled[Segment.static]["stay_id"]) == set(pooled[Segment.dynamic]["stay_id"]) == stays
    assert len(pooled[Segment.dynamic]) == 2 * len(stays)
    assert pooled[Segment.outcome]["label"].sum() == SAMPLES  # Stratified: half of each dataset's sample is positive

    for name in ["a", "b"]:
        held_out = read_segments(tmp_path / f"{name}_test_{STAYS - SAMPLES}")
        held_out_stays = set(held_out[Segment.outcome]["stay_id"])
        assert len(held_out_stays) == STAYS - SAMPLES and not held_out_stays & stays
        assert set(held_out[Segment.dynamic]["stay_id"]) == held_out_stays


def test_stratification_labels_follow_the_stay_order(tmp_path):
    # The labels were sorted by stay ID while the stays kept the outcome's row order, so an unsorted outcome frame
    # stratified on the wrong labels. Here a quarter of the stays are positive and the outcome is in reverse order.
    stay_ids = np.arange(1, STAYS + 1)
    label = (stay_ids > STAYS * 3 // 4).astype(int)
    segments = {
        Segment.outcome: pd.DataFrame({"stay_id": stay_ids, "label": label}).iloc[::-1],
        Segment.static: pd.DataFrame({"stay_id": stay_ids, "age": 0.0}),
        Segment.dynamic: pd.DataFrame({"stay_id": stay_ids, "time": 0, "hr": 0.0}),
    }
    (tmp_path / "a").mkdir()
    for segment, file_name in FILES.items():
        segments[segment].to_parquet(tmp_path / "a" / file_name)
    samples = 8
    PooledData(tmp_path, VARS, ["a"], FILES, shuffle=True).generate(["a"], samples=samples)
    pooled = read_segments(tmp_path / f"a_{samples}")
    assert pooled[Segment.outcome]["label"].sum() == samples // 4
