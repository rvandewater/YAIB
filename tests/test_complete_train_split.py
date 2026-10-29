"""The complete_train split must be reproducible and stratified on the labels of the right stays (#250)."""

import numpy as np
import polars as pl

from icu_benchmarks.data.constants import DataSegment as Segment, DataSplit as Split
from icu_benchmarks.data.split_process_data import make_train_val_polars

VARS = {"GROUP": "stay_id", "SEQUENCE": "time", "LABEL": "label", "DYNAMIC": ["x"]}


def split_stays(order_seed):
    # The same 2000 stays (10% positive), in a different row order per call
    stays = np.arange(1000, 3000)
    outcome = pl.DataFrame({"stay_id": stays, "label": (stays % 10 == 0).astype(int)})
    outcome = outcome.sample(fraction=1.0, shuffle=True, seed=order_seed)
    dynamic = outcome.select("stay_id", time=pl.lit(0), x=pl.col("label") * 1.0)
    data = make_train_val_polars({Segment.outcome: outcome, Segment.dynamic: dynamic}, VARS, train_size=0.8, seed=42)
    return data[Split.val][Segment.outcome]


def test_complete_train_split_is_reproducible_and_stratified():
    val = split_stays(order_seed=0)
    # Stratified on aligned labels, the 400 validation stays hold exactly the overall 10% positives
    assert len(val) == 400 and val["label"].sum() == 40
    for order_seed in (1, 2):
        assert split_stays(order_seed)["stay_id"].to_list() == val["stay_id"].to_list()
