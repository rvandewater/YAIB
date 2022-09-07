import os
from itertools import groupby
import numpy as np
import pandas as pd
import pyarrow as pa

from icu_benchmarks.common import constants


def generate_stay_windows_lookup_table(ricu_data_root, stay_windows_lookup_path):
    """
    This function generates a table with the start and end index in the outcome (and dynamic) table for each stay.
    """
    dyn = pa.parquet.ParquetFile(os.path.join(ricu_data_root, "dyn.parquet"))
    dyn_table = dyn.read()
    # TODO stay_id as constant
    stay_ids = dyn_table.column("stay_id").to_numpy()

    stay_id_length = [(stay_id, len(list(val))) for stay_id, val in groupby(stay_ids)]
    stay_id_idx = np.zeros((len(np.unique(stay_ids)), 3), dtype=int)
    start_idx = 0
    for i, (stay_id, run_length) in enumerate(stay_id_length):
        end_idx = start_idx + run_length - 1
        stay_id_idx[i] = (stay_id, start_idx, end_idx)
        start_idx = end_idx + 1
        
    stays_df = pd.DataFrame(stay_id_idx, columns=["stay_id", "start_idx", "end_idx"])
    return pa.parquet.write_table(pa.Table.from_pandas(stays_df), stay_windows_lookup_path)


def impute_forward_then_backward(ricu_data_root, imputated_data_path):
    """
    1. Get all rows of a specific stay as separate Object, making sure it is sorted by time.
    2. Forward fill missing values for all columns containing measurements.
    3. Backward fill leftover missing values at the start of a stay.
    """
    dyn = pa.parquet.ParquetFile(os.path.join(ricu_data_root, "dyn.parquet"))
    dyn_table = dyn.read()
    dyn_df = dyn_table.to_pandas()

    # TODO stay_id, time as constant
    dyn_df[constants.DYN_FEATURES] = dyn_df.sort_values(['stay_id','time']).groupby('stay_id')[constants.DYN_FEATURES].ffill().bfill()
    
    return pa.parquet.write_table(pa.Table.from_pandas(dyn_df), imputated_data_path)