import os
from itertools import groupby
import numpy as np
import pandas as pd
import pyarrow as pa

from icu_benchmarks.common import constants


def impute_forward_then_backward(ricu_data_root, dyn_imputed_path):
    """
    1. Get all rows of a specific stay as separate Object, making sure it is sorted by time.
    2. Forward fill missing values for all columns containing measurements.
    3. Fill leftover missing values at the start of a stay with the mean of the variable.
    """
    dyn_df = pa.parquet.read_table(os.path.join(ricu_data_root, "dyn.parquet")).to_pandas()

    # TODO stay_id, time as constant
    dyn_data_grouped_df = dyn_df.sort_values(['stay_id','time']).groupby('stay_id')[constants.DYN_FEATURES]
    dyn_data_mean_per_stay = dyn_data_grouped_df.transform('mean')
    dyn_df[constants.DYN_FEATURES] = dyn_data_grouped_df.ffill().fillna(value=dyn_data_mean_per_stay)
    
    return pa.parquet.write_table(pa.Table.from_pandas(dyn_df), dyn_imputed_path)


def generate_splits(ricu_data_root, dyn_imputed_path, stays_splits_path, labels_splits_path, dyn_splits_path):
    """
    1. Generate training, validation and test splits in the data and write out.
    2. Merge dynamic data with splits for easy accessing.
    """
    sta_df = pa.parquet.read_table(os.path.join(ricu_data_root, "sta.parquet")).to_pandas()
    stay_ids_df = sta_df[['stay_id']]

    train_df = stay_ids_df.sample(frac=0.7, random_state=3333)
    train_df['split'] = 'train'

    validation_df = stay_ids_df.drop(train_df.index).sample(frac=0.5, random_state=25)
    validation_df['split'] = 'val'

    test_df = stay_ids_df.drop(train_df.index).drop(validation_df.index)
    test_df['split'] = 'test'

    splits_df = pd.concat([train_df, validation_df, test_df]).sort_index()
    splits_reindexed_df = splits_df.set_index('split')
    pa.parquet.write_table(pa.Table.from_pandas(splits_reindexed_df), stays_splits_path)

    dyn_df = pa.parquet.read_table(dyn_imputed_path).to_pandas()
    dyn_with_splits_df = dyn_df.merge(splits_df, on='stay_id').set_index(['split', 'stay_id'])

    # TODO check whether this works for different label format too
    outc_df = pa.parquet.read_table(os.path.join(ricu_data_root, "outc.parquet")).to_pandas()
    labels_splits_df = outc_df.merge(splits_df, on='stay_id').set_index(['split', 'stay_id'])
    pa.parquet.write_table(pa.Table.from_pandas(labels_splits_df), labels_splits_path)

    return pa.parquet.write_table(pa.Table.from_pandas(dyn_with_splits_df), dyn_splits_path)
