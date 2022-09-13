import os
import pandas as pd
import pyarrow as pa

from icu_benchmarks.common import constants

VARS = constants.VARS
FILE_NAMES = constants.FILE_NAMES


def impute_forward_then_backward(ricu_data_root, dyn_imputed_path):
    """
    1. Get all rows of a specific stay as separate Object, making sure it is sorted by time.
    2. Forward fill missing values for all columns containing measurements.
    3. Fill leftover missing values at the start of a stay with the global mean of the variable.
    """
    dyn_df = pa.parquet.read_table(os.path.join(ricu_data_root, FILE_NAMES['DYNAMIC'])).to_pandas()
    dyn_data_grouped_df = dyn_df.sort_values([VARS['STAY_ID'], VARS['TIME']]).groupby(VARS['STAY_ID'])[VARS['DYN_FEATURES']]
    dyn_df[VARS['DYN_FEATURES']] = dyn_data_grouped_df.ffill().fillna(value=dyn_df.mean())
    
    return pa.parquet.write_table(pa.Table.from_pandas(dyn_df), dyn_imputed_path)


def generate_splits(ricu_data_root, dyn_imputed_path, stays_splits_path, labels_splits_path, dyn_splits_path):
    """
    1. Generate training, validation and test splits in the data and write out.
    2. Merge dynamic data with splits for easy accessing.
    """
    sta_df = pa.parquet.read_table(os.path.join(ricu_data_root, FILE_NAMES['STATIC'])).to_pandas()
    stay_ids_df = sta_df[[VARS['STAY_ID']]]

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
    dyn_with_splits_df = dyn_df.merge(splits_df, on=VARS['STAY_ID']).set_index(['split', VARS['STAY_ID']])

    # TODO check whether this works for different label format too
    outc_df = pa.parquet.read_table(os.path.join(ricu_data_root, FILE_NAMES['OUTCOME'])).to_pandas()
    labels_splits_df = outc_df.merge(splits_df, on=VARS['STAY_ID']).set_index(['split', VARS['STAY_ID']])
    pa.parquet.write_table(pa.Table.from_pandas(labels_splits_df), labels_splits_path)

    return pa.parquet.write_table(pa.Table.from_pandas(dyn_with_splits_df), dyn_splits_path)
