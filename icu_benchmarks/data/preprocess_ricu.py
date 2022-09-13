import pandas as pd
import pyarrow as pa

from icu_benchmarks.common import constants

VARS = constants.VARS
FILE_NAMES = constants.FILE_NAMES


def generate_splits(sta_path, outc_path, dyn_path, stays_splits_path, labels_splits_path, dyn_splits_path):
    """
    1. Generate training, validation and test splits in the data and write out.
    2. Merge dynamic data with splits for easy accessing.
    """
    sta_df = pa.parquet.read_table(sta_path).to_pandas()
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

    dyn_df = pa.parquet.read_table(dyn_path).to_pandas()
    dyn_with_splits_df = dyn_df.merge(splits_df, on=VARS['STAY_ID']).set_index(['split', VARS['STAY_ID']])

    # TODO check whether this works for different label format too
    outc_df = pa.parquet.read_table(outc_path).to_pandas()
    labels_splits_df = outc_df.merge(splits_df, on=VARS['STAY_ID']).set_index(['split', VARS['STAY_ID']])
    pa.parquet.write_table(pa.Table.from_pandas(labels_splits_df), labels_splits_path)

    return pa.parquet.write_table(pa.Table.from_pandas(dyn_with_splits_df), dyn_splits_path)


def extract_features(dyn_splits_path, extracted_features_path):
    """
    Calculate historical min, max, number of measurements and mean per stay and write to table.
    """
    dyn_df = pa.parquet.read_table(dyn_splits_path).to_pandas().sort_values([VARS['STAY_ID'], VARS['TIME']])
    dyn_data_grouped_df = dyn_df.groupby(VARS['STAY_ID'])[VARS['DYNAMIC_VARS']]
    features_df = dyn_df.drop(labels=VARS['DYNAMIC_VARS'] + [VARS['TIME']], axis=1)
    features_df[['min_' + c for c in VARS['DYNAMIC_VARS']]] = dyn_data_grouped_df.cummin()
    features_df[['max_' + c for c in VARS['DYNAMIC_VARS']]] = dyn_data_grouped_df.cummax()
    features_df[['n_meas_' + c for c in VARS['DYNAMIC_VARS']]] = dyn_df[VARS['DYNAMIC_VARS']].notna().groupby(VARS['STAY_ID']).cumsum().values
    features_df[['mean_' + c for c in VARS['DYNAMIC_VARS']]] = dyn_data_grouped_df.cumsum().values / features_df[['n_meas_' + c for c in VARS['DYNAMIC_VARS']]].values
    # forward fill missing values, replace leftover NaNs at the beginning of measurements with zero
    features_df = features_df.groupby(VARS['STAY_ID']).ffill().fillna(value=0)
    
    return pa.parquet.write_table(pa.Table.from_pandas(features_df), extracted_features_path)


def impute_forward_then_backward(dyn_splits_path, dyn_imputed_path):
    """
    1. Get all rows of a specific stay as separate Object, making sure it is sorted by time.
    2. Forward fill missing values for all columns containing measurements.
    3. Fill leftover missing values at the start of a stay with the global mean of the variable.
    """
    dyn_df = pa.parquet.read_table(dyn_splits_path).to_pandas()
    dyn_data_grouped_df = dyn_df.sort_values([VARS['STAY_ID'], VARS['TIME']]).groupby(VARS['STAY_ID'])[VARS['DYNAMIC_VARS']]
    mean_per_split = dyn_df.groupby('split').mean()
    dyn_df[VARS['DYNAMIC_VARS']] = dyn_data_grouped_df.ffill().fillna(value=mean_per_split)
    
    return pa.parquet.write_table(pa.Table.from_pandas(dyn_df), dyn_imputed_path)
