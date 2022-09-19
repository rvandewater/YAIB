import pandas as pd
import pyarrow as pa

from icu_benchmarks.common import constants

VARS = constants.VARS
FILE_NAMES = constants.FILE_NAMES

# TODO finish doc strings
# TODO use type hints


# TODO make function take df like the rest
# TODO make proportions of splits as parameter
def generate_splits(sta_path, outc_path, dyn_path, static_splits_path, labels_splits_path, dyn_splits_path):
    """
    1. Generates training, validation and test splits in the data.
    2. Merges dynamic data with splits for easy accessing.
    """
    static_df = pa.parquet.read_table(sta_path).to_pandas()

    train_df = static_df.sample(frac=0.7, random_state=3333)
    train_df['split'] = 'train'

    validation_df = static_df.drop(train_df.index).sample(frac=0.5, random_state=25)
    validation_df['split'] = 'val'

    test_df = static_df.drop(train_df.index).drop(validation_df.index)
    test_df['split'] = 'test'

    static_splits_df = pd.concat([train_df, validation_df, test_df]).sort_index()
    splits_reindexed_df = static_splits_df.set_index('split')
    pa.parquet.write_table(pa.Table.from_pandas(splits_reindexed_df), static_splits_path)

    splits_df = static_splits_df[['split', VARS['STAY_ID']]]

    dyn_df = pa.parquet.read_table(dyn_path).to_pandas()
    dyn_with_splits_df = dyn_df.merge(splits_df, on=VARS['STAY_ID']).set_index(['split', VARS['STAY_ID']])

    # TODO check whether this works for different label format too
    outc_df = pa.parquet.read_table(outc_path).to_pandas()
    labels_splits_df = outc_df.merge(splits_df, on=VARS['STAY_ID']).set_index(['split', VARS['STAY_ID']])
    pa.parquet.write_table(pa.Table.from_pandas(labels_splits_df), labels_splits_path)

    return pa.parquet.write_table(pa.Table.from_pandas(dyn_with_splits_df), dyn_splits_path)


def extract_features(dyn_df):
    """Calculates historical min, max, number of measurements and mean per stay and write to table.
    """
    dyn_data_grouped_df = dyn_df.groupby(VARS['STAY_ID'])[VARS['DYNAMIC_VARS']]
    features_df = dyn_df.drop(labels=VARS['DYNAMIC_VARS'] + [VARS['TIME']], axis=1)
    features_df[['min_' + c for c in VARS['DYNAMIC_VARS']]] = dyn_data_grouped_df.cummin()
    features_df[['max_' + c for c in VARS['DYNAMIC_VARS']]] = dyn_data_grouped_df.cummax()
    features_df[['n_meas_' + c for c in VARS['DYNAMIC_VARS']]] = dyn_df[VARS['DYNAMIC_VARS']].notna().groupby(VARS['STAY_ID']).cumsum().values
    features_df[['mean_' + c for c in VARS['DYNAMIC_VARS']]] = dyn_data_grouped_df.cumsum().values / features_df[['n_meas_' + c for c in VARS['DYNAMIC_VARS']]].values
    
    return features_df


def forward_fill(raw_df, impute_cols):
    return raw_df[impute_cols].groupby(VARS['STAY_ID']).ffill()


def impute(raw_df, impute_function=None, exclude_cols=[], sort_col=None, fill_method='zero'):
    """Imputes a df by forward filling and replacing remaining NaNs by a default.

    1. Get all rows of a specific stay as separate Object, making sure it is sorted if needed.
    2. Forward fill missing values for all columns but exclude_cols.
    3. Fill leftover missing values at the start of a stay with a default (0 or the global mean) of the variable.
    """
    imputed_df = raw_df.copy()
    
    if sort_col:
        imputed_df = imputed_df.sort_values(sort_col)

    impute_cols = imputed_df.columns.difference(exclude_cols)
    if impute_function:
        imputed_df[impute_cols] = impute_function(imputed_df, impute_cols)
    
    # TODO think of better solution for fill_method (gin function, default value, combine with impute function (as extra args for impute)?)
    if fill_method == 'zero':
        fill_value = 0
    elif fill_method == 'mean':
        fill_value = raw_df[impute_cols].loc['train'].mean()
    else:
        raise ValueError('Wrong fill_method, choose between "zero" and "mean".')
    
    imputed_df[impute_cols] = imputed_df[impute_cols].fillna(value=fill_value)
    
    return imputed_df
