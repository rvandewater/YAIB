import pandas as pd

from icu_benchmarks.common import constants
from icu_benchmarks.data.preprocess import get_var_types

ADD_EXCLUDE = ['admissiontime', 'sex', 'age', 'height']

def extract_feature_df(df, df_var_ref):
    cat_values, binary_values, _, _ = get_var_types(df.columns, df_var_ref)
    to_exclude = cat_values + binary_values + ADD_EXCLUDE

    cols = df.columns
    assert constants.DATETIME in cols
    assert constants.PID in cols

    df_feat = df.groupby(constants.PID).apply(lambda x: extract_feat_patient(x, exclude=to_exclude))
    df_feat = df_feat.reset_index(level=0, drop=True)
    return df_feat


def extract_feat_patient(patient_sample, exclude):
    cols = [c for c in patient_sample.columns if c not in exclude + [constants.PID, constants.DATETIME]]
    patient_feat = patient_sample.copy()
    max_cols = ['max_' + c for c in cols]
    min_cols = ['min_' + c for c in cols]
    mean_cols = ['mean_' + c for c in cols]
    meas_cols = ['n_meas_' + c for c in cols]
    patient_feat[max_cols] = patient_sample[cols].cummax().ffill().values
    patient_feat[min_cols] = patient_sample[cols].cummin().ffill().values
    patient_feat[meas_cols] = pd.notna(patient_sample[cols]).astype(float).cumsum().ffill().values
    patient_feat[mean_cols] = patient_sample[cols].cumsum().ffill().values / patient_feat[meas_cols].values
    return patient_feat[[constants.PID, constants.DATETIME] + min_cols + max_cols + mean_cols + meas_cols]
