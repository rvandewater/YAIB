import pandas as pd

from icu_benchmarks.common import constants


def read_static(static_path):
    df_static = pd.read_parquet(static_path).reset_index()
    assert 'admissiontime' in df_static.columns
    df_static['admissiontime'] = df_static['admissiontime'].dt.strftime('%m-%d').astype(str).apply(convert_to_days)
    return df_static[[constants.PID, 'admissiontime', 'age', 'sex', 'height']]


def convert_to_days(x):
    month, day = x.split('-')
    return float(month) * 30 + float(day)
