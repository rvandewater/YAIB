from pathlib import Path

import pandas as pd
import pytest
import numpy as np
import itertools

from icu_benchmarks.common import lookups
from icu_benchmarks.common.constants import PID, DATETIME
from icu_benchmarks.data import feature_extraction

TEST_ROOT = Path(__file__).parent.parent

PREPROCESSING_RES = TEST_ROOT.parent / 'preprocessing' / 'resources'


@pytest.fixture()
def df_var_ref():
    varref = lookups.read_var_ref_table(PREPROCESSING_RES / 'varref.tsv')
    return varref


def test_extract_feature_df(df_var_ref):
    length = 10
    n_patient = 2

    df = pd.DataFrame({
        PID: [1234] * length + [2345] * length,
        DATETIME: list(range(length)) * n_patient,
        'age': [22] * length * n_patient,
        'constant': [200] * length * n_patient,
        'slope': list(range(length)) * n_patient
    })

    features_df = feature_extraction.extract_feature_df(df, df_var_ref)
    should_exist_cols = [k + m for k, m in list(itertools.product(['max_', 'min_', 'mean_', 'n_meas_'],
                                                                  ['constant', 'slope']))]
    should_exist_cols += [PID, DATETIME]
    cols = list(features_df.columns)
    for col in cols:
        assert col in should_exist_cols
        c_words = col.split('_')
        if c_words[0] == 'max':
            if c_words[1] == 'constant':
                assert np.all(features_df[col].values == np.ones(length * n_patient) * 200)
            elif c_words[1] == 'slope':
                assert np.all(features_df[col].values == np.array(list(range(length)) * n_patient))
            else:
                assert False  # Checks age, PID, datetime are not used

        if c_words[0] == 'min':
            if c_words[1] == 'constant':
                assert np.all(features_df[col].values == np.ones(length * n_patient) * 200)
            elif c_words[1] == 'slope':
                assert np.all(features_df[col].values == np.zeros(length * n_patient))
            else:
                assert False

        if c_words[0] == 'mean':
            if c_words[1] == 'constant':
                assert np.all(features_df[col].values == np.ones(length * n_patient) * 200)

        if c_words[0] == 'n':
            if c_words[2] == 'constant':
                assert np.all(features_df[col].values == np.array(list(range(length)) * n_patient) + 1)
            elif c_words[2] == 'slope':
                assert np.all(features_df[col].values == np.array(list(range(length)) * n_patient) + 1)
            else:
                assert False
