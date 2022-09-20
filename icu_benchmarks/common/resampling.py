import numpy as np
import pandas as pd

from icu_benchmarks.common import constants
import gc


def irregular_to_gridded(df, df_static, df_var_ref, freq_string='5T'):
    df = resample_df(df, freq_string)
    gc.collect()
    df = add_static_df(df, df_static)
    df = rename_to_human_df(df, df_var_ref)
    return df


def resample_df(df, freq_string='5T'):
    cols = df.columns
    assert constants.DATETIME in cols
    assert constants.PID in cols

    def reorder_time(patient_sample):
        pid = patient_sample[constants.PID].iloc[0]

        patient_sample = patient_sample.reset_index(drop=True)
        HRs_non_zero = np.where(~np.isnan(patient_sample.vm1))[0]
        if len(HRs_non_zero) > 0:

            HR_start_idx, HR_stop_idx = HRs_non_zero[0], HRs_non_zero[-1]

            patient_sample.loc[:HR_start_idx] = patient_sample.loc[:HR_start_idx].ffill()
        else:
            HR_start_idx, HR_stop_idx = 0, patient_sample.shape[0] - 1
        stay_stop_time, stay_start_time = patient_sample.loc[HR_stop_idx, constants.DATETIME], patient_sample.loc[
            HR_start_idx, constants.DATETIME]
        patient_sample = patient_sample.loc[HR_start_idx:HR_stop_idx].reset_index(drop=True)
        offset = np.timedelta64(stay_start_time.minute, 'm') + np.timedelta64(stay_start_time.second,
                                                                              's') + np.timedelta64(
            stay_start_time.microsecond, 'us') + np.timedelta64(1, 'us')
        patient_sample.loc[:, constants.DATETIME] = patient_sample[constants.DATETIME] - offset
        grided = patient_sample.set_index(constants.DATETIME).resample(freq_string, axis=0, closed='left',
                                                                       label='right').last().reset_index()
        grided.loc[:, constants.DATETIME] -= (stay_start_time - offset + np.timedelta64(1, 'us'))

        grided = grided.reset_index(drop=True)
        grided[constants.PID] = pid
        return grided

    dfs_pat = []
    for p in df[constants.PID].unique():
        dfs_pat.append(reorder_time(df.query(f'{constants.PID} == {p}')))
        gc.collect()

    df_part = pd.concat(dfs_pat).reset_index(drop=True)
    df_part[constants.PID] = df_part[constants.PID].astype('int64')
    df_part = df_part[[constants.PID] + [c for c in df_part.columns if c != constants.PID]]

    df_part[constants.DATETIME] /= np.timedelta64(60, 's')
    return df_part


def add_static_df(df, df_static):
    df = df.join(df_static.set_index(constants.PID), on=constants.PID)
    return df


def rename_to_human_df(df, df_var_ref):
    to_rename = [k for k in df.columns if k[2:].isdigit()]
    meta_ids = [int(m[2:]) for m in to_rename]
    names = [str(df_var_ref[df_var_ref.metavariableid == i]['metavariablename'].values[0]) for i in meta_ids]
    final_names = [to_rename[i] if names[i] == 'nan' else names[i] for i in range(len(names))]
    rename_mapping = {old: new for old, new in zip(to_rename, final_names)}
    df = df.rename(columns=rename_mapping)
    return df
