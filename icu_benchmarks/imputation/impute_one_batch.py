import gc
import glob
import logging
import os
import os.path

import numpy as np
import pandas as pd

from icu_benchmarks.common.constants import PID, DATETIME, REL_DATETIME, MAX_IMPUTE_DAYS, IMPUTATION_PERIOD_SEC,\
    VAR_IDS_EP
import icu_benchmarks.imputation.forward_filling as endpoint_ff



def value_empty(size, default_val, dtype=None):
    """ Returns a vector filled with elements of a specific value"""

    if dtype is not None:
        tmp_arr = np.empty(size, dtype=dtype)
    else:
        tmp_arr = np.empty(size)

    tmp_arr[:] = default_val
    return tmp_arr


def empty_nan(sz):
    """ Returns an empty NAN vector of specified size"""
    arr = np.empty(sz)
    arr[:] = np.nan
    return arr


def impute_dynamic_df(patient_df, pid=None):
    """ Transformer method, taking as input a data-frame with irregularly sampled input data. The method
        assumes that the data-frame contains a time-stamp column, and the data-frame is sorted along the first
        axis in non-decreasing order with respect to the timestamp column. Pass the <pid> of the patient stay
        as additional information"""
    max_grid_length_secs = MAX_IMPUTE_DAYS * 24 * 3600
    # Set non-observed value to NAN
    global_impute_val = np.nan

    # If either the endpoints or the features don't exist, log the failure but do nothing, the missing patients can be
    # latter added as a new group to the output H5
    if patient_df.shape[0] == 0:
        logging.info("WARNING: p{} has missing features, skipping output generation...".format(pid))
        return None

    all_keys = list(set(patient_df.columns.values.tolist()).difference(
        set([DATETIME, PID, "a_temp", "m_pm_1", "m_pm_2"])))

    ts = patient_df[DATETIME]
    ts_arr = np.array(ts)
    n_ts = ts_arr.size

    hr = np.array(patient_df[VAR_IDS_EP["HR"]])
    finite_hr = ts_arr[np.isfinite(hr)]

    if finite_hr.size == 0:
        logging.info("WARNING: Patient {} has no HR, ignoring patient...".format(pid))
        return None

    # Respiratory / circulatory failure, define grid over ICU stay
    ts_min = ts_arr[np.isfinite(hr)][0]
    ts_max = ts_arr[np.isfinite(hr)][-1]

    max_ts_diff = (ts_max - ts_min) / np.timedelta64(1, 's')
    time_grid = np.arange(0.0, min(max_ts_diff + IMPUTATION_PERIOD_SEC, max_grid_length_secs), IMPUTATION_PERIOD_SEC)
    time_grid_abs = [ts_min + pd.Timedelta(seconds=time_grid[idx]) for idx in range(time_grid.size)]
    imputed_df_dict = {}
    imputed_df_dict[PID] = [int(pid)] * time_grid.size
    imputed_df_dict[REL_DATETIME] = time_grid
    imputed_df_dict[DATETIME] = time_grid_abs

    # There is nothing to do if the patient has no records, just return...
    if n_ts == 0:
        logging.info("WARNING: p{} has an empty record, skipping output generation...".format(pid))
        return None

    # Initialize the storage for the imputed time grid, NANs for the non-pharma, 0 for pharma.
    for col in all_keys:
        if col[:2] == "pm":
            imputed_df_dict[col] = np.zeros(time_grid.size)
        elif col[:2] == "vm":
            imputed_df_dict[col] = empty_nan(time_grid.size)
        else:
            logging.info("ERROR: Invalid variable type")
            assert (False)

    imputed_df = pd.DataFrame(imputed_df_dict)
    norm_ts = np.array(ts - ts_min) / np.timedelta64(1, 's')
    weight_id = VAR_IDS_EP["Weight"][0]
    all_keys.remove(weight_id)
    all_keys = [weight_id] + all_keys

    # Impute all variables independently, with the two relevant cases pharma variable and other variable,
    # distinguishable from the variable prefix. We enforce that weight is the first variable to be imputed, so that
    # its time-gridded information can later be used by other custom formulae imputations that depend on it.
    for var_idx, variable in enumerate(all_keys):
        df_var = patient_df[variable]
        assert (n_ts == df_var.shape[0] == norm_ts.size)

        raw_col = np.array(df_var)
        assert (raw_col.size == norm_ts.size)

        observ_idx = np.isfinite(raw_col)
        observ_ts = norm_ts[observ_idx]
        observ_val = raw_col[observ_idx]

        # No values have been observed for this variable, it has to be imputed using the normal value.
        if observ_val.size == 0:
            est_vals = value_empty(time_grid.size, global_impute_val)
            imputed_df[variable] = est_vals
            imputed_df["{}_IMPUTED_STATUS_CUM_COUNT".format(variable)] = np.zeros(time_grid.size)
            imputed_df["{}_IMPUTED_STATUS_TIME_TO".format(variable)] = value_empty(time_grid.size, -1.0)
            continue

        assert (np.isfinite(observ_val).all())
        assert (np.isfinite(observ_ts).all())

        est_vals, cum_count_ts, time_to_last_ms = endpoint_ff.impute_forward_fill_simple(observ_ts, observ_val,
                                                                                         time_grid,
                                                                                         global_impute_val)
        imputed_df[variable] = est_vals
        imputed_df["{}_IMPUTED_STATUS_CUM_COUNT".format(variable)] = cum_count_ts
        imputed_df["{}_IMPUTED_STATUS_TIME_TO".format(variable)] = time_to_last_ms
    return imputed_df


def is_df_sorted(df, colname):
    return (np.array(df[colname].diff().dropna(), dtype=np.float64) >= 0).all()


def execute(batch_id, merged_path, imputed_path):
    """ Batch wrapper that loops through the patients of one the 250 batches"""

    n_skipped_patients = 0
    cand_files = glob.glob(os.path.join(merged_path, "part-{}.parquet".format(batch_id)))
    assert (len(cand_files) == 1)
    source_fpath = cand_files[0]
    no_patient_output = 0
    output_dfs = []
    all_patient_df = pd.read_parquet(source_fpath)

    all_pids = all_patient_df[PID].unique()
    logging.info("Number of patient IDs: {}".format(len(all_pids)))

    for pidx, pid in enumerate(all_pids):
        patient_df = all_patient_df[all_patient_df[PID] == pid]

        if patient_df.shape[0] == 0:
            n_skipped_patients += 1

        if not is_df_sorted(patient_df, DATETIME):
            patient_df = patient_df.sort_values(by=DATETIME, kind="mergesort")

        imputed_df = impute_dynamic_df(patient_df, pid=pid)

        # No data could be output for this patient...
        if imputed_df is None:
            no_patient_output += 1
            continue

        output_dfs.append(imputed_df)

        gc.collect()

        if (pidx + 1) % 100 == 0:
            logging.info("Batch {}: {:.2f} %".format(batch_id, (pidx + 1) / len(all_pids) * 100))
            logging.info("Number of skipped patients: {}".format(n_skipped_patients))
            logging.info("Number of no patients output: {}".format(no_patient_output))

    # Now dump the dictionary of all data-frames horizontally merged
    all_dfs = pd.concat(output_dfs, axis=0)
    all_dfs.to_parquet(os.path.join(imputed_path, "batch_{}.parquet".format(batch_id)))

    return 0
