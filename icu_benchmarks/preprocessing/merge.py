import functools
import logging
from pathlib import Path
from typing import List, Sequence, Mapping

import numpy as np
import pandas as pd

from icu_benchmarks.common import lookups, processing
from icu_benchmarks.common.constants import VARREF_LOWERBOUND, VARREF_UPPERBOUND, PID, VARID, DATETIME, VALUE, \
    INVALID_PHARMA_STATUS, INSTANTANEOUS_STATE, PSEUDO_INSTANTANEOUS_STATE, METAVAR_ID, METAVAR_UNIT, INFID, \
    PHARMAID, PHARMA_ENTERTIME, ENTERTIME, HR_METAVAR_ID, UNITCONVERT_FACTOR, PHARMA_STATUS, PHARMA_DATETIME
from icu_benchmarks.common.datasets import Dataset
from icu_benchmarks.preprocessing.preprocess_pharma import convert_cumul_value_to_rate, drop_duplicates_pharma, \
    process_single_infusion


def _merge_duplicate_measurements(tmp, stddev_dict):
    assert tmp[VALUE].isna().sum() == 0
    assert not tmp.empty

    val_std = tmp[VALUE].std()
    all_vals_equal = (len(tmp) == 1) | (val_std == 0)
    vid = tmp[VARID].iloc[0]

    ret = tmp.iloc[-1]

    if all_vals_equal:
        return ret
    elif vid in stddev_dict.keys() and val_std <= 0.05 * stddev_dict[vid]:
        ret[VALUE] = tmp[VALUE].mean()
        return ret

    ret[DATETIME] = np.nan  # mark for removal
    return ret


def drop_duplicates_non_pharma(df, stddev_dict):
    duplicated_index = df.duplicated([DATETIME, VARID], keep=False)

    df_dup = df[duplicated_index]
    df_non_dup = df[~duplicated_index]

    if df_dup.empty:
        df_ret = df
    else:
        df_dedup = (df_dup.reset_index().
                    groupby([DATETIME, VARID]).
                    apply(_merge_duplicate_measurements, stddev_dict=stddev_dict)
                    )

        df_dedup = df_dedup[~df_dedup[DATETIME].isna()].set_index('index')
        df_ret = pd.concat([df_dedup, df_non_dup]).sort_values([DATETIME, VARID])

    assert (df_ret.duplicated([DATETIME, VARID], keep=False).sum() == 0)

    return df_ret


def drop_out_of_range_values(df, varref):
    """
    df: long-format dataframe of a patient
    varref: variable reference table that contain the lower-bound and upper-bound of values for a subset of variables
    """
    bound_cols = [VARREF_LOWERBOUND, VARREF_UPPERBOUND]

    df_with_bounds = df.merge(varref[bound_cols], left_on=VARID, right_index=True, how='inner')
    df_filtered = df_with_bounds.query(
        f'{VARREF_LOWERBOUND}.isnull() or {VARREF_UPPERBOUND}.isnull() or ({VARREF_LOWERBOUND} <= value and value <= {VARREF_UPPERBOUND})',
        engine='python')

    return df_filtered.drop(columns=bound_cols)


def aggregate_cols(wide_observ, varref):
    metavar_varid_dict = {m: [f"v{vid}" for vid in vars] for (m, vars) in
                          varref.reset_index().groupby(METAVAR_ID)[VARID]}

    metavar_cols = {}
    for vmid, varid_cols in metavar_varid_dict.items():
        varid_cols_avail = list(set(varid_cols).intersection(set(wide_observ.columns)))

        if len(varid_cols_avail) > 1:
            c = wide_observ.loc[:, varid_cols_avail].median(axis=1, numeric_only=True, skipna=True)
        elif len(varid_cols_avail) == 1:
            c = wide_observ.loc[:, varid_cols_avail[0]]
        else:
            c = np.zeros(wide_observ.shape[0])
            c[:] = np.nan

        metavar_cols[f"vm{vmid}"] = c

    wide_observ_new = pd.DataFrame(metavar_cols, index=wide_observ.index)

    return wide_observ_new


def transform_obs_table_fn(observ: pd.DataFrame, lst_vmid: List[int], lst_cumul_vid, varref, general_table):
    valid_variables = set(varref.index)

    observ = observ.loc[observ[VARID].isin(valid_variables)]
    observ = observ[~observ[VALUE].isna()].sort_values([VARID, DATETIME, ENTERTIME])

    observ = drop_out_of_range_values(observ, varref)

    stddev_dict = {v: std for (v, std) in varref["standard_deviation"].items()}
    observ = drop_duplicates_non_pharma(observ, stddev_dict)

    if observ[VARID].isin(lst_cumul_vid).sum() > 0:
        observ = convert_cumul_value_to_rate(observ, lst_cumul_vid, general_table)

    observ.loc[:, VARID] = observ.variableid.apply(lambda x: "v%d" % x)

    if observ.empty:
        return pd.DataFrame()

    wide_observ = (pd.pivot_table(observ, values=VALUE, columns=VARID, index=DATETIME).
                   sort_index())

    wide_aggregated = aggregate_cols(wide_observ, varref)

    binary_vmids = ["vm%d" % x for x in
                    varref[varref[METAVAR_UNIT].apply(lambda x: x == "Binary")][METAVAR_ID].values]
    for col in binary_vmids:
        wide_aggregated.loc[:, col] = wide_aggregated[col].apply(lambda x: x if np.isnan(x) else float(x != 0))

    return wide_aggregated


def transform_pharma_table_fn(pharma: pd.DataFrame, pharmaref, lst_pmid):
    pharma_ids = set(pharmaref[PHARMAID].unique())
    pharma = pharma.loc[pharma[PHARMAID].isin(pharma_ids)].copy()

    pharma.drop(pharma.index[pharma[PHARMA_STATUS].isin(INVALID_PHARMA_STATUS)], inplace=True)
    pharma.sort_values([PHARMAID, PHARMA_DATETIME, PHARMA_ENTERTIME], inplace=True)
    pharma.loc[:, PHARMA_STATUS] = pharma[PHARMA_STATUS].replace(PSEUDO_INSTANTANEOUS_STATE, INSTANTANEOUS_STATE)
    pharma = drop_duplicates_pharma(pharma)

    if pharma.empty:
        return pd.DataFrame()
    else:
        wide_pharma = []
        for pharmaid in pharma[PHARMAID].unique():
            pharma_acting_period = pharmaref[pharmaref[PHARMAID] == pharmaid].iloc[0].pharmaactingperiod_min
            infusion_rate = []
            for infusionid in pharma[pharma[PHARMAID] == pharmaid][INFID].unique():
                tmp_pharma = pharma[(pharma[PHARMAID] == pharmaid) & (pharma[INFID] == infusionid)].copy()
                infusion_rate.append(process_single_infusion(tmp_pharma, pharma_acting_period))
            infusion_rate = pd.concat(infusion_rate, axis=1).sort_index()
            infusion_rate = infusion_rate.sum(axis=1).to_frame(name="p%d" % pharmaid)
            wide_pharma.append(infusion_rate)
        wide_pharma = pd.concat(wide_pharma, axis=1).sort_index()
        for pmid in lst_pmid:
            cols = ['p%d' % x for x in pharmaref[pharmaref[METAVAR_ID] == pmid][PHARMAID]]
            if np.isin(wide_pharma.columns, cols).sum() == 0:
                wide_pharma.loc[:, "pm%d" % pmid] = np.nan
            else:
                if pharmaref[pharmaref[METAVAR_ID] == pmid][UNITCONVERT_FACTOR].notnull().sum() > 0:
                    unitconverters = [pharmaref[(pharmaref[METAVAR_ID] == pmid) & (
                            pharmaref[PHARMAID] == int(c[1:]))].iloc[0][UNITCONVERT_FACTOR] for c in
                                      wide_pharma.columns[
                                          np.isin(wide_pharma.columns, cols)]]
                    wide_pharma.loc[:, "pm%d" % pmid] = (wide_pharma[
                                                             wide_pharma.columns[np.isin(wide_pharma.columns,
                                                                                         cols)]] * unitconverters).sum(
                        axis=1)
                else:
                    wide_pharma.loc[:, "pm%d" % pmid] = wide_pharma[
                        wide_pharma.columns[np.isin(wide_pharma.columns, cols)]].sum(axis=1)
                wide_pharma.loc[wide_pharma.index[
                                    wide_pharma[wide_pharma.columns[np.isin(wide_pharma.columns, cols)]].notnull().sum(
                                        axis=1) == 0], "pm%d" % pmid] = np.nan
                wide_pharma.drop(wide_pharma.columns[np.isin(wide_pharma.columns, cols)], axis=1, inplace=True)

        binary_pmids = ["pm%d" % x for x in
                        pharmaref[pharmaref[METAVAR_UNIT].apply(lambda x: x == "Binary")][METAVAR_ID].values]
        for col in binary_pmids:
            wide_pharma.loc[:, col] = wide_pharma[col].apply(lambda x: x if np.isnan(x) else float(x != 0))

        return wide_pharma


def length_of_stay_filtering(df, admission_time):
    rec_adm_time = admission_time
    if df[HR_METAVAR_ID].notnull().sum() > 0:
        hr_first_meas_time = df.loc[df[df[HR_METAVAR_ID].notnull()].index[0], DATETIME]
        esti_adm_time = max(rec_adm_time, hr_first_meas_time)
        esti_disc_time = df.loc[df[df[HR_METAVAR_ID].notnull()].index[-1], DATETIME]
    else:
        esti_adm_time = rec_adm_time
        esti_disc_time = None

    df = df.drop(df.index[df[DATETIME] < esti_adm_time])
    if esti_disc_time is not None:
        df = df.drop(df.index[df[DATETIME] > esti_disc_time])

    if not df.empty:
        los = (df.iloc[-1][DATETIME] - df.iloc[0][DATETIME]) / np.timedelta64(24, "h")
        assert (los < 32)
    return df


def combine_obs_and_pharma_tables(dfs: Sequence[Mapping[int, pd.DataFrame]], columns, admission_times):
    assert len(dfs) == 2, "Expecting exactly two dictionaries"

    all_ids = set.union(*(set(m.keys()) for m in dfs))

    obs_dfs, pharma_dfs = dfs

    df_patients = []
    for pid in all_ids:
        ph_df = pharma_dfs.get(pid, pd.DataFrame())

        obs_df = obs_dfs.get(pid, pd.DataFrame())
        df_pid = pd.concat([obs_df, ph_df], axis=1)

        if df_pid.empty:
            continue

        df_pid[PID] = pid
        df_pid = df_pid.reset_index().rename(columns={"index": DATETIME, PHARMA_DATETIME: DATETIME})
        df_pid.loc[:, list(set(columns).difference(set(df_pid.columns)))] = np.nan
        df_pid = df_pid[columns]  # reorder cols

        assert ((df_pid.iloc[:, 2:].notnull().sum(axis=1) == 0).sum() == 0)

        df_pid = length_of_stay_filtering(df_pid, admission_times[pid])

        df_patients.append(df_pid)

    df = pd.concat(df_patients).sort_values([PID, DATETIME])

    # explicitly setting type to be consistent across chunks (important to process using pyspark)
    df[PID] = df[PID].astype('int32')
    df.iloc[:, 2:] = df.iloc[:, 2:].astype('float64')

    return df


def merge_tables(observation_tables_path: Path, pharma_path: Path, general_data_path: Path,
                 varref_path: Path, output: Path, workers: int = 1):
    """
    entry point for preprocessing the data as published on physionet
    """

    logging.getLogger().setLevel(logging.INFO)

    observation_ds = Dataset(observation_tables_path)
    pharma_ds = Dataset(pharma_path)

    observed_tables = observation_ds.list_parts()
    pharma_tables = pharma_ds.list_parts()

    output_ds = Dataset(output)
    output_ds.prepare(single_part=False)

    assert len(observed_tables) == len(pharma_tables)
    assert [f.name for f in observed_tables] == [f.name for f in pharma_tables]

    varref, pharmaref = lookups.read_reference_table(varref_path)

    general_table = lookups.read_general_table(general_data_path)

    lst_vmid = np.sort(varref[METAVAR_ID].unique())
    lst_pmid = np.sort(pharmaref[METAVAR_ID].unique())
    lst_cumul_vid = varref[
        varref.variablename.apply(lambda x: "/c" in x.lower() or "cumul" in x.lower())].index.tolist()

    obs_per_pat = functools.partial(transform_obs_table_fn, lst_vmid=lst_vmid, lst_cumul_vid=lst_cumul_vid,
                                    varref=varref, general_table=general_table)
    transform_pharma_table_fn_per_pat = functools.partial(transform_pharma_table_fn, pharmaref=pharmaref,
                                                          lst_pmid=lst_pmid)

    simple_read_parquet = lambda path: pd.read_parquet(path)
    simple_write_parquet = lambda df, part: df.to_parquet(output / part, index=False)

    logging.info(f"start processing using {workers} worker")

    output_cols = [PID, DATETIME] + [f"vm{vid}" for vid in sorted(varref[METAVAR_ID].unique())] + \
                  [f"pm{vid}" for vid in sorted(pharmaref[METAVAR_ID].unique())]

    admission_times = {pid: adm_time for (pid, adm_time) in general_table['admissiontime'].items()}

    processing.map_and_combine_patient_dfs(
        [obs_per_pat, transform_pharma_table_fn_per_pat],
        [observed_tables, pharma_tables],
        [simple_read_parquet, simple_read_parquet],
        functools.partial(combine_obs_and_pharma_tables, columns=output_cols, admission_times=admission_times),
        simple_write_parquet,
        workers
    )

    output_ds.mark_done()
