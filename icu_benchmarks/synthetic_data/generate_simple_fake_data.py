#!/usr/bin/env python
# coding: utf-8

import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from icu_benchmarks.common import constants


def get_date_in_range(start: datetime.datetime, end: datetime.datetime, n):
    return pd.to_datetime(np.random.randint(start.timestamp(), end.timestamp(), n), unit='s')


def generate_summaries_dict(summaries_df, varref_df):
    merged = summaries_df.merge(varref_df, how='left', on=constants.VARID, suffixes=('_calc', '_ref'))

    # take data from varref if available, otherwise take stats as observed in the entire dataset
    merged['mean_ref'] = merged['mean_ref'].fillna(merged['mean_calc'])
    merged['standard_deviation_ref'] = merged['standard_deviation_ref'].fillna(merged['standard_deviation_calc'])
    merged['lowerbound_ref'] = merged['lowerbound'].fillna(merged['min'])
    merged['upperbound_ref'] = merged['upperbound'].fillna(merged['max'])

    return {r[constants.VARID]: (r['mean_ref'], r['standard_deviation_ref'], r['lowerbound_ref'], r['upperbound_ref'],
                              r['rounding_remainders'] < 0.5) # proxy for knowing whether the variable has integer values
            for _, r in merged.iterrows()}


def get_timestamps_diffs(duration, dists_df, max_length):
    cur_time = 0

    diffs = []

    weights = dists_df['count'] / dists_df['count'].sum()

    while cur_time < duration and len(diffs) < max_length:
        d = np.random.choice(dists_df['time_diff'], p=weights)
        duration += d

        diffs.append(d)

    return diffs


def get_varids(n, summaries_df):
    weights = summaries_df['count'] / summaries_df['count'].sum()

    return np.random.choice(summaries_df[constants.VARID], p=weights, size=n)


def sample_values(varid, summaries):
    m, std, min_val, max_val, is_integer_val = summaries[varid]

    digits = 0 if is_integer_val else 2

    vals = np.round(np.random.normal(loc=m, scale=std), digits)
    return np.clip(vals, a_min=min_val, a_max=max_val)


def generate_fake_general_data(nr_patients):
    pids = pd.Series(range(0, nr_patients)) + 1

    admission_times = get_date_in_range(datetime.datetime(2100, 1, 1), datetime.datetime(2199, 1, 1), nr_patients)

    return pd.DataFrame({constants.PID: pids,
                         'admissiontime': admission_times,
                         'sex': np.random.choice(['M', 'F'], size=nr_patients),
                         'age': np.random.randint(4, 18, size=nr_patients) * 5,  # ages 20-90 in 5y
                         'discharge_status': np.random.choice(['alive', 'dead'], size=nr_patients)})


def get_fake_obs_data(pid, duration, admission_time, summaries_df, time_diff_dists, varref_df):
    ts = get_timestamps_diffs(duration, time_diff_dists, 10000)

    length = len(ts)
    datetimes = pd.Series(pd.to_datetime(np.cumsum(ts) + admission_time.timestamp(), unit='s')).dt.floor('s')

    var_ids = get_varids(length, summaries_df)
    summaries_dict = generate_summaries_dict(summaries_df, varref_df)

    values = [sample_values(vid, summaries_dict) for vid in var_ids]

    return pd.DataFrame({
        'datetime': datetimes,
        'entertime': datetimes,  # set equal to 'datetime' for simplicity
        constants.PID: pid,
        'status': 8,
        'stringvalue': None,
        'type': None,
        'value': values,
        constants.VARID: var_ids
    })


def get_fake_pharma_data(pid, duration, admission_time, summaries_df, time_diff_dists, varref_df):
    ts = get_timestamps_diffs(duration, time_diff_dists, 500)

    length = len(ts)
    datetimes = pd.Series(pd.to_datetime(np.cumsum(ts) + admission_time.timestamp(), unit='s')).dt.floor('s')

    var_ids = get_varids(length, summaries_df)
    summaries_dict = generate_summaries_dict(summaries_df, varref_df)

    values = [sample_values(vid, summaries_dict) for vid in var_ids]

    return pd.DataFrame({
        constants.PID: pid,
        'pharmaid': var_ids,
        'givenat': datetimes,
        'enteredentryat': datetimes,  # set equal to 'datetime' for simplicity
        'givendose': values,
        'cumulativedose': np.nan,
        'fluidamount_calc': np.nan,
        'cumulfluidamount_calc': np.nan,
        'doseunit': np.nan,
        'recordstatus': 8,
        'infusionid': np.random.randint(0, 10000, size=length),
        'typeid': 1,
    })


def _write_part(df, path):
    path.mkdir(exist_ok=True, parents=True)
    df.to_parquet(path / 'part-0.parquet', index=False)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Generate fake data')

    parser.add_argument('stats_dir', help="output dir of collect_stats.py", type=Path)
    parser.add_argument('output_dir', help="output dir of collect_stats.py", type=Path)
    parser.add_argument('--var-ref-path', help="Path to load the variable references from", type=Path)
    parser.add_argument('--seed', help="random seed", type=int, default=40510)
    parser.add_argument('--nr-patients', help='number of patients to generate', default=10)

    return parser

def main():
    args = get_parser().parse_args()

    stats_dir = args.stats_dir
    output_dir = args.output_dir

    var_ref_path = args.var_ref_path
    nr_patients = args.nr_patients

    np.random.seed(args.seed)

    df_general_fake = generate_fake_general_data(nr_patients)
    _write_part(df_general_fake, output_dir / 'general_table')

    length_of_stay = np.random.randint(4 * 3600, 10 * 3600, nr_patients)  # between 4 and 10 hours

    time_diff_dists_obs = pd.read_parquet(stats_dir / 'time_diff_stats_obs.parquet')
    time_diff_dists_pharma = pd.read_parquet(stats_dir / 'time_diff_stats_pharma.parquet')

    obs_summaries_df = pd.read_parquet(stats_dir / 'observation_tables_stats.parquet')

    pharma_summaries_df = pd.read_parquet(stats_dir / 'pharma_records_stats.parquet')

    varref_df = pd.read_csv(var_ref_path, sep='\t')

    dfs = []
    for pid in df_general_fake[constants.PID]:
        df_fake = get_fake_obs_data(pid,
                                    length_of_stay[pid - 1],
                                    df_general_fake.query(f'patientid == {pid}')['admissiontime'].iloc[0],
                                    obs_summaries_df,
                                    time_diff_dists_obs,
                                    varref_df)

        dfs.append(df_fake)

    df_obs_fake = pd.concat(dfs)
    _write_part(df_obs_fake, output_dir / 'observation_tables')

    dfs = []
    for pid in df_general_fake[constants.PID]:
        df_fake = get_fake_pharma_data(pid,
                                       length_of_stay[pid - 1],
                                       df_general_fake.query(f'patientid == {pid}')['admissiontime'].iloc[0],
                                       pharma_summaries_df,
                                       time_diff_dists_pharma,
                                       varref_df)

        dfs.append(df_fake)

    df_pharma_fake = pd.concat(dfs)
    _write_part(df_pharma_fake, output_dir / 'pharma_records')


if __name__ == "__main__":
    main()
