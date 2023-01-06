from pathlib import Path

import pandas as pd

from icu_benchmarks.common.constants import PID, VARID, VALUE


def _read_general_table(general_table_path: Path):
    if general_table_path.name.endswith('.csv'):
        df = pd.read_csv(general_table_path)
        df['admissiontime'] = pd.to_datetime(df['admissiontime'])
        return df
    else:
        return pd.read_parquet(general_table_path)


def generate_extended_general_table(observation_tables_path, general_table_path, output_path):
    df_general_table = _read_general_table(Path(general_table_path))

    df_obs_tables = pd.read_parquet(observation_tables_path,
                                    engine="pyarrow",
                                    columns=[PID, VARID, VALUE],
                                    filters=[(VARID, "in", [10000450, 9990002, 9990004])])

    df_dropped = df_obs_tables.drop_duplicates([PID, VARID])

    df_additional_cols = (pd.pivot_table(df_dropped, index=PID, columns=VARID, values=VALUE).
                          rename(columns={10000450: "height",
                                          9990002: "APACHE II Group",
                                          9990004: "APACHE IV Group"}))

    df_out = df_general_table.merge(df_additional_cols, how="left", left_on=PID, right_index=True).reset_index(
        drop=True)
    df_out[PID] = df_out[PID].astype('int32')
    df_out.to_parquet(output_path)
