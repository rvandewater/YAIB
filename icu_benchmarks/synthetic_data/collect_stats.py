#!/usr/bin/env python

import argparse
from pathlib import Path

# pip install pyspark (not including into environment.yml as it is a rather big dependency and otherwise unused in the code
import pyspark.sql
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql import functions as sf

from icu_benchmarks.common import constants


def get_spark_session(cores, memory_per_executor):
    driver_mem = cores * memory_per_executor + 2000  # + some driver overhead

    cfg = (SparkConf().set("spark.driver.memory", "{}m".format(driver_mem)).
           set("spark.executor.memory", "{}m".format(memory_per_executor)).
           set("spark.master", "local[{}]".format(cores)).
           set("spark.sql.execution.arrow.enabled", str(True))
           )

    return (SparkSession.
            builder.
            config(conf=cfg).
            getOrCreate())


def _collect_variable_stats(df, grp_var, val_var) -> pyspark.sql.DataFrame:
    return df.groupby(grp_var).agg(sf.mean(val_var).alias('mean'),
                                   sf.stddev(val_var).alias('standard_deviation'),
                                   sf.count(val_var).alias('count'),
                                   sf.min(val_var).alias('min'),
                                   sf.max(val_var).alias('max'),
                                   (sf.sum(sf.col(val_var) - sf.floor(val_var)).alias(
                                       'rounding_remainders')))


def collect_stats_from_dataset(dataset_path, grp_var, val_var, output_path, spark):
    df = spark.read.parquet(str(dataset_path))

    stats = (_collect_variable_stats(df, grp_var, val_var).
             toPandas().
             rename(columns={grp_var: constants.VARID}))

    stats.to_parquet(output_path, index=False)


def time_between_events(dataset_path, time_col, output_path, spark):
    df = spark.read.parquet(str(dataset_path))

    w = Window.partitionBy(constants.PID).orderBy(sf.col(time_col).asc())

    time_diffs = df.withColumn('time_diff', (sf.unix_timestamp(time_col) -
                                             sf.unix_timestamp(sf.lag(time_col, 1).over(w))))

    time_diffs_counts = (time_diffs.groupby('time_diff').
                         count().
                         toPandas().
                         sort_values('time_diff'))

    time_diffs_counts = time_diffs_counts[~time_diffs_counts['time_diff'].isna()]
    time_diffs_counts.to_parquet(output_path, index=False)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Aggregating stats from original HiRID datast')

    parser.add_argument('hirid_data_root', help="Path to the decompressed parquet data directory as published on physionet", type=Path)
    parser.add_argument('output_dir', help="output directory", type=Path)
    parser.add_argument('--cores', help="nr cores to use", type=int, default=4)
    parser.add_argument('--mem-per-core', help="memory per core", type=int, default=2048)
    return parser

def main():
    args = get_parser().parse_args()

    hirid_data_root = args.hirid_data_root
    obs_tables_path = hirid_data_root / 'observation_tables'
    pharma_records_path = hirid_data_root / 'pharma_records'

    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    with get_spark_session(args.cores, args.mem_per_core) as spark:
        collect_stats_from_dataset(obs_tables_path,
                                   constants.VARID, constants.VALUE,
                                   output_dir / 'observation_tables_stats.parquet',
                                   spark)

        collect_stats_from_dataset(pharma_records_path,
                                   'pharmaid', 'givendose',
                                   output_dir / 'pharma_records_stats.parquet',
                                   spark)

        time_between_events(obs_tables_path, 'datetime', output_dir / 'time_diff_stats_obs.parquet', spark)
        time_between_events(pharma_records_path, 'givenat', output_dir / 'time_diff_stats_pharma.parquet', spark)


if __name__ == "__main__":
    main()
