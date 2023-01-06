import functools
import re
from pathlib import Path

import gin

from icu_benchmarks.common import processing, utils
from icu_benchmarks.common.datasets import Dataset
from icu_benchmarks.labels import label_benchmark

batch_parquet_pattern = re.compile('batch_([0-9]+).parquet')


def process_chunk(part_path: Path, endpoints_path: Path, imputation_for_endpoints_path: Path,
                  output_dir: Path, static_path: Path,
                  horizon):
    batch_id = int(re.match(batch_parquet_pattern, part_path.name).groups()[0])

    label_benchmark.label_gen_benchmark(batch_id=batch_id, label_path=output_dir, endpoint_path=endpoints_path,
                                        imputed_path=imputation_for_endpoints_path, static_path=static_path,
                                        horizon=horizon)


def generate_labels(endpoints_path: Path, imputation_for_endpoints_path: Path,
                    exended_general_data_path: Path, output_dir: Path, nr_workers=1, horizon=12):
    input_ds = Dataset(endpoints_path, part_re=batch_parquet_pattern)
    parts = input_ds.list_parts()
    output_ds = Dataset(output_dir)
    output_ds.prepare()

    processing.exec_parallel_on_parts(functools.partial(process_chunk, endpoints_path=endpoints_path,
                                                        imputation_for_endpoints_path=imputation_for_endpoints_path,
                                                        output_dir=output_dir, static_path=exended_general_data_path,
                                                        horizon=horizon),
                                      parts, nr_workers)

    output_ds.mark_done()
