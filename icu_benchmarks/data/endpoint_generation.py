import functools
import re
from pathlib import Path

import gin

from icu_benchmarks.common import processing, utils
from icu_benchmarks.common.datasets import Dataset
from icu_benchmarks.endpoints import endpoint_benchmark


def process_endpoint_chunk(part_path: Path, imputed_path: Path, merged_path: Path, output_dir: Path):

    # modify config
    batch_id = int(re.match('part-([0-9]+).parquet', part_path.name).groups()[0])

    endpoint_benchmark.endpoint_gen_benchmark(batch_id=batch_id, endpoint_path=output_dir,
                                              imputed_path=imputed_path, merged_path=merged_path)


def generate_endpoints(merged_path, imputed_path, output_dir, nr_workers=1):

    merged_ds = Dataset(merged_path)
    parts = merged_ds.list_parts()

    output_ds = Dataset(output_dir)
    output_ds.prepare()

    processing.exec_parallel_on_parts(functools.partial(process_endpoint_chunk, imputed_path=imputed_path,
                                                        merged_path=merged_path, output_dir=output_dir),
                                      parts, nr_workers)
    output_ds.mark_done()
