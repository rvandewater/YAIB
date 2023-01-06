import functools
import re
from pathlib import Path

import gin

from icu_benchmarks.common import processing, utils
from icu_benchmarks.common.datasets import Dataset
from icu_benchmarks.imputation import impute_one_batch


def process_chunk(part_path: Path, output_dir: Path):
    batch_id = int(re.match('part-([0-9]+).parquet', part_path.name).groups()[0])
    merged_path = str(part_path.parent)
    imputed_path = output_dir

    impute_one_batch.execute(batch_id=batch_id, merged_path=merged_path, imputed_path=imputed_path)


def impute_for_endpoints(merged_path, output_dir, nr_workers=1):
    input_ds = Dataset(merged_path)
    parts = input_ds.list_parts()
    output_ds = Dataset(output_dir)
    output_ds.prepare()

    processing.exec_parallel_on_parts(functools.partial(process_chunk, output_dir=output_dir),
                                      parts, nr_workers)

    output_ds.mark_done()
