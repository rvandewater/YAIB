import gc
from pathlib import Path
from typing import Sequence

import pathos
import tqdm

from icu_benchmarks.common import constants


def map_and_combine_patient_dfs(map_pat_fns, part_lists, reading_fns, combine_fn, writing_fn, workers) -> None:

    def _process_parts(paths_same_part: Sequence[Path]):
        assert len(paths_same_part) == len(reading_fns) == len(map_pat_fns)

        part_name = paths_same_part[0].name
        dfs_mapped = []
        for (path, read_fn, map_fn) in zip(paths_same_part, reading_fns, map_pat_fns):
            df = read_fn(path)

            df_mapped = {pid: map_fn(df_pat) for (pid, df_pat) in df.groupby(constants.PID)}
            dfs_mapped.append(df_mapped)

        df_ret = combine_fn(dfs_mapped)
        writing_fn(df_ret, part_name)

    all_paths_same_part = zip(*part_lists)
    exec_parallel_on_parts(_process_parts, all_paths_same_part, workers)


def map_patient_df(map_pat_fn, part_list, reading_fn, writing_fn, workers) -> None:
    map_df(lambda df: df.groupby(constants.PID).apply(map_pat_fn),
           part_list,
           reading_fn,
           writing_fn,
           workers)


def map_df(map_fn, part_list, reading_fn, writing_fn, workers) -> None:
    def _process_part(path: Path):
        # we are doing the file I/O in the subprocess, not in the main process in order to avoid having to transfer
        # the data frame in memory from the main process to the subprocess
        df = reading_fn(path)

        df_ret = map_fn(df)

        part = path.name
        writing_fn(df_ret, part)
        gc.collect()

    exec_parallel_on_parts(_process_part, part_list, workers)


def map_reduce_patient_df(map_pat_fn, part_list, reading_fn, reduce_fn, workers):
    def _process_part(path: Path):
        df = reading_fn(path)

        return reduce_fn(map_pat_fn(df_pat) for (_, df_pat) in df.groupby(constants.PID))
    return exec_parallel_on_parts(_process_part, part_list, workers)


def exec_parallel_on_parts(fnc, part_list, workers):
    if workers > 1:
        # using pathos as it uses a more robust serialization than the default multiprocessing
        with pathos.multiprocessing.Pool(workers) as pool:
            return list(tqdm.tqdm(pool.imap(fnc, part_list)))
    else:
        return [fnc(part) for part in tqdm.tqdm(part_list)]
