import logging
import gin
import json
import hashlib
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import pickle

from sklearn.model_selection import KFold

from icu_benchmarks.data.preprocessor import Preprocessor, DefaultPreprocessor


def make_single_split(
    data: dict[pd.DataFrame],
    vars: dict[str],
    num_folds: int,
    fold_index: int,
    seed: int = 42,
    debug: bool = False,
) -> dict[dict[pd.DataFrame]]:
    """Randomly split the data into training, validation, and test set.

    Args:
        data: dictionary containing data divided int OUTCOME, STATIC, and DYNAMIC.
        vars: Contains the names of columns in the data.
        num_folds: Number of folds for cross validation.
        seed: Random seed.
        debug: Load less data if true.

    Returns:
        Input data divided into 'train', 'val', and 'test'.
    """
    id = vars["GROUP"]
    fraction_to_load = 1 if not debug else 0.01
    stays = data["STATIC"][[id]].sample(frac=fraction_to_load, random_state=seed)

    outer = KFold(num_folds, shuffle=True, random_state=seed)

    train, test_and_val = list(outer.split(stays))[fold_index]
    test, val = np.array_split(test_and_val, 2)

    split = {
        "train": stays.iloc[train],
        "val": stays.iloc[test],
        "test": stays.iloc[val],
    }
    data_split = {}

    for fold in split.keys():  # Loop through train / val / test
        # Loop through DYNAMIC / STATIC / OUTCOME
        # set sort to true to make sure that IDs are reordered after scrambling earlier
        data_split[fold] = {
            data_type: data[data_type].merge(split[fold], on=id, how="right", sort=True) for data_type in data.keys()
        }

    return data_split






@gin.configurable("preprocess")
def preprocess_data(
    data_dir: Path,
    file_names: dict[str] = gin.REQUIRED,
    preprocessor: Preprocessor = DefaultPreprocessor,
    vars: dict[str] = gin.REQUIRED,
    seed: int = 42,
    debug: bool = False,
    use_cache: bool = False,
    num_folds: int = 5,
    fold_index: int = 0,
) -> dict[dict[pd.DataFrame]]:
    """Perform loading, splitting, imputing and normalising of task data.

    Args:
        scaling: Use scaling if true.
        preprocessor: Define the preprocessor.
        data_dir: Path to the directory holding the data.
        file_names: Contains the parquet file names in data_dir.
        vars: Contains the names of columns in the data.
        use_features: Whether to generate features on the dynamic data.
        seed: Random seed.
        debug: Load less data if true.
        use_cache: Cache and use cached preprocessed data if true.
        num_folds: Number of folds to use for cross validation.
        fold_index: Index of the fold to return.

    Returns:
        Preprocessed data as DataFrame in a hierarchical dict with data type (STATIC/DYNAMIC/OUTCOME)
            nested within split (train/val/test).
    """
    # print(preprocessor)
    cache_dir = data_dir / "cache"
    dumped_file_names = json.dumps(file_names, sort_keys=True)
    dumped_vars = json.dumps(vars, sort_keys=True)

    use_features = True
    config_string = f"{dumped_file_names}{dumped_vars}{use_features}{seed}{fold_index}{debug}".encode("utf-8")
    cache_file = cache_dir / hashlib.md5(config_string).hexdigest()

    use_cache = False

    if use_cache:
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                logging.info(f"Loading cached data from {cache_file}.")
                return pickle.load(f)
        else:
            logging.info(f"No cached data found in {cache_file}, loading raw data.")

    data = {f: pq.read_table(data_dir / file_names[f]).to_pandas() for f in file_names.keys()}

    logging.info("Generating splits.")
    data = make_single_split(data, vars, num_folds, fold_index, seed=seed, debug=debug)

    preprocessing = preprocessor(data, seed, vars)
    data = preprocessing.apply()

    caching(cache_dir, cache_file, data, use_cache)

    logging.info("Finished preprocessing.")

    return data


def caching(cache_dir, cache_file, data, use_cache):
    if use_cache and not cache_file.exists():
        if not cache_dir.exists():
            cache_dir.mkdir()
        cache_file.touch()
        with open(cache_file, "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        logging.info(f"Cached data in {cache_file}.")
