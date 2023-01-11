import logging
import gin
import json
import hashlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import pickle

from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from recipys.recipe import Recipe
from recipys.selector import all_of, all_numeric_predictors, has_type
from recipys.step import Accumulator, StepHistorical, StepImputeFill, StepScale, StepSklearn


def make_single_split(
    data: dict[pd.DataFrame],
    vars: dict[str],
    cv_repetitions: int,
    repetition_index: int,
    cv_folds: int,
    fold_index: int,
    seed: int = 42,
    debug: bool = False,
    fold_size: int = None,
    test_all: bool = False,
) -> dict[dict[pd.DataFrame]]:
    """Randomly split the data into training, validation, and test set.

    Args:
        data: dictionary containing data divided int OUTCOME, STATIC, and DYNAMIC.
        vars: Contains the names of columns in the data.
        cv_repetitions: Number of times to repeat cross validation.
        repetition_index: Index of the repetition to return.
        cv_folds: Number of folds for cross validation.
        fold_index: Index of the fold to return.
        seed: Random seed.
        debug: Load less data if true.
        test_all: If true, the test set will be the entire dataset.

    Returns:
        Input data divided into 'train', 'val', and 'test'.
    """
    id = vars["GROUP"]
    stays = data["STATIC"][id]
    if debug:
        # Only use 1% of the data
        stays = stays.sample(frac=0.01, random_state=seed)
    labels = data["OUTCOME"][vars["LABEL"]].loc[stays.index]

    if test_all:
        split = {
            "train": stays.iloc[0:0],
            "val": stays.iloc[0:0],
            "test": stays,
        }
    else:
        outer_CV = StratifiedKFold(cv_repetitions, shuffle=True, random_state=seed)
        dev, test = list(outer_CV.split(stays, labels))[repetition_index]

        if fold_size:
            start_index = 0
            end_index = fold_size
            pre_dev = dev[start_index:end_index]
            leave_for_test = dev[end_index:]
            pre_dev_labels = labels.iloc[pre_dev]
            while pre_dev_labels.sum() < cv_folds:
                start_index += fold_size
                end_index += fold_size
                pre_dev = dev[start_index:end_index]
                pre_dev_labels = labels.iloc[pre_dev]
                leave_for_test = np.append(dev[0:start_index], dev[end_index:])
            dev = pre_dev
            test = np.append(test, leave_for_test)

        dev_stays = stays.iloc[dev]
        dev_labels = labels.iloc[dev]

        inner_CV = StratifiedKFold(cv_folds, shuffle=True, random_state=seed)
        train, val = list(inner_CV.split(dev_stays, dev_labels))[fold_index]

        split = {
            "train": dev_stays.iloc[train],
            "val": dev_stays.iloc[val],
            "test": stays.iloc[test],
        }

    data_split = {}
    for fold_name, fold in split.items():  # Loop through train / val / test
        # Loop through DYNAMIC / STATIC / OUTCOME
        # set sort to true to make sure that IDs are reordered after scrambling earlier
        data_split[fold_name] = {
            data_type: data[data_type].merge(fold, on=id, how="right", sort=True) for data_type in data.keys()
        }

    return data_split


def apply_recipe_to_splits(
    recipe: Recipe, data: dict[dict[pd.DataFrame]], type: str, test_all: bool = False
) -> dict[dict[pd.DataFrame]]:
    """Fits and transforms the training data, then transforms the validation and test data with the recipe.

    Args:
        recipe: Object containing info about the data and steps.
        data: Dict containing 'train', 'val', and 'test' and types of data per split.
        type: Whether to apply recipe to dynamic data, static data or outcomes.
        test_all: If true, the test set will be the entire dataset.

    Returns:
        Transformed data divided into 'train', 'val', and 'test'.
    """
    if test_all:
        data["test"][type] = recipe.prep(data["test"][type])
        return data

    data["train"][type] = recipe.prep()
    data["val"][type] = recipe.bake(data["val"][type])
    data["test"][type] = recipe.bake(data["test"][type])
    return data


@gin.configurable("preprocess")
def preprocess_data(
    data_dir: Path,
    file_names: dict[str] = gin.REQUIRED,
    vars: dict[str] = gin.REQUIRED,
    use_features: bool = gin.REQUIRED,
    seed: int = 42,
    debug: bool = False,
    use_cache: bool = False,
    cv_repetitions: int = 5,
    repetition_index: int = 0,
    cv_folds: int = 5,
    fold_size: int = None,
    fold_index: int = 0,
    test_all: bool = False,
) -> dict[dict[pd.DataFrame]]:
    """Perform loading, splitting, imputing and normalising of task data.

    Args:
        data_dir: Path to the directory holding the data.
        file_names: Contains the parquet file names in data_dir.
        vars: Contains the names of columns in the data.
        use_features: Whether to generate features on the dynamic data.
        seed: Random seed.
        debug: Load less data if true.
        use_cache: Cache and use cached preprocessed data if true.
        cv_repetitions: Number of times to repeat cross validation.
        repetition_index: Index of the repetition to return.
        cv_folds: Number of folds to use for cross validation.
        fold_index: Index of the fold to return.
        test_all: If true, the test set will be the entire dataset.

    Returns:
        Preprocessed data as DataFrame in a hierarchical dict with data type (STATIC/DYNAMIC/OUTCOME)
            nested within split (train/val/test).
    """
    cache_dir = data_dir / "cache"
    if fold_size:
        cache_dir = cache_dir / f"T{fold_size}"
    if test_all:
        cache_dir = cache_dir / "test_complete"
    dumped_file_names = json.dumps(file_names, sort_keys=True)
    dumped_vars = json.dumps(vars, sort_keys=True)
    config_string = f"{dumped_file_names}{dumped_vars}{use_features}{seed}{repetition_index}{fold_index}{debug}".encode(
        "utf-8"
    )
    cache_file = cache_dir / hashlib.md5(config_string).hexdigest()

    if use_cache:
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                logging.info(f"Loading cached data from {cache_file}.")
                return pickle.load(f)
        else:
            logging.info(f"No cached data found in {cache_file}, loading raw data.")

    data = {f: pq.read_table(data_dir / file_names[f]).to_pandas() for f in ["STATIC", "DYNAMIC", "OUTCOME"]}

    logging.info("Generating splits.")
    data = make_single_split(
        data,
        vars,
        cv_repetitions,
        repetition_index,
        cv_folds,
        fold_index,
        seed=seed,
        debug=debug,
        fold_size=fold_size,
        test_all=test_all,
    )

    logging.info("Preprocessing static data.")
    sta_rec = Recipe(data["train"]["STATIC"], [], vars["STATIC"])
    sta_rec.add_step(StepScale())
    sta_rec.add_step(StepImputeFill(sel=all_numeric_predictors(), value=0))
    sta_rec.add_step(StepSklearn(SimpleImputer(missing_values=None, strategy="most_frequent"), sel=has_type("object")))
    sta_rec.add_step(StepSklearn(LabelEncoder(), sel=has_type("object"), columnwise=True))

    data = apply_recipe_to_splits(sta_rec, data, "STATIC", test_all=test_all)

    logging.info("Preprocessing dynamic data.")
    dyn_rec = Recipe(data["train"]["DYNAMIC"], [], vars["DYNAMIC"], vars["GROUP"], vars["SEQUENCE"])
    dyn_rec.add_step(StepScale())
    dyn_rec.add_step(StepSklearn(MissingIndicator(), sel=all_of(vars["DYNAMIC"]), in_place=False))
    if use_features:
        dyn_rec.add_step(StepHistorical(sel=all_of(vars["DYNAMIC"]), fun=Accumulator.MIN, suffix="min_hist"))
        dyn_rec.add_step(StepHistorical(sel=all_of(vars["DYNAMIC"]), fun=Accumulator.MAX, suffix="max_hist"))
        dyn_rec.add_step(StepHistorical(sel=all_of(vars["DYNAMIC"]), fun=Accumulator.COUNT, suffix="count_hist"))
        dyn_rec.add_step(StepHistorical(sel=all_of(vars["DYNAMIC"]), fun=Accumulator.MEAN, suffix="mean_hist"))
    dyn_rec.add_step(StepImputeFill(method="ffill"))
    dyn_rec.add_step(StepImputeFill(value=0))

    data = apply_recipe_to_splits(dyn_rec, data, "DYNAMIC", test_all=test_all)

    if use_cache and not cache_file.exists():
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)
        cache_file.touch()
        with open(cache_file, "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        logging.info(f"Cached data in {cache_file}.")

    logging.info("Finished preprocessing.")

    return data
