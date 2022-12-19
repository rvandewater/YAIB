import logging
import gin
import json
import hashlib
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import pickle

from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

from recipys.recipe import Recipe
from recipys.selector import all_of, all_numeric_predictors, has_type
from recipys.step import Accumulator, StepHistorical, StepImputeFill, StepScale, StepSklearn


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


def apply_recipe_to_splits(recipe: Recipe, data: dict[dict[pd.DataFrame]], type: str) -> dict[dict[pd.DataFrame]]:
    """Fits and transforms the training data, then transforms the validation and test data with the recipe.

    Args:
        recipe: Object containing info about the data and steps.
        data: Dict containing 'train', 'val', and 'test' and types of data per split.
        type: Whether to apply recipe to dynamic data, static data or outcomes.

    Returns:
        Transformed data divided into 'train', 'val', and 'test'.
    """
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
    num_folds: int = 5,
    fold_index: int = 0,
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
        num_folds: Number of folds to use for cross validation.
        fold_index: Index of the fold to return.

    Returns:
        Preprocessed data as DataFrame in a hierarchical dict with data type (STATIC/DYNAMIC/OUTCOME)
            nested within split (train/val/test).
    """
    cache_dir = data_dir / "cache"
    dumped_file_names = json.dumps(file_names, sort_keys=True)
    dumped_vars = json.dumps(vars, sort_keys=True)
    config_string = f"{dumped_file_names}{dumped_vars}{use_features}{seed}{fold_index}{debug}".encode("utf-8")
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
    data = make_single_split(data, vars, num_folds, fold_index, seed=seed, debug=debug)

    logging.info("Preprocessing static data.")
    sta_rec = Recipe(data["train"]["STATIC"], [], vars["STATIC"])
    sta_rec.add_step(StepScale())
    sta_rec.add_step(StepImputeFill(sel=all_numeric_predictors(), value=0))
    sta_rec.add_step(StepSklearn(SimpleImputer(missing_values=None, strategy="most_frequent"), sel=has_type("object")))
    sta_rec.add_step(StepSklearn(LabelEncoder(), sel=has_type("object"), columnwise=True))

    data = apply_recipe_to_splits(sta_rec, data, "STATIC")

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

    data = apply_recipe_to_splits(dyn_rec, data, "DYNAMIC")

    if use_cache and not cache_file.exists():
        if not cache_dir.exists():
            cache_dir.mkdir()
        cache_file.touch()
        with open(cache_file, "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        logging.info(f"Cached data in {cache_file}.")

    logging.info("Finished preprocessing.")

    return data
