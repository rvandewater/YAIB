import logging
import gin
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import pickle

from icu_benchmarks.recipes.recipe import Recipe
from icu_benchmarks.recipes.selector import all_of
from icu_benchmarks.recipes.step import Accumulator, StepHistorical, StepImputeFill, StepScale


@gin.configurable("splits")
def make_single_split(
    data: dict[pd.DataFrame],
    train_pct: float = 0.7,
    val_pct: float = 0.1,
    seed: int = 42,
    vars: dict[str] = gin.REQUIRED,
    debug: bool = False,
) -> dict[dict[pd.DataFrame]]:
    """Randomly split the data into training, validation, and test set.

    Args:
        data: dictionary containing data divided int OUTCOME, STATIC, and DYNAMIC.
        train_pct: Proportion of stays assigned to training fold.
        val_pct: Proportion of stays assigned to validation fold.
        seed: Random seed.
        vars: Contains the names of columns in the data.
        debug: Load less data if true.

    Returns:
        Input data divided into 'train', 'val', and 'test'.
    """
    id = vars["GROUP"]
    fraction_to_load = 1 if not debug else 0.01
    stays = data["STATIC"][[id]].sample(frac=fraction_to_load, random_state=seed)

    num_stays = len(stays)
    delims = (num_stays * np.array([0, train_pct, train_pct + val_pct, 1])).astype(int)

    splits = {"train": {}, "val": {}, "test": {}}
    for i, fold in enumerate(splits.keys()):
        # Loop through train / val / test
        stays_in_fold = stays.iloc[delims[i]:delims[i + 1], :]
        for data_type in data.keys():
            # Loop through DYNAMIC / STATIC / OUTCOME
            # set sort to true to make sure that IDs are reordered after scrambling earlier
            splits[fold][data_type] = data[data_type].merge(stays_in_fold, on=id, how="right", sort=True)

    return splits


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
    data["test"][type] = recipe.prep(data["test"][type])
    return data


@gin.configurable("preprocess")
def preprocess_data(
    data_dir: Path,
    file_names: dict[str] = gin.REQUIRED,
    use_features: bool = gin.REQUIRED,
    vars: dict[str] = gin.REQUIRED,
    seed: int = 42,
    debug: bool = False,
    use_cache: bool = False,
) -> dict[dict[pd.DataFrame]]:
    """Perform loading, splitting, imputing and normalising of task data.

    Args:
        data_dir: Path to the directory holding the data.
        file_names: Contains the parquet file names in data_dir.
        use_features: Whether to generate features on the dynamic data.
        vars: Contains the names of columns in the data.
        seed: Random seed.
        debug: Load less data if true.
        use_cache: Cache and use cached preprocessed data if true.

    Returns:
        Preprocessed data as DataFrame in a hierarchical dict with data type (STATIC/DYNAMIC/OUTCOME)
            nested within split (train/val/test).
    """
    cache_dir = data_dir / "cache"
    cache_file = cache_dir / f"seed_{seed}"

    if use_cache:
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                logging.info(f"Loading cached data from {cache_file}.")
                return pickle.load(f)
        else:
            logging.info(f"No cached data found in {cache_file}, loading raw data.")

    data = {f: pq.read_table(data_dir / file_names[f]).to_pandas() for f in ["STATIC", "DYNAMIC", "OUTCOME"]}

    logging.info("Generating splits.")
    data = make_single_split(data, seed=seed, debug=debug)

    logging.info("Preprocessing static data.")
    sta_rec = Recipe(data["train"]["STATIC"], [], vars["STATIC"])
    sta_rec.add_step(StepScale())
    sta_rec.add_step(StepImputeFill(value=0))

    data = apply_recipe_to_splits(sta_rec, data, "STATIC")

    logging.info("Preprocessing dynamic data.")
    dyn_rec = Recipe(data["train"]["DYNAMIC"], [], vars["DYNAMIC"], vars["GROUP"], vars["SEQUENCE"])
    dyn_rec.add_step(StepScale())
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
