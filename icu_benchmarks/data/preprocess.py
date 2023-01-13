import logging
import gin
import json
import hashlib
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import pickle
import wandb
from typing import Dict, List

from recipys.recipe import Recipe
from recipys.selector import all_of
from recipys.step import Accumulator, StepHistorical, StepImputeFill, StepImputeModel, StepScale
import torch


def make_single_split(
    data: Dict[str, pd.DataFrame],
    vars: Dict[str, str],
    train_pct: float = 0.7,
    val_pct: float = 0.1,
    seed: int = 42,
    debug: bool = False,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Randomly split the data into training, validation, and test set.

    Args:
        data: dictionary containing data divided int OUTCOME, STATIC, and DYNAMIC.
        vars: Contains the names of columns in the data.
        train_pct: Proportion of stays assigned to training fold.
        val_pct: Proportion of stays assigned to validation fold.
        seed: Random seed.
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
        stays_in_fold = stays.iloc[delims[i] : delims[i + 1], :]
        for data_type in data.keys():
            # Loop through DYNAMIC / STATIC / OUTCOME
            # set sort to true to make sure that IDs are reordered after scrambling earlier

            # this operation effectively selects the rows corresponding to stays_in_fold and stores them in splits
            splits[fold][data_type] = data[data_type].merge(stays_in_fold, on=id, how="right", sort=True)

    return splits


def apply_recipe_to_splits(
    recipe: Recipe, data: Dict[str, Dict[str, pd.DataFrame]], type: str
) -> Dict[str, Dict[str, pd.DataFrame]]:
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
    file_names: Dict[str, str] = gin.REQUIRED,
    vars: Dict[str, str] = gin.REQUIRED,
    use_features: bool = gin.REQUIRED,
    seed: int = 42,
    debug: bool = False,
    use_cache: bool = False,
    train_pct: float = 0.7,
    val_pct: float = 0.1,
    mode: str = "Classification",
    pretrained_imputation_model: str = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Perform loading, splitting, imputing and normalising of task data.

    Args:
        data_dir: Path to the directory holding the data.
        file_names: Contains the parquet file names in data_dir.
        vars: Contains the names of columns in the data.
        use_features: Whether to generate features on the dynamic data.
        seed: Random seed.
        debug: Load less data if true.
        use_cache: Cache and use cached preprocessed data if true.
        train_pct: Proportion of stays assigned to training fold.
        val_pct: Proportion of stays assigned to validation fold.

    Returns:
        Preprocessed data as DataFrame in a hierarchical dict with data type (STATIC/DYNAMIC/OUTCOME)
            nested within split (train/val/test).
    """

    logging.info("Preprocess static data")
    cache_dir = data_dir / "cache"
    dumped_file_names = json.dumps(file_names, sort_keys=True)
    dumped_vars = json.dumps(vars, sort_keys=True)
    config_string = f"{dumped_file_names}{dumped_vars}{use_features}{seed}{use_cache}{train_pct}{val_pct}".encode("utf-8")
    cache_file = cache_dir / hashlib.md5(config_string).hexdigest()

    if use_cache:
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                logging.info(f"Loading cached data from {cache_file}.")
                return pickle.load(f)
        else:
            logging.info(f"No cached data found in {cache_file}, loading raw data.")

    data = {f: pq.read_table(data_dir / file_names[f]).to_pandas() for f in file_names.keys()}

    if mode == "Imputation":
        rows_to_remove = data["DYNAMIC"][vars["DYNAMIC"]].isna().sum(axis=1) != 0
        ids_to_remove = data["DYNAMIC"].loc[rows_to_remove][vars["GROUP"]].unique()
        data = {table_name: table.loc[~table[vars["GROUP"]].isin(ids_to_remove)] for table_name, table in data.items()}
    logging.info("Generating splits.")
    data = make_single_split(data, vars, train_pct=train_pct, val_pct=val_pct, seed=seed, debug=debug)

    logging.info("Preprocessing static data.")
    sta_rec = Recipe(data["train"]["STATIC"], [], vars["STATIC"])
    # sta_rec.steps = stat_recipe_steps
    sta_rec.add_step(StepScale())
    if mode == "classification":
        sta_rec.add_step(StepImputeFill(value=0) if pretrained_imputation_model is None else StepImputeModel(model=pretrained_imputation_model.predict))

    data = apply_recipe_to_splits(sta_rec, data, "STATIC")

    logging.info("Preprocessing dynamic data.")
    dyn_rec = Recipe(data["train"]["DYNAMIC"], [], vars["DYNAMIC"], vars["GROUP"], vars["SEQUENCE"])
    dyn_rec.add_step(StepScale())
    if mode == "Classification":
        print("use features: ", use_features)
        if use_features:
            logging.info("Generating features....")
            # dyn_rec.add_step(StepHistorical(sel=all_of(vars["DYNAMIC"]), fun=Accumulator.MIN, suffix="min_hist"))
            # dyn_rec.add_step(StepHistorical(sel=all_of(vars["DYNAMIC"]), fun=Accumulator.MAX, suffix="max_hist"))
            # dyn_rec.add_step(StepHistorical(sel=all_of(vars["DYNAMIC"]), fun=Accumulator.COUNT, suffix="count_hist"))
            # dyn_rec.add_step(StepHistorical(sel=all_of(vars["DYNAMIC"]), fun=Accumulator.MEAN, suffix="mean_hist"))
        if pretrained_imputation_model is not None:
            logging.info("Imputing missing values with pretrained model...")
            dyn_rec.add_step(StepImputeModel(model=pretrained_imputation_model.predict))
        else:
            dyn_rec.add_step(StepImputeFill(method="ffill"))
            dyn_rec.add_step(StepImputeFill(value=0))
    # dyn_rec = dyn_recipe_steps

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
