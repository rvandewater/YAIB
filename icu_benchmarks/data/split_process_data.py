import copy
import logging
import os

import gin
import json
import hashlib
import pandas as pd
import polars as pl
from pathlib import Path
import pickle
from timeit import default_timer as timer
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedShuffleSplit, ShuffleSplit
from icu_benchmarks.data.preprocessor import Preprocessor, PandasClassificationPreprocessor, PolarsClassificationPreprocessor
from icu_benchmarks.constants import RunMode
from icu_benchmarks.run_utils import check_required_keys
from .constants import DataSplit as Split, DataSegment as Segment, VarType as Var


@gin.configurable("preprocess")
def preprocess_data(
    data_dir: Path,
    file_names: dict[str] = gin.REQUIRED,
    preprocessor: Preprocessor = PolarsClassificationPreprocessor,
    use_static: bool = True,
    vars: dict[str] = gin.REQUIRED,
    modality_mapping: dict[str] = {},
    selected_modalities: list[str] = "all",
    seed: int = 42,
    debug: bool = False,
    cv_repetitions: int = 5,
    repetition_index: int = 0,
    cv_folds: int = 5,
    train_size: int = None,
    load_cache: bool = False,
    generate_cache: bool = False,
    fold_index: int = 0,
    pretrained_imputation_model: str = None,
    complete_train: bool = False,
    runmode: RunMode = RunMode.classification,
    label: str = None,
    required_var_types=["GROUP", "SEQUENCE", "LABEL"],
    required_segments=[Segment.static, Segment.dynamic, Segment.outcome],
) -> dict[dict[pl.DataFrame]] or dict[dict[pd.DataFrame]]:
    """Perform loading, splitting, imputing and normalising of task data.

    Args:
        use_static: Whether to use static features (for DL models).
        complete_train: Whether to use all data for training/validation.
        runmode: Run mode. Can be one of the values of RunMode
        preprocessor: Define the preprocessor.
        data_dir: Path to the directory holding the data.
        file_names: Contains the parquet file names in data_dir.
        vars: Contains the names of columns in the data.
        seed: Random seed.
        debug: Load less data if true.
        cv_repetitions: Number of times to repeat cross validation.
        repetition_index: Index of the repetition to return.
        cv_folds: Number of folds to use for cross validation.
        train_size: Fixed size of train split (including validation data).
        load_cache: Use cached preprocessed data if true.
        generate_cache: Generate cached preprocessed data if true.
        fold_index: Index of the fold to return.
        pretrained_imputation_model: pretrained imputation model to use. if None, standard imputation is used.

    Returns:
        Preprocessed data as DataFrame in a hierarchical dict with features type (STATIC) / DYNAMIC/ OUTCOME
            nested within split (train/val/test).
    """

    cache_dir = data_dir / "cache"
    check_required_keys(vars, required_var_types)
    check_required_keys(file_names, required_segments)
    if not use_static:
        file_names.pop(Segment.static)
        vars.pop(Segment.static)
    if isinstance(vars[Var.label], list) and len(vars[Var.label]) > 1:
        if label is not None:
            vars[Var.label] = [label]
        else:
            logging.debug(f"Multiple labels found and no value provided. Using first label: {vars[Var.label]}")
            vars[Var.label] = vars[Var.label][0]
        logging.info(f"Using label: {vars[Var.label]}")
    if not vars[Var.label]:
        raise ValueError("No label selected after filtering.")
    dumped_file_names = json.dumps(file_names, sort_keys=True)
    dumped_vars = json.dumps(vars, sort_keys=True)

    cache_filename = f"s_{seed}_r_{repetition_index}_f_{fold_index}_t_{train_size}_d_{debug}"

    logging.log(logging.INFO, f"Using preprocessor: {preprocessor.__name__}")
    vars_to_exclude = (
        modality_mapping.get("cat_clinical_notes") + modality_mapping.get("cat_med_embeddings_map")
        if (
            modality_mapping.get("cat_clinical_notes") is not None
            and modality_mapping.get("cat_med_embeddings_map") is not None
        )
        else None
    )

    preprocessor = preprocessor(
        use_static_features=use_static,
        save_cache=data_dir / "preproc" / (cache_filename + "_recipe"),
        vars_to_exclude=vars_to_exclude,
    )
    if isinstance(preprocessor, PandasClassificationPreprocessor):
        preprocessor.set_imputation_model(pretrained_imputation_model)

    hash_config = hashlib.md5(f"{preprocessor.to_cache_string()}{dumped_file_names}{dumped_vars}".encode("utf-8"))
    cache_filename += f"_{hash_config.hexdigest()}"
    cache_file = cache_dir / cache_filename

    if load_cache:
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                logging.info(f"Loading cached data from {cache_file}.")
                return pickle.load(f)
        else:
            logging.info(f"No cached data found in {cache_file}, loading raw features.")

    # Read parquet files into pandas dataframes and remove the parquet file from memory
    logging.info(f"Loading data from directory {data_dir.absolute()}")
    data = {
        f: pl.read_parquet(data_dir / file_names[f]) for f in file_names.keys() if os.path.exists(data_dir / file_names[f])
    }
    logging.info(f"Loaded data: {list(data.keys())}")
    data = check_sanitize_data(data, vars)

    if not (Segment.dynamic in data.keys()):
        logging.warning("No dynamic data found, using only static data.")

    logging.debug(f"Modality mapping: {modality_mapping}")
    if len(modality_mapping) > 0:
        # Optional modality selection
        if selected_modalities not in [None, "all", ["all"]]:
            data, vars = modality_selection(data, modality_mapping, selected_modalities, vars)
        else:
            logging.info("Selecting all modalities.")

    # Generate the splits
    logging.info("Generating splits.")
    # complete_train = True
    if not complete_train:
        data = make_single_split(
            data,
            vars,
            cv_repetitions,
            repetition_index,
            cv_folds,
            fold_index,
            train_size=train_size,
            seed=seed,
            debug=debug,
            runmode=runmode,
        )
    else:
        # If full train is set, we use all data for training/validation
        data = make_train_val(data, vars, train_size=None, seed=seed, debug=debug, runmode=runmode)

    # Apply preprocessing

    start = timer()
    data = preprocessor.apply(data, vars)
    end = timer()
    logging.info(f"Preprocessing took {end - start:.2f} seconds.")
    logging.info(f"Checking for NaNs and nulls in {data.keys()}.")
    for dict in data.values():
        for key, val in dict.items():
            logging.debug(f"Data type: {key}")
            logging.debug("Is NaN:")
            sel = dict[key].select(pl.selectors.numeric().is_nan().max())
            logging.debug(sel.select(col.name for col in sel if col.item(0)))
            # logging.info(dict[key].select(pl.all().has_nulls()).sum_horizontal())
            logging.debug("Has nulls:")
            sel = dict[key].select(pl.all().has_nulls())
            logging.debug(sel.select(col.name for col in sel if col.item(0)))
            # dict[key] = val[:, [not (s.null_count() > 0) for s in val]]
            dict[key] = val.fill_null(strategy="zero")
            dict[key] = val.fill_nan(0)
            logging.debug("Dropping columns with nulls")
            sel = dict[key].select(pl.all().has_nulls())
            logging.debug(sel.select(col.name for col in sel if col.item(0)))

    # Generate cache
    if generate_cache:
        caching(cache_dir, cache_file, data, load_cache)
    else:
        logging.info("Cache will not be saved.")

    logging.info("Finished preprocessing.")

    return data


def check_sanitize_data(data, vars):
    """Check for duplicates in the loaded data and remove them."""
    group = vars[Var.group] if Var.group in vars.keys() else None
    sequence = vars[Var.sequence] if Var.sequence in vars.keys() else None
    keep = "last"
    if Segment.static in data.keys():
        old_len = len(data[Segment.static])
        data[Segment.static] = data[Segment.static].unique(subset=group, keep=keep, maintain_order=True)
        logging.warning(f"Removed {old_len - len(data[Segment.static])} duplicates from static data.")
    if Segment.dynamic in data.keys():
        old_len = len(data[Segment.dynamic])
        data[Segment.dynamic] = data[Segment.dynamic].unique(subset=[group, sequence], keep=keep, maintain_order=True)
        logging.warning(f"Removed {old_len - len(data[Segment.dynamic])} duplicates from dynamic data.")
    if Segment.outcome in data.keys():
        old_len = len(data[Segment.outcome])
        if sequence in data[Segment.outcome].columns:
            # We have a dynamic outcome with group and sequence
            data[Segment.outcome] = data[Segment.outcome].unique(subset=[group, sequence], keep=keep, maintain_order=True)
        else:
            data[Segment.outcome] = data[Segment.outcome].unique(subset=[group], keep=keep, maintain_order=True)
        logging.warning(f"Removed {old_len - len(data[Segment.outcome])} duplicates from outcome data.")
    return data


def modality_selection(
    data: dict[pl.DataFrame], modality_mapping: dict[str], selected_modalities: list[str], vars
) -> dict[pl.DataFrame]:
    logging.info(f"Selected modalities: {selected_modalities}")
    selected_columns = [modality_mapping[cols] for cols in selected_modalities if cols in modality_mapping.keys()]
    if not any(col in modality_mapping.keys() for col in selected_modalities):
        raise ValueError("None of the selected modalities found in modality mapping.")
    if selected_columns == []:
        logging.info("No columns selected. Using all columns.")
        return data, vars
    selected_columns = sum(selected_columns, [])
    selected_columns.extend([vars[Var.group], vars[Var.label], vars[Var.sequence]])
    old_columns = []
    # Update vars dict
    for key, value in vars.items():
        if key not in [Var.group, Var.label, Var.sequence]:
            old_columns.extend(value)
            vars[key] = [col for col in value if col in selected_columns]
    # -3 because of standard columns
    logging.info(f"Selected columns: {len(selected_columns) - 3}, old columns: {len(old_columns)}")
    logging.debug(f"Difference: {set(old_columns) - set(selected_columns)}")
    # Update data dict
    for key in data.keys():
        sel_col = [col for col in data[key].columns if col in selected_columns]
        data[key] = data[key].select(sel_col)
        logging.debug(f"Selected columns in {key}: {len(data[key].columns)}")
    return data, vars


def make_train_val(
    data: dict[pd.DataFrame],
    vars: dict[str],
    train_size=0.8,
    seed: int = 42,
    debug: bool = False,
    runmode: RunMode = RunMode.classification,
    polars: bool = True,
) -> dict[dict[pl.DataFrame]]:
    """Randomly split the data into training and validation sets for fitting a full model.

    Args:
        data: dictionary containing data divided int OUTCOME, STATIC, and DYNAMIC.
        vars: Contains the names of columns in the data.
        train_size: Fixed size of train split (including validation data).
        seed: Random seed.
        debug: Load less data if true.
    Returns:
        Input data divided into 'train', 'val', and 'test'.
    """
    # ID variable
    id = vars[Var.group]

    if debug:
        # Only use 1% of the data
        logging.info("Using only 1% of the data for debugging. Note that this might lead to errors for small datasets.")
        if polars:
            data[Segment.outcome] = data[Segment.outcome].sample(fraction=0.01, seed=seed)
        else:
            data[Segment.outcome] = data[Segment.outcome].sample(frac=0.01, random_state=seed)

    # Get stay IDs from outcome segment
    stays = _get_stays(data, id, polars)

    # If there are labels, and the task is classification, use stratified k-fold
    if Var.label in vars and runmode is RunMode.classification:
        # Get labels from outcome data (takes the highest value (or True) in case seq2seq classification)
        labels = _get_labels(data, id, vars, polars)
        train_val = StratifiedShuffleSplit(train_size=train_size, random_state=seed, n_splits=1)
        train, val = list(train_val.split(stays, labels))[0]

    else:
        # If there are no labels, use random split
        train_val = ShuffleSplit(train_size=train_size, random_state=seed)
        train, val = list(train_val.split(stays))[0]

    if polars:
        split = {
            Split.train: stays[train].cast(pl.datatypes.Int64).to_frame(),
            Split.val: stays[val].cast(pl.datatypes.Int64).to_frame(),
        }
    else:
        split = {Split.train: stays.iloc[train], Split.val: stays.iloc[val]}

    data_split = {}

    for fold in split.keys():  # Loop through splits (train / val / test)
        # Loop through segments (DYNAMIC / STATIC / OUTCOME)
        # set sort to true to make sure that IDs are reordered after scrambling earlier
        if polars:
            data_split[fold] = {
                data_type: split[fold]
                .join(data[data_type].with_columns(pl.col(id).cast(pl.datatypes.Int64)), on=id, how="left")
                .sort(by=id)
                for data_type in data.keys()
            }
        else:
            data_split[fold] = {
                data_type: data[data_type].merge(split[fold], on=id, how="right", sort=True) for data_type in data.keys()
            }

    # Maintain compatibility with test split
    data_split[Split.test] = copy.deepcopy(data_split[Split.val])
    return data_split


def _get_stays(data, id, polars):
    return (
        pl.Series(name=id, values=data[Segment.outcome][id].unique())
        if polars
        else pd.Series(data[Segment.outcome][id].unique(), name=id)
    )


def _get_labels(data, id, vars, polars):
    # Get labels from outcome data (takes the highest value (or True) in case seq2seq classification)
    if polars:
        return data[Segment.outcome].group_by(id).max()[vars[Var.label]]
    else:
        return data[Segment.outcome].groupby(id).max()[vars[Var.label]].reset_index(drop=True)


# Use these helper functions in both make_train_val and make_single_split


def make_single_split(
    data: dict[pd.DataFrame],
    vars: dict[str],
    cv_repetitions: int,
    repetition_index: int,
    cv_folds: int,
    fold_index: int,
    train_size: int = None,
    seed: int = 42,
    debug: bool = False,
    runmode: RunMode = RunMode.classification,
    polars: bool = True,
) -> dict[dict[pl.DataFrame]]:
    """Randomly split the data into training, validation, and test set.

    Args:
        runmode: Run mode. Can be one of the values of RunMode
        data: dictionary containing data divided int OUTCOME, STATIC, and DYNAMIC.
        vars: Contains the names of columns in the data.
        cv_repetitions: Number of times to repeat cross validation.
        repetition_index: Index of the repetition to return.
        cv_folds: Number of folds for cross validation.
        fold_index: Index of the fold to return.
        train_size: Fixed size of train split (including validation data).
        seed: Random seed.
        debug: Load less data if true.

    Returns:
        Input data divided into 'train', 'val', and 'test'.
    """
    # ID variable
    id = vars[Var.group]

    if debug:
        # Only use 1% of the data
        logging.info("Using only 1% of the data for debugging. Note that this might lead to errors for small datasets.")
        if polars:
            data[Segment.outcome] = data[Segment.outcome].sample(fraction=0.01, seed=seed)
        else:
            data[Segment.outcome] = data[Segment.outcome].sample(frac=0.01, random_state=seed)
    # Get stay IDs from outcome segment
    if polars:
        stays = pl.Series(name=id, values=data[Segment.outcome][id].unique())
    else:
        stays = pd.Series(data[Segment.outcome][id].unique(), name=id)

    # If there are labels, and the task is classification, use stratified k-fold
    if Var.label in vars and runmode is RunMode.classification:
        # Get labels from outcome data (takes the highest value (or True) in case seq2seq classification)
        if polars:
            labels = data[Segment.outcome].group_by(id).max()[vars[Var.label]]
            if labels.value_counts().min().item(0, 1) < cv_folds:
                raise Exception(
                    f"The smallest amount of samples in a class is: {labels.value_counts().min()}, "
                    f"but {cv_folds} folds are requested. Reduce the number of folds or use more data."
                )
        else:
            labels = data[Segment.outcome].groupby(id).max()[vars[Var.label]].reset_index(drop=True)
            if labels.value_counts().min() < cv_folds:
                raise Exception(
                    f"The smallest amount of samples in a class is: {labels.value_counts().min()}, "
                    f"but {cv_folds} folds are requested. Reduce the number of folds or use more data."
                )

        if train_size:
            outer_cv = StratifiedShuffleSplit(cv_repetitions, train_size=train_size)
        else:
            outer_cv = StratifiedKFold(cv_repetitions, shuffle=True, random_state=seed)
        inner_cv = StratifiedKFold(cv_folds, shuffle=True, random_state=seed)

        dev, test = list(outer_cv.split(stays, labels))[repetition_index]
        if polars:
            dev_stays = stays[dev]
            train, val = list(inner_cv.split(dev_stays, labels[dev]))[fold_index]
        else:
            dev_stays = stays.iloc[dev]
            train, val = list(inner_cv.split(dev_stays, labels.iloc[dev]))[fold_index]
    else:
        # If there are no labels, or the task is regression, use regular k-fold.
        if train_size:
            outer_cv = ShuffleSplit(cv_repetitions, train_size=train_size)
        else:
            outer_cv = KFold(cv_repetitions, shuffle=True, random_state=seed)
        inner_cv = KFold(cv_folds, shuffle=True, random_state=seed)

        dev, test = list(outer_cv.split(stays))[repetition_index]
        if polars:
            dev_stays = stays[dev]
        else:
            dev_stays = stays.iloc[dev]
        train, val = list(inner_cv.split(dev_stays))[fold_index]
    if polars:
        split = {
            Split.train: dev_stays[train].cast(pl.datatypes.Int64).to_frame(),
            Split.val: dev_stays[val].cast(pl.datatypes.Int64).to_frame(),
            Split.test: stays[test].cast(pl.datatypes.Int64).to_frame(),
        }
    else:
        split = {
            Split.train: dev_stays.iloc[train],
            Split.val: dev_stays.iloc[val],
            Split.test: stays.iloc[test],
        }
    data_split = {}

    for fold in split.keys():  # Loop through splits (train / val / test)
        # Loop through segments (DYNAMIC / STATIC / OUTCOME)
        # set sort to true to make sure that IDs are reordered after scrambling earlier
        if polars:
            data_split[fold] = {
                data_type: split[fold]
                .join(data[data_type].with_columns(pl.col(id).cast(pl.datatypes.Int64)), on=id, how="left")
                .sort(by=id)
                for data_type in data.keys()
            }
        else:
            data_split[fold] = {
                data_type: data[data_type].merge(split[fold], on=id, how="right", sort=True) for data_type in data.keys()
            }
    logging.info(f"Data split: {data_split}")
    return data_split


def caching(cache_dir, cache_file, data, use_cache, overwrite=True):
    if use_cache and (not overwrite or not cache_file.exists()):
        if not cache_dir.exists():
            cache_dir.mkdir()
        cache_file.touch()
        with open(cache_file, "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        logging.info(f"Cached data in {cache_file}.")
