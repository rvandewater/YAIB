import copy
import hashlib
import json
import logging
import os
from timeit import default_timer as timer
from typing import Any, Iterable, Optional, Union
import gin
import pandas as pd
import polars as pl
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
import polars.selectors as cs
from pathlib import Path
import pickle
from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.preprocessor import (
    PandasClassificationPreprocessor,
    PolarsClassificationPreprocessor,
    PolarsRegressionPreprocessor,
    Preprocessor,
)

from .constants import DataSegment, DataSplit, VarType
from .utils import check_sanitize_data, modality_selection


@gin.configurable("preprocess")
def preprocess_data(
    data_dir: Path,
    file_names: dict[str, str] | Any = gin.REQUIRED,
    preprocessor: type[Preprocessor] = PolarsClassificationPreprocessor,
    use_static: bool = True,
    vars: dict[str, str | list[str]] | Any = gin.REQUIRED,
    modality_mapping: Optional[dict[str, list[str]]] = None,
    selected_modalities: Optional[list[str]] = None,
    exclude_preproc: list[str] = None,
    seed: int = 42,
    debug: bool = False,
    cv_repetitions: int = 5,
    repetition_index: int = 0,
    cv_folds: int = 5,
    train_size: Optional[float] = None,
    load_cache: bool = False,
    generate_cache: bool = False,
    fold_index: int = 0,
    pretrained_imputation_model: Optional[str] = None,
    complete_train: bool = False,
    runmode: RunMode = RunMode.classification,
    label: Optional[str] = None,
    required_var_types: Optional[list[str]] = None,
    required_segments: Optional[list[str]] = None,
) -> dict[str, dict[str, pl.DataFrame]]:
    """
    Perform loading, splitting, imputing and normalising of task data.

    Args:
        use_static: Whether to use static features (for DL models).
        complete_train: Whether to use all data for training/validation.
        runmode: Run mode. Can be one of the values of RunMode
        preprocessor: Define the preprocessor.
        data_dir: Path to the directory holding the data.
        file_names: Contains the parquet file names in data_dir.
        vars: Contains the names of columns in the data.
        modality_mapping: Mapping of modalities to column names.
        selected_modalities: List of selected modalities to use.
        exclude_preproc: List of modalities to exclude from preprocessing.
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
    if modality_mapping is None:
        modality_mapping = {}
    if selected_modalities is None:
        selected_modalities = ["all"]
    if required_var_types is None:
        required_var_types = ["GROUP", "LABEL"]
    if required_segments is None:
        required_segments = [DataSegment.static, DataSegment.dynamic, DataSegment.outcome]

    check_required_keys(vars, required_var_types)
    check_required_keys(file_names, required_segments)

    if not use_static:
        file_names.pop(DataSegment.static)
        vars.pop(DataSegment.static)

    if isinstance(vars[VarType.label], list) and len(vars[VarType.label]) > 1:
        if label is not None:
            vars[VarType.label] = [label]
        else:
            logging.debug(f"Multiple labels found and no value provided. Using first label: {vars[VarType.label]}")
            vars[VarType.label] = vars[VarType.label][0]
        logging.info(f"Using label: {vars[VarType.label]}")

    if not vars[VarType.label]:
        raise ValueError("No label selected after filtering.")

    dumped_file_names = json.dumps(file_names, sort_keys=True)
    dumped_vars = json.dumps(vars, sort_keys=True)

    logging.info(f"Using preprocessor: {preprocessor.__name__}")

    cat_clinical_notes = modality_mapping.get("cat_clinical_notes")
    cat_med_embeddings_map = modality_mapping.get("cat_med_embeddings_map")
    if cat_clinical_notes is not None and cat_med_embeddings_map is not None:
        vars_to_exclude = cat_clinical_notes + cat_med_embeddings_map
    else:
        vars_to_exclude = None

    cache_dir = data_dir / "cache"
    cache_filename = f"s_{seed}_r_{repetition_index}_f_{fold_index}_t_{train_size}_d_{debug}"

    logging.log(logging.INFO, f"Using preprocessor: {preprocessor.__name__}")

    excluded_vars = []
    if exclude_preproc is not None:
        # Exclude variables from preprocessing based on modality:
        # useful if modality has already undergone extensive preprocessing.
        if modality_mapping is not None and len(modality_mapping) > 0:
            for modality in exclude_preproc:
                if modality in modality_mapping:
                    excluded_vars.extend(modality_mapping.get(modality))
                else:
                    logging.warning(f"Modality '{modality}' not found in modality mapping.")
            logging.info(
                f"Excluding modalities in {exclude_preproc}. Total vars excluded from preprocessing: {len(excluded_vars)}"
            )
        else:
            logging.warning("No modality mapping provided. Excluding variables from preprocessing will have no effect.")
    preprocessor_instance: Preprocessor = preprocessor(
        use_static_features=use_static,
        save_cache=data_dir / "preproc" / (cache_filename + "_recipe") if generate_cache else None,
        vars_to_exclude=vars_to_exclude,
    )
    if isinstance(preprocessor_instance, PandasClassificationPreprocessor):
        preprocessor_instance.set_imputation_model(pretrained_imputation_model)

    hash_config = hashlib.md5(f"{preprocessor_instance.to_cache_string()}{dumped_file_names}{dumped_vars}".encode("utf-8"))
    cache_filename += f"_{hash_config.hexdigest()}"
    cache_file = cache_dir / cache_filename

    if load_cache:
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                logging.info(f"Loading cached data from {cache_file}.")
                return pickle.load(f)
        else:
            logging.info(f"No cached data found in {cache_file}, loading raw features.")

    # Read parquet files into dataframes and remove the parquet file from memory
    logging.info(f"Loading data from directory {data_dir.absolute()}")
    data: dict[str, pl.DataFrame] = {
        f: pl.read_parquet(data_dir / file_names[f]) for f in file_names.keys() if os.path.exists(data_dir / file_names[f])
    }

    logging.info(f"Loaded data: {list(data.keys())}")
    sanitized_data, vars = check_sanitize_data(data, vars)

    if DataSegment.dynamic not in sanitized_data.keys():
        logging.warning("No dynamic data found, using only static data.")

    logging.debug(f"Modality mapping: {modality_mapping}")
    if len(modality_mapping) > 0:
        # Optional modality selection
        if selected_modalities not in [None, "all", ["all"]]:
            data, vars = modality_selection(sanitized_data, modality_mapping, selected_modalities, vars)
        else:
            logging.info("Selecting all modalities.")

    # Generate the splits
    logging.info("Generating splits.")
    if not complete_train:
        sanitized_data = make_single_split_polars(
            sanitized_data,
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
        sanitized_data = make_train_val_polars(data, vars, train_size=None, seed=seed, debug=debug, runmode=runmode)

    # Apply preprocessing
    start = timer()
    sanitized_data = preprocessor_instance.apply(sanitized_data, vars)
    end = timer()
    logging.info(f"Preprocessing took {end - start:.2f} seconds.")
    logging.info(f"Checking for NaNs and nulls in {data.keys()}.")
    for _dict in sanitized_data.values():
        for key, dataframe in _dict.items():
            logging.debug(f"Data type: {key}")
            logging.debug("Is NaN:")
            sel = _dict[key].select(pl.selectors.numeric().is_nan().max())
            logging.debug(sel.select(col.name for col in sel if col.item(0)))
            logging.debug("Has nulls:")
            sel = _dict[key].select(pl.all().has_nulls())
            logging.debug(sel.select(col.name for col in sel if col.item(0)))
            _dict[key] = dataframe.fill_null(strategy="zero")
            _dict[key] = dataframe.fill_nan(0)
            logging.debug("Dropping columns with nulls")
            sel = _dict[key].select(pl.all().has_nulls())
            logging.debug(sel.select(col.name for col in sel if col.item(0)))
            logging.info("Checking for infinite values.")
            for col in dataframe.select(cs.numeric()).columns:
                if dataframe[col].is_infinite().any():
                    logging.info(f"Column '{col}' contains infinite values. Datatype: {dataframe[col].dtype}")

            max_float64 = 0
            # Replace infinite values with the maximum value for float64
            dataframe = dataframe.with_columns(
                [
                    pl.when(pl.col(col).is_infinite()).then(max_float64).otherwise(pl.col(col)).alias(col)
                    for col in dataframe.columns
                    if dataframe[col].dtype == pl.Float64
                ]
            )
            _dict[key] = dataframe
            logging.info(f"Amount of columns: {len(dataframe.columns)}")

    # Generate cache
    if generate_cache:
        caching(cache_dir, cache_file, data, load_cache)
    else:
        logging.info("Cache will not be saved.")

    logging.info("Finished preprocessing.")

    return sanitized_data


def flatten_column_names(*args: object) -> list[str]:
    result: list[str] = []
    for arg in args:
        if isinstance(arg, str):
            result.append(arg)
        elif isinstance(arg, Iterable) and not isinstance(arg, (str, bytes)):
            result.extend([a for a in arg if isinstance(a, str)])
    return result


def check_sanitize_data(data: dict[str, pl.DataFrame], vars: dict[str, str | list[str]]) -> dict[str, pl.DataFrame]:  # noqa: F811
    """Check for duplicates in the loaded data and remove them."""
    group: Optional[Union[str, list[str]]] = vars.get(VarType.group)
    sequence: Optional[Union[str, list[str]]] = vars.get(VarType.sequence)
    keep = "last"
    if DataSegment.static in data.keys():
        old_len = len(data[DataSegment.static])
        data[DataSegment.static] = data[DataSegment.static].unique(subset=group, keep=keep, maintain_order=True)
        if old_len != len(data[DataSegment.static]):
            logging.warning(f"Removed {old_len - len(data[DataSegment.static])} duplicates from static data.")
    if DataSegment.dynamic in data.keys():
        old_len = len(data[DataSegment.dynamic])

        data[DataSegment.dynamic] = data[DataSegment.dynamic].unique(
            subset=flatten_column_names(group, sequence), keep=keep, maintain_order=True
        )
        if old_len != len(data[DataSegment.dynamic]):
            logging.warning(f"Removed {old_len - len(data[DataSegment.dynamic])} duplicates from dynamic data.")
    if DataSegment.outcome in data.keys():
        old_len = len(data[DataSegment.outcome])
        if sequence in data[DataSegment.outcome].columns:
            # We have a dynamic outcome with group and sequence
            data[DataSegment.outcome] = data[DataSegment.outcome].unique(
                subset=flatten_column_names(group, sequence), keep=keep, maintain_order=True
            )
        else:
            data[DataSegment.outcome] = data[DataSegment.outcome].unique(subset=[group], keep=keep, maintain_order=True)
        if old_len != len(data[DataSegment.outcome]):
            logging.warning(f"Removed {old_len - len(data[DataSegment.outcome])} duplicates from outcome data.")
    return data, vars


def modality_selection(  # noqa: F811
    data: dict[str, pl.DataFrame],
    modality_mapping: dict[str, list[str]],
    selected_modalities: list[str],
    vars: dict[str, Union[str, list[str]]],
) -> tuple[dict[str, pl.DataFrame], dict[str, Union[str, list[str]]]]:
    logging.info(f"Selected modalities: {selected_modalities}")
    selected_columns = [modality_mapping[cols] for cols in selected_modalities if cols in modality_mapping.keys()]
    if not any(col in modality_mapping.keys() for col in selected_modalities):
        raise ValueError("None of the selected modalities found in modality mapping.")
    if selected_columns == []:
        logging.info("No columns selected. Using all columns.")
        return data, vars
    selected_columns = [col for cols in selected_columns for col in cols]

    group_val = vars[VarType.group]
    label_val = vars[VarType.label]
    sequence_val = vars[VarType.sequence]

    if not (isinstance(group_val, str) and isinstance(label_val, str) and isinstance(sequence_val, str)):
        raise TypeError(
            f'Expected keys "{VarType.group}", "{VarType.label}" and "{VarType.sequence}" to be of type str, '
            f"got {type(group_val)}, {type(label_val)} and {type(sequence_val)} instead."
        )

    selected_columns.extend([group_val, label_val, sequence_val])
    old_columns = []
    # Update vars dict
    for key, value in vars.items():
        if key not in [VarType.group, VarType.label, VarType.sequence]:
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


def make_train_val_pandas(
    data: dict[str, pd.DataFrame],
    vars: dict[str, Union[str, list[str]]],
    train_size: Optional[float] = 0.8,
    seed: int = 42,
    debug: bool = False,
    runmode: RunMode = RunMode.classification,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Randomly splits the data into training and validation sets for fitting a full model,
    specifically designed for Pandas DataFrames.

    For a more detailed documentation refer to make_train_val(...)
    """
    _id = vars[VarType.group]
    label = vars[VarType.label]
    if not (isinstance(_id, str) and isinstance(label, str)):
        raise TypeError(
            f'Expected keys "{VarType.group}" and "{VarType.label}" to be of type str, '
            f"got {type(_id)} and {type(label)} instead."
        )

    if debug:
        logging.info("Using only 1% of the data for debugging. Note that this might lead to errors for small datasets.")
        data[DataSegment.outcome] = data[DataSegment.outcome].sample(frac=0.01, random_state=seed)

    stays = data[DataSegment.outcome][_id].unique()

    if VarType.label in vars and runmode is RunMode.classification:
        labels = data[DataSegment.outcome].groupby(_id)[label].max()
        train_val_splitter = StratifiedShuffleSplit(train_size=train_size, random_state=seed, n_splits=1)
        train_indices, val_indices = list(train_val_splitter.split(stays, labels))[0]
    else:
        train_val_splitter = ShuffleSplit(train_size=train_size, random_state=seed, n_splits=1)
        train_indices, val_indices = list(train_val_splitter.split(stays))[0]

    split_ids = {
        DataSplit.train: pd.DataFrame({_id: stays[train_indices]}),
        DataSplit.val: pd.DataFrame({_id: stays[val_indices]}),
    }

    data_split: dict[str, dict[str, pd.DataFrame]] = {}

    for fold in split_ids.keys():
        data_split[fold] = {}
        for data_type in data.keys():
            merged_df = data[data_type].merge(split_ids[fold], on=_id, how="right", sort=True)
            data_split[fold][data_type] = merged_df

    data_split[DataSplit.test] = copy.deepcopy(data_split[DataSplit.val])
    return data_split


def make_train_val_polars(
    data: dict[str, pl.DataFrame],
    vars: dict[str, Union[str, list[str]]],
    train_size: Optional[float] = 0.8,
    seed: int = 42,
    debug: bool = False,
    runmode: RunMode = RunMode.classification,
) -> dict[str, dict[str, pl.DataFrame]]:
    """
    Randomly splits the data into training and validation sets for fitting a full model,
    specifically designed for Polars DataFrames.

    For a more detailed documentation refer to make_train_val(...)
    """
    _id = vars[VarType.group]
    label = vars[VarType.label]
    if not (isinstance(_id, str) and isinstance(label, str)):
        raise TypeError(
            f'Expected keys "{VarType.group}" and "{VarType.label}" to be of type str, '
            f"got {type(_id)} and {type(label)} instead."
        )

    if debug:
        logging.info("Using only 1% of the data for debugging. Note that this might lead to errors for small datasets.")
        data[DataSegment.outcome] = data[DataSegment.outcome].sample(fraction=0.01, seed=seed)

    stays = pl.Series(name=_id, values=data[DataSegment.outcome][_id].unique())

    if VarType.label in vars and runmode is RunMode.classification:
        labels = data[DataSegment.outcome].group_by(_id).max()[label]
        train_val_splitter = StratifiedShuffleSplit(train_size=train_size, random_state=seed, n_splits=1)
        train_indices, val_indices = list(train_val_splitter.split(stays, labels))[0]
    else:
        train_val_splitter = ShuffleSplit(train_size=train_size, random_state=seed, n_splits=1)
        train_indices, val_indices = list(train_val_splitter.split(stays))[0]

    split = {
        DataSplit.train: stays[train_indices].cast(pl.datatypes.Int64).to_frame(),
        DataSplit.val: stays[val_indices].cast(pl.datatypes.Int64).to_frame(),
    }

    data_split: dict[str, dict[str, pl.DataFrame]] = {}

    for fold in split.keys():
        data_split[fold] = {
            data_type: split[fold]
            .join(data[data_type].with_columns(pl.col(_id).cast(pl.datatypes.Int64)), on=_id, how="left")
            .sort(by=_id)
            for data_type in data.keys()
        }

    data_split[DataSplit.test] = copy.deepcopy(data_split[DataSplit.val])
    return data_split


def make_train_val(
    data: dict[str, Union[pd.DataFrame, pl.DataFrame]],
    vars: dict[str, Union[str, list[str]]],
    train_size: Optional[float] = 0.8,
    seed: int = 42,
    debug: bool = False,
    runmode: RunMode = RunMode.classification,
    polars: bool = True,
) -> dict[str, dict[str, pl.DataFrame]] | dict[str, dict[str, pd.DataFrame]]:
    """
    Randomly splits the data into training and validation sets for fitting a full model.
    Dispatches to either a Polars or Pandas backend based on the 'polars' flag.

    Args:
        data: A dictionary containing DataFrames (either Pandas or Polars),
              divided into segments like OUTCOME, STATIC, and DYNAMIC.
        vars: A dictionary containing the names of columns (variables) in the data.
        train_size: The proportion of the dataset to include in the train split.
        seed: Random seed for reproducibility.
        debug: If True, uses only a small fraction (1%) of the data for debugging.
        runmode: The type of machine learning task (e.g., classification, regression).
        polars: If True, uses the Polars backend; otherwise, uses the Pandas backend.

    Returns:
        A dictionary containing the input data divided into 'train', 'val', and 'test'
        splits. Each split is itself a dictionary of DataFrames (Pandas or Polars)
        corresponding to the original data segments.
    """
    if polars:
        polars_data = {k: v if isinstance(v, pl.DataFrame) else pl.DataFrame(v) for k, v in data.items()}
        return make_train_val_polars(polars_data, vars, train_size, seed, debug, runmode)
    else:
        pandas_data = {k: v if isinstance(v, pd.DataFrame) else v.to_pandas() for k, v in data.items()}
        return make_train_val_pandas(pandas_data, vars, train_size, seed, debug, runmode)


# Use these helper functions in both make_train_val and make_single_split
def make_single_split_pandas(
    data: dict[str, pd.DataFrame],
    vars: dict[str, Union[str, list[str]]],
    cv_repetitions: int,
    repetition_index: int,
    cv_folds: int,
    fold_index: int,
    train_size: Optional[float] = 0.80,
    seed: int = 42,
    debug: bool = False,
    runmode: RunMode = RunMode.classification,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Randomly splits the data into training, validation, and test sets,
    specifically designed for Pandas DataFrames.

    For a more detailed documentation refer to make_single_splits(...)
    """
    _id = vars[VarType.group]
    label = vars[VarType.label]

    if not (isinstance(_id, str) and isinstance(label, str)):
        raise TypeError(
            f'Expected keys "{VarType.group}" and "{VarType.label}" to be of type str, '
            f"got {type(_id)} and {type(label)} instead."
        )

    if debug:
        logging.info("Using only 1% of the data for debugging. Note that this might lead to errors for small datasets.")
        data[DataSegment.outcome] = data[DataSegment.outcome].sample(frac=0.01, random_state=seed)

    stays = data[DataSegment.outcome][_id].unique()

    if VarType.label in vars and runmode is RunMode.classification:
        labels = data[DataSegment.outcome].groupby(_id)[label].max().reset_index(drop=True)
        if labels.value_counts().min() < cv_folds:
            raise Exception(
                f"The smallest amount of samples in a class is: {labels.value_counts().min()}, "
                f"but {cv_folds} folds are requested. Reduce the number of folds or use more data."
            )

        if train_size:
            outer_cv = StratifiedShuffleSplit(cv_repetitions, train_size=train_size, random_state=seed)
        else:
            outer_cv = StratifiedKFold(cv_repetitions, shuffle=True, random_state=seed)
        inner_cv = StratifiedKFold(cv_folds, shuffle=True, random_state=seed)

        dev_indices, test_indices = list(outer_cv.split(stays, labels))[repetition_index]
        dev_stays = stays[dev_indices]
        train_indices, val_indices = list(inner_cv.split(dev_stays, labels[dev_indices]))[fold_index]
    else:
        if train_size:
            outer_cv = ShuffleSplit(cv_repetitions, train_size=train_size, random_state=seed)
        else:
            outer_cv = KFold(cv_repetitions, shuffle=True, random_state=seed)
        inner_cv = KFold(cv_folds, shuffle=True, random_state=seed)

        dev_indices, test_indices = list(outer_cv.split(stays))[repetition_index]
        dev_stays = stays[dev_indices]
        train_indices, val_indices = list(inner_cv.split(dev_stays))[fold_index]

    split_ids = {
        DataSplit.train: pd.DataFrame({_id: dev_stays[train_indices]}),
        DataSplit.val: pd.DataFrame({_id: dev_stays[val_indices]}),
        DataSplit.test: pd.DataFrame({_id: stays[test_indices]}),
    }

    data_split: dict[str, dict[str, pd.DataFrame]] = {}

    for fold in split_ids.keys():
        data_split[fold] = {}
        for data_type in data.keys():
            merged_df = data[data_type].merge(split_ids[fold], on=_id, how="right", sort=True)
            data_split[fold][data_type] = merged_df

    logging.debug(f"Data split: {data_split}")
    return data_split


def make_single_split_polars(
    data: dict[str, pl.DataFrame],
    vars: dict[str, Union[str, list[str]]],
    cv_repetitions: int,
    repetition_index: int,
    cv_folds: int,
    fold_index: int,
    train_size: Optional[float] = None,
    seed: int = 42,
    debug: bool = False,
    runmode: RunMode = RunMode.classification,
) -> dict[str, dict[str, pl.DataFrame]]:
    """
    Randomly splits the data into training, validation, and test set,
    specifically designed for Polars DataFrames.

    For a more detailed documentation refer to make_single_splits(...)
    """
    # ID variable
    id = vars[VarType.group]
    if debug:
        # Only use 1% of the data
        logging.info("Using only 1% of the data for debugging. Note that this might lead to errors for small datasets.")
        data[DataSegment.outcome] = data[DataSegment.outcome].sample(fraction=0.01, seed=seed)

    # Get stay IDs from outcome segment
    stays = pl.Series(name=id, values=data[DataSegment.outcome][id].unique())
    # If there are labels, and the task is classification, use stratified k-fold
    if VarType.label in vars and runmode is RunMode.classification:
        # Get labels from outcome data (takes the highest value (or True) in case seq2seq classification)
        labels: pl.Series = data[DataSegment.outcome].group_by(id).max().sort(id)[vars[VarType.label]]
        if labels.value_counts().min().item(0, 1) < cv_folds:
            raise Exception(
                f"The smallest amount of samples in a class is: {labels.value_counts().min()}, "
                f"but {cv_folds} folds are requested. Reduce the number of folds or use more data."
            )

        if train_size:
            outer_cv = StratifiedShuffleSplit(cv_repetitions, train_size=train_size)
        else:
            outer_cv = StratifiedKFold(cv_repetitions, shuffle=True, random_state=seed)
        if cv_folds > 2:
            inner_cv = StratifiedKFold(cv_folds, shuffle=True, random_state=seed)
        else:
            inner_cv = StratifiedShuffleSplit(cv_folds, train_size=0.75, random_state=seed)
        dev, test = list(outer_cv.split(stays, labels))[repetition_index]
        dev_stays = stays[dev]
        train, val = list(inner_cv.split(dev_stays, labels[dev]))[fold_index]

    else:
        # If there are no labels, or the task is regression, use regular k-fold.
        if train_size:
            outer_cv = ShuffleSplit(cv_repetitions, train_size=train_size)
        else:
            outer_cv = KFold(cv_repetitions, shuffle=True, random_state=seed)

        inner_cv = KFold(cv_folds, shuffle=True, random_state=seed)
        dev, test = list(outer_cv.split(stays))[repetition_index]
        dev_stays = stays[dev]
        train, val = list(inner_cv.split(dev_stays))[fold_index]

    split = {
        DataSplit.train: dev_stays[train].cast(pl.datatypes.Int64).to_frame(),
        DataSplit.val: dev_stays[val].cast(pl.datatypes.Int64).to_frame(),
        DataSplit.test: stays[test].cast(pl.datatypes.Int64).to_frame(),
    }

    data_split = {}
    for fold in split.keys():  # Loop through splits (train / val / test)
        # Loop through segments (DYNAMIC / STATIC / OUTCOME)
        # set sort to true to make sure that IDs are reordered after scrambling earlier
        data_split[fold] = {
            data_type: split[fold]
            .join(data[data_type].with_columns(pl.col(id).cast(pl.datatypes.Int64)), on=id, how="left")
            .sort(by=id)
            for data_type in data.keys()
        }

    logging.debug(f"Data split: {data_split}")

    return data_split


def make_single_split(
    data: dict[str, Union[pd.DataFrame, pl.DataFrame]],
    vars: dict[str, Union[str, list[str]]],
    cv_repetitions: int,
    repetition_index: int,
    cv_folds: int,
    fold_index: int,
    train_size: Optional[float] = None,
    seed: int = 42,
    debug: bool = False,
    runmode: RunMode = RunMode.classification,
    polars: bool = True,
) -> dict[str, dict[str, pl.DataFrame]] | dict[str, dict[str, pd.DataFrame]]:
    """
    Randomly splits the data into training, validation, and test sets.
    Dispatches to either a Polars or Pandas backend based on the 'polars' flag.

    Args:
        data: A dictionary containing DataFrames (either Pandas or Polars),
              divided into segments like OUTCOME, STATIC, and DYNAMIC.
        vars: Contains the names of columns in the data.
        cv_repetitions: Number of times to repeat cross validation.
        repetition_index: Index of the repetition to return.
        cv_folds: Number of folds for cross validation.
        fold_index: Index of the fold to return.
        train_size: Fixed size of train split (including validation data).
        seed: Random seed.
        debug: Load less data if true.
        runmode: Run mode. Can be one of the values of RunMode
        polars: If True, uses the Polars backend; otherwise, uses the Pandas backend.

    Returns:
        Input data divided into 'train', 'val', and 'test'.
    """
    if polars:
        polars_data = {k: v if isinstance(v, pl.DataFrame) else pl.DataFrame(v) for k, v in data.items()}
        return make_single_split_polars(
            polars_data, vars, cv_repetitions, repetition_index, cv_folds, fold_index, train_size, seed, debug, runmode
        )
    else:
        pandas_data = {k: v if isinstance(v, pd.DataFrame) else v.to_pandas() for k, v in data.items()}
        return make_single_split_pandas(
            pandas_data, vars, cv_repetitions, repetition_index, cv_folds, fold_index, train_size, seed, debug, runmode
        )


def caching(cache_dir, cache_file, data, use_cache, overwrite=True):
    if use_cache and (not overwrite or not cache_file.exists()):
        if not cache_dir.exists():
            cache_dir.mkdir()
        cache_file.touch()
        with open(cache_file, "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        logging.info(f"Cached data in {cache_file}.")


def check_required_keys(vars, required_keys):
    """
    Checks if all required keys are present in the vars dictionary.

    Args:
        vars (dict): The dictionary to check.
        required_keys (list): The list of required keys.

    Raises:
        KeyError: If any required key is missing.
    """
    missing_keys = [key for key in required_keys if key not in vars]
    if missing_keys:
        raise KeyError(f"Missing required keys in vars: {', '.join(missing_keys)}")
