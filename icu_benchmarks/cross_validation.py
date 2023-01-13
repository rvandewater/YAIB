import json
from datetime import datetime
import logging
import gin
from pathlib import Path

from icu_benchmarks.data.preprocess import preprocess_data
from icu_benchmarks.models.train import train_common
from icu_benchmarks.models.utils import JsonResultLoggingEncoder
from icu_benchmarks.run_utils import log_full_line


@gin.configurable
def execute_repeated_cv(
    data_dir: Path,
    log_dir: Path,
    seed: int,
    load_weights: bool = False,
    source_dir: Path = None,
    cv_repetitions: int = 5,
    cv_repetitions_to_train: int = None,
    cv_folds: int = 5,
    cv_folds_to_train: int = None,
    reproducible: bool = True,
    debug: bool = False,
    use_cache: bool = False,
    test_on: str = "test",
) -> float:
    """Preprocesses data and trains a model for each fold.

    Args:
        data_dir: Path to the data directory.
        log_dir: Path to the log directory.
        seed: Random seed.
        load_weights: Whether to load weights from source_dir.
        source_dir: Path to the source directory.
        cv_folds: Number of folds for cross validation.
        cv_folds_to_train: Number of folds to use during training. If None, all folds are trained on.
        reproducible: Whether to make torch reproducible.
        debug: Whether to load less data and enable more logging.
        use_cache: Whether to cache and use cached data.
        test_on: Dataset to test on. Can be "test" or "val" (e.g. for hyperparameter tuning).

    Returns:
        The average loss of all folds.
    """
    if not cv_repetitions_to_train:
        cv_repetitions_to_train = cv_repetitions
    if not cv_folds_to_train:
        cv_folds_to_train = cv_folds
    agg_loss = 0
    for repetition in range(cv_repetitions_to_train):
        for fold_index in range(cv_folds_to_train):
            start_time = datetime.now()
            data = preprocess_data(
                data_dir,
                seed=seed,
                debug=debug,
                use_cache=use_cache,
                cv_repetitions=cv_repetitions,
                repetition_index=repetition,
                cv_folds=cv_folds,
                fold_index=fold_index,
            )

            repetition_fold_dir = log_dir / f"repetition_{repetition}" / f"fold_{fold_index}"
            repetition_fold_dir.mkdir(parents=True, exist_ok=True)

            preprocess_time = datetime.now() - start_time
            start_time = datetime.now()
            agg_loss += train_common(
                data,
                log_dir=repetition_fold_dir,
                load_weights=load_weights,
                source_dir=source_dir,
                seed=seed,
                reproducible=reproducible,
                test_on=test_on,
            )
            train_time = datetime.now() - start_time

            log_full_line(
                f"FINISHED FOLD {fold_index}| PREPROCESSING DURATION {preprocess_time}| TRAINING DURATION {train_time}",
                level=logging.INFO,
            )
            durations = {"preprocessing_duration": preprocess_time, "train_duration": train_time}

            with open(repetition_fold_dir / "durations.json", "w") as f:
                json.dump(durations, f, cls=JsonResultLoggingEncoder)
        log_full_line(f"FINISHED CV REPETITION {repetition}", level=logging.INFO, char="=", num_newlines=3)

    return agg_loss / (cv_repetitions_to_train * cv_folds_to_train)
