import json
from datetime import datetime
import logging
import gin
from pathlib import Path
from pytorch_lightning import seed_everything

from icu_benchmarks.wandb_utils import wandb_log
from icu_benchmarks.run_utils import aggregate_results
from icu_benchmarks.data.split_process_data import preprocess_data
from icu_benchmarks.models.train import train_common
from icu_benchmarks.models.utils import JsonResultLoggingEncoder
from icu_benchmarks.run_utils import log_full_line
from icu_benchmarks.contants import RunMode


@gin.configurable
def execute_repeated_cv(
    data_dir: Path,
    log_dir: Path,
    seed: int,
    eval_only: bool = False,
    train_size: int = None,
    load_weights: bool = False,
    source_dir: Path = None,
    cv_repetitions: int = 1,
    cv_repetitions_to_train: int = None,
    cv_folds: int = 5,
    cv_folds_to_train: int = None,
    reproducible: bool = True,
    debug: bool = False,
    generate_cache: bool = False,
    load_cache: bool = False,
    test_on: str = "test",
    mode: str = RunMode.classification,
    pretrained_imputation_model: object = None,
    cpu: bool = False,
    verbose: bool = False,
    wandb: bool = False,
    complete_train: bool = False,
    explain: bool = False,
    pytorch_forecasting: bool = False,
    XAI_metric: bool = False,
    random_labels: bool = False,
    random_model_dir: str = None,
) -> float:
    """Preprocesses data and trains a model for each fold.

    Args:

        complete_train: Use the full data for training instead of held out test splits.
        wandb: Use wandb for logging.
        data_dir: Path to the data directory.
        log_dir: Path to the log directory.
        seed: Random seed.
        eval_only: Whether to only evaluate the model.
        train_size: Fixed size of train split (including validation data).
        load_weights: Whether to load weights from source_dir.
        source_dir: Path to the source directory.
        cv_folds: Number of folds for cross validation.
        cv_folds_to_train: Number of folds to use during training. If None, all folds are trained on.
        cv_repetitions: Amount of cross validation repetitions.
        cv_repetitions_to_train: Amount of training repetitions. If None, all repetitions are trained on.
        reproducible: Whether to make torch reproducible.
        debug: Whether to load less data and enable more logging.
        generate_cache: Whether to generate and save cache.
        load_cache: Whether to load previously cached data.
        test_on: Dataset to test on. Can be "test" or "val" (e.g. for hyperparameter tuning).
        mode: Run mode. Can be one of the values of RunMode
        pretrained_imputation_model: Use a pretrained imputation model.
        cpu: Whether to run on CPU.
        verbose: Enable detailed logging.
    Returns:
        The average loss of all folds.
    """
    if not cv_repetitions_to_train:
        cv_repetitions_to_train = cv_repetitions
    if not cv_folds_to_train:
        cv_folds_to_train = cv_folds
    agg_loss = 0
    seed_everything(seed, reproducible)
    if complete_train:
        logging.info("Will train full model without cross validation.")
        cv_repetitions_to_train = 1
        cv_folds_to_train = 1

    else:
        logging.info(f"Starting nested CV with {cv_repetitions_to_train} repetitions of {cv_folds_to_train} folds.")

    for repetition in range(cv_repetitions_to_train):
        for fold_index in range(cv_folds_to_train):
            repetition_fold_dir = log_dir / f"repetition_{repetition}" / f"fold_{fold_index}"
            repetition_fold_dir.mkdir(parents=True, exist_ok=True)

            start_time = datetime.now()
            data = preprocess_data(
                data_dir,
                seed=seed,
                debug=debug,
                load_cache=load_cache,
                generate_cache=generate_cache,
                cv_repetitions=cv_repetitions,
                repetition_index=repetition,
                train_size=train_size,
                cv_folds=cv_folds,
                fold_index=fold_index,
                pretrained_imputation_model=pretrained_imputation_model,
                runmode=mode,
                complete_train=complete_train,
            )

            preprocess_time = datetime.now() - start_time
            start_time = datetime.now()
            agg_loss += train_common(
                data,
                log_dir=repetition_fold_dir,
                eval_only=eval_only,
                load_weights=load_weights,
                source_dir=source_dir,
                reproducible=reproducible,
                test_on=test_on,
                mode=mode,
                cpu=cpu,
                verbose=verbose,
                use_wandb=wandb,
                train_only=complete_train,
                explain=explain,
                pytorch_forecasting=pytorch_forecasting,
                XAI_metric=XAI_metric,
                random_labels=random_labels,
                random_model_dir=random_model_dir,
            )

            train_time = datetime.now() - start_time

            log_full_line(
                f"FINISHED FOLD {fold_index}| PREPROCESSING DURATION {preprocess_time}| PROCEDURE DURATION {train_time}",
                level=logging.INFO,
            )
            durations = {
                "preprocessing_duration": preprocess_time,
                "train_duration": train_time,
            }

            with open(repetition_fold_dir / "durations.json", "w") as f:
                json.dump(durations, f, cls=JsonResultLoggingEncoder)
            if wandb:
                wandb_log({"Iteration": repetition * cv_folds_to_train + fold_index})
            if repetition * cv_folds_to_train + fold_index > 1:
                aggregate_results(log_dir)
        log_full_line(
            f"FINISHED CV REPETITION {repetition}",
            level=logging.INFO,
            char="=",
            num_newlines=3,
        )

    return agg_loss / (cv_repetitions_to_train * cv_folds_to_train)
