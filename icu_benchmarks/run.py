# -*- coding: utf-8 -*-
import argparse
from argparse import BooleanOptionalAction
from datetime import datetime
import gin
import logging
import sys
from pathlib import Path

from icu_benchmarks.data.preprocess import preprocess_data
from icu_benchmarks.models.train import train_common
from icu_benchmarks.hyperparameter_tuning import choose_and_bind_hyperparameters

SEEDS = [1111]


def build_parser() -> argparse.ArgumentParser:
    """Builds an ArgumentParser for the command line.

    Returns:
        The configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark lib for processing and evaluation of deep learning models on ICU data"
    )

    parent_parser = argparse.ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(title="Commands", dest="command", required=True)

    # ARGUMENTS FOR ALL COMMANDS
    general_args = parent_parser.add_argument_group("General arguments")
    general_args.add_argument("-d", "--data-dir", required=True, type=Path, help="Path to the parquet data directory.")
    general_args.add_argument("-n", "--name", required=True, help="Name of the (target) dataset.")
    general_args.add_argument("-t", "--task", default="BinaryClassification", help="Name of the task gin.")
    general_args.add_argument("-tn", "--task-name", required=True, help="Name of the task.")
    general_args.add_argument("-m", "--model", default="LGBMClassifier", help="Name of the model gin.")
    general_args.add_argument("-e", "--experiment", help="Name of the experiment gin.")
    general_args.add_argument(
        "-l", "--log-dir", required=True, type=Path, help="Path to the log directory with model weights."
    )
    general_args.add_argument("-s", "--seed", default=SEEDS, nargs="+", type=int, help="Random seed at train and eval.")
    general_args.add_argument("-db", "--debug", action=BooleanOptionalAction, help="Set to load less data.")
    general_args.add_argument("-c", "--cache", action=BooleanOptionalAction, help="Set to cache and use preprocessed data.")

    # MODEL TRAINING ARGUMENTS
    parser_prep_and_train = subparsers.add_parser("train", help="Preprocess data and train model.", parents=[parent_parser])
    parser_prep_and_train.add_argument(
        "--reproducible", default=True, action=BooleanOptionalAction, help="Set torch to be reproducible."
    )
    parser_prep_and_train.add_argument("--cpu", default=False, action=BooleanOptionalAction, help="Set to train on CPU.")
    parser_prep_and_train.add_argument("-hp", "--hyperparams", nargs="+", help="Hyperparameters for model.")
    parser_prep_and_train.add_argument("--tune", action=BooleanOptionalAction, help="Find best hyperparameters.")

    # EVALUATION PARSER
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate trained model on data.", parents=[parent_parser])
    evaluate_parser.add_argument("-sn", "--source-name", required=True, type=Path, help="Name of the source dataset.")
    evaluate_parser.add_argument("--source-dir", required=True, type=Path, help="Directory containing gin and model weights.")

    return parser


def create_run_dir(log_dir: Path, randomly_searched_params: str = None) -> Path:
    """Creates a log directory with the current time as name.

    Also creates a file in the log directory, if any parameters were randomly searched.
    The filename contains the fixed hyperparameters to check against in future runs.

    Args:
        log_dir: Parent directory to create run directory in.
        randomly_searched_params: String representing the randomly searched params.

    Returns:
        Path to the created run log directory.
    """
    log_dir_run = log_dir / str(datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
    log_dir_run.mkdir(parents=True)
    if randomly_searched_params:
        (log_dir_run / randomly_searched_params).touch()
    return log_dir_run


@gin.configurable
def preprocess_and_train_for_folds(
    data_dir,
    log_dir,
    seed,
    load_weights=False,
    source_dir=None,
    num_folds=gin.REQUIRED,
    num_folds_to_train=None,
    reproducible=False,
    debug=False,
    use_cache=False,
    test_on="test",
):
    if not num_folds_to_train:
        num_folds_to_train = num_folds
    agg_loss = 0
    for fold_index in range(num_folds_to_train):
        data = preprocess_data(
            data_dir, seed=seed, debug=debug, use_cache=use_cache, num_folds=num_folds, fold_index=fold_index
        )

        run_dir_seed = log_dir / f"seed_{seed}" / f"fold_{fold_index}"
        run_dir_seed.mkdir(parents=True, exist_ok=True)

        agg_loss += train_common(
            data,
            log_dir=run_dir_seed,
            load_weights=load_weights,
            source_dir=source_dir,
            seed=seed,
            reproducible=reproducible,
            test_on=test_on,
        )

    return agg_loss / num_folds


def main(my_args=tuple(sys.argv[1:])):
    args = build_parser().parse_args(my_args)

    debug = args.debug
    cache = args.cache
    if debug and cache:
        raise ValueError("Caching is not supported in debug mode.")

    log_fmt = "%(asctime)s - %(levelname)s: %(message)s"
    logging.basicConfig(format=log_fmt)
    logging.getLogger().setLevel(logging.INFO)

    load_weights = args.command == "evaluate"
    name = args.name
    task = args.task
    model = args.model
    experiment = args.experiment
    log_dir_name = args.log_dir / name
    log_dir = (log_dir_name / experiment) if experiment else (log_dir_name / args.task_name / model)

    if load_weights:
        log_dir /= f"from_{args.source_name}"
        source_dir = args.source_dir
        reproducible = False
        gin_config_files = [source_dir / "train_config.gin"]
    else:
        source_dir = None
        reproducible = args.reproducible
        if args.experiment:
            gin_config_files = [Path(f"configs/experiments/{args.experiment}.gin")]
        else:
            gin_config_files = [Path(f"configs/models/{model}.gin"), Path(f"configs/tasks/{task}.gin")]

    run_dir = create_run_dir(log_dir)
    gin.parse_config_files_and_bindings(gin_config_files, None, finalize_config=False)

    for seed in args.seed:
        choose_and_bind_hyperparameters(args.tune, args.data_dir, run_dir, seed)
        preprocess_and_train_for_folds(
            args.data_dir,
            run_dir,
            seed,
            load_weights=load_weights,
            source_dir=source_dir,
            reproducible=reproducible,
            debug=args.debug,
            use_cache=args.cache,
        )


"""Main module."""
if __name__ == "__main__":
    main()
