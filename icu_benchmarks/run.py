# -*- coding: utf-8 -*-
import argparse
from argparse import BooleanOptionalAction
from datetime import datetime
import gin
import logging
import sys
from pathlib import Path


from icu_benchmarks.data.preprocess import preprocess_data
from icu_benchmarks.models.train import train_with_gin
from icu_benchmarks.gin_parser import random_search_configs

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
    general_args.add_argument("-dir", "--data-dir", required=True, type=Path, help="Path to the parquet data directory.")
    general_args.add_argument("-d", "--data", required=True, help="Name of the (source) dataset.")
    general_args.add_argument("-t", "--task", default="Mortality_At24Hours", help="Name of the task gin.")
    general_args.add_argument("-m", "--model", default="LGBMClassifier", help="Name of the model gin.")
    general_args.add_argument("-e", "--experiment", help="Name of the experiment gin.")
    general_args.add_argument("-l", "--log-dir", type=Path, help="Path to the log directory with model weights.")
    general_args.add_argument("--seed", default=SEEDS, nargs="+", type=int, help="Random seed at train and eval.")

    # MODEL TRAINING ARGUMENTS
    parser_prep_and_train = subparsers.add_parser("train", help="Preprocess data and train model.", parents=[parent_parser])
    parser_prep_and_train.add_argument(
        "--reproducible", default=True, action=BooleanOptionalAction, help="Set torch to be reproducible."
    )
    parser_prep_and_train.add_argument("-hp", "--hyperparams", nargs="+", help="Hyperparameters for model.")

    # EVALUATION PARSER
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate trained model on data.", parents=[parent_parser])
    evaluate_parser.add_argument(
        "-s", "--source-dir", required=True, type=Path, help="Directory containing train gin and model weights."
    )
    evaluate_parser.add_argument("--target", required=True, type=Path, help="Name of the taget dataset.")

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


def main(my_args=tuple(sys.argv[1:])):
    args = build_parser().parse_args(my_args)

    log_fmt = "%(asctime)s - %(levelname)s: %(message)s"
    logging.basicConfig(format=log_fmt)
    logging.getLogger().setLevel(logging.INFO)

    load_weights = args.command == "evaluate"
    data = args.data
    task = args.task
    model = args.model
    log_dir_base = args.data_dir / "logs" if args.log_dir is None else args.log_dir

    if load_weights:
        log_dir_model = log_dir_base / data / task / args.target / model
        source_dir = args.source_dir
        reproducible = False
        with open(source_dir / "train_config.gin") as f:
            gin_configs = f.read()
        log_dir = create_run_dir(log_dir_model)
    else:
        log_dir_model = log_dir_base / data / task / model
        source_dir = None
        reproducible = args.reproducible
        if args.experiment:
            gin_config_files = [Path(f"configs/experiments/{args.experiment}.gin")]
        else:
            gin_config_files = [Path(f"configs/models/{model}.gin"), Path(f"configs/tasks/{task}.gin")]
        gin_configs, randomly_searched_params = random_search_configs(gin_config_files, args.hyperparams, log_dir_model)
        log_dir = create_run_dir(log_dir_model, randomly_searched_params)
        gin_configs += [f"TASK = '{task}'"]

    gin.parse_config(gin_configs)
    data = preprocess_data(args.data_dir)

    for seed in args.seed:
        log_dir_seed = log_dir / str(seed)
        log_dir_seed.mkdir()
        train_with_gin(
            log_dir=log_dir_seed,
            data=data,
            load_weights=load_weights,
            source_dir=source_dir,
            gin_configs=gin_configs,
            seed=seed,
            reproducible=reproducible,
        )


"""Main module."""
if __name__ == "__main__":
    main()
