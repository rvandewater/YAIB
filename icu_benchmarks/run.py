# -*- coding: utf-8 -*-
import argparse
from argparse import BooleanOptionalAction
import gin
import logging
import sys
from pathlib import Path


from icu_benchmarks.data.preprocess import preprocess_data
from icu_benchmarks.models.train import train_with_gin
from icu_benchmarks.gin_parser import random_search_configs_and_create_log_dir

SEEDS = [1111]


def build_parser():
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
    general_args.add_argument("-t", "--task", default="Mortality_At24Hours", help="Name of the task gin.")
    general_args.add_argument("-m", "--model", default="LGBMClassifier", help="Name of the model gin.")
    general_args.add_argument("-e", "--experiment", help="Name of the experiment gin.")
    general_args.add_argument("-l", "--log-dir", type=Path, help="Path to the log directory with model weights.")
    general_args.add_argument("-s", "--seed", default=SEEDS, nargs="+", type=int, help="Random seed at train and eval.")

    # MODEL TRAINING ARGUMENTS
    parser_prep_and_train = subparsers.add_parser("train", help="Preprocess data and train model.", parents=[parent_parser])
    train_args = parser_prep_and_train.add_argument_group("Train arguments")
    train_args.add_argument("--reproducible", default=True, action=BooleanOptionalAction, help="Set torch to be reproducible.")
    train_args.add_argument("-hp", "--hyperparams", nargs="+", help="Hyperparameters for model.")

    # EVALUATION PARSER
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate trained model on data.", parents=[parent_parser])
    evaluate_parser.add_argument("-c", "--train-config", required=True, help="Original train gin.")

    return parser


def main(my_args=tuple(sys.argv[1:])):
    args = build_parser().parse_args(my_args)

    log_fmt = "%(asctime)s - %(levelname)s: %(message)s"
    logging.basicConfig(format=log_fmt)
    logging.getLogger().setLevel(logging.INFO)

    load_weights = args.command == "evaluate"
    task = args.task
    model = args.model

    if load_weights:
        reproducible = False
        gin_config_files = [args.train_config]
    elif args.experiment:
        reproducible = args.reproducible
        gin_config_files = [Path(f"configs/experiments/{args.experiment}.gin")]
    else:
        reproducible = args.reproducible
        gin_config_files = [Path(f"configs/models/{model}.gin"), Path(f"configs/tasks/{task}.gin")]

    log_dir_base = args.data_dir / "logs" if args.log_dir is None else args.log_dir
    log_dir_model = log_dir_base / task / model
    gin_configs, log_dir = random_search_configs_and_create_log_dir(gin_config_files, args.hyperparams, log_dir_model)
    gin_configs += [f"TASK = '{task}'"]
    gin.parse_config(gin_configs)
    data = preprocess_data(args.data_dir)

    for seed in args.seed:
        log_dir_seed = log_dir / str(seed)
        train_with_gin(
            model_dir=log_dir_seed,
            data=data,
            load_weights=load_weights,
            gin_configs=gin_configs,
            seed=seed,
            reproducible=reproducible,
        )


"""Main module."""
if __name__ == "__main__":
    main()
