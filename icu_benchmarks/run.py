# -*- coding: utf-8 -*-
import argparse
from argparse import BooleanOptionalAction
from ast import literal_eval
import logging
import re
import sys
from pathlib import Path

import gin
import numpy as np

from icu_benchmarks.data.preprocess import preprocess_data
from icu_benchmarks.models.train import train_with_gin
from icu_benchmarks.models.utils import get_bindings

MAX_ATTEMPTS = 300
SEEDS = [1111]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark lib for processing and evaluation of deep learning models on ICU data"
    )

    parent_parser = argparse.ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(title="Commands", dest="command", required=True)

    # ARGUMENTS FOR ALL COMMANDS
    general_args = parent_parser.add_argument_group("General arguments")
    general_args.add_argument("-dir", "--data-dir", required=True, type=Path, help="Path to the parquet data directory.")
    general_args.add_argument("-t", "--task-config", default="Mortality_At24Hours", help="Name of the task gin.")
    general_args.add_argument("-m", "--model-config", default="LGBMClassifier", help="Name of the model gin.")
    general_args.add_argument("-e", "--experiment-config", help="Name of the experiment gin.")
    general_args.add_argument("-l", "--log-dir", type=Path, help="Path to the log directory with model weights.")
    general_args.add_argument("-s", "--seed", default=SEEDS, nargs="+", type=int, help="Random seed at train and eval.")

    # MODEL TRAINING ARGUMENTS
    parser_prep_and_train = subparsers.add_parser("train", help="Preprocess data and train model.", parents=[parent_parser])
    train_args = parser_prep_and_train.add_argument_group("Train arguments")
    train_args.add_argument("-o", "--overwrite", default=False, action=BooleanOptionalAction, help="Overwrite previous model.")
    train_args.add_argument("--reproducible", default=True, action=BooleanOptionalAction, help="Set torch to be reproducible.")
    train_args.add_argument("-rs", "--random-search", default=True, action=BooleanOptionalAction, help="Enable random search.")
    train_args.add_argument("-hp", "--hyperparams", nargs="+", help="Hyperparameters for model.")

    # EVALUATION PARSER
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate trained model on data.", parents=[parent_parser])
    evaluate_parser.add_argument("-c", "--train-config", required=True, help="Original train gin.")

    return parser


def to_correct_type(value):
    try:
        val = literal_eval(value)
        return val if isinstance(val, list) else [val]
    except ValueError:
        return value


def rs_from_match(matchobj):
    values = to_correct_type(matchobj.group(0)[3:-1])
    return str(values[np.random.randint(len(values))])


def rs_gin_configs(gin_config_files):
    parsed_configs = []
    for gin_file in gin_config_files:
        with open(gin_file, encoding="utf-8") as f:
            contents = f.read()
            parsed_contents = re.sub(r'RS\((.*)\)', rs_from_match, contents, flags=re.MULTILINE)
            parsed_configs += [parsed_contents]
    return parsed_configs


def main(my_args=tuple(sys.argv[1:])):
    args = build_parser().parse_args(my_args)
    hyperparams = {param.split('=')[0]: to_correct_type(param.split('=')[1]) for param in args.hyperparams} if args.hyperparams else {}

    log_fmt = "%(asctime)s - %(levelname)s: %(message)s"
    logging.basicConfig(format=log_fmt)
    logging.getLogger().setLevel(logging.INFO)

    load_weights = args.command == "evaluate"
    task = args.task_config

    log_dir_base = args.data_dir / "logs" if args.log_dir is None else args.log_dir
    log_dir_model = log_dir_base / task / args.model_config
    if load_weights:
        reproducible = False
        overwrite = False
        gin.parse_config_file(args.train_config)
        gin_config_files = [args.train_config]
        gin_bindings, log_dir_bindings = get_bindings(hyperparams, log_dir_model)
    else:
        reproducible = args.reproducible
        overwrite = args.overwrite
        if args.experiment_config:
            experiment_config = Path(f"configs/experiments/{args.experiment_config}.gin")
            gin_config_files = [experiment_config]
        else:
            model_config = Path(f"configs/models/{args.model_config}.gin")
            task_config = Path(f"configs/tasks/{args.task_config}.gin")
            gin_config_files = [model_config, task_config]
        gin_bindings, log_dir_bindings = get_bindings(hyperparams, log_dir_model)
        if args.random_search:
            reproducible = False
            attempt = 0
            while Path.exists(log_dir_bindings) and attempt < MAX_ATTEMPTS:
                gin_bindings, log_dir_bindings = get_bindings(hyperparams, log_dir_model, do_rs=True)
                attempt += 1
            if Path.exists(log_dir_bindings):
                raise Exception("Reached max attempt to find unexplored set of parameters parameters")

    logging.info(f"Selected hyper parameters: {gin_bindings}")
    logging.info(f"Log directory: {log_dir_bindings}")
    gin_configs = rs_gin_configs(gin_config_files)
    gin.parse_config(gin_configs)
    data = preprocess_data(args.data_dir)

    gin_bindings_task = gin_bindings + [f"TASK = '{task}'"]
    all_configs = gin_configs + gin_bindings_task
    for seed in args.seed:
        log_dir_seed = log_dir_bindings / str(seed)
        train_with_gin(
            model_dir=log_dir_seed,
            data=data,
            overwrite=overwrite,
            load_weights=load_weights,
            gin_configs=all_configs,
            seed=seed,
            reproducible=reproducible,
        )


"""Main module."""
if __name__ == "__main__":
    main()
