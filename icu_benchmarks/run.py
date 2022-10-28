# -*- coding: utf-8 -*-
import argparse
import logging
import sys
from pathlib import Path

import gin

from icu_benchmarks.data.preprocess import preprocess_data
from icu_benchmarks.models.train import train_with_gin
from icu_benchmarks.models.utils import get_bindings

max_rs_attempt = 300
default_seeds = [1111]
cli_hyper_params = [
    ("bs", "batch_size", int, None, "Batchsize for the model"),
    ("lr", "learning_rate", float, None, "Learning rate for the model"),
    ("nc", "num_class", int, None, "Number of classes considered for the task"),
    ("emb", "embeddings", int, None, "Embedding size of the input data"),
    ("k", "kernel_size", int, None, "Kernel size for Temporal CNN"),
    ("hi", "hidden", int, None, "Dimensionality of hidden layer in Neural Networks"),
    ("de", "depth", int, None, "Number of layers in Neural Network"),
    ("ho", "horizon", int, None, "History length for Neural Networks"),
    ("do", "drop_out", float, None, "Dropout probability"),
    ("do_att", "drop_out_att", float, None, "Dropout probability for the Self-Attention layer only"),
    ("he", "heads", int, None, "Number of heads in Sel-Attention layer"),
    ("la", "latent", int, None, "Dimension of fully-conected layer in Transformer block"),
    ("ssd", "subsample_data", float, None, "Parameter in Gradient Boosting, subsample ratio of the training"),
    ("ssf", "subsample_feat", float, None, "Subsample ratio of columns when constructing each tree in LGBM"),
    ("reg", "l1_reg", float, None, "L1 regularization coefficient for Transformer"),
    ("cp", "c_parameter", float, None, "C parameter in Logistic Regression"),
    ("pen", "penalty", float, None, "Penalty parameter for Logistic Regression"),
    ("lw", "loss_weight", str, None, "Loss weigthing parameter"),
    # ("r", "resampling", int, None, "Resampling for the data"),
    # ("rl", "resampling_label", int, None, "Resampling for the prediction"),
]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark lib for processing and evaluation of deep learning models on ICU data"
    )

    parent_parser = argparse.ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(title="Commands", dest="command", required=True)

    #
    # ARGUMENTS FOR ALL COMMANDS
    #
    general_args = parent_parser.add_argument_group("General arguments")
    general_args.add_argument(
        "-dir", "--data-dir", required=True, dest="data_dir", type=Path, help="Path to the parquet data directory."
    )
    general_args.add_argument("-t", "--task", default="Mortality_At24Hours", dest="task_config", help="Name of the task gin.")
    general_args.add_argument("-m", "--model", default="LGBMClassifier", dest="model_config", help="Name of the model gin.")
    general_args.add_argument("-e", "--experiment", dest="experiment_config", help="Name of the experiment gin.")
    general_args.add_argument(
        "-l", "--log-dir", dest="log_dir", type=Path, help="Path to the log directory with model weights."
    )
    general_args.add_argument(
        "-s", "--seed", default=default_seeds, dest="seed", nargs="+", type=int, help="Random seed at training and evaluation."
    )

    #
    # MODEL TRAINING ARGUMENTS
    #
    parser_prep_and_train = subparsers.add_parser("train", help="Preprocess data and train model.", parents=[parent_parser])
    train_args = parser_prep_and_train.add_argument_group("Train arguments")
    train_args.add_argument(
        "--reproducible", default=True, type=bool, dest="reproducible", help="Set torch to be reproducible."
    )
    train_args.add_argument("-rs", "--random-search", default=True, dest="rs", type=bool, help="Enable random search.")
    train_args.add_argument("-o", "--overwrite", default=False, dest="overwrite", type=bool, help="Overwrite previous model.")

    hyper_args = parser_prep_and_train.add_argument_group("Hyperparameters")
    for short, dest, tp, default, desc in cli_hyper_params:
        hyper_args.add_argument(
            f"-{short}", f"--{dest}".replace("_", "-"), default=default, dest=dest, type=tp, nargs="+", help=desc
        )

    #
    # EVALUATION PARSER
    #
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate trained model on data.", parents=[parent_parser])
    evaluate_parser.add_argument("-c", "--train-config", required=True, dest="train_config", help="Original train gin.")

    return parser


def main(my_args=tuple(sys.argv[1:])):
    args = build_parser().parse_args(my_args)
    hyper_params = [param[1] for param in cli_hyper_params]

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
        gin_bindings, log_dir_bindings = get_bindings(hyper_params, args, log_dir_model)
    else:
        reproducible = args.reproducible
        overwrite = args.overwrite
        if args.experiment_config:
            experiment_config = Path(f"configs/experiments/{args.experiment_config}.gin")
            print(experiment_config)
            gin.parse_config_file(experiment_config)
            gin_config_files = [experiment_config]
        else:
            model_config = Path(f"configs/models/{args.model_config}.gin")
            task_config = Path(f"configs/tasks/{args.task_config}.gin")
            gin.parse_config_file(model_config)
            gin.parse_config_file(task_config)
            gin_config_files = [model_config, task_config]
        gin_bindings, log_dir_bindings = get_bindings(hyper_params, args, log_dir_model)
        if args.rs:
            reproducible = False
            attempt = 0
            while Path.exists(log_dir_bindings) and attempt < max_rs_attempt:
                gin_bindings, log_dir_bindings = get_bindings(hyper_params, args, log_dir_model, do_rs=True)
                attempt += 1
            if Path.exists(log_dir_bindings):
                raise Exception("Reached max attempt to find unexplored set of parameters parameters")

    logging.info(f"Selected hyper parameters: {gin_bindings}")
    logging.info(f"Log directory: {log_dir_bindings}")

    data = preprocess_data(args.data_dir)

    gin_bindings_task = gin_bindings + [f"TASK = '{task}'"]
    for seed in args.seed:
        log_dir_seed = log_dir_bindings / str(seed)
        train_with_gin(
            model_dir=log_dir_seed,
            data=data,
            overwrite=overwrite,
            load_weights=load_weights,
            gin_config_files=gin_config_files,
            gin_bindings=gin_bindings_task,
            seed=seed,
            reproducible=reproducible,
        )


"""Main module."""
if __name__ == "__main__":
    main()
