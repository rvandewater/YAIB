# -*- coding: utf-8 -*-
import argparse
import logging
import os
import sys

import gin

from icu_benchmarks.data.preprocess import preprocess_data
from icu_benchmarks.models.train import train_with_gin
from icu_benchmarks.models.utils import get_bindings_w_rs

default_seeds = [1111]


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
    general_args.add_argument("-dir", "--data-dir", required=True, dest="data_dir", help="Path to the parquet data directory.")
    general_args.add_argument("-t", "--task", default="Mortality_At24Hours", dest="task_config", help="Name of the task gin.")
    general_args.add_argument("-m", "--model", default="LGBMClassifier", dest="model_config", help="Name of the model gin.")
    general_args.add_argument("-l", "--log-dir", dest="log_dir", help="Path to the log directory with model weights.")

    #
    # MODEL TRAINING ARGUMENTS
    #
    parser_prep_and_train = subparsers.add_parser("train", help="Preprocess data and train model.", parents=[parent_parser])
    train_args = parser_prep_and_train.add_argument_group("Train arguments")
    train_args.add_argument("--reproducible", default=True, type=bool, dest="reproducible", help="Set torch to be reproducible.")
    train_args.add_argument(
        "-rs", "--random-search", default=True, dest="rs", required=False, type=bool, help="Enable random search."
    )
    train_args.add_argument("-o", "--overwrite", default=False, dest="overwrite", type=bool, help="Overwrite previous model.")

    hyper_args = parser_prep_and_train.add_argument_group("Hyperparameters")
    # hyper_args.add_argument('-r', '--resampling', default=None, dest="res",
    #                              required=False, type=int,
    #                              help="resampling for the data")
    # hyper_args.add_argument('-rl', '--resampling_label', default=None,
    #                              dest="res_lab", required=False, type=int,
    #                              help="resampling for the prediction")
    params = [
        ("sd", "seed", int, default_seeds, "Random seed at training and evaluation, default : 1111"),
        ("bs", "batch_size", int, None, "Batchsize for the model"),
        ("lr", "learning_rate", float, None, "Learning rate for the model"),
        ("nc", "num_class", int, None, "Number of classes considered for the task"),
        ("emb", "embeddings", int, None, "Embedding size of the input data"),
        ("k", "kernel", int, None, "Kernel size for Temporal CNN"),
        ("hi", "hidden", int, None, "Dimensionality of hidden layer in Neural Networks"),
        ("de", "depth", int, None, "Number of layers in Neural Network"),
        ("ho", "horizon", int, None, "History length for Neural Networks"),
        ("do", "drop_out", float, None, "Dropout probability"),
        ("do_att", "drop_out_att", float, None, "Dropout probability for the Self-Attention layer only"),
        ("he", "heads", int, None, "Number of heads in Sel-Attention layer"),
        ("la", "latent", int, None, "Dimension of fully-conected layer in Transformer block"),
        (
            "ssd",
            "subsample_data",
            float,
            None,
            "Parameter in Gradient Boosting, subsample ratio of the training",
        ),
        (
            "ssf",
            "subsample_feat",
            float,
            None,
            "Colsample_bytree in Gradient Boosting, subsample ratio of columns when constructing each tree",
        ),
        ("reg", "l1_reg", float, None, "L1 regularization coefficient for Transformer"),
        ("cp", "c_parameter", float, None, "C parameter in Logistic Regression"),
        ("pen", "penalty", float, None, "Penalty parameter for Logistic Regression"),
        ("lw", "loss_weight", str, None, "Loss weigthing parameter"),
    ]

    for short, dest, tp, default, desc in params:
        hyper_args.add_argument(
            f"-{short}",
            f"--{dest}".replace("_", "-"),
            default=default,
            dest=dest,
            type=tp,
            nargs="+",
            help=desc,
        )

    #
    # TRANSFER PARSER
    #
    transfer_parser = subparsers.add_parser("transfer", help="Evaluate trained model on data.", parents=[parent_parser])
    transfer_parser.add_argument("-c", "--train-config", required=True, dest="train_config", help="Original train gin.")

    return parser


def make_config_path(prefix, name):
    return f"configs/{prefix}/{name}.gin"


def main(my_args=tuple(sys.argv[1:])):
    args = build_parser().parse_args(my_args)

    log_fmt = "%(asctime)s - %(levelname)s: %(message)s"
    logging.basicConfig(format=log_fmt)
    logging.getLogger().setLevel(logging.INFO)

    load_weights = args.command == "transfer"
    task = args.task_config
    
    log_dir_base = f"{args.data_dir}/logs" if args.log_dir is None else args.log_dir
    log_dir_model = os.path.join(log_dir_base, task, args.model_config)
    if load_weights:
        reproducible = False
        overwrite = False
        gin.parse_config_file(args.train_config)
        gin_config_files = [args.train_config]
        gin_bindings, log_dir_bindings = get_bindings_w_rs(args, log_dir_model)
    else:
        reproducible = args.reproducible
        overwrite = args.overwrite
        model_config = make_config_path("models", args.model_config)
        task_config = make_config_path("tasks", args.task_config)
        gin.parse_config_file(model_config)
        gin.parse_config_file(task_config)
        gin_config_files = [model_config, task_config]
        gin_bindings, log_dir_bindings = get_bindings_w_rs(args, log_dir_model)
        # if args.rs:
        #     reproducible = False
        #     attempt = 0
        #     max_attempt = 300
        #     is_already_run = os.path.isdir(log_dir_bindings)
        #     gin.parse_config_file(model_config)
        #     while is_already_run and attempt < max_attempt:
        #         gin_bindings, log_dir_bindings = get_bindings_w_rs(args, log_dir_model)
        #         is_already_run = os.path.isdir(log_dir_bindings)
        #         attempt += 1
        #     if is_already_run:
        #         raise Exception("Reached max attempt to find unexplored set of parameters parameters")

    data = preprocess_data(args.data_dir)

    gin_bindings_task = gin_bindings + ["TASK = " + "'" + str(task) + "'"]
    seeds = args.seed if hasattr(args, 'seeds') else default_seeds
    for seed in seeds:
        log_dir_seed = os.path.join(log_dir_bindings, str(seed))
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
