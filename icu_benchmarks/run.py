# -*- coding: utf-8 -*-
import argparse
import gin
import logging
import os
import sys

from icu_benchmarks.data.preprocess import preprocess_data
from icu_benchmarks.models.train import train_with_gin
from icu_benchmarks.models.utils import get_bindings_and_params

default_seed = 42


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark lib for processing and evaluation of deep learning models on ICU data"
    )

    parent_parser = argparse.ArgumentParser(add_help=False)

    subparsers = parser.add_subparsers(title="Commands", dest="command", required=True)

    parser_prep_and_train = subparsers.add_parser(
        "train", help="Calls sequentially merge and resample.", parents=[parent_parser]
    )

    #
    # DATA AND PREPROCESSING ARGUMENTS
    #
    preprocess_args = parser_prep_and_train.add_argument_group("Preprocess arguments")
    preprocess_args.add_argument(
        "-dir", "--data-dir", required=True, dest="data_dir", type=str, help="Path to the parquet data directory."
    )
    preprocess_args.add_argument(
        "-d", "--data", required=False, default="ricu", dest="data_config", type=str, help="Name of the data gin file."
    )

    #
    # MODEL ARGUMENTS
    #
    model_args = parser_prep_and_train.add_argument_group("Model arguments")
    model_args.add_argument(
        "-m",
        "--model",
        required=False,
        default="LGBMClassifier",
        dest="model_config",
        type=str,
        help="Name of the model gin file.",
    )
    model_args.add_argument(
        "-t",
        "--task",
        required=False,
        default="Mortality_At24Hours",
        dest="task_config",
        type=str,
        help="Name to the task gin file.",
    )
    model_args.add_argument("-l", "--logdir", dest="logdir", required=False, type=str, help="Path to the log directory.")
    model_args.add_argument(
        "--reproducible",
        default=True,
        dest="reproducible",
        required=False,
        type=str,
        help="Set torch to be reproducible.",
    )
    model_args.add_argument(
        "-rs", "--random-search", default=True, dest="rs", required=False, type=bool, help="Random Search setting"
    )
    model_args.add_argument(
        "-o",
        "--overwrite",
        default=False,
        dest="overwrite",
        required=False,
        type=bool,
        help="Set to overwrite previous model",
    )
    # model_args.add_argument('-r', '--resampling', default=None, dest="res",
    #                              required=False, type=int,
    #                              help="resampling for the data")
    # model_args.add_argument('-rl', '--resampling_label', default=None,
    #                              dest="res_lab", required=False, type=int,
    #                              help="resampling for the prediction")
    params = [
        ("sd", "seed", int, 1111, "Random seed at training and evaluation, default : 1111"),
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
        model_args.add_argument(
            f"-{short}",
            f"--{dest}".replace("_", "-"),
            required=False,
            default=default,
            dest=dest,
            type=tp,
            nargs="+",
            help=desc,
        )

    #
    # TRANSFER ARGUMENTS
    #
    transfer_parser = subparsers.add_parser("transfer", help="transfer", parents=[parent_parser])
    transfer_parser.add_argument(
        "-dir", "--data-dir", required=True, dest="data_dir", type=str, help="Path to the parquet data directory."
    )
    transfer_parser.add_argument(
        "-d", "--data", required=False, default="ricu", dest="data_config", type=str, help="Path to the gin data config file."
    )
    transfer_parser.add_argument(
        "-t",
        "--task",
        required=True,
        dest="task_configs",
        nargs="+",
        type=str,
        help="Paths to the gin task config file.",
    )
    transfer_parser.add_argument(
        "-m",
        "--model",
        required=True,
        dest="model_configs",
        nargs="+",
        type=str,
        help="Path to the gin model config file.",
    )
    transfer_parser.add_argument(
        "-w", "--model-weights", required=True, dest="model_weights", type=str, help="Path to the model weights directory."
    )

    return parser


def make_config_path(prefix, name):
    return f"configs/{prefix}/{name}.gin"


def main(my_args=tuple(sys.argv[1:])):
    args = build_parser().parse_args(my_args)

    log_fmt = "%(asctime)s - %(levelname)s: %(message)s"
    logging.basicConfig(format=log_fmt)
    logging.getLogger().setLevel(logging.INFO)

    data_config = make_config_path("data", args.data_config)
    for model in args.model_configs:
        model_config = make_config_path("models", model)
        gin.parse_config_file(data_config)
        gin.parse_config_file(model_config)
        data = preprocess_data(args.data_dir)

        load_weights = args.command == "evaluate"
        reproducible = str(args.reproducible) == "True"
        seeds = args.seed if isinstance(args.seed, list) else [args.seed]
        log_dir_base = f"{args.data_dir}/logs" if args.logdir is None else args.logdir
        log_dir_model = f"{log_dir_base}/{model}"
        gin_bindings, log_dir = get_bindings_and_params(args, log_dir_model)
        if load_weights:
            log_dir = log_dir_model
        if args.rs:
            reproducible = False
            max_attempt = 0
            is_already_ran = os.path.isdir(log_dir)
            while is_already_ran and max_attempt < 500:
                gin_bindings, log_dir = get_bindings_and_params(args)
                is_already_ran = os.path.isdir(log_dir)
                max_attempt += 1
            if max_attempt >= 300:
                raise Exception("Reached max attempt to find unexplored set of parameters parameters")

        for task in args.task_configs:
            gin_bindings_task = gin_bindings + ["TASK = " + "'" + str(task) + "'"]
            gin_config_files = [data_config, model_config, make_config_path("tasks", task)]
            log_dir_task = os.path.join(log_dir, str(task))
            for seed in seeds:
                log_dir_seed = log_dir_task if load_weights else os.path.join(log_dir_task, str(seed))
                train_with_gin(
                    model_dir=log_dir_seed,
                    data=data,
                    overwrite=args.overwrite,
                    load_weights=load_weights,
                    gin_config_files=gin_config_files,
                    gin_bindings=gin_bindings_task,
                    seed=seed,
                    reproducible=reproducible,
                )


"""Main module."""

if __name__ == "__main__":
    main()
