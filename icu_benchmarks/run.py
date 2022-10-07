# -*- coding: utf-8 -*-
import argparse
import logging
import os
import sys
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

# from sklearn.impute import MissingIndicator
from icu_benchmarks.common import constants

from icu_benchmarks.common.constants import VARS
from icu_benchmarks.data.preprocess import generate_splits
from icu_benchmarks.recipes.recipe import Recipe
from icu_benchmarks.recipes.selector import all_numeric_predictors, ends_with, all_of
from icu_benchmarks.recipes.step import Accumulator, StepHistorical, StepImputeFill
from icu_benchmarks.models.train import train_with_gin
from icu_benchmarks.models.utils import get_bindings_and_params

default_seed = 42


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark lib for processing and evaluation of deep learning models on HiRID ICU data"
    )

    parent_parser = argparse.ArgumentParser(add_help=False)

    subparsers = parser.add_subparsers(title="Commands", dest="command", required=True)

    parser_prep_ml = subparsers.add_parser(
        "preprocess", help="Calls sequentially merge and resample.", parents=[parent_parser]
    )

    preprocess_arguments = parser_prep_ml.add_argument_group("Preprocess arguments")

    preprocess_arguments.add_argument(
        "--data-dir", required=True, type=Path, help="Path to the parquet data directory as preprocessed by RICU."
    )
    preprocess_arguments.add_argument(
        "-nw",
        "--nr-workers",
        default=1,
        required=False,
        type=int,
        dest="nr_workers",
        help="Number of process to use at preprocessing, Default to 1 ",
    )
    preprocess_arguments.add_argument(
        "--seed", dest="seed", default=default_seed, required=False, type=int, help="Seed for the train/val/test split"
    )
    # preprocess_arguments.add_argument('--imputation', dest="imputation",
    #                                   default='ffill', required=False, type=str,
    #                                   help="Type of imputation. Default: 'ffill' ")
    # preprocess_arguments.add_argument('--horizon', dest="horizon",
    #                                   default=12, required=False, type=int,
    #                                   help="Horizon of prediction in hours for failure tasks")

    model_arguments = parent_parser.add_argument_group("Model arguments")
    model_arguments.add_argument("-l", "--logdir", dest="logdir", required=False, type=str, help="Path to the log directory ")
    model_arguments.add_argument(
        "--reproducible",
        default=True,
        dest="reproducible",
        required=False,
        type=str,
        help="Whether to configure torch to be reproducible.",
    )
    model_arguments.add_argument(
        "-sd",
        "--seed",
        default=1111,
        dest="seed",
        required=False,
        nargs="+",
        type=int,
        help="Random seed at training and evaluation, default : 1111",
    )
    model_arguments.add_argument(
        "-t", "--task", default=None, dest="task", required=False, nargs="+", type=str, help="Name of the task : Default None"
    )
    # model_arguments.add_argument('-r', '--resampling', default=None, dest="res",
    #                              required=False, type=int,
    #                              help="resampling for the data")
    # model_arguments.add_argument('-rl', '--resampling_label', default=None,
    #                              dest="res_lab", required=False, type=int,
    #                              help="resampling for the prediction")
    model_arguments.add_argument(
        "-bs",
        "--batch-size",
        default=None,
        dest="batch_size",
        required=False,
        type=int,
        nargs="+",
        help="Batchsize for the model",
    )
    model_arguments.add_argument(
        "-lr",
        "--learning-rate",
        default=None,
        nargs="+",
        dest="lr",
        required=False,
        type=float,
        help="Learning rate for the model",
    )
    model_arguments.add_argument(
        "--num-class",
        default=None,
        dest="num_class",
        required=False,
        type=int,
        help="Number of classes considered for the task",
    )
    model_arguments.add_argument(
        "-emb", "--emb", default=None, dest="emb", required=False, nargs="+", type=int, help="Embedding size of the input data"
    )
    model_arguments.add_argument(
        "-kernel",
        "--kernel",
        default=None,
        dest="kernel",
        required=False,
        nargs="+",
        type=int,
        help="Kernel size for Temporal CNN",
    )
    model_arguments.add_argument(
        "-do",
        "--do",
        default=None,
        dest="do",
        required=False,
        nargs="+",
        type=float,
        help="Dropout probability for the Transformer block",
    )
    model_arguments.add_argument(
        "-do_att",
        "--do_att",
        default=None,
        dest="do_att",
        required=False,
        nargs="+",
        type=float,
        help="Dropout probability for the Self-Attention layer only",
    )
    model_arguments.add_argument(
        "-depth",
        "--depth",
        default=None,
        dest="depth",
        required=False,
        nargs="+",
        type=int,
        help="Number of layers in Neyral Network",
    )
    model_arguments.add_argument(
        "-heads",
        "--heads",
        default=None,
        dest="heads",
        required=False,
        nargs="+",
        type=int,
        help="Number of heads in Sel-Attention layer",
    )
    model_arguments.add_argument(
        "-latent",
        "--latent",
        default=None,
        dest="latent",
        required=False,
        nargs="+",
        type=int,
        help="Dimension of fully-conected layer in Transformer block",
    )
    model_arguments.add_argument(
        "-horizon",
        "--horizon",
        default=None,
        dest="horizon",
        required=False,
        nargs="+",
        type=int,
        help="History length for Neural Networks",
    )
    model_arguments.add_argument(
        "-hidden",
        "--hidden",
        default=None,
        dest="hidden",
        required=False,
        nargs="+",
        type=int,
        help="Dimensionality of hidden layer in Neural Networks",
    )
    model_arguments.add_argument(
        "--subsample-data",
        default=None,
        dest="subsample_data",
        required=False,
        nargs="+",
        type=float,
        help="Subsample parameter in Gradient Boosting, subsample ratio of the training instance",
    )
    model_arguments.add_argument(
        "--subsample-feat",
        default=None,
        dest="subsample_feat",
        required=False,
        nargs="+",
        type=float,
        help="Colsample_bytree parameter in Gradient Boosting, " "subsample ratio of columns when constructing each tree",
    )
    model_arguments.add_argument(
        "--regularization",
        default=None,
        dest="regularization",
        required=False,
        nargs="+",
        type=float,
        help="L1 or L2 regularization type",
    )
    model_arguments.add_argument(
        "-rs", "--random-search", default=False, dest="rs", required=False, type=bool, help="Random Search setting"
    )
    model_arguments.add_argument(
        "-c_parameter",
        "--c_parameter",
        default=None,
        dest="c_parameter",
        required=False,
        nargs="+",
        help="C parameter in Logistic Regression",
    )
    model_arguments.add_argument(
        "-penalty",
        "--penalty",
        default=None,
        dest="penalty",
        required=False,
        nargs="+",
        help="Penalty parameter for Logistic Regression",
    )
    model_arguments.add_argument(
        "--loss-weight", default=None, dest="loss_weight", required=False, nargs="+", type=str, help="Loss weigthing parameter"
    )
    model_arguments.add_argument(
        "-o",
        "--overwrite",
        default=False,
        dest="overwrite",
        required=False,
        type=bool,
        help="Boolean to overwrite previous model in logdir",
    )
    model_arguments.add_argument(
        "-c", "--config", default=None, dest="config", nargs="+", type=str, help="Path to the gin train config file."
    )

    # parser_evaluate = subparsers.add_parser('evaluate', help='evaluate',
    #                                         parents=[parent_parser])

    subparsers.add_parser("train", help="train", parents=[parent_parser])
    return parser


def run_preprocessing(work_dir):
    sta_path = work_dir / constants.FILE_NAMES["STATIC"]
    dyn_path = work_dir / constants.FILE_NAMES["DYNAMIC"]
    outc_path = work_dir / constants.FILE_NAMES["OUTCOME"]

    static_splits_path = work_dir / constants.FILE_NAMES["STATIC_SPLITS"]
    labels_splits_path = work_dir / constants.FILE_NAMES["LABELS_SPLITS"]
    dyn_splits_path = work_dir / constants.FILE_NAMES["DYNAMIC_SPLITS"]

    sta_imputed_path = work_dir / constants.FILE_NAMES["STATIC_IMPUTED"]
    dyn_imputed_path = work_dir / constants.FILE_NAMES["DYNAMIC_IMPUTED"]
    dyn_w_features_imputed_path = work_dir / constants.FILE_NAMES["FEATURES_IMPUTED"]

    if not static_splits_path.exists() or not labels_splits_path.exists() or not dyn_splits_path.exists():
        logging.info("Generating splits")
        generate_splits(sta_path, outc_path, dyn_path, static_splits_path, labels_splits_path, dyn_splits_path)
    else:
        logging.info(f"Splits in {work_dir} exist, skipping")

    if not sta_imputed_path.exists():
        logging.info("Imputing dynamic data")
        sta_df = pq.read_table(static_splits_path).to_pandas()
        print(sta_df)
        sta_rec = Recipe(sta_df, [], VARS["STATIC_VARS"], VARS["STAY_ID"])
        # # sta_rec.add_step(StepSklearn(MissingIndicator(), sel=all_predictors(), in_place=False))
        sta_rec.add_step(StepImputeFill(sel=all_numeric_predictors(), value=sta_df.loc["train"].mean()))
        sta_df_imputed = sta_rec.prep()
        pq.write_table(pa.Table.from_pandas(sta_df_imputed), sta_imputed_path)
    else:
        logging.info(f"Imputated static data in {sta_imputed_path} exists, skipping")

    dyn_df = pq.read_table(dyn_splits_path).to_pandas()
    if not dyn_imputed_path.exists():
        logging.info("Imputing dynamic data")
        dyn_rec = Recipe(dyn_df, [], VARS["DYNAMIC_VARS"], VARS["STAY_ID"])
        # # dyn_rec.add_step(StepSklearn(MissingIndicator(), sel=all_predictors(), in_place=False))
        dyn_rec.add_step(StepImputeFill(sel=all_of(VARS["DYNAMIC_VARS"]), method="ffill"))
        dyn_rec.add_step(StepImputeFill(sel=all_of(VARS["DYNAMIC_VARS"]), value=dyn_df.loc["train"].mean()))
        dyn_df_imputed = dyn_rec.prep()
        pq.write_table(pa.Table.from_pandas(dyn_df_imputed), dyn_imputed_path)
    else:
        logging.info(f"Imputated dynamic data in {dyn_imputed_path} exists, skipping")

    if not dyn_w_features_imputed_path.exists():
        logging.info("Generating features and imputing data")
        # train_df = dyn_df.loc['train']
        dyn_rec = Recipe(dyn_df, [], VARS["DYNAMIC_VARS"], VARS["STAY_ID"])

        dyn_rec.add_step(StepHistorical(sel=all_of(VARS["DYNAMIC_VARS"]), fun=Accumulator.MIN, suffix="min_hist"))
        dyn_rec.add_step(StepHistorical(sel=all_of(VARS["DYNAMIC_VARS"]), fun=Accumulator.MAX, suffix="max_hist"))
        dyn_rec.add_step(StepHistorical(sel=all_of(VARS["DYNAMIC_VARS"]), fun=Accumulator.COUNT, suffix="count_hist"))
        dyn_rec.add_step(StepHistorical(sel=all_of(VARS["DYNAMIC_VARS"]), fun=Accumulator.MEAN, suffix="mean_hist"))
        # # dyn_rec.add_step(StepSklearn(MissingIndicator(), sel=all_predictors(), in_place=False))
        dyn_rec.add_step(StepImputeFill(sel=all_of(VARS["DYNAMIC_VARS"]), method="ffill"))
        dyn_rec.add_step(StepImputeFill(sel=all_of(VARS["DYNAMIC_VARS"]), value=dyn_df.loc["train"].mean()))
        dyn_rec.add_step(StepImputeFill(sel=ends_with("_hist"), value=0))

        dyn_df_w_features_imputed = dyn_rec.prep()
        pq.write_table(pa.Table.from_pandas(dyn_df_w_features_imputed), dyn_w_features_imputed_path)
    else:
        logging.info(f"Imputated dynamic data with features in {dyn_w_features_imputed_path} exists, skipping")


def main(my_args=tuple(sys.argv[1:])):
    args = build_parser().parse_args(my_args)

    log_fmt = "%(asctime)s - %(levelname)s: %(message)s"
    logging.basicConfig(format=log_fmt)
    logging.getLogger().setLevel(logging.INFO)

    # Dispatch
    if args.command == "preprocess":
        run_preprocessing(args.data_dir)

    if args.command in ["train", "evaluate"]:
        load_weights = args.command == "evaluate"
        reproducible = str(args.reproducible) == "True"
        if not isinstance(args.seed, list):
            seeds = [args.seed]
        else:
            seeds = args.seed
        if not load_weights:
            gin_bindings, log_dir = get_bindings_and_params(args)
        else:
            gin_bindings, _ = get_bindings_and_params(args)
            log_dir = args.logdir
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

        if args.task is not None:
            for task in args.task:
                gin_bindings_task = gin_bindings + ["TASK = " + "'" + str(task) + "'"]
                log_dir_task = os.path.join(log_dir, str(task))
                for seed in seeds:
                    if not load_weights:
                        log_dir_seed = os.path.join(log_dir_task, str(seed))
                    else:
                        log_dir_seed = log_dir_task
                    train_with_gin(
                        model_dir=log_dir_seed,
                        overwrite=args.overwrite,
                        load_weights=load_weights,
                        gin_config_files=args.config,
                        gin_bindings=gin_bindings_task,
                        seed=seed,
                        reproducible=reproducible,
                    )
        else:
            for seed in seeds:
                if not load_weights:
                    log_dir_seed = os.path.join(log_dir, str(seed))
                train_with_gin(
                    model_dir=log_dir_seed,
                    overwrite=args.overwrite,
                    load_weights=load_weights,
                    gin_config_files=args.config,
                    gin_bindings=gin_bindings,
                    seed=seed,
                    reproducible=reproducible,
                )


"""Main module."""

if __name__ == "__main__":
    main()
