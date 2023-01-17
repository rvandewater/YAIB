# -*- coding: utf-8 -*-
import argparse
from datetime import datetime
import gin
import logging
import sys
import torch
import wandb
from pathlib import Path
from pytorch_lightning import seed_everything

from icu_benchmarks.data.preprocess import preprocess_data
from icu_benchmarks.models.train import train_common
from icu_benchmarks.gin_utils import parse_gin_and_random_search
from icu_benchmarks.imputation import name_mapping

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
    general_args.add_argument("-t", "--task", default="Mortality_At24Hours", help="Name of the task gin.")
    general_args.add_argument("-m", "--model", default="LGBMClassifier", help="Name of the model gin.")
    general_args.add_argument("-e", "--experiment", help="Name of the experiment gin.")
    general_args.add_argument("-l", "--log-dir", type=Path, help="Path to the log directory with model weights.")
    general_args.add_argument("-s", "--seed", default=SEEDS, nargs="+", type=int, help="Random seed at train and eval.")
    general_args.add_argument("-db", "--debug", action="store_true", help="Set to load less data.")
    general_args.add_argument("-c", "--cache", action="store_true", help="Set to cache and use preprocessed data.")
    general_args.add_argument("--wandb-sweep", action="store_true", help="activates wandb hyper parameter sweep")
    general_args.add_argument("--use_pretrained_imputation", required=False, type=str, help="Path to pretrained imputation model.")

    # MODEL TRAINING ARGUMENTS
    parser_prep_and_train = subparsers.add_parser("train", help="Preprocess data and train model.", parents=[parent_parser])
    parser_prep_and_train.add_argument(
        "--reproducible", default=False, action="store_true", help="Set torch to be reproducible."
    )
    parser_prep_and_train.add_argument(
        "--hyperparameter-search", default=False, action="store_true", help="Train the model with an untried hyperparameter set."
    )
    parser_prep_and_train.add_argument("--cpu", default=False, action="store_true", help="Set to train on CPU.")
    parser_prep_and_train.add_argument("-hp", "--hyperparams", default=[], nargs="+", help="Hyperparameters for model.")

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

@gin.configurable("Run")
def get_mode(mode: gin.REQUIRED):
    return mode

def main(my_args=tuple(sys.argv[1:])):
    args = build_parser().parse_args(my_args)
    if args.wandb_sweep:
        wandb.init()
        sweep_config = wandb.config
        args.__dict__.update(sweep_config)
        for key, value in sweep_config.items():
            args.hyperparams.append(f"{key}=" + (('\'' + value + '\'') if isinstance(value, str) else str(value)))

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
    if isinstance(args.seed, int):
        args.seed = [args.seed]

    gin.parse_config_file(f"configs/tasks/{task}.gin")
    mode = get_mode()
    logging.info(f"Task mode: {mode}")
    experiment = args.experiment
    log_dir_base = args.data_dir / "logs" if args.log_dir is None else args.log_dir
    log_dir_name = log_dir_base / name
    log_dir = (log_dir_name / experiment) if experiment else (log_dir_name / task / model)
    
    print("using pretrained from", args.use_pretrained_imputation)
    if args.use_pretrained_imputation is not None and not Path(args.use_pretrained_imputation).exists():
        #     args.use_pretrained_imputation = Path(args.use_pretrained_imputation).parent / "model.ckpt"
        # print("exists:", args.use_pretrained_imputation.exists())
        # else:
        #     args.use_pretrained_imputation = Path(args.use_pretrained_imputation)
        print("doesnt exist")
        args.use_pretrained_imputation = None
    print("now using pretrained from", args.use_pretrained_imputation)
    
    # print("now loading from the following path: >"+str(args.use_pretrained_imputation.resolve())+"< and exists:"+str(args.use_pretrained_imputation.exists()))
    pretrained_imputation_model = torch.load(args.use_pretrained_imputation, map_location=torch.device('cpu')) if args.use_pretrained_imputation is not None else None
    if isinstance(pretrained_imputation_model, dict):
        model_name = Path(args.use_pretrained_imputation).parent.parent.parent.name
        model_class = name_mapping[model_name]
        # print("is dict? keys:", pretrained_imputation_model.keys())
        pretrained_imputation_model = model_class.load_from_checkpoint(args.use_pretrained_imputation)
        # for k, v in pretrained_imputation_model.items():
        #     if isinstance(v, str) and "NP" in v:
        #         print(k, ":", v)
    if wandb.run is not None:
        print("updating wandb config:", {"pretrained_imputation_model": pretrained_imputation_model.__class__.__name__ if pretrained_imputation_model is not None else "None"})
        wandb.config.update({"pretrained_imputation_model": pretrained_imputation_model.__class__.__name__ if pretrained_imputation_model is not None else "None"})

    if load_weights:
        log_dir /= f"from_{args.source_name}"
        source_dir = args.source_dir
        reproducible = False
        gin.parse_config_file(source_dir / "train_config.gin")
        run_dir = create_run_dir(log_dir)
    else:
        source_dir = None
        reproducible = args.reproducible
        if args.experiment:
            gin_config_files = [Path(f"configs/experiments/{args.experiment}.gin")]
        else:
            model_path = Path("configs") / ("imputation_models" if mode == "Imputation" else "classification_models")
            model_path = model_path / f"{model}.gin"
            gin_config_files = [model_path, Path(f"configs/tasks/{task}.gin")]
        print("HYPERPARAMS:", args.hyperparams)
        randomly_searched_params = parse_gin_and_random_search(gin_config_files, args.hyperparams, args.cpu, log_dir)
        run_dir = create_run_dir(log_dir, randomly_searched_params)

    for seed in args.seed:
        seed_everything(seed)
        data = preprocess_data(args.data_dir, seed=seed, debug=debug, use_cache=cache, mode=mode, pretrained_imputation_model=pretrained_imputation_model)
        run_dir_seed = run_dir / f"seed_{str(seed)}"
        run_dir_seed.mkdir()
        train_common(
            log_dir=run_dir_seed,
            data=data,
            load_weights=load_weights,
            source_dir=source_dir,
            reproducible=reproducible,
            dataset_name=name,
            mode=mode,
        )


"""Main module."""
if __name__ == "__main__":
    main()
