# -*- coding: utf-8 -*-
from datetime import datetime

import gin
import logging
import sys
from pathlib import Path
import importlib.util

import torch.cuda

from icu_benchmarks.wandb_utils import update_wandb_config, apply_wandb_sweep, set_wandb_run_name
from icu_benchmarks.tuning.hyperparameters import choose_and_bind_hyperparameters
from scripts.plotting.utils import plot_aggregated_results
from icu_benchmarks.cross_validation import execute_repeated_cv
from icu_benchmarks.run_utils import (
    build_parser,
    create_run_dir,
    aggregate_results,
    log_full_line,
    load_pretrained_imputation_model,
    setup_logging,
)
from icu_benchmarks.contants import RunMode


@gin.configurable("Run")
def get_mode(mode: gin.REQUIRED):
    # Check if enum is mode.
    assert RunMode(mode)
    return RunMode(mode)


def main(my_args=tuple(sys.argv[1:])):
    args, _ = build_parser().parse_known_args(my_args)

    # Set arguments for wandb sweep
    if args.wandb_sweep:
        args = apply_wandb_sweep(args)

    # Initialize loggers
    log_format = "%(asctime)s - %(levelname)s - %(name)s : %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    verbose = args.verbose
    setup_logging(date_format, log_format, verbose)
    verbose = True
    # Get arguments
    data_dir = Path(args.data_dir)
    name = args.name
    task = args.task
    model = args.model
    reproducible = args.reproducible

    # Set experiment name
    if name is None:
        name = data_dir.name
    logging.info(f"Running experiment {name}.")

    # Load task config
    gin.parse_config_file(f"configs/tasks/{task}.gin")

    mode = get_mode()

    evaluate = args.eval

    # Set train size to fine tune size if fine tune is set, else use custom train size
    train_size = args.fine_tune if args.fine_tune is not None else args.samples
    load_weights = evaluate or args.fine_tune is not None

    if args.wandb_sweep:
        run_name = f"{mode}_{model}_{name}"
        if load_weights:
            if args.fine_tune:
                run_name += f"_source_{args.source_name}_fine-tune_{args.fine_tune}_samples"
            else:
                run_name += f"_source_{args.source_name}"
        set_wandb_run_name(run_name)

    logging.info(f"Task mode: {mode}.")
    experiment = args.experiment

    pretrained_imputation_model = load_pretrained_imputation_model(args.pretrained_imputation)

    # Log imputation model to wandb
    update_wandb_config(
        {
            "pretrained_imputation_model": pretrained_imputation_model.__class__.__name__
            if pretrained_imputation_model is not None
            else "None"
        }
    )
    source_dir = None
    log_dir_name = args.log_dir / name
    log_dir = (
        (log_dir_name / experiment)
        if experiment
        else (log_dir_name / (args.task_name if args.task_name is not None else args.task) / model)
    )
    if torch.cuda.is_available():
        for name in range(0, torch.cuda.device_count()):
            log_full_line(f"Available GPU {name}: {torch.cuda.get_device_name(name)}", level=logging.INFO)
    else:
        log_full_line(
            "No GPUs available: please check your device and Torch,Cuda installation if unintended.", level=logging.WARNING
        )

    log_full_line(f"Logging to {log_dir}.", logging.INFO)

    if args.preprocessor:
        # Import custom supplied preprocessor
        log_full_line(f"Importing custom preprocessor from {args.preprocessor}.", logging.INFO)
        try:
            spec = importlib.util.spec_from_file_location("CustomPreprocessor", args.preprocessor)
            module = importlib.util.module_from_spec(spec)
            sys.modules["preprocessor"] = module
            spec.loader.exec_module(module)
            gin.bind_parameter("preprocess.preprocessor", module.CustomPreprocessor)
        except Exception as e:
            logging.error(f"Could not import custom preprocessor from {args.preprocessor}: {e}")

    # Load pretrained model in evaluate mode or when finetuning
    if load_weights:
        if args.source_dir is None:
            raise ValueError("Please specify a source directory when evaluating or fine-tuning.")
        log_dir /= f"_from_{args.source_name}"
        gin.bind_parameter("train_common.dataset_names", {"train": args.source_name, "val": args.source_name, "test": args.name})
        if args.fine_tune:
            log_dir /= f"fine_tune_{args.fine_tune}"
            gin.bind_parameter("train_common.dataset_names", {"train": args.name, "val": args.name, "test": args.name})
        run_dir = create_run_dir(log_dir)
        source_dir = args.source_dir
        logging.info(f"Will load weights from {source_dir} and bind train gin-config. Note: this might override your config.")
        gin.parse_config_file(source_dir / "train_config.gin")
    elif args.samples and args.source_dir is not None: # Train model with limited samples
        gin.parse_config_file(args.source_dir / "train_config.gin")
        log_dir /= f"samples_{args.fine_tune}"
        gin.bind_parameter("train_common.dataset_names", {"train": args.name, "val": args.name, "test": args.name})
        run_dir = create_run_dir(log_dir)
    else:
        # Normal train and evaluate
        gin.bind_parameter("train_common.dataset_names", {"train": args.name, "val": args.name, "test": args.name})
        hp_checkpoint = log_dir / args.hp_checkpoint if args.hp_checkpoint else None
        model_path = (
                Path("configs") / ("imputation_models" if mode == RunMode.imputation else "prediction_models") / f"{model}.gin"
        )
        gin_config_files = (
            [Path(f"configs/experiments/{args.experiment}.gin")]
            if args.experiment
            else [model_path, Path(f"configs/tasks/{task}.gin")]
        )
        gin.parse_config_files_and_bindings(gin_config_files, args.hyperparams, finalize_config=False)
        log_full_line(f"Data directory: {data_dir.resolve()}", level=logging.INFO)
        run_dir = create_run_dir(log_dir)
        choose_and_bind_hyperparameters(
            args.tune,
            data_dir,
            run_dir,
            args.seed,
            run_mode=mode,
            checkpoint=hp_checkpoint,
            debug=args.debug,
            generate_cache=args.generate_cache,
            load_cache=args.load_cache,
            verbose=verbose,
        )

    log_full_line(f"Logging to {run_dir.resolve()}", level=logging.INFO)
    if evaluate:
        mode_string = "STARTING EVALUATION"
    elif args.fine_tune:
        mode_string = "STARTING FINE TUNING"
    else:
        mode_string = "STARTING TRAINING"
    log_full_line(mode_string, level=logging.INFO, char="=", num_newlines=3)

    start_time = datetime.now()
    execute_repeated_cv(
        data_dir,
        run_dir,
        args.seed,
        eval_only=evaluate,
        train_size=train_size,
        load_weights=load_weights,
        source_dir=source_dir,
        reproducible=reproducible,
        debug=args.debug,
        verbose=args.verbose,
        load_cache=args.load_cache,
        generate_cache=args.generate_cache,
        mode=mode,
        pretrained_imputation_model=pretrained_imputation_model,
        cpu=args.cpu,
        wandb=args.wandb_sweep,
    )

    log_full_line("FINISHED TRAINING", level=logging.INFO, char="=", num_newlines=3)
    execution_time = datetime.now() - start_time
    log_full_line(f"DURATION: {execution_time}", level=logging.INFO, char="")
    aggregate_results(run_dir, execution_time)
    if args.plot:
        plot_aggregated_results(run_dir, "aggregated_test_metrics.json")


"""Main module."""
if __name__ == "__main__":
    main()
