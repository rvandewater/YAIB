# -*- coding: utf-8 -*-
import gin
import logging
import sys
from pathlib import Path
import importlib.util

from icu_benchmarks.hyperparameter_tuning import choose_and_bind_hyperparameters
from utils.plotting.utils import plot_agg_results
from icu_benchmarks.run_utils import (
    build_parser,
    create_run_dir,
    preprocess_and_train_for_folds,
    aggregate_results,
    log_full_line,
)


def main(my_args=tuple(sys.argv[1:])):
    args = build_parser().parse_args(my_args)

    log_fmt = "%(asctime)s - %(levelname)s: %(message)s"
    logging.basicConfig(format=log_fmt)
    logging.getLogger().setLevel(logging.INFO)

    load_weights = args.command == "evaluate"
    name = args.name
    task = args.task
    model = args.model
    experiment = args.experiment
    source_dir = None
    reproducible = False
    log_dir_name = args.log_dir / name
    log_dir = (log_dir_name / experiment) if experiment else (log_dir_name / args.task_name / model)
    train_on_cpu = args.cpu

    if args.preprocessor:
        # Import custom supplied preprocessor
        try:
            spec = importlib.util.spec_from_file_location("CustomPreprocessor", args.preprocessor)
            module = importlib.util.module_from_spec(spec)
            sys.modules["preprocessor"] = module
            spec.loader.exec_module(module)
            gin.bind_parameter("preprocess.preprocessor", module.CustomPreprocessor)
        except Exception as e:
            logging.error(f"Could not import custom preprocessor from {args.preprocessor}: {e}")

    if train_on_cpu:
        gin.bind_parameter("DLWrapper.device", "cpu")
    if load_weights:
        log_dir /= f"from_{args.source_name}"
        run_dir = create_run_dir(log_dir)
        source_dir = args.source_dir
        gin.parse_config_file(source_dir / "train_config.gin")
    else:
        reproducible = args.reproducible
        checkpoint = log_dir / args.checkpoint if args.checkpoint else None
        gin_config_files = (
            [Path(f"configs/experiments/{args.experiment}.gin")]
            if args.experiment
            else [Path(f"configs/models/{model}.gin"), Path(f"configs/tasks/{task}.gin")]
        )
        gin.parse_config_files_and_bindings(gin_config_files, args.hyperparams, finalize_config=False)
        run_dir = create_run_dir(log_dir)
        choose_and_bind_hyperparameters(
            args.tune, args.data_dir, run_dir, args.seeds[0], checkpoint=checkpoint, debug=args.debug
        )

    logging.info(f"Logging to {run_dir.resolve()}")
    log_full_line("STARTING TRAINING", level=logging.INFO, char="=", num_newlines=3)

    for seed in args.seeds:
        preprocess_and_train_for_folds(
            args.data_dir,
            run_dir,
            seed,
            load_weights=load_weights,
            source_dir=source_dir,
            reproducible=reproducible,
            debug=args.debug,
            load_cache=args.cache,
        )
        log_full_line(f"FINISHED SEED {seed}", level=logging.INFO, char="=", num_newlines=3)

    log_full_line("FINISHED TRAINING", level=logging.INFO, char="=", num_newlines=3)
    aggregate_results(run_dir)
    if args.plot:
        plot_agg_results(run_dir, "aggregated_test_metrics")


"""Main module."""
if __name__ == "__main__":
    main()
