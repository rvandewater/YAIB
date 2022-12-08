# -*- coding: utf-8 -*-
import gin
import logging
import sys
from pathlib import Path

from icu_benchmarks.hyperparameter_tuning import choose_and_bind_hyperparameters
from icu_benchmarks.run_utils import build_parser, create_run_dir, preprocess_and_train_for_folds


def main(my_args=tuple(sys.argv[1:])):
    args = build_parser().parse_args(my_args)

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
    experiment = args.experiment
    log_dir_name = args.log_dir / name
    log_dir = (log_dir_name / experiment) if experiment else (log_dir_name / args.task_name / model)

    if load_weights:
        log_dir /= f"from_{args.source_name}"
        source_dir = args.source_dir
        reproducible = False
        gin_config_files = [source_dir / "train_config.gin"]
    else:
        source_dir = None
        reproducible = args.reproducible
        if args.experiment:
            gin_config_files = [Path(f"configs/experiments/{args.experiment}.gin")]
        else:
            gin_config_files = [Path(f"configs/models/{model}.gin"), Path(f"configs/tasks/{task}.gin")]

    run_dir = create_run_dir(log_dir)

    gin.parse_config_files_and_bindings(gin_config_files, args.hyperparams, finalize_config=False)
    choose_and_bind_hyperparameters(args.tune, args.data_dir, run_dir, args.seed)
    preprocess_and_train_for_folds(
        args.data_dir,
        run_dir,
        args.seed,
        load_weights=load_weights,
        source_dir=source_dir,
        reproducible=reproducible,
        debug=args.debug,
        use_cache=args.cache,
    )


"""Main module."""
if __name__ == "__main__":
    main()
