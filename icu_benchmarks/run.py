# -*- coding: utf-8 -*-
from datetime import datetime
import gin
import logging
import sys
from pathlib import Path
import torch.cuda
from icu_benchmarks.wandb_utils import (
    update_wandb_config,
    apply_wandb_sweep,
    set_wandb_experiment_name,
)
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
    import_preprocessor,
    name_datasets,
)
from icu_benchmarks.contants import RunMode


@gin.configurable("Run")
def get_mode(mode: gin.REQUIRED):
    # Check if enum is mode.
    assert RunMode(mode)
    return RunMode(mode)


def main(my_args=tuple(sys.argv[1:])):
    args, _ = build_parser().parse_known_args(my_args)
    if args.wandb_sweep:
        args = apply_wandb_sweep(args)
        set_wandb_experiment_name(args, "run")
    # Initialize loggers
    log_format = "%(asctime)s - %(levelname)s - %(name)s : %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    verbose = args.verbose
    setup_logging(date_format, log_format, verbose)
    # Get arguments
    data_dir = Path(args.data_dir)
    name = args.name
    task = args.task
    model = args.model
    reproducible = args.reproducible
    evaluate = args.eval
    experiment = args.experiment
    source_dir = args.source_dir
    # Load task config
    gin.parse_config_file(f"configs/tasks/{task}.gin")
    mode = get_mode()
    # Set arguments for wandb sweep

    # Set experiment name
    if name is None:
        name = data_dir.name
    logging.info(f"Running experiment {name}.")
    logging.info(f"Task mode: {mode}.")

    # Set train size to fine tune size if fine tune is set, else use custom train size
    train_size = args.fine_tune if args.fine_tune is not None else args.samples if args.samples is not None else None
    # Whether to load weights from a previous run
    load_weights = evaluate or args.fine_tune is not None

    pretrained_imputation_model = load_pretrained_imputation_model(args.pretrained_imputation)
    # Log imputation model to wandb
    update_wandb_config(
        {
            "pretrained_imputation_model": pretrained_imputation_model.__class__.__name__
            if pretrained_imputation_model is not None
            else "None"
        }
    )

    log_dir_name = args.log_dir / name
    log_dir = (
        (log_dir_name / experiment)
        if experiment
        else (log_dir_name / (args.task_name if args.task_name is not None else args.task) / model)
    )
    log_full_line(f"Logging to {log_dir}.", logging.INFO)

    # Check cuda availability
    if torch.cuda.is_available():
        for name in range(0, torch.cuda.device_count()):
            log_full_line(
                f"Available GPU {name}: {torch.cuda.get_device_name(name)}",
                level=logging.INFO,
            )
    else:
        log_full_line(
            "No GPUs available: please check your device and Torch,Cuda installation if unintended.",
            level=logging.WARNING,
        )

    if args.preprocessor:
        import_preprocessor(args.preprocessor)

    # Load pretrained model in evaluate mode or when finetuning
    if load_weights:
        if args.source_dir is None:
            raise ValueError("Please specify a source directory when evaluating or fine-tuning.")
        log_dir /= f"_from_{args.source_name}"
        name_datasets(args.source_name, args.source_name, args.name)
        if args.fine_tune:
            log_dir /= f"fine_tune_{args.fine_tune}"
            name_datasets(args.name, args.name, args.name)
        run_dir = create_run_dir(log_dir)
        source_dir = args.source_dir
        logging.info(f"Will load weights from {source_dir} and bind train gin-config. Note: this might override your config.")
        gin.parse_config_file(source_dir / "train_config.gin")
    elif args.samples and args.source_dir is not None:  # Train model with limited samples and bind existing config
        logging.info("Binding train gin-config. Note: this might override your config.")
        gin.parse_config_file(args.source_dir / "train_config.gin")
        log_dir /= f"samples_{args.fine_tune}"
        name_datasets(args.name, args.name, args.name)
        run_dir = create_run_dir(log_dir)
    else:
        # Normal train and evaluate
        name_datasets(args.name, args.name, args.name)
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
        complete_train=args.complete_train,
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
