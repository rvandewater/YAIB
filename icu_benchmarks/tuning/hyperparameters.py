import json
import gin
import logging
from logging import NOTSET
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from skopt import gp_minimize
import tempfile
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
from icu_benchmarks.models.utils import JsonResultLoggingEncoder, log_table_row, Align
from icu_benchmarks.cross_validation import execute_repeated_cv
from icu_benchmarks.run_utils import log_full_line
from icu_benchmarks.tuning.gin_utils import get_gin_hyperparameters, bind_gin_params
from icu_benchmarks.contants import RunMode
from icu_benchmarks.wandb_utils import wandb_log
from optuna.visualization import plot_param_importances, plot_optimization_history

TUNE = 25
logging.addLevelName(25, "TUNE")


@gin.configurable("tune_hyperparameters_deprecated")
def choose_and_bind_hyperparameters(
    do_tune: bool,
    data_dir: Path,
    log_dir: Path,
    seed: int,
    run_mode: RunMode = RunMode.classification,
    checkpoint: str = None,
    scopes: list[str] = [],
    n_initial_points: int = 3,
    n_calls: int = 20,
    folds_to_tune_on: int = None,
    checkpoint_file: str = "hyperparameter_tuning_logs.json",
    generate_cache: bool = False,
    load_cache: bool = False,
    debug: bool = False,
    verbose: bool = False,
    wandb: bool = False,
):
    """Choose hyperparameters to tune and bind them to gin.

    Args:
        wandb: Whether we use wandb or not.
        load_cache: Load cached data if available.
        generate_cache: Generate cache data.
        do_tune: Whether to tune hyperparameters or not.
        data_dir: Path to the data directory.
        log_dir: Path to the log directory.
        seed: Random seed.
        run_mode: The run mode of the experiment.
        checkpoint: Name of the checkpoint run to load previously explored hyperparameters from.
        scopes: List of gin scopes to search for hyperparameters to tune.
        n_initial_points: Number of initial points to explore.
        n_calls: Number of iterations to optimize the hyperparameters.
        folds_to_tune_on: Number of folds to tune on.
        checkpoint_file: Name of the checkpoint file.
        debug: Whether to load less data.
        verbose: Set to true to increase log output.

    Raises:
        ValueError: If checkpoint is not None and the checkpoint does not exist.
    """
    hyperparams = {}

    if len(scopes) == 0 or folds_to_tune_on is None:
        logging.warning("No scopes and/or folds to tune on, skipping tuning.")
        return

    # Collect hyperparameters.
    hyperparams_bounds, hyperparams_names = collect_bound_hyperparameters(hyperparams, scopes)

    if do_tune and not hyperparams_bounds:
        logging.info("No hyperparameters to tune, skipping tuning.")
        return

    # Attempt checkpoint loading
    configuration, evaluation = None, None
    if checkpoint:
        checkpoint_path = checkpoint / checkpoint_file
        if not checkpoint_path.exists():
            logging.warning(f"Hyperparameter checkpoint {checkpoint_path} does not exist.")
            logging.info("Attempting to find latest checkpoint file.")
            checkpoint_path = find_checkpoint(log_dir.parent, checkpoint_file)
        # Check if we found a checkpoint file
        if checkpoint_path:
            n_calls, configuration, evaluation = load_checkpoint(checkpoint_path, n_calls)
            # Check if we surpassed maximum tuning iterations
            if n_calls <= 0:
                logging.log(TUNE, "No more hyperparameter tuning iterations left, skipping tuning.")
                logging.info("Training with these hyperparameters:")
                bind_gin_params(hyperparams_names, configuration[np.argmin(evaluation)])  # bind best hyperparameters
                return
        else:
            logging.warning("No checkpoint file found, starting from scratch.")

    # Function that trains the model with the given hyperparameters.
    def bind_params_and_train(hyperparams):
        with tempfile.TemporaryDirectory(dir=log_dir) as temp_dir:
            bind_gin_params(hyperparams_names, hyperparams)
            if not do_tune:
                return 0
            return execute_repeated_cv(
                data_dir,
                Path(temp_dir),
                seed,
                mode=run_mode,
                cv_repetitions_to_train=1,
                cv_folds_to_train=folds_to_tune_on,
                generate_cache=generate_cache,
                load_cache=load_cache,
                test_on="val",
                debug=debug,
                verbose=verbose,
                wandb=wandb,
            )

    header = ["ITERATION"] + hyperparams_names + ["LOSS AT ITERATION"]

    def tune_step_callback(res):
        with open(log_dir / checkpoint_file, "w") as f:
            data = {
                "x_iters": res.x_iters,
                "func_vals": res.func_vals,
            }
            f.write(json.dumps(data, cls=JsonResultLoggingEncoder))
            table_cells = [len(res.x_iters)] + res.x_iters[-1] + [res.func_vals[-1]]
            highlight = res.x_iters[-1] == res.x  # highlight if best so far
            log_table_row(header, TUNE)
            log_table_row(table_cells, TUNE, align=Align.RIGHT, header=header, highlight=highlight)
            wandb_log({"hp-iteration": len(res.x_iters)})

    if do_tune:
        log_full_line("STARTING TUNING", level=TUNE, char="=")
        logging.log(
            TUNE,
            f"Applying Bayesian Optimization from {n_initial_points} points in {n_calls} "
            f"iterations on {folds_to_tune_on} folds.",
        )
        log_table_row(header, TUNE)
    else:
        logging.log(TUNE, "Hyperparameter tuning disabled")
        if configuration:
            # We have loaded a checkpoint, use the best hyperparameters.
            logging.info("Training with the best hyperparameters from loaded checkpoint:")
            bind_gin_params(hyperparams_names, configuration[np.argmin(evaluation)])
        else:
            logging.log(TUNE, "Choosing hyperparameters randomly from bounds.")
            n_initial_points = 1
            n_calls = 1

    # Call gaussian process. To choose a random set of hyperparameters this functions is also called.
    res = gp_minimize(
        bind_params_and_train,
        hyperparams_bounds,
        x0=configuration,
        y0=evaluation,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=seed,
        noise=1e-10,  # The models are deterministic, but noise is needed for the gp to work.
        callback=tune_step_callback if do_tune else None,
    )
    logging.disable(level=NOTSET)

    if do_tune:
        log_full_line("FINISHED TUNING", level=TUNE, char="=", num_newlines=4)

    logging.info("Training with these hyperparameters:")
    bind_gin_params(hyperparams_names, res.x)


@gin.configurable("tune_hyperparameters")
def choose_and_bind_hyperparameters_optuna(
    do_tune: bool,
    data_dir: Path,
    log_dir: Path,
    seed: int,
    run_mode: RunMode = RunMode.classification,
    checkpoint: str = None,
    scopes: list[str] = [],
    n_initial_points: int = 3,
    n_calls: int = 20,
    sampler=optuna.samplers.GPSampler,
    folds_to_tune_on: int = None,
    checkpoint_file: str = "hyperparameter_tuning_logs.db",
    generate_cache: bool = False,
    load_cache: bool = False,
    debug: bool = False,
    verbose: bool = False,
    wandb: bool = False,
    plot: bool = True,
):
    """Choose hyperparameters to tune and bind them to gin. Uses Optuna for hyperparameter optimization.

    Args:
        sampler:
        wandb: Whether we use wandb or not.
        load_cache: Load cached data if available.
        generate_cache: Generate cache data.
        do_tune: Whether to tune hyperparameters or not.
        data_dir: Path to the data directory.
        log_dir: Path to the log directory.
        seed: Random seed.
        run_mode: The run mode of the experiment.
        checkpoint: Name of the checkpoint run to load previously explored hyperparameters from.
        scopes: List of gin scopes to search for hyperparameters to tune.
        n_initial_points: Number of initial points to explore.
        n_calls: Number of iterations to optimize the hyperparameters.
        folds_to_tune_on: Number of folds to tune on.
        checkpoint_file: Name of the checkpoint file.
        debug: Whether to load less data.
        verbose: Set to true to increase log output.

    Raises:
        ValueError: If checkpoint is not None and the checkpoint does not exist.
    """
    hyperparams = {}

    if len(scopes) == 0 or folds_to_tune_on is None:
        logging.warning("No scopes and/or folds to tune on, skipping tuning.")
        return

    # Collect hyperparameters.
    hyperparams_bounds, hyperparams_names = collect_bound_hyperparameters(hyperparams, scopes)

    if do_tune and not hyperparams_bounds:
        logging.info("No hyperparameters to tune, skipping tuning.")
        return

    # Function that trains the model with the given hyperparameters.

    header = ["ITERATION"] + hyperparams_names + ["LOSS AT ITERATION"]

    # Optuna objective function
    def objective(trail, hyperparams_bounds, hyperparams_names):
        # Optuna objective function
        hyperparams = {}
        logging.info(f"Bounds: {hyperparams_bounds}, Names: {hyperparams_names}")
        for name, value in zip(hyperparams_names, hyperparams_bounds):
            if isinstance(value, tuple):
                # Check for range or "list-type" hyperparameter bounds
                if isinstance(value[0], (int, float)) and isinstance(value[1], (int, float)):
                    if len(value) == 3 and isinstance(value[2], str):
                        if isinstance(value[0], int) and isinstance(value[1], int):
                            hyperparams[name] = trail.suggest_int(name, value[0], value[1], log=value[2] == "log")
                        elif isinstance(value[0], (int, float)) and isinstance(value[1], (int, float)):
                            hyperparams[name] = trail.suggest_float(name, value[0], value[1], log=value[2] == "log")
                        else:
                            hyperparams[name] = trail.suggest_categorical(name, value)
                    elif len(value) == 2:
                        if isinstance(value[0], int) and isinstance(value[1], int):
                            hyperparams[name] = trail.suggest_int(name, value[0], value[1])
                        elif isinstance(value[0], (int, float)) and isinstance(value[1], (int, float)):
                            hyperparams[name] = trail.suggest_float(name, value[0], value[1])
                        else:
                            hyperparams[name] = trail.suggest_categorical(name, value)
                    else:
                        hyperparams[name] = trail.suggest_categorical(name, value)
            else:
                hyperparams[name] = trail.suggest_categorical(name, value)
        return bind_params_and_train(hyperparams)

    def tune_step_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        table_cells = [str(len(study.trials)), *list(study.trials[-1].params.values()), study.trials[-1].value]
        highlight = study.trials[-1] == study.best_trial  # highlight if best so far
        log_table_row(header, TUNE)
        log_table_row(table_cells, TUNE, align=Align.RIGHT, header=header, highlight=highlight)
        wandb_log({"HP-optimization-iteration": len(study.trials)})

    if do_tune:
        log_full_line("STARTING TUNING", level=TUNE, char="=")
        logging.log(
            TUNE,
            f"Applying {sampler} from {n_initial_points} points in {n_calls} " f"iterations on {folds_to_tune_on} folds.",
        )
        log_table_row(header, TUNE)
    else:
        logging.log(TUNE, "Hyperparameter tuning disabled")
        if checkpoint:
            study = optuna.load_study(study_name="tuning", storage="sqlite:///" + str(checkpoint))
            configuration = study.best_params
            # We have loaded a checkpoint, use the best hyperparameters.
            logging.info("Training with the best hyperparameters from loaded checkpoint:")
            bind_gin_params(configuration)
            return
        else:
            logging.log(
                TUNE, "Choosing hyperparameters randomly from bounds using hp tuning as no earlier checkpoint " "supplied."
            )
            n_initial_points = 1
            n_calls = 1

    def bind_params_and_train(hyperparams):
        with tempfile.TemporaryDirectory(dir=log_dir) as temp_dir:
            bind_gin_params(hyperparams)
            if not do_tune:
                return 0
            score = execute_repeated_cv(
                data_dir,
                Path(temp_dir),
                seed,
                mode=run_mode,
                cv_repetitions_to_train=1,
                cv_folds_to_train=folds_to_tune_on,
                generate_cache=generate_cache,
                load_cache=load_cache,
                test_on="val",
                debug=debug,
                verbose=verbose,
                wandb=wandb,
            )
            logging.info(f"Score: {score}")
            return score

    if isinstance(sampler, optuna.samplers.GPSampler):
        sampler = sampler(seed=seed, n_startup_trials=n_initial_points, deterministic_objective=True)
    else:
        sampler = sampler(seed=seed)
    pruner = optuna.pruners.HyperbandPruner()
    # Optuna study
    # Attempt checkpoint loading
    if checkpoint and checkpoint.exists():
        logging.warning(f"Hyperparameter checkpoint {checkpoint} does not exist.")
        # logging.info("Attempting to find latest checkpoint file.")
        # checkpoint_path = find_checkpoint(log_dir.parent, checkpoint_file)
        # Check if we found a checkpoint file
        logging.info(f"Loading checkpoint at {checkpoint}")
        study = optuna.load_study(study_name="tuning", storage="sqlite:///" + str(checkpoint), sampler=sampler, pruner=pruner)
        n_calls = n_calls - len(study.trials)
    else:
        if checkpoint:
            logging.warning("Checkpoint path given as flag but not found, starting from scratch.")
        study = optuna.create_study(
            sampler=sampler,
            storage="sqlite:///" + str(log_dir / checkpoint_file),
            study_name="tuning",
            pruner=pruner,
            load_if_exists=True,
        )

    callbacks = [tune_step_callback]
    if wandb:
        wandb_kwargs = {
            "config": {"sampler": sampler},
            "allow_val_change": True,
        }
        wandbc = WeightsAndBiasesCallback(metric_name="loss", wandb_kwargs=wandb_kwargs)
        callbacks.append(wandbc)

    logging.info(f"Starting or resuming Optuna study with {n_calls} trails and callbacks: {callbacks}.")
    if n_calls > 0:
        study.optimize(
            lambda trail: objective(trail, hyperparams_bounds, hyperparams_names),
            n_trials=n_calls,
            callbacks=callbacks,
            gc_after_trial=True,
        )
    else:
        logging.info("No more hyperparameter tuning iterations left, skipping tuning.")
        logging.info("Training with these hyperparameters:")
        bind_gin_params(study.best_params)
        return
    logging.disable(level=NOTSET)

    if do_tune:
        log_full_line("FINISHED TUNING", level=TUNE, char="=", num_newlines=4)

    logging.info("Training with these hyperparameters:")
    bind_gin_params(study.best_params)

    if plot:
        try:
            logging.info("Plotting hyperparameter importances.")
            plot_param_importances(study)
            plt.savefig(log_dir / "param_importances.png")
            plot_optimization_history(study)
            plt.savefig(log_dir / "optimization_history.png")
        except Exception as e:
            logging.error(f"Failed to plot hyperparameter importances: {e}")


def collect_bound_hyperparameters(hyperparams, scopes):
    for scope in scopes:
        with gin.config_scope(scope):
            hyperparams.update(get_gin_hyperparameters())
    hyperparams_names = list(hyperparams.keys())
    hyperparams_bounds = list(hyperparams.values())
    return hyperparams_bounds, hyperparams_names


def load_checkpoint(checkpoint_path, n_calls):
    logging.info(f"Loading checkpoint at {checkpoint_path}")
    with open(checkpoint_path, "r") as f:
        data = json.loads(f.read())
        x0 = data["x_iters"]
        y0 = data["func_vals"]
    n_calls -= len(x0)
    logging.log(TUNE, f"Checkpoint contains {len(x0)} points.")
    return n_calls, x0, y0


def find_checkpoint(log_dir, checkpoint_file):
    """Find the latest checkpoint in the log directory."""
    hyperparameters = sorted(log_dir.glob(f"*/{checkpoint_file}"), reverse=True)
    if not hyperparameters:
        return None
    return hyperparameters[0]
