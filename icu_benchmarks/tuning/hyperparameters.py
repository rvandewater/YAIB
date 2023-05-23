import json
import gin
import logging
from logging import NOTSET
import numpy as np
from pathlib import Path
from skopt import gp_minimize
import tempfile

from icu_benchmarks.models.utils import JsonResultLoggingEncoder, log_table_row, Align
from icu_benchmarks.cross_validation import execute_repeated_cv
from icu_benchmarks.run_utils import log_full_line
from icu_benchmarks.tuning.gin_utils import get_gin_hyperparameters, bind_gin_params
from icu_benchmarks.contants import RunMode
from icu_benchmarks.wandb_utils import wandb_log
TUNE = 25
logging.addLevelName(25, "TUNE")


@gin.configurable("tune_hyperparameters")
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
    wandb: bool = False
):
    """Choose hyperparameters to tune and bind them to gin.

    Args:
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

    # Function to
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
                wandb=wandb
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
            log_table_row(table_cells, TUNE, align=Align.RIGHT, header=header, highlight=highlight)
            wandb_log({"hp-iteration" : len(res.x_iters)})

    if do_tune:
        log_full_line("STARTING TUNING", level=TUNE, char="=")
        logging.log(TUNE, f"Applying Bayesian Optimization from {n_initial_points} points in {n_calls} "
                          f"iterations on {folds_to_tune_on} folds.")
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
