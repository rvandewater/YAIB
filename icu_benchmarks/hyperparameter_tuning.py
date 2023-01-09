import json
import gin
import logging
from logging import INFO, NOTSET
import numpy as np
from pathlib import Path
from skopt import gp_minimize
import tempfile

from icu_benchmarks.models.utils import JsonNumpyEncoder, log_table_row, Align
from icu_benchmarks.run_utils import log_full_line, execute_repeated_cv

TUNE = 25
logging.addLevelName(25, "TUNE")


@gin.configurable("hyperparameter")
def hyperparameters_to_tune(class_to_tune: str = gin.REQUIRED, **hyperparams: dict) -> dict:
    """Get hyperparameters to tune from gin config.

    Hyperparameters that are already present in the gin config are ignored.
    Hyperparameters that are not a list or tuple are bound directly to the class.
    Hyperparameters that are a list or tuple are returned to be tuned.

    Args:
        class_to_tune: Name of the class to tune hyperparameters for.
        **hyperparams: Dictionary of hyperparameters to potentially tune.

    Returns:
        Dictionary of hyperparameters to tune.
    """
    hyperparams_to_tune = {}
    for param, values in hyperparams.items():
        name = f"{class_to_tune.__name__}.{param}"
        if f"{name}=" in gin.config_str().replace(" ", ""):
            # check if parameter is already bound, directly binding to class always takes precedence
            continue
        if not isinstance(values, (list, tuple)):
            # if parameter is not a tuple, bind it directly
            gin.bind_parameter(name, values)
            continue
        hyperparams_to_tune[name] = values
    return hyperparams_to_tune


def bind_params(hyperparams_names: list[str], hyperparams_values: list):
    """Binds hyperparameters to gin config and logs them.

    Args:
        hyperparams_names: List of hyperparameter names.
        hyperparams_values: List of hyperparameter values.
    """
    for param, value in zip(hyperparams_names, hyperparams_values):
        gin.bind_parameter(param, value)
        logging.info(f"{param} = {value}")


@gin.configurable("tune_hyperparameters")
def choose_and_bind_hyperparameters(
    do_tune: bool,
    data_dir: Path,
    log_dir: Path,
    seed: int,
    checkpoint: str = None,
    scopes: list[str] = gin.REQUIRED,
    n_initial_points: int = 3,
    n_calls: int = 20,
    folds_to_tune_on: int = gin.REQUIRED,
    debug: bool = False,
):
    """Choose hyperparameters to tune and bind them to gin.

    Args:
        do_tune: Whether to tune hyperparameters or not.
        data_dir: Path to the data directory.
        log_dir: Path to the log directory.
        seed: Random seed.
        checkpoint: Name of the checkpoint run to load previously explored hyperparameters from.
        scopes: List of gin scopes to search for hyperparameters to tune.
        n_initial_points: Number of initial points to explore.
        n_calls: Number of iterations to optimize the hyperparameters.
        folds_to_tune_on: Number of folds to tune on.
        debug: Whether to load less data and enable more logging.

    Raises:
        ValueError: If checkpoint is not None and the checkpoint does not exist.
    """
    hyperparams = {}
    # Collect hyperparameters.
    for scope in scopes:
        with gin.config_scope(scope):
            hyperparams.update(hyperparameters_to_tune())
    hyperparams_names = list(hyperparams.keys())
    hyperparams_bounds = list(hyperparams.values())

    if do_tune and not hyperparams_bounds:
        logging.info("No hyperparameters to tune, skipping tuning.")
        return

    x0, y0 = None, None
    checkpoint_file = "hyperparameter_tuning_logs.json"
    if checkpoint:
        checkpoint_path = checkpoint / checkpoint_file
        if not checkpoint_path.exists():
            raise ValueError(f"No checkpoint found in {checkpoint_path} to restart from.")
        with open(checkpoint_path, "r") as f:
            data = json.loads(f.read())
            x0 = data["x_iters"]
            y0 = data["func_vals"]
        n_calls -= len(x0)
        logging.log(TUNE, f"Restarting hyperparameter tuning from {len(x0)} points.")
        if n_calls <= 0:
            logging.log(TUNE, "No more hyperparameter tuning iterations left, skipping tuning.")
            logging.info("Training with these hyperparameters:")
            bind_params(hyperparams_names, x0[np.argmin(y0)])  # bind best hyperparameters
            return

    with tempfile.TemporaryDirectory() as temp_dir:

        def bind_params_and_train(hyperparams):
            bind_params(hyperparams_names, hyperparams)
            if not do_tune:
                return 0
            return execute_repeated_cv(
                data_dir,
                Path(temp_dir),
                seed,
                cv_repetitions_to_train=1,
                cv_folds_to_train=folds_to_tune_on,
                use_cache=True,
                test_on="val",
                debug=debug,
            )

    header = ["ITERATION"] + hyperparams_names + ["LOSS AT ITERATION"]

    def tune_step_callback(res):
        with open(log_dir / checkpoint_file, "w") as f:
            data = {
                "x_iters": res.x_iters,
                "func_vals": res.func_vals,
            }
            f.write(json.dumps(data, cls=JsonNumpyEncoder))
            if do_tune:
                table_cells = [len(res.x_iters)] + res.x_iters[-1] + [res.func_vals[-1]]
                highlight = res.x_iters[-1] == res.x  # highlight if best so far
                log_table_row(table_cells, TUNE, align=Align.RIGHT, header=header, highlight=highlight)

    if do_tune:
        log_full_line("STARTING TUNING", level=TUNE, char="=")
        logging.log(TUNE, f"Tuning from {n_initial_points} points in {n_calls} iterations on {folds_to_tune_on} folds.")
        log_table_row(header, TUNE)
    else:
        logging.log(TUNE, "Hyperparameter tuning disabled, choosing randomly from bounds.")
        n_initial_points = 1
        n_calls = 1
    if not debug:
        logging.disable(level=INFO)

    res = gp_minimize(
        bind_params_and_train,
        hyperparams_bounds,
        x0=x0,
        y0=y0,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=seed,
        noise=1e-10,  # the models are deterministic, but noise is needed for the gp to work
        callback=tune_step_callback if do_tune else None,
    )  # to choose a random set of hyperparameters this functions is also called when tuning is disabled
    logging.disable(level=NOTSET)

    if do_tune:
        log_full_line("FINISHED TUNING", level=TUNE, char="=", num_newlines=4)

    logging.info("Training with these hyperparameters:")
    bind_params(hyperparams_names, res.x)
