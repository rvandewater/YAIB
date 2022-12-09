import json
import gin
import logging
from logging import INFO, NOTSET
import numpy as np
from pathlib import Path
from skopt import gp_minimize
import tempfile

from icu_benchmarks.run_utils import preprocess_and_train_for_folds

TUNE = 25


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


@gin.configurable("hyperparameter")
def hyperparameters_to_tune(class_to_tune=gin.REQUIRED, **hyperparams):
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


@gin.configurable("tune_hyperparameters")
def choose_and_bind_hyperparameters(
    do_tune,
    data_dir,
    log_dir,
    seed,
    restart_from_checkpoint=None,
    scopes=gin.REQUIRED,
    n_initial_points=3,
    n_calls=20,
    folds_to_tune_on=gin.REQUIRED,
):
    hyperparams = {}
    for scope in scopes:
        with gin.config_scope(scope):
            hyperparams.update(hyperparameters_to_tune())

    hyperparams_names = list(hyperparams.keys())
    hyperparams_bounds = list(hyperparams.values())

    def bind_params(hyperparams_values):
        for param, value in zip(hyperparams_names, hyperparams_values):
            gin.bind_parameter(param, value)
            logging.info(f"{param}: {value}")

    if do_tune:
        logging.info(
            f"Tuning hyperparameters from {n_initial_points} points in {n_calls} iterations on {folds_to_tune_on} folds."
        )
    else:
        logging.info("Hyperparameter tuning disabled, choosing randomly from bounds.")
        n_initial_points = 1
        n_calls = 1

    with tempfile.TemporaryDirectory() as temp_dir:

        def bind_params_and_train(hyperparams):
            bind_params(hyperparams)
            if not do_tune:
                return 0
            return preprocess_and_train_for_folds(
                data_dir,
                Path(temp_dir),
                seed,
                num_folds_to_train=folds_to_tune_on,
                use_cache=True,
                test_on="val",
            )

    def tune_step_callback(res):
        logging.info(f"Best hyperparameters so far: {res.x}")
        with open(log_dir / checkpoint_file, "w") as f:
            data = {
                "x_iters": res.x_iters,
                "func_vals": res.func_vals,
            }
            logging.log(TUNE, f"{res.x_iters[-1]} yielded a loss of {res.func_vals[-1]}")
            logging.log(TUNE, f"Best hyperparameters so far: {res.x}")
            f.write(json.dumps(data, cls=NpEncoder))

    x0, y0 = None, None
    checkpoint_file = "hyperparameter_tuning_logs.json"
    if restart_from_checkpoint:
        checkpoint = restart_from_checkpoint / checkpoint_file
        if not checkpoint.exists():
            raise ValueError(f"No checkpoint found in {checkpoint} to restart from.")
        with open(checkpoint, "r") as f:
            data = json.loads(f.read())
            x0 = data["x_iters"]
            y0 = data["func_vals"]
        n_calls -= len(x0)
        logging.info(f"Restarting hyperparameter tuning from {len(x0)} points.")

    logging.disable(level=INFO)
    res = gp_minimize(
        bind_params_and_train,
        hyperparams_bounds,
        x0=x0,
        y0=y0,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=seed,
        callback=tune_step_callback if do_tune else None,
    )

    logging.disable(level=NOTSET)
    logging.info("Training with these hyperparameters:")
    bind_params(res.x)
