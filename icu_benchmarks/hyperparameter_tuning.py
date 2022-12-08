from bayes_opt import BayesianOptimization
import gin
import logging
from logging import INFO, NOTSET
from pathlib import Path
import tempfile

from icu_benchmarks.run_utils import preprocess_and_train_for_folds


@gin.configurable("hyperparameter")
def hyperparameters_to_tune(class_to_tune=gin.REQUIRED, cast_to_int=None, **hyperparams):
    if cast_to_int is None:
        cast_to_int = []
    hyperparams_to_tune = {}
    cast_list = []
    for param, values in hyperparams.items():
        name = f"{class_to_tune.__name__}.{param}"
        if f"{name}=" in gin.config_str().replace(" ", ""):
            # check if parameter is already bound, directly binding to class always takes precedence
            continue
        if not type(values) is tuple:
            # if parameter is not a tuple, bind it directly
            gin.bind_parameter(name, values)
            continue
        hyperparams_to_tune[name] = values
        if param in cast_to_int:
            cast_list += [name]
    return hyperparams_to_tune, cast_list


@gin.configurable("tune_hyperparameters")
def choose_and_bind_hyperparameters(
    do_tune,
    data_dir,
    seed,
    scopes=gin.REQUIRED,
    init_points=3,
    n_iter=20,
    folds_to_tune_on=gin.REQUIRED,
):
    hyperparams = {}
    cast_to_int = []
    for scope in scopes:
        with gin.config_scope(scope):
            hyperparams_to_tune, cast_list = hyperparameters_to_tune()
            hyperparams.update(hyperparams_to_tune)
            cast_to_int += cast_list

    def bind_params_from_dict(params_dict):
        for param, value in params_dict.items():
            value = int(value) if param in cast_to_int else value
            gin.bind_parameter(param, value)
            logging.info(f"{param}: {value}")

    if do_tune:
        logging.info(f"Tuning hyperparameters from {init_points} points in {n_iter} iterations on {folds_to_tune_on} folds.")
    else:
        logging.info("Hyperparameter tuning disabled, choosing randomly from bounds.")
        init_points = 1
        n_iter = 0

    logging.disable(level=INFO)
    with tempfile.TemporaryDirectory() as temp_dir:

        def bind_params_and_train(**hyperparams):
            bind_params_from_dict(hyperparams)
            if not do_tune:
                return 0
            # return negative loss because BO maximizes
            return -preprocess_and_train_for_folds(
                data_dir, Path(temp_dir), seed, num_folds_to_train=folds_to_tune_on, use_cache=True, test_on="val"
            )

        bo = BayesianOptimization(bind_params_and_train, hyperparams, random_state=seed)
        bo.set_gp_params(alpha=1e-3)
        bo.maximize(init_points=init_points, n_iter=n_iter)

    logging.disable(level=NOTSET)
    logging.info("Training with these hyperparameters:")
    bind_params_from_dict(bo.max["params"])
