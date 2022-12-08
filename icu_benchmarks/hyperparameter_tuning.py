from bayes_opt import BayesianOptimization
import gin
import logging
from logging import INFO, NOTSET

import icu_benchmarks.run


@gin.configurable
def hyperparameter(class_to_tune=gin.REQUIRED, **kwargs):
    return {f"{class_to_tune.__name__}.{param}": value for param, value in kwargs.items()}


@gin.configurable("tune_hyperparameters")
def choose_and_bind_hyperparameters(
    do_tune,
    data_dir,
    log_dir,
    seed,
    scopes=gin.REQUIRED,
    init_points=3,
    n_iter=20,
    folds_to_tune_on=gin.REQUIRED,
    cast_to_int=None,
):
    def bind_params_from_dict(params_dict):
        for param, value in params_dict.items():
            value = int(value) if param in cast_to_int else value
            gin.bind_parameter(param, value)
            logging.info(f"{param}: {value}")

    hyperparams_dir = log_dir / "hyperparameter_tuning"
    def bind_params_and_train(**hyperparams):
        bind_params_from_dict(hyperparams)
        if not do_tune:
            return 0
        # return negative loss because BO maximizes
        return - icu_benchmarks.run.preprocess_and_train_for_folds(
            data_dir, hyperparams_dir, seed, num_folds_to_train=folds_to_tune_on, use_cache=True, test_on="val"
        )

    hyperparams = {}
    for scope in scopes:
        with gin.config_scope(scope):
            hyperparams.update(hyperparameter())
    
    if do_tune:
        logging.info(f"Tuning hyperparameters from {init_points} points in {n_iter} iterations on {folds_to_tune_on} folds.")
    else:
        logging.info("Hyperparameter tuning disabled, choosing randomly from bounds.")
        init_points = 1
        n_iter = 0

    bo = BayesianOptimization(bind_params_and_train, hyperparams)
    bo.set_gp_params(alpha=1e-3)
    logging.disable(level=INFO)
    bo.maximize(init_points=init_points, n_iter=n_iter)
    logging.disable(level=NOTSET)
    logging.info("Training with these hyperparameters:")
    bind_params_from_dict(bo.max["params"])
