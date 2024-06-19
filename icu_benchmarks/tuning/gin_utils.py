import logging
from ..wandb_utils import wandb_log
import gin


@gin.configurable("hyperparameter")
def get_gin_hyperparameters(class_to_tune: str = gin.REQUIRED, **hyperparams: dict) -> dict:
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


# def bind_gin_params(hyperparams_names: list[str], hyperparams_values: list):
#     """Binds hyperparameters to gin config and logs them.
#
#     Args:
#         hyperparams_names: List of hyperparameter names.
#         hyperparams_values: List of hyperparameter values.
#     """
#     logging.info("Binding Hyperparameters:")
#     for param, value in zip(hyperparams_names, hyperparams_values):
#         gin.bind_parameter(param, value)
#         logging.info(f"{param} = {value}")
#         wandb_log({param: value})


def bind_gin_params(hyperparams: dict[str, any]):
    """Binds hyperparameter dict to gin config and logs them.

    Args:
        hyperparams: Dictionary of hyperparameters.
    """
    logging.info("Binding Hyperparameters:")
    for param, value in hyperparams.items():
        gin.bind_parameter(param, value)
        logging.info(f"{param} = {value}")
        wandb_log({param: value})
