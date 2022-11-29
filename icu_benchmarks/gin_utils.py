import gin
import numpy as np
from pathlib import Path


@gin.configurable
def random_search(class_to_configure: type = gin.REQUIRED, **kwargs: dict[str, list]) -> list[str]:
    """Randomly searches parameters for a class and sets gin bindings.

    Args:
        class_to_configure: The class that gets configured with the parameters.
        kwargs: A dict containing the name of the parameter and a list of possible values.

    Returns:
        The randomly searched parameters.
    """
    randomly_searched_params = []
    for param, values in kwargs.items():
        param_to_set = f"{class_to_configure.__name__}.{param}"
        if param_to_set in gin.config_str():
            continue  # hyperparameter is already set in the config (e.g. from experiment), so skip random search
        value = values[np.random.randint(len(values))]
        randomly_searched_params += [(param_to_set, value)]
    return randomly_searched_params


@gin.configurable
def run_random_searches(scopes: list[str] = gin.REQUIRED) -> list[str]:
    """Executes random searches for the different scopes defined in gin configs.

    Args:
        scopes: The gin scopes to explicitly set.

    Returns:
        The randomly searched parameters.
    """
    randomly_searched_params = []
    for scope in scopes:
        with gin.config_scope(scope):
            randomly_searched_params += random_search()
    return randomly_searched_params


def parse_gin_and_random_search(gin_config_files: list[Path], log_dir: Path, max_attempts: int = 1000) -> str:
    """Parses and binds gin configs and finds unexplored parameters via random search.

    Tries to find an unexplored set of hyperparameters a maximum of max_attempts by comparing filenames.

    Args:
        gin_config_files: A list of all configuration files to pparse.
        log_dir: Directory in which the runs are logged.
        max_attempts: Maximum number of tries to find unxplored set of parameters.

    Raises:
        RuntimeError: If no unexplored set of hyperparameters was found in max_attempts.

    Returns:
        A string representing the randomly searched hyperparameters.
    """
    gin.parse_config_files_and_bindings(gin_config_files, None, finalize_config=False)
    for _ in range(max_attempts):
        randomly_searched_params = run_random_searches()
        randomly_searched_params_str = ("-").join(
            [f"{param.split('.')[-1]}_{value}" for param, value in randomly_searched_params]
        )
        if not randomly_searched_params or not log_dir.exists():
            hyperparams_already_tried = False  # no hyperparams to randomly search or no previous runs, so proceed
            break
        # look through all previous runs to see if this set of hyperparameters exists already
        hyperparams_already_tried = any([(run / randomly_searched_params_str).exists() for run in log_dir.iterdir()])
        if not hyperparams_already_tried:
            break  # unexplored set of hyperparameters found
    if hyperparams_already_tried:
        raise RuntimeError(f"Could not find unexplored set of hyperparameters in {max_attempts} attempts.")
    for param, value in randomly_searched_params:
        gin.bind_parameter(param, value)
    # parse gin again so overwriting parameters in experiments takes precedence
    gin.parse_config_files_and_bindings(gin_config_files, None)  
    return randomly_searched_params_str
