import numpy as np
from ast import literal_eval
from pathlib import Path

MAX_ATTEMTPS = 1000


def random_search_config_lines(config_lines: list[str]) -> tuple[list[str], str]:
    """Parses all configuration lines and randomly searches where applicable.

    Args:
        config_lines: A list of all configuration lines, from gin files and the command line.

    Raises:
        ValueError: If the argument to RS() can't be parsed to a list by literal_eval().

    Returns:
        A list containing all parsed gin config lines.
        A string representing the randomly searched hyperparameters.
    """
    parsed_lines = []
    randomly_searched_params = {}
    for line in config_lines:
        try:
            name, value_string = (sub.strip() for sub in line.split("="))
        except:  # line is empty or contains import, include etc.
            parsed_lines += [line.rstrip()]
            continue

        if value_string.startswith("{") or value_string.startswith("@") or value_string.startswith("%"):
            # variable is set to dict, reference or macro, don't do anything
            parsed_lines += [line.rstrip()]
            continue
        elif "RS([" in value_string:
            values = literal_eval(value_string[3:-1])  # value_string should be RS([...]), only evaluate list in RS()
            if not isinstance(values, list):
                raise ValueError("Wrong parameter for random search, expects list.")
            param = values[np.random.randint(len(values))]  # do random search in possible values
            randomly_searched_params[name] = param
        else:  # line doesn't contain parameter to randomly search
            if name in randomly_searched_params:
                del randomly_searched_params[name]  # parameter was randomly searched before, but is now set explicitly
            try:
                param = literal_eval(value_string)
            except:
                param = value_string

        # repr adds quotes for strings
        parsed_lines += [f"{name} = {repr(param) if type(param) == str else param}"]

    randomly_searched_params = [f"{name.split('.')[-1]}_{param}" for name, param in randomly_searched_params.items()]
    return parsed_lines, ("-").join(randomly_searched_params)


def random_search_configs(gin_files: list[Path], hyperparams_from_cli: list[str], log_dir: Path) -> tuple[list[str], Path]:
    """Does random search in gin configs and hyperparameters from the command line.

    Parses all gin config files and hyperparameters from the command line and randomly searches them.
    Tries to find an unexplored set of hyperparameters from previous runs a maximum of MAX_ATTEMPTS by comparing filenames.

    Args:
        gin_files: A list of all configuration files to possibly do random search in.
        hyperparams_from_cli: A list of all hyperparameters from the command line to possibly do random search in.
        log_dir: Directory in which the runs are logged.

    Raises:
        RuntimeError: If no unexplored set of hyperparameters was found in MAX_ATTEMPTS.

    Returns:
        A list containing all parsed gin config lines.
        A string representing the randomly searched hyperparameters.
    """
    configs_to_read = []
    for file in gin_files:
        with open(file, encoding="utf-8") as f:
            configs_to_read += f.read().splitlines()
    if hyperparams_from_cli:
        configs_to_read += hyperparams_from_cli

    for _ in range(MAX_ATTEMTPS):
        parsed_lines, randomly_searched_params = random_search_config_lines(configs_to_read)
        if not randomly_searched_params or not log_dir.exists():
            hyperparams_already_tried = False  # no hyperparams to randomly search or no previous runs, so proceed
            break
        # look through all previous runs to see if this set of hyperparameters exists already
        hyperparams_already_tried = any([(run / randomly_searched_params).exists() for run in log_dir.iterdir()])
        if not hyperparams_already_tried:
            break
    if hyperparams_already_tried:
        raise RuntimeError(f"Could not find unexplored set of hyperparameters in {MAX_ATTEMTPS} attempts.")

    return parsed_lines, randomly_searched_params
