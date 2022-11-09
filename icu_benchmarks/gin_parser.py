import numpy as np
from ast import literal_eval
from datetime import datetime
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
        if "=" not in line:  # line is empty or contains import, include etc.
            parsed_lines += [line.rstrip()]
            continue

        name, value_string = (sub.strip() for sub in line.split("="))
        if "RS([" in value_string:
            values = literal_eval(value_string[3:-1])  # value_string should be RS([...]), only evaluate list in RS()
            if not isinstance(values, list):
                raise ValueError("Wrong parameter for random search, expects list.")
            param = str(values[np.random.randint(len(values))])  # do random search in possible values
            randomly_searched_params[name] = param
        else:  # line doesn't contain parameter to randomly search
            if name in randomly_searched_params:
                del randomly_searched_params[name]  # parameter was randomly searched before, but is now set explicitly
            param = value_string

        parsed_lines += [f"{name} = {param}"]

    randomly_searched_params = [f"{name.split('.')[-1]}_{param}" for name, param in randomly_searched_params.items()]
    return parsed_lines, ("-").join(randomly_searched_params)


def random_search_configs_and_create_log_dir(
    gin_files: list[Path], hyperparams_from_cli: list[str], log_dir: Path
) -> tuple[list[str], Path]:
    """Does random search in gin configs and creates log directory.

    Parses all gin config files and hyperparameters from the command line and randomly searches them.
    Tries to find an unexplored set of hyperparameters from previous runs a maximum of MAX_ATTEMPTS by comparing filenames.
    Creates log directory with current timestamp and creates file whose name represents randomly searched hyperparameters.

    Args:
        gin_files: A list of all configuration files to possibly do random search in.
        hyperparams_from_cli: A list of all hyperparameters from the command line to possibly do random search in.
        log_dir: Directory in which the runs are logged.

    Raises:
        RuntimeError: If no unexplored set of hyperparameters was found in MAX_ATTEMPTS.

    Returns:
        A list containing all parsed gin config lines.
        A Path to the newly created log dir of the current run.
    """
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    configs_to_read = []
    for file in gin_files:
        with open(file, encoding="utf-8") as f:
            configs_to_read += f.read().splitlines()
    if hyperparams_from_cli:
        configs_to_read += hyperparams_from_cli

    for _ in range(MAX_ATTEMTPS):
        parsed_lines, randomly_searched_params = random_search_config_lines(configs_to_read)
        if not randomly_searched_params:
            hyperparams_already_tried = False  # no hyperparams to randomly search, so proceed
            break
        # look through all previous runs to see if this set of hyperparameters exists already
        hyperparams_already_tried = any([(run / randomly_searched_params).exists() for run in log_dir.iterdir()])
        if not hyperparams_already_tried:
            break
    if hyperparams_already_tried:
        raise RuntimeError(f"Could not find unexplored set of hyperparameters in {MAX_ATTEMTPS} attempts.")

    log_dir_run = log_dir / str(datetime.now())
    log_dir_run.mkdir()
    if randomly_searched_params:
        (log_dir_run / randomly_searched_params).touch()

    return parsed_lines, log_dir_run
