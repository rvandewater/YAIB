import numpy as np
from ast import literal_eval

MAX_ATTEMTPS = 300


def parse_config_lines(config_lines):
    parsed_lines = []
    randomly_searched_params = {}
    for line in config_lines:
        if "=" not in line:
            parsed_lines += [line.rstrip()]
            continue

        name, value_string = (sub.strip() for sub in line.split("="))
        if "RS([" not in value_string:
            if name in randomly_searched_params:
                del randomly_searched_params[name]
            param = value_string
        else:
            values = literal_eval(value_string[3:-1])
            if not isinstance(values, list):
                raise ValueError("Wrong parameter for random search, expects list.")
            param = str(values[np.random.randint(len(values))])
            randomly_searched_params[name] = param

        parsed_lines += [f"{name} = {param}"]

    randomly_searched_params = [f"{name.split('.')[-1]}_{param}" for name, param in randomly_searched_params.items()]
    return ("\n").join(parsed_lines), ("-").join(randomly_searched_params)


def parse_gin_config_files_and_bindings(gin_files, hyperparams_from_cli, log_dir):
    configs_to_read = []
    for file in gin_files:
        with open(file, encoding="utf-8") as f:
            configs_to_read += f.read().splitlines()
    if hyperparams_from_cli:
        configs_to_read += hyperparams_from_cli

    for _ in range(MAX_ATTEMTPS):
        parsed_lines, randomly_searched_params = parse_config_lines(configs_to_read)
        hyperparams_already_tried = any([(run / randomly_searched_params).exists() for run in log_dir.iterdir()])
        if not hyperparams_already_tried:
            break
    if hyperparams_already_tried:
        raise Exception(f"Could not find unexplored set of hyperparameters in {MAX_ATTEMTPS} attempts.")

    return [parsed_lines], randomly_searched_params
