import numpy as np
from ast import literal_eval


def parse_config_lines(config_lines):
    parsed_lines = []
    for line in config_lines:
        if "RS([" not in line:
            parsed_lines += [line.rstrip()]
            continue
        name, values = (sub.strip() for sub in line.split("="))
        values = literal_eval(values[3:-1])
        if not isinstance(values, list):
            raise ValueError("Wrong parameter for random search, expects list.")
        param = str(values[np.random.randint(len(values))])
        parsed_lines += [f"{name} = {param}"]
    return parsed_lines


def parse_gin_config_files(gin_files):
    parsed_files = []
    for file in gin_files:
        with open(file, encoding="utf-8") as f:
            parsed_files += [("\n").join(parse_config_lines(f))]
    return parsed_files
