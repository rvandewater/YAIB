import numpy as np
import re

from ast import literal_eval


def rs_from_string(string):
    values = literal_eval(string[3:-1])
    if not isinstance(values, list):
        raise ValueError("Wrong parameter for random search, expects list.")
    return str(values[np.random.randint(len(values))])


def parse_config_lines(config_lines):
    parsed_lines = []
    for line in config_lines:
        if "RS([" not in line:
            parsed_lines += [line.rstrip()]
            continue
        name, params = (sub.strip() for sub in line.split("="))
        param = rs_from_string(params) if params.startswith("RS") else literal_eval(params)
        parsed_lines += [f"{name} = {param}"]
    return parsed_lines


def rs_gin_config(gin_file):
    with open(gin_file, encoding="utf-8") as f:
        return ("\n").join(parse_config_lines(f))
