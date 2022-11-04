import numpy as np
import re

from ast import literal_eval


def to_correct_type(value):
    try:
        val = literal_eval(value)
        return val if isinstance(val, list) else [val]
    except ValueError:
        return value


def rs_from_match(matchobj):
    values = to_correct_type(matchobj.group(0)[3:-1])
    return str(values[np.random.randint(len(values))])


def rs_gin_configs(gin_config_files):
    parsed_configs = []
    for gin_file in gin_config_files:
        with open(gin_file, encoding="utf-8") as f:
            contents = f.read()
            parsed_contents = re.sub(r'RS\((.*)\)', rs_from_match, contents, flags=re.MULTILINE)
            parsed_configs += [parsed_contents]
    return parsed_configs


def get_bindings(hyperparams, log_dir, do_rs=False):
    hyperparams = {param.split('=')[0]: to_correct_type(param.split('=')[1]) for param in hyperparams} if hyperparams else {}
    gin_bindings = []
    for name, params in hyperparams.items():
        # randomly choose one param from list if random search enable, else take first
        param = params[np.random.randint(len(params))] if do_rs else params[0]
        gin_bindings += [f"{name} = {param}"]
        log_dir /= f"{name}_{param}"

        if name == "depth":
            num_leaves = 2**param
            gin_bindings += [f"NUM_LEAVES = {num_leaves}"]

    return gin_bindings, log_dir