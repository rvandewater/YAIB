from cmath import log
import logging
import os
import gin
import numpy as np
import torch


def save_model(model, optimizer, epoch, save_file):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(state, save_file)
    del state


def load_model_state(filepath, model, optimizer=None):
    state = torch.load(filepath)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    logging.info("Loaded model and optimizer")


def save_config_file(log_dir):
    with open(os.path.join(log_dir, "train_config.gin"), "w") as f:
        f.write(gin.operative_config_str())


@gin.configurable("random_search")
def get_bindings_w_rs(cli_params, args, log_dir, do_rs_for_conf=True, **rs_params_from_config):
    # only handle cli params that are set (exist in args and aren't None)
    cli_params = {param: args[param] for param in cli_params if getattr(args, param, None) is not None}
    # merge params for random search from config with cli params
    merged_params = rs_params_from_config | cli_params if do_rs_for_conf else cli_params
    gin_bindings = []
    for name, params in merged_params.items():
        # randomly choose one param from list
        param = params[np.random.randint(len(params))]
        gin_bindings += [f"{name.upper()} = {param}"]
        log_dir += f"/{name}_{param}"

        if name == "depth":
            num_leaves = 2**param
            gin_bindings += [f"NUM_LEAVES = {num_leaves}"]

    print(gin_bindings)
    print(log_dir)
    return gin_bindings, log_dir
