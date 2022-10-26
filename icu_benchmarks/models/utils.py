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
def get_bindings_w_rs(args, log_dir, do_rs_for_conf=True, **rs_params_from_config):
    gin_bindings = []
    # if args.num_class:
    #     num_class = args.num_class
    #     gin_bindings += ["NUM_CLASSES = " + str(num_class)]

    # if args.res:
    #     res = args.res
    #     gin_bindings += ['RES = ' + str(res)]
    #     log_dir = os.path.join(log_dir, 'data-res_' + str(res))

    # if args.res_lab:
    #     res_lab = args.res_lab
    #     gin_bindings += ['RES_LAB = ' + str(res_lab)]
    #     log_dir = os.path.join(log_dir, 'pre-res_' + str(res_lab))

    cli_params = {
        "horizon": getattr(args, "horizon", None),
        "l1_reg": getattr(args, "l1_reg", None),
        "batch_size": getattr(args, "batch_size", None),
        "learning_rate": getattr(args, "learning_rate", None),
        "embeddings": getattr(args, "embeddings", None),
        "drop_out": getattr(args, "drop_out", None),
        "drop_out_att": getattr(args, "drop_out_att", None),
        "kernel_size": getattr(args, "kernel_size", None),
        "depth": getattr(args, "depth", None),
        "heads": getattr(args, "heads", None),
        "latent": getattr(args, "latent", None),
        "hidden": getattr(args, "hidden", None),
        "subsample_data": getattr(args, "subsample_data", None),
        "subsample_feat": getattr(args, "subsample_feat", None),
        "c_parameter": getattr(args, "c_parameter", None),
        "penalty": getattr(args, "penalty", None),
        "loss_weight": getattr(args, "loss_weight", None),
    }
    existing_cli_params = {name: value for name, value in cli_params.items() if value is not None}
    merged_params = rs_params_from_config | existing_cli_params if do_rs_for_conf else existing_cli_params
    for name, params in merged_params.items():
        param = params[np.random.randint(len(params))]
        gin_bindings += [f"{name.upper()} = {param}"]
        # log_dir += f"/{name}_{param}"

        if name == "depth":
            num_leaves = 2**param
            gin_bindings += ["NUM_LEAVES = " + str(num_leaves)]

    print(gin_bindings)
    print(log_dir)
    return gin_bindings, log_dir
