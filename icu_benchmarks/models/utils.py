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


def get_bindings_and_params(args, log_dir_base):  # noqa: C901
    gin_bindings = []
    log_dir = log_dir_base
    if args.num_class:
        num_class = args.num_class
        gin_bindings += ["NUM_CLASSES = " + str(num_class)]

    # if args.res:
    #     res = args.res
    #     gin_bindings += ['RES = ' + str(res)]
    #     log_dir = os.path.join(log_dir, 'data-res_' + str(res))

    # if args.res_lab:
    #     res_lab = args.res_lab
    #     gin_bindings += ['RES_LAB = ' + str(res_lab)]
    #     log_dir = os.path.join(log_dir, 'pre-res_' + str(res_lab))

    params = [
        ("horizon", args.horizon),
        ("l1_reg", args.regularization),
        ("batch_size", args.batch_size),
        ("learning_rate", args.lr),
        ("embeddings", args.emb),
        ("drop_out", args.do),
        ("drop_out_att", args.do_att),
        ("kernel", args.kernel),
        ("depth", args.depth),
        ("heads", args.heads),
        ("latent", args.latent),
        ("hidden", args.hidden),
        ("subsample_data", args.subsample_data),
        ("subsample_feat", args.subsample_feat),
        ("c_parameter", args.c_parameter),
        ("penalty", args.penalty),
        ("loss_weight", args.loss_weight),
    ]

    for name, cli_param in params:
        if cli_param:
            if args.rs:
                param = cli_param[np.random.randint(len(cli_param))]
            else:
                param = cli_param[0]
            gin_bindings += [f"{name.upper()} = {str(param)}"]
            log_dir = f"{log_dir}/{name}_{str(param)}"

            if cli_param is args.depth:
                num_leaves = 2**param
                gin_bindings += ["NUM_LEAVES = " + str(num_leaves)]

    return gin_bindings, log_dir
