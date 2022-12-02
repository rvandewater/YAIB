import logging
import gin
import torch
import os
import json

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
    config_path = log_dir / "train_config.gin"
    with config_path.open("w") as f:
        f.write(gin.operative_config_str())

def append_results(experiment_parent, results, seed):
    try:
        with open(experiment_parent) as f:
            file = json.load(f)
    except IOError:
        file = {}
    with open(experiment_parent, "w") as f:
        file[seed] = results
        json.dump(file, f)
