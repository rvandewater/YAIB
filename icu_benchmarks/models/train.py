import os
import random
import shutil
import sys
import gin
import torch
import logging
import numpy as np

from icu_benchmarks.data.loader import RICUDataset
from icu_benchmarks.models.wrappers import MLWrapper
from icu_benchmarks.models.utils import save_config_file


def train_with_gin(
    model_dir=None,
    data=None,
    load_weights=False,
    gin_configs=None,
    seed=1234,
    reproducible=True,
):
    """Trains a model based on the provided gin configuration.

    This function will set the provided gin bindings, call the train() function
    and clear the gin config. Please see train() for required gin bindings.

    Args:
        model_dir: String with path to directory where model output should be saved.
        gin_config_files: List of gin config files to load.
        gin_bindings: List of gin bindings to use.
        seed: Integer corresponding to the common seed used for any random operation.
    """

    # Setting the seed before gin parsing
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if reproducible:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if gin_configs is None:
        gin_configs = []

    gin.parse_config(gin_configs)
    train_common(model_dir, data, load_weights)
    gin.clear_config()


@gin.configurable("train_common")
def train_common(
    log_dir,
    data,
    load_weights=False,
    model=MLWrapper,
    weight=None,
    do_test=False,
):
    """
    Common wrapper to train all benchmarked models.
    """
    if not load_weights:
        os.makedirs(log_dir)
    dataset = RICUDataset(data, split="train")
    val_dataset = RICUDataset(data, split="val")

    model.set_logdir(log_dir)

    if load_weights:
        if os.path.isfile(os.path.join(log_dir, "model.torch")):
            model.load_weights(os.path.join(log_dir, "model.torch"))
        elif os.path.isfile(os.path.join(log_dir, "model.txt")):
            model.load_weights(os.path.join(log_dir, "model.txt"))
        elif os.path.isfile(os.path.join(log_dir, "model.joblib")):
            model.load_weights(os.path.join(log_dir, "model.joblib"))
        else:
            raise Exception("No weights to load at path : {}".format(os.path.join(log_dir, "model.*")))
        do_test = True

    else:
        save_config_file(log_dir)  # We save the operative config before and also after training
        try:
            model.train(dataset, val_dataset, weight)
        except ValueError as e:
            logging.exception(e)
            if "Only one class present" in str(e):
                logging.error(
                    "There seems to be a problem with the evaluation metric. In case you are attempting "
                    "to train with the synthetic data, this is expected behaviour"
                )
            sys.exit(1)

    if do_test:
        test_dataset = RICUDataset(data, split="test")
        weight = dataset.get_balance()
        model.test(test_dataset, weight)
    save_config_file(log_dir)
