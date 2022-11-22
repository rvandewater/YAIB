import os
import random
import sys
import gin
import torch
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from icu_benchmarks.data.loader import RICUDataset
from icu_benchmarks.models.wrappers import MLWrapper
from icu_benchmarks.models.utils import save_config_file


def train_with_gin(
    log_dir: Path = None,
    data: dict[str, pd.DataFrame] = None,
    load_weights: bool = False,
    source_dir: Path = None,
    seed: int = 1234,
    reproducible: bool = True,
):
    """Trains a model based on the provided gin configuration.

    This function will set the provided gin bindings, call the train() function
    and clear the gin config. Please see train() for required gin bindings.

    Args:
        log_dir: Path to directory where model output should be saved.
        data: Dict containing data to be trained on.
        load_weights: If set to true, skip training and load weights from source_dir instead.
        source_dir: If set to load weights, path to directory containing trained weights.
        seed: Common seed used for any random operation.
        reproducible: If set to true, set torch to run reproducibly.
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

    train_common(log_dir, data, load_weights, source_dir)


@gin.configurable("train_common")
def train_common(
    log_dir: Path,
    data: dict[str, pd.DataFrame],
    load_weights: bool = False,
    source_dir: Path = None,
    model: object = MLWrapper,
    weight: str = None,
    do_test: bool = False,
):
    """Common wrapper to train all benchmarked models."""
    model.set_logdir(log_dir)
    save_config_file(log_dir)  # We save the operative config before and also after training

    dataset = RICUDataset(data, split="train")
    val_dataset = RICUDataset(data, split="val")

    if load_weights:
        if (source_dir / "model.torch").is_file():
            model.load_weights(source_dir / "model.torch")
        elif (source_dir / "model.txt").is_file():
            model.load_weights(source_dir / "model.txt")
        elif (source_dir / "model.joblib").is_file():
            model.load_weights(source_dir / "model.joblib")
        else:
            raise Exception("No weights to load at path : {}".format(source_dir / "model.*"))
        do_test = True

    else:
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

def train_imputation_method(
        log_dir: Path,
        data: dict[str, pd.DataFrame],
        load_weights: bool = False,
        source_dir: Path = None,
        model: object = MLWrapper,
        weight: str = None,
        do_test: bool = False) -> None:
    ...