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


@gin.configurable("train_common")
def train_common(
    data: dict[str, pd.DataFrame],
    log_dir: Path,
    load_weights: bool = False,
    source_dir: Path = None,
    seed: int = 1234,
    reproducible: bool = True,
    model: object = MLWrapper,
    weight: str = None,
    test_on: str = "test",
    use_static: bool = True,
):
    """Common wrapper to train all benchmarked models.

    Args:
        data: Dict containing data to be trained on.
        log_dir: Path to directory where model output should be saved.
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

    model.set_log_dir(log_dir)
    save_config_file(log_dir)

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

    else:
        try:
            model.train(dataset, val_dataset, weight, seed)
        except ValueError as e:
            logging.exception(e)
            sys.exit(1)

    test_dataset = RICUDataset(data, split=test_on)
    weight = dataset.get_balance()

    # save config file again to capture missing gin parameters
    save_config_file(log_dir)
    return model.test(test_dataset, weight, seed)
