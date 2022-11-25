import os
import random
import sys
import gin
import torch
import logging
import numpy as np
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from typing import Type
from torch.utils.data import DataLoader
from pathlib import Path

from icu_benchmarks.data.loader import RICUDataset, ImputationDataset
from icu_benchmarks.models.wrappers import MLWrapper, DLWrapper, ImputationWrapper
from icu_benchmarks.models.utils import save_config_file

@gin.configurable("training")
def train_with_gin(
    log_dir: Path = None,
    data: dict[str, pd.DataFrame] = None,
    load_weights: bool = False,
    source_dir: Path = None,
    seed: int = 1234,
    reproducible: bool = True,
    mode: str = "classification",
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

    if mode == "classification":
        train_common(log_dir, data, load_weights, source_dir)
    elif mode == "imputation":
        train_imputation_method(log_dir, data, load_weights, source_dir)


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

@gin.configurable("train_imputation_method")
def train_imputation_method(
        log_dir: Path,
        data: dict[str, pd.DataFrame],
        load_weights: bool = False,
        source_dir: Path = None,
        model: Type = ImputationWrapper,
        weight: str = None,
        do_test: bool = False,
        epochs: int = 10,
        num_workers: int = os.cpu_count(),
        batch_size: int = 64,
        patience: int = 10,
        min_delta = 1e-4) -> None:
    
    train_dataset = ImputationDataset(data, split="train")
    validation_dataset = ImputationDataset(data, split="val")
    
    train_loader = DataLoader(train_dataset, num_workers=num_workers, pin_memory=True, batch_size=batch_size)
    # usually a much larger batch size for validation can be used, as not gradient updates have to be performed on them
    validation_loader = DataLoader(validation_dataset, num_workers=num_workers, pin_memory=True, batch_size=batch_size * 4)
    
    data_shape = iter(train_loader).next().shape()
    
    if load_weights:
        model = model.load_from_chekpoint(source_dir)
    else:
        model = model(input_size=data_shape)

    trainer = Trainer(
        max_epochs=epochs,
        callbacks=[
            EarlyStopping(monitor="loss", min_delta=min_delta, patience=patience),
            ModelCheckpoint(log_dir, monitor="rmse", save_top_k=1, save_last=True),
        ],
        precision=16,
        accelerator="auto",
        gpus=torch.cuda.device_count(),
    )
    