import os
import random
import sys
import gin
import torch
import logging
import numpy as np
import pandas as pd
from typing import Dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from typing import Type
from torch.utils.data import DataLoader
from pathlib import Path

from icu_benchmarks.data.loader import RICUDataset, ImputationDataset
from icu_benchmarks.models.wrappers import MLWrapper, DLWrapper, ImputationWrapper
from icu_benchmarks.models.utils import save_config_file


@gin.configurable("train_common")
def train_common(
    log_dir: Path,
    data: Dict[str, pd.DataFrame],
    load_weights: bool = False,
    only_evaluate = False,
    source_dir: Path = None,
    reproducible: bool = True,
    mode: str = "Classification",
    dataset_name: str = "",
    model: object = MLWrapper,
    weight: str = None,
    do_test: bool = False,
    batch_size=64,
    epochs=1000,
    patience=10,
    min_delta=1e-4,
    wandb: bool = True,
    num_workers: int = os.cpu_count(),
):
    """Common wrapper to train all benchmarked models.

    Args:
        log_dir: Path to directory where model output should be saved.
        data: Dict containing data to be trained on.
        load_weights: If set to true, skip training and load weights from source_dir instead.
        source_dir: If set to load weights, path to directory containing trained weights.
        seed: Common seed used for any random operation.
        reproducible: If set to true, set torch to run reproducibly.
    """
    DatasetClass = ImputationDataset if mode == "Imputation" else RICUDataset

    save_config_file(log_dir)  # We save the operative config before and also after training

    train_dataset = DatasetClass(data, split="train")
    val_dataset = DatasetClass(data, split="val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 4,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    data_shape = next(iter(train_loader))[0].shape

    if load_weights:
        if source_dir.exists():
            if not model.needs_training:
                model = torch.load(source_dir / "model.ckpt")
            else:
                model = model.from_checkpoint(source_dir / "model.ckpt")
        else:
            raise Exception(f"No weights to load at path : {source_dir}")
        do_test = True
    else:
        model = model(weight=weight, input_size=data_shape, epochs=epochs)
        if mode == "Classification":
            model.set_weight(weight, train_dataset)
    
    if not only_evaluate:
        loggers = [TensorBoardLogger(log_dir)]
        if wandb:
            run_name = f"{type(model).__name__}-{dataset_name}"
            loggers.append(WandbLogger(run_name, save_dir=log_dir))

        trainer = Trainer(
            max_epochs=epochs,
            callbacks=[
                EarlyStopping(monitor="train/loss", min_delta=min_delta, patience=patience),
                ModelCheckpoint(log_dir, filename="model", save_top_k=1, save_last=True),
            ],
            # precision=16,
            accelerator="auto",
            devices=max(torch.cuda.device_count(), 1),
            deterministic=reproducible,
            logger=loggers,
        )

        if model.needs_fit:
            logging.info("fitting model to data...")
            model.fit(train_dataset)
            if not model.needs_training:
                torch.save(model, log_dir / "model.ckpt")
            logging.info("fitting complete!")

        if model.needs_training:
            logging.info("training model on data...")
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            logging.info("training complete!")

    if only_evaluate or do_test:
        logging.info("testing...")
        test_dataset = DatasetClass(data, split="test")
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size * 4,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        if mode == "Classification":
            model.set_weight("balanced", train_dataset)
        trainer.test(model, dataloaders=test_loader)
    save_config_file(log_dir)
