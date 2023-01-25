import os
import random
import sys
import gin
import torch
import logging
import numpy as np
import pandas as pd
import wandb
from typing import Dict
from torch.optim import Adam
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader
from pathlib import Path

from icu_benchmarks.data.loader import RICUDataset, ImputationDataset
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
    model: object = gin.REQUIRED,
    weight: str = None,
    optimizer: type = Adam,
    do_test: bool = False,
    batch_size=64,
    epochs=1000,
    patience=20,
    min_delta=1e-5,
    use_wandb: bool = True,
    num_workers: int = min(len(os.sched_getaffinity(0)), torch.cuda.device_count() * 8 if torch.cuda.is_available() else 16),
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
    logging.info(f"Training model: {model.__name__}")
    DatasetClass = ImputationDataset if mode == "Imputation" else RICUDataset

    logging.info(f"Logging to directory: {log_dir}")
    save_config_file(log_dir)  # We save the operative config before and also after training

    train_dataset = DatasetClass(data, split="train")
    val_dataset = DatasetClass(data, split="val")
    print("IN TRAIN ", batch_size, "epochs:", epochs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    data_shape = next(iter(train_loader))[0].shape
    logging.info(f"performing task on model {model.__name__}...")

    if load_weights:
        if source_dir.exists():
            # if not model.needs_training:
            #     model = torch.load(source_dir / "model.ckpt")
            # else:
            model = model.from_checkpoint(source_dir / "model.ckpt")
        else:
            raise Exception(f"No weights to load at path : {source_dir}")
        do_test = True
    else:
        model = model(optimizer=optimizer, input_size=data_shape, epochs=epochs)
        if mode == "Classification":
            model.set_weight(weight, train_dataset)
    
    if not only_evaluate:
        loggers = [TensorBoardLogger(log_dir)]
        if use_wandb:
            run_name = f"{type(model).__name__}-{dataset_name}"
            loggers.append(WandbLogger(run_name, save_dir=log_dir, project="Data_Imputation"))
            wandb.config.update({"run-name": run_name})
            wandb.run.name = run_name
            wandb.run.save()

        trainer = Trainer(
            # model=model,
            max_epochs=epochs if model.needs_training else 1,
            callbacks=[
                EarlyStopping(monitor=f"val/loss", min_delta=min_delta, patience=patience, strict=False),
                ModelCheckpoint(log_dir, filename="model", save_top_k=1, save_last=True),
            ],
            # precision=16,
            accelerator="auto",
            devices=max(torch.cuda.device_count(), 1),
            deterministic=reproducible,
            benchmark=not reproducible,
            logger=loggers,
            num_sanity_val_steps=0,
        )

        if model.needs_fit:
            logging.info("fitting model to data...")
            if mode == "Imputation":
                model.fit(train_dataset)
            else:
                trainer.fit(
                    model,
                    train_dataloaders=DataLoader([train_dataset.get_data_and_labels() + val_dataset.get_data_and_labels()], batch_size=1),
                    val_dataloaders=DataLoader([val_dataset.get_data_and_labels()], batch_size=1)
                )
            if not model.needs_training:
                try:
                    torch.save(model, log_dir / "model.ckpt")
                except Exception as e:
                    logging.error(f"cannot save model to path {str((log_dir / 'model.ckpt').resolve())}: {e}")
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
        trainer.test(
            model, 
            dataloaders = (
                test_loader if (mode == "Imputation" or model.needs_training) 
                else DataLoader([test_dataset.get_data_and_labels()], batch_size=1)
            )
        )
    save_config_file(log_dir)
