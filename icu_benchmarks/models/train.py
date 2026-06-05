import logging
import os
from pathlib import Path
from typing import Literal, Optional

import gin
import numpy as np
import polars as pl
import torch
from joblib import load
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.optim import Adam
from torch.utils.data import DataLoader

from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.constants import DataSplit as DataSplit
from icu_benchmarks.data.loader import (
    ImputationPandasDataset,
    PredictionPandasDataset,
    PredictionPolarsDataset,
)
from icu_benchmarks.models import DLModel, MLModelClassifier, MLModelRegression
from icu_benchmarks.models.utils import JSONMetricsLogger, save_config_file

cpu_core_count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
cpu_core_count = 1 if not cpu_core_count else cpu_core_count  #  os.cpu_count possibly None


def assure_minimum_length(dataset: pl.DataFrame) -> pl.DataFrame:
    if len(dataset) < 2:
        return pl.concat([dataset, dataset])
    return dataset


@gin.configurable("train_common")
def train_common(
    data: dict[str, dict[str, pl.DataFrame]],
    log_dir: Path,
    eval_only: bool = False,
    load_weights: bool = False,
    source_dir: Path = Path(""),
    reproducible: bool = True,
    mode: str = RunMode.classification,
    model: DLModel | MLModelClassifier | MLModelRegression | object = gin.REQUIRED,
    weight: str = "",
    optimizer: type = Adam,
    precision: Optional[Literal[16, 32, 64, "16-mixed", "bf16", "bf16-mixed", "16-true"]] = 32,
    batch_size: int = 1,
    epochs: int = 100,
    patience: int = 20,
    min_delta: float = 1e-5,
    test_on: str = DataSplit.test,
    dataset_names: Optional[dict] = None,
    use_wandb: bool = False,
    cpu: bool = False,
    verbose: bool = False,
    ram_cache: bool = False,
    pl_model: bool = True,
    train_only: bool = False,
    num_workers: int = min(
        cpu_core_count,
        torch.cuda.device_count() * 8 * int(torch.cuda.is_available()),
        32,
    ),
    polars: bool = True,
    persistent_workers: bool = False,
):
    """Common wrapper to train all benchmarked models.

    Args:
        data: Dict containing data to be trained on.
        log_dir: Path to directory where model output should be saved.
        eval_only: If set to true, skip training and only evaluate the model.
        load_weights: If set to true, skip training and load weights from source_dir instead.
        source_dir: If set to load weights, path to directory containing trained weights.
        reproducible: If set to true, set torch to run reproducibly.
        mode: Mode of the model. Can be one of the values of RunMode.
        model: Model to be trained.
        weight: Weight to be used for the loss function.
        optimizer: Optimizer to be used for training.
        precision: Pytorch precision to be used for training. Can be 16 or 32.
        batch_size: Batch size to be used for training.
        epochs: Number of epochs to train for.
        patience: Number of epochs to wait for improvement before early stopping.
        min_delta: Minimum change in loss to be considered an improvement.
        test_on: If set to "test", evaluate the model on the test set. If set to "val", evaluate on the validation set.
        use_wandb: If set to true, log to wandb.
        cpu: If set to true, run on cpu.
        verbose: Enable detailed logging.
        ram_cache: Whether to cache the data in RAM.
        pl_model: Loading a pytorch lightning model.
        num_workers: Number of workers to use for data loading.
    """

    if dataset_names is None:
        dataset_names = {"train": "default", "val": "default", "test": "default"}

    logging.info(f"Training model: {model.__name__}.")
    # TODO: add support for polars versions of datasets
    dataset_classes: dict = {
        RunMode.imputation: ImputationPandasDataset,
        RunMode.classification: PredictionPolarsDataset if polars else PredictionPandasDataset,
        RunMode.regression: PredictionPolarsDataset if polars else PredictionPandasDataset,
    }
    dataset_class = dataset_classes[mode]

    logging.info(f"Using dataset class: {dataset_class.__name__}.")
    logging.info(f"Logging to directory: {log_dir}.")
    save_config_file(log_dir)  # We save the operative config before and also after training
    train_dataset = dataset_class(data, split=DataSplit.train, ram_cache=ram_cache, name=dataset_names.get("train", "default"))
    val_dataset = dataset_class(data, split=DataSplit.val, ram_cache=ram_cache, name=dataset_names.get("val", "default"))
    train_dataset, val_dataset = (
        assure_minimum_length(train_dataset),
        assure_minimum_length(val_dataset),
    )
    batch_size = min(batch_size, len(train_dataset), len(val_dataset))

    if not eval_only:
        logging.info(
            f"Training on {train_dataset.name} with {len(train_dataset)} samples and validating on {val_dataset.name} with"
            f" {len(val_dataset)} samples."
        )
    logging.info(f"Using {num_workers} workers for data loading.")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=persistent_workers,
        pin_memory=not cpu,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=persistent_workers,
        pin_memory=not cpu,
    )

    data_shape = next(iter(train_loader))[0].shape

    if load_weights:
        model: DLModel | MLModelClassifier | MLModelRegression = load_model(model, source_dir, pl_model=pl_model)
    else:
        model: DLModel | MLModelClassifier | MLModelRegression = model(
            optimizer=optimizer,
            input_size=data_shape,
            epochs=epochs,
            run_mode=mode,
            cpu=cpu,
        )

    model.set_weight(weight, train_dataset)
    model.set_trained_columns(train_dataset.get_feature_names())
    loggers = [TensorBoardLogger(log_dir), JSONMetricsLogger(log_dir)]
    devices = max(torch.cuda.device_count(), 1)

    if use_wandb:
        loggers.append(WandbLogger(save_dir=log_dir))
        logging.info("Use of wandb is detected. Only single gpu training is supported with wandb.")
        devices = 1

    callbacks = [
        EarlyStopping(
            monitor="val/loss",
            min_delta=min_delta,
            patience=patience,
            strict=False,
            verbose=verbose,
        ),
        ModelCheckpoint(log_dir, filename="model", save_top_k=1, save_last=True),
        LearningRateMonitor(logging_interval="step"),
    ]
    if verbose:
        callbacks.append(TQDMProgressBar(refresh_rate=min(100, len(train_loader) // 2)))
    if precision in (16, "16-mixed", "bf16", "bf16-mixed"):
        torch.set_float32_matmul_precision("high")

    trainer = Trainer(
        max_epochs=epochs if model.requires_backprop else 1,
        min_epochs=1,  # We need at least one epoch to get results.
        callbacks=callbacks,
        precision=precision,
        accelerator="auto" if not cpu else "cpu",
        devices=devices,
        deterministic="warn" if reproducible else False,
        benchmark=not reproducible,
        enable_progress_bar=verbose,
        logger=loggers,
        num_sanity_val_steps=2,  # Helps catch errors in the validation loop before training begins.
        log_every_n_steps=5,
    )
    if not eval_only:
        if model.requires_backprop:
            logging.info("Training DL model.")
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            logging.info("Training complete.")
        else:
            logging.info("Training ML model.")
            model.fit(train_dataset, val_dataset)
            model.save_model(log_dir, "last")
            logging.info("Training complete.")
    if train_only:
        logging.info("Finished training full model.")
        save_config_file(log_dir)
        return 0
    test_dataset = dataset_class(data, split=test_on, name=dataset_names.get("test", "default"), ram_cache=ram_cache)
    test_dataset = assure_minimum_length(test_dataset)
    logging.info(f"Testing on {test_dataset.name}  with {len(test_dataset)} samples.")
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=min(batch_size * 4, len(test_dataset)),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=not cpu,
            drop_last=True,
            persistent_workers=persistent_workers,
        )
        if model.requires_backprop
        else DataLoader([test_dataset.to_tensor()], batch_size=1)
    )

    model.set_weight("balanced", train_dataset)
    test_loss = trainer.test(model, dataloaders=test_loader, verbose=verbose)[0]["test/loss"]
    persist_shap_data(trainer, log_dir)
    save_config_file(log_dir)
    return test_loss


def persist_shap_data(trainer: Trainer, log_dir: Path):
    """
    Persist shap values to disk.
    Args:
        trainer: Pytorch lightning trainer object
        log_dir: Log directory
    """
    try:
        if trainer.lightning_module.test_shap_values is not None:
            shap_values = trainer.lightning_module.test_shap_values
            shaps_test = pl.DataFrame(
                schema=trainer.lightning_module.trained_columns,
                data=np.transpose(shap_values.values),
            )
            with (log_dir / "test_shap_values.parquet").open("wb") as f:
                shaps_test.write_parquet(f)
            logging.info(f"Saved shap values to {log_dir / 'test_shap_values.parquet'}")
        if trainer.lightning_module.train_shap_values is not None:
            shap_values = trainer.lightning_module.train_shap_values
            shaps_train = pl.DataFrame(
                schema=trainer.lightning_module.trained_columns,
                data=np.transpose(shap_values.values),
            )
            with (log_dir / "train_shap_values.parquet").open("wb") as f:
                shaps_train.write_parquet(f)

    except Exception as e:
        logging.error(f"Failed to save shap values: {e}")


def load_model(model, source_dir, pl_model=True) -> DLModel | MLModelClassifier | MLModelRegression:
    if source_dir.exists():
        if model.requires_backprop:
            if (source_dir / "model.ckpt").exists():
                model_path = source_dir / "model.ckpt"
            elif (source_dir / "model-v1.ckpt").exists():
                model_path = source_dir / "model-v1.ckpt"
            elif (source_dir / "last.ckpt").exists():
                model_path = source_dir / "last.ckpt"
            else:
                raise Exception(f"No weights to load at path : {source_dir}")
            if pl_model:
                model = model.load_from_checkpoint(model_path)
            else:
                checkpoint = torch.load(model_path)
                model.load_from_checkpoint(checkpoint)
        else:
            model_path = source_dir / "model.joblib"
            model = load(model_path)
    else:
        raise Exception(f"No weights to load at path : {source_dir}")
    logging.info(f"Loaded {type(model)} model from {model_path}")
    return model
