import os
import gin
import torch
import logging
import pandas as pd
from joblib import load
from torch.optim import Adam
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pathlib import Path
from icu_benchmarks.data.loader import PredictionDataset, ImputationDataset
from icu_benchmarks.models.utils import save_config_file, JSONMetricsLogger
from icu_benchmarks.contants import RunMode
from icu_benchmarks.data.constants import DataSplit as Split
from icu_benchmarks.models.dl_models import GRUNet
# from finetuning_scheduler import FinetuningScheduler
cpu_core_count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
from pytorch_lightning.loggers import WandbLogger
def assure_minimum_length(dataset):
    if len(dataset) < 2:
        return [dataset[0], dataset[0]]
    return dataset


@gin.configurable("train_common")
def train_common(
        data: dict[str, pd.DataFrame],
        log_dir: Path,
        eval_only: bool = False,
        load_weights: bool = False,
        source_dir: Path = None,
        reproducible: bool = True,
        mode: str = RunMode.classification,
        model: object = gin.REQUIRED,
        weight: str = None,
        optimizer: type = Adam,
        precision=32,
        batch_size=64,
        epochs=1000,
        patience=20,
        min_delta=1e-5,
        test_on: str = Split.test,
        dataset_names = None,
        use_wandb: bool = False,
        cpu: bool = False,
        verbose=False,
        ram_cache=False,
        pl_model=True,
        num_workers: int = min(cpu_core_count, torch.cuda.device_count() * 8 * int(torch.cuda.is_available()), 32),
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

    logging.info(f"Training model: {model.__name__}.")
    dataset_class = ImputationDataset if mode == RunMode.imputation else PredictionDataset

    logging.info(f"Logging to directory: {log_dir}.")
    save_config_file(log_dir)  # We save the operative config before and also after training

    train_dataset = dataset_class(data, split=Split.train, ram_cache=ram_cache, name=dataset_names["train"])
    val_dataset = dataset_class(data, split=Split.val, ram_cache=ram_cache, name=dataset_names["val"])
    train_dataset, val_dataset = assure_minimum_length(train_dataset), assure_minimum_length(val_dataset)
    batch_size = min(batch_size, len(train_dataset), len(val_dataset))

    logging.info(f"Training on {train_dataset.name} with {len(train_dataset)} samples and validating on {val_dataset.name} with"
                 f" {len(val_dataset)} samples.")
    logging.info(f"Using {num_workers} workers for data loading.")

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

    # model = model(optimizer=optimizer, input_size=data_shape, epochs=epochs, run_mode=mode)
    if load_weights:
        model = GRUNet.load_from_checkpoint(source_dir/"last.ckpt")
        # model = load_model(model, source_dir, pl_model)

    model.set_weight(weight, train_dataset)
    model.set_trained_columns(train_dataset.get_feature_names())

    loggers = [TensorBoardLogger(log_dir), JSONMetricsLogger(log_dir)]
    if use_wandb:
        loggers.append(WandbLogger(save_dir=log_dir))
    callbacks = [
        EarlyStopping(monitor="val/loss", min_delta=min_delta, patience=patience, strict=False),
        ModelCheckpoint(log_dir, filename="model", save_top_k=1, save_last=True),
        #FinetuningScheduler()
    ]
    if verbose:
        callbacks.append(TQDMProgressBar(refresh_rate=min(100, len(train_loader) // 2)))
    if precision == 16 or "16-mixed":
        torch.set_float32_matmul_precision("medium")

    trainer = Trainer(
        max_epochs=epochs if model.requires_backprop else 1,
        callbacks=callbacks,
        precision=precision,
        accelerator="auto" if not cpu else "cpu",
        devices=max(torch.cuda.device_count(), 1),
        deterministic="warn" if reproducible else False,
        benchmark=not reproducible,
        enable_progress_bar=verbose,
        logger=loggers,
        num_sanity_val_steps=0,

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

    test_dataset = dataset_class(data, split=test_on, name=dataset_names["test"])
    test_dataset = assure_minimum_length(test_dataset)
    logging.info(f"Testing on {test_dataset.name}  with {len(test_dataset)} samples.")
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=min(batch_size * 4, len(test_dataset)),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        if model.requires_backprop
        else DataLoader([test_dataset.to_tensor()], batch_size=1)
    )

    model.set_weight("balanced", train_dataset)
    test_loss = trainer.test(model, dataloaders=test_loader, verbose=verbose)[0]["test/loss"]
    save_config_file(log_dir)
    return test_loss

def load_model(model, source_dir, pl_model=True):
    if source_dir.exists():
        if model.requires_backprop:
            if (source_dir / "last.ckpt").exists():
                model_path = source_dir / "last.ckpt"
            elif (source_dir / "model.ckpt").exists():
                model_path = source_dir / "model.ckpt"
            elif (source_dir / "model-v1.ckpt").exists():
                model_path = source_dir / "model-v1.ckpt"
            else:
                return Exception(f"No weights to load at path : {source_dir}")
            if pl_model:
                model = model.load_from_checkpoint(model_path)
            else:
                checkpoint = torch.load(model_path)
                model.load_state_dict(checkpoint)
        else:
            model = load(source_dir / "model.joblib")
    else:
        raise Exception(f"No weights to load at path : {source_dir}")
    logging.info(f"Loaded {type(model)} model from {source_dir}")
    return model
