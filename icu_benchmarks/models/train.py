import os
import gin
import torch
import logging
import json
import pandas as pd
from joblib import load
from torch.optim import Adam
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TQDMProgressBar,
    LearningRateMonitor,
)
from pathlib import Path
from icu_benchmarks.data.loader import (
    PredictionDataset,
    ImputationDataset,
    PredictionDatasetpytorch,
)


from icu_benchmarks.models.utils import save_config_file, JSONMetricsLogger
from icu_benchmarks.contants import RunMode
from icu_benchmarks.data.constants import DataSplit as Split
from captum.attr import IntegratedGradients, Saliency, FeatureAblation, Lime


cpu_core_count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()


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
    gradient_clip_val=0,
    test_on: str = Split.test,
    dataset_names=None,
    use_wandb: bool = False,
    cpu: bool = False,
    verbose=True,
    ram_cache=False,
    pl_model=True,
    train_only=False,
    num_workers: int = min(
        cpu_core_count,
        torch.cuda.device_count() * 8 * int(torch.cuda.is_available()),
        32,
    ),
    explain: bool = False,
    pytorch_forecasting: bool = False,
    XAI_metric: bool = False,
    random_labels: bool = False,
    random_model_dir: str = None,
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

    # choose dataset_class based on the model
    dataset_class = (
        ImputationDataset
        if mode == RunMode.imputation
        else (PredictionDatasetpytorch if (pytorch_forecasting) else PredictionDataset)
    )

    logging.info(f"Logging to directory: {log_dir}.")
    save_config_file(log_dir)  # We save the operative config before and also after training

    train_dataset = dataset_class(data, split=Split.train, ram_cache=ram_cache, name=dataset_names["train"])
    val_dataset = dataset_class(data, split=Split.val, ram_cache=ram_cache, name=dataset_names["val"])
    train_dataset, val_dataset = assure_minimum_length(train_dataset), assure_minimum_length(val_dataset)
    batch_size = min(batch_size, len(train_dataset), len(val_dataset))
    test_dataset = dataset_class(data, split=test_on, name=dataset_names["test"])
    test_dataset = assure_minimum_length(test_dataset)

    if not eval_only:
        logging.info(
            f"Training on {train_dataset.name} with {len(train_dataset)} samples and validating on {val_dataset.name} with"
            f" {len(val_dataset)} samples."
        )
    batch_size = int(batch_size)
    logging.info(f"Using {num_workers} workers for data loading.")
    train_loader, val_loader, test_loader, model = prepare_data_loaders(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        pytorch_forecasting=pytorch_forecasting,
        load_weights=load_weights,
        source_dir=source_dir,
        pl_model=pl_model,
        optimizer=optimizer,
        epochs=epochs,
        mode=mode,
        random_labels=random_labels,
    )

    model.set_weight(weight, train_dataset)
    model.set_trained_columns(train_dataset.get_feature_names())
    loggers = [TensorBoardLogger(log_dir), JSONMetricsLogger(log_dir)]
    if use_wandb:
        loggers.append(WandbLogger(save_dir=log_dir))
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
        num_sanity_val_steps=-1,
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

    if explain:
        path = Path(random_model_dir)
        random_model = load_model(
            model,
            source_dir=path,
            pl_model=pl_model,
            train_dataset=train_dataset,
            optimizer=optimizer,
        )

        XAI_dict = {}  # dictrionary to log attributions metrics

        # choose which  methods to get attributions
        methods = {
            "G": Saliency,
            "L": Lime,
            "IG": IntegratedGradients,
            "FA": FeatureAblation,
            "R": "Random",
            "Att": "Attention",
        }
        for key, item in methods.items():
            # If conditions needed here as different explantations require different inputs
            if key == "IG":
                (
                    all_attrs,
                    features_attrs,
                    timestep_attrs,
                    ts_v_score,
                    ts_score,
                    v_score,
                    r_score,
                    st_i_score,
                    st_o_score,
                ) = model.explantation(
                    dataloader=test_loader,
                    method=item,
                    log_dir=log_dir,
                    plot=True,
                    n_steps=50,
                    XAI_metric=XAI_metric,
                    random_model=random_model,
                )
            elif key == "L" or key == "FA":
                """for Lime and feature ablation we need to define
                what is a feature we define each variable
                per timestep as a feature"""
                shapes = [
                    torch.Size([64, 24, 0]),
                    torch.Size([64, 24, 53]),
                    torch.Size([64, 24]),
                    torch.Size([64]),
                    torch.Size([64, 1, 0]),
                    torch.Size([64, 1, 53]),
                    torch.Size([64, 1]),
                    torch.Size([64]),
                    torch.Size([64, 1]),
                    torch.Size([64, 1]),
                    torch.Size([64, 2]),
                ]

                # Create a feature mask for the second tensor that includes both features and timesteps
                num_timesteps = shapes[1][1]
                num_features = shapes[1][2]
                feature_mask_second = torch.arange(num_timesteps * num_features).reshape(num_timesteps, num_features)
                feature_mask_second = feature_mask_second.unsqueeze(0).repeat(shapes[1][0], 1, 1)
                # Create a tuple of masks
                feature_masks = tuple(
                    [create_default_mask(shape) if i != 1 else feature_mask_second for i, shape in enumerate(shapes)]
                )
                (
                    all_attrs,
                    features_attrs,
                    timestep_attrs,
                    ts_v_score,
                    ts_score,
                    v_score,
                    r_score,
                    st_i_score,
                    st_o_score,
                ) = model.explantation(
                    dataloader=test_loader,
                    method=item,
                    log_dir=log_dir,
                    plot=True,
                    feature_mask=feature_masks,
                    return_input_shape=True,
                    XAI_metric=XAI_metric,
                    random_model=random_model,
                )

            else:
                (
                    all_attrs,
                    features_attrs,
                    timestep_attrs,
                    ts_v_score,
                    ts_score,
                    v_score,
                    r_score,
                    st_i_score,
                    st_o_score,
                ) = model.explantation(
                    dataloader=test_loader,
                    method=item,
                    log_dir=log_dir,
                    plot=True,
                    XAI_metric=XAI_metric,
                    random_model=random_model,
                )

            if XAI_metric:
                # logging metric scores
                print("{} Attributions Faithfulness Timesteps ".format(key), ts_score)
                XAI_dict["{}_Faith Timesteps".format(key)] = ts_score
                print("{}_ROS ".format(key), st_o_score)
                XAI_dict["{}_ROS".format(key)] = st_o_score
                print("{}_RIS ".format(key), st_i_score)
                XAI_dict["{}_RIS".format(key)] = st_i_score

                print("{} Attributions faithfulness featrues ".format(key), v_score)
                XAI_dict["{}_Faith Features".format(key)] = v_score

                print(
                    "{}_Attributions Faithfulness Variable Per Timestep ".format(key),
                    ts_v_score,
                )
                XAI_dict["{}_Faith Variable Per Timestep".format(key)] = ts_v_score
                print("{}_Data Randomization Distance ".format(key), r_score)
                XAI_dict["{}_Data Randomization Distance".format(key)] = r_score

        # Path to the JSON file in log_dir
        json_file_path = f"{log_dir}/XAI_metrics.json"

        # Write the dictionary to a JSON file
        with open(json_file_path, "w") as json_file:
            json.dump(XAI_dict, json_file)

    model.set_weight("balanced", train_dataset)
    test_loss = trainer.test(model, dataloaders=test_loader, verbose=verbose)[0]["test/loss"]
    save_config_file(log_dir)
    return test_loss


def prepare_data_loaders(
    model,
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size,
    num_workers,
    pin_memory,
    drop_last=True,
    shuffle_train=True,
    pytorch_forecasting=False,
    load_weights=False,
    source_dir=None,
    pl_model=None,
    optimizer=None,
    epochs=None,
    mode=None,
    random_labels=False,
):
    """
    Prepares PyTorch data loaders based on the provided datasets and configuration.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Test dataset.
        batch_size: Batch size for data loaders.
        num_workers: Number of worker processes for data loading.
        pin_memory: Whether to use pin_memory for faster data transfer to GPU.
        drop_last: Whether to drop the last incomplete batch.
        shuffle_train: Whether to shuffle the training data loader.
        load_weights: Whether to load weights from a pre-trained model.
        source_dir: Directory to load weights from.
        pl_model: PyTorch Lightning model (used for loading weights).
        optimizer: Optimizer for the model.
        epochs: Number of training epochs.
        mode: Run mode for the model.
        random_labels: Whether to randomize labels for the datasets.

    Returns:
        tuple: Tuple containing train_loader, val_loader, and test_loader.
    """
    if pytorch_forecasting:
        train_loader = train_dataset.to_dataloader(
            train=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            shuffle=shuffle_train,
        )
        val_loader = val_dataset.to_dataloader(
            train=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        test_loader = test_dataset.to_dataloader(
            train=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            shuffle=False,
        )
        if load_weights:
            model = load_model(
                model,
                source_dir,
                pl_model=pl_model,
                train_dataset=train_dataset,
                optimizer=optimizer,
            )
        else:
            model = model(
                train_dataset,
                optimizer=optimizer,
                epochs=epochs,
                run_mode=mode,
                batch_size=batch_size,
            )
        if random_labels:
            train_dataset.randomize_labels(num_classes=model.num_classes)
            val_dataset.randomize_labels(num_classes=model.num_classes)
            test_dataset.randomize_labels(num_classes=model.num_classes)

    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

        test_loader = (
            DataLoader(
                test_dataset,
                batch_size=min(batch_size * 4, len(test_dataset)),
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
            )
            if model.requires_backprop
            else DataLoader([test_dataset.to_tensor()], batch_size=1)
        )

        data_shape = next(iter(train_loader))[0].shape
        if load_weights:
            model = load_model(model, source_dir, pl_model=pl_model)
        else:
            model = model(optimizer=optimizer, input_size=data_shape, epochs=epochs, run_mode=mode)

    return train_loader, val_loader, test_loader, model


def create_default_mask(shape):
    if len(shape) == 3:
        return torch.zeros(shape[0], shape[1], max(1, shape[2]), dtype=torch.int32)
    elif len(shape) == 2:
        return torch.zeros(shape[0], max(1, shape[1]), dtype=torch.int32)
    else:  # len(shape) == 1
        return torch.zeros(shape[0], dtype=torch.int32)


def load_model(model, source_dir, pl_model=True, train_dataset=None, optimizer=None):
    if source_dir is None:
        return None

    if source_dir.exists():
        if model.requires_backprop:
            if (source_dir / "model.ckpt").exists():
                model_path = source_dir / "model.ckpt"
            elif (source_dir / "model-v1.ckpt").exists():
                model_path = source_dir / "model-v1.ckpt"
            elif (source_dir / "last.ckpt").exists():
                model_path = source_dir / "last.ckpt"
            else:
                return Exception(f"No weights to load at path : {source_dir}")
            if pl_model:
                if train_dataset is not None:
                    model = model.load_from_checkpoint(model_path, dataset=train_dataset, optimizer=optimizer)

                else:
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
