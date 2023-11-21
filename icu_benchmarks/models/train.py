import os
import gin
import torch
import pickle
import logging
import json
import pandas as pd
from joblib import load
from torch.optim import Adam
import numpy as np
import quantus
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
from collections import OrderedDict
from captum.attr import IntegratedGradients, ShapleyValueSampling, Saliency, GuidedBackprop, LRP, FeatureAblation, Lime


cpu_core_count = (
    len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
)


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
    random_model_dir: str = None
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
        else (
            PredictionDatasetpytorch if (pytorch_forecasting) else PredictionDataset
        )
    )

    logging.info(f"Logging to directory: {log_dir}.")
    save_config_file(
        log_dir
    )  # We save the operative config before and also after training

    train_dataset = dataset_class(
        data, split=Split.train, ram_cache=ram_cache, name=dataset_names["train"]
    )
    val_dataset = dataset_class(
        data, split=Split.val, ram_cache=ram_cache, name=dataset_names["val"]
    )
    train_dataset, val_dataset = assure_minimum_length(
        train_dataset
    ), assure_minimum_length(val_dataset)
    batch_size = min(batch_size, len(train_dataset), len(val_dataset))
    test_dataset = dataset_class(data, split=test_on, name=dataset_names["test"])
    test_dataset = assure_minimum_length(test_dataset)
    if not eval_only:
        logging.info(
            f"Training on {train_dataset.name} with {len(train_dataset)} samples and validating on {val_dataset.name} with"
            f" {len(val_dataset)} samples."
        )

    logging.info(f"Using {num_workers} workers for data loading.")
    if pytorch_forecasting:
        train_loader = train_dataset.to_dataloader(
            train=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=True,
        )
        val_loader = val_dataset.to_dataloader(
            train=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=True,
        )
        test_loader = test_dataset.to_dataloader(
            train=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=True,
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

        data_shape = next(iter(train_loader))[0].shape
        if load_weights:
            model = load_model(model, source_dir, pl_model=pl_model)
        else:
            model = model(
                optimizer=optimizer, input_size=data_shape, epochs=epochs, run_mode=mode
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

            trainer.fit(
                model, train_dataloaders=train_loader, val_dataloaders=val_loader
            )
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
        attributions_dict = {}
        Interpertations = model.interpertations(test_loader, log_dir, plot=True)
        attributions_dict["attention_weights"] = Interpertations["attention"].tolist()
        attributions_dict["static_variables"] = Interpertations["static_variables"].tolist()
        attributions_dict["encoder_variables"] = Interpertations["encoder_variables"].tolist()
        print("attention", Interpertations)
        model.train()
        XAI_dict = {}
        if XAI_metric:
            XAI_dict = {}
            # random attribution per timestep
            random_attributions_ts = np.random.normal(size=24)
            F_baseline_ts = model.Faithfulness_Correlation(
                test_loader, random_attributions_ts, pertrub='Noise', subset_size=4, time_step=True, nr_runs=10)
            print('Random normal faithfulness correlation for timesteps', F_baseline_ts)

            F_attention = model.Faithfulness_Correlation(
                test_loader, Interpertations["attention"], pertrub='Noise', subset_size=4, time_step=True, nr_runs=10)
            print('Attention faithfulness correlation', F_attention)
            # random attribution per variable per timestep
            random_attributions_v_ts = np.random.normal(size=[24, 53])
            F_baseline_v_ts = model.Faithfulness_Correlation(
                test_loader, random_attributions_v_ts, pertrub='Noise', feature_timestep=True, nr_runs=10, subset_size=[4, 9])
            print('Random normal faithfulness correlation for variables per timesteps', F_baseline_v_ts)

        XAI_dict["attention_faith"] = F_attention
        XAI_dict["random_faith_timestep"] = random_attributions_ts
        XAI_dict["random_faith_var_timestep"] = random_attributions_v_ts.tolist()
        methods = {
            "Saliency": Saliency,
            #  "Lime": Lime,
            "IG": IntegratedGradients,


        }
        for key, item in methods.items():
            if key == "IG":
                all_attrs, features_attrs, timestep_attrs = model.explantation_captum(
                    test_loader=test_loader,
                    method=item, log_dir=log_dir, plot=True, n_steps=20
                )
            else:
                all_attrs, features_attrs, timestep_attrs = model.explantation_captum(
                    test_loader=test_loader,
                    method=item, log_dir=log_dir, plot=True
                )
            attributions_dict["{}_all".format(key)] = all_attrs.tolist()
            attributions_dict["{}_timesteps".format(key)] = timestep_attrs.tolist()
            attributions_dict["{}_features".format(key)] = features_attrs.tolist()

            print("{}".format(key), all_attrs, features_attrs, timestep_attrs)
            if XAI_metric:
                faithfulness_timesteps = model.Faithfulness_Correlation(
                    test_loader, timestep_attrs, pertrub='Noise', time_step=True, subset_size=4, nr_runs=10)
                print('Attributions faithfulness timesteps correlation', faithfulness_timesteps)
                XAI_dict["{}_faith_timesteps".format(key)] = faithfulness_timesteps.tolist()
                random_attributions = np.random.normal(np.shape(all_attrs))

                faithfulness_timesteps_variable = model.Faithfulness_Correlation(
                    test_loader, all_attrs, pertrub='Noise', feature_timestep=True, subset_size=[4, 9], nr_runs=10)
                print('Attributions faithfulness variable per timestep correlation', faithfulness_timesteps_variable)
                XAI_dict["{}_faith_variable_per_timestep".format(key)] = faithfulness_timesteps_variable.tolist()

        # Path to the JSON file in log_dir
        json_file_path = f"{log_dir}/Attributions.json"

        # Write the dictionary to a JSON file
        with open(json_file_path, 'w') as json_file:
            json.dump(attributions_dict, json_file)

        # Path to the JSON file in log_dir
        json_file_path = f"{log_dir}/XAI_metrics.json"

        # Write the dictionary to a JSON file
        with open(json_file_path, 'w') as json_file:
            json.dump(XAI_dict, json_file)

        # path = Path(random_model_dir)

        # random_model = load_model(
        #    model,
        #    source_dir=path,
        #    pl_model=pl_model,
        #    train_dataset=train_dataset,
        #    optimizer=optimizer,
        # )
        # R_attribution = model.Data_Randomization(
        #    test_loader, attributions_IG, IntegratedGradients, random_model)
        # print('Distance Data randmoization score attribution', R_attribution)
        # R_attention = model.Data_Randomization(
        #    test_loader, Attention_weights["attention"], "Attention", random_model)
        # print('Distance Data randmoization score attention', R_attention)

    model.set_weight("balanced", train_dataset)
    test_loss = trainer.test(model, dataloaders=test_loader, verbose=verbose)[0][
        "test/loss"
    ]
    save_config_file(log_dir)
    return test_loss


def load_model(model, source_dir, pl_model=True, train_dataset=None, optimizer=None):
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
                    model = model.load_from_checkpoint(
                        model_path, dataset=train_dataset, optimizer=optimizer
                    )

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
