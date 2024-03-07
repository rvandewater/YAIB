import logging
from abc import ABC
from typing import Dict, Any, List, Optional, Union
import torchmetrics
from sklearn.metrics import log_loss, mean_squared_error
import torch
from torch.nn import MSELoss, CrossEntropyLoss
import torch.nn as nn
from torch import Tensor, FloatTensor
from torch.optim import Optimizer, Adam
import inspect
import gin
import numpy as np
from ignite.exceptions import NotComputableError
from icu_benchmarks.models.constants import ImputationInit
from icu_benchmarks.models.utils import create_optimizer, create_scheduler
from joblib import dump
from pytorch_lightning import LightningModule
from icu_benchmarks.models.constants import MLMetrics, DLMetrics
from icu_benchmarks.contants import RunMode
import matplotlib.pyplot as plt
from icu_benchmarks.models.similarity_func import correlation_spearman, cosine
import captum
from captum._utils.models.linear_model import SkLearnLasso

gin.config.external_configurable(nn.functional.nll_loss, module="torch.nn.functional")
gin.config.external_configurable(nn.functional.cross_entropy, module="torch.nn.functional")
gin.config.external_configurable(nn.functional.mse_loss, module="torch.nn.functional")
gin.config.external_configurable(nn.functional.l1_loss, module="torch.nn.functional")
gin.config.external_configurable(mean_squared_error, module="sklearn.metrics")
gin.config.external_configurable(log_loss, module="sklearn.metrics")


@gin.configurable("BaseModule")
class BaseModule(LightningModule):
    # DL type models, requires backpropagation
    requires_backprop = False
    # Loss function weight initialization type
    weight = None
    # Metrics to be logged
    metrics = {}
    trained_columns = None
    # Type of run mode
    run_mode = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def step_fn(self, batch, step_prefix=""):
        raise NotImplementedError()

    def finalize_step(self, step_prefix=""):
        pass

    def set_metrics(self, *args, **kwargs):
        self.metrics = {}

    def set_trained_columns(self, columns: List[str]):
        self.trained_columns = columns

    def set_weight(self, weight, *args, **kwargs):
        pass

    def training_step(self, batch, batch_idx):
        return self.step_fn(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step_fn(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step_fn(batch, "test")

    def on_train_epoch_end(self) -> None:
        self.finalize_step("train")

    def on_validation_epoch_end(self) -> None:
        self.finalize_step("val")

    def on_test_epoch_end(self) -> None:
        self.finalize_step("test")

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["class"] = self.__class__
        checkpoint["trained_columns"] = self.trained_columns
        return super().on_save_checkpoint(checkpoint)

    def save_model(self, save_path, file_name, file_extension):
        raise NotImplementedError()

    def check_supported_runmode(self, runmode: RunMode):
        if runmode not in self._supported_run_modes:
            raise ValueError(f"Runmode {runmode} not supported for {self.__class__.__name__}")
        return True


@gin.configurable("DLWrapper")
class DLWrapper(BaseModule, ABC):
    requires_backprop = True
    _metrics_warning_printed = set()
    _supported_run_modes = [
        RunMode.classification,
        RunMode.regression,
        RunMode.imputation,
    ]

    def __init__(
        self,
        loss=CrossEntropyLoss(),
        optimizer=Adam,
        run_mode: RunMode = RunMode.classification,
        input_shape=None,
        lr: float = 0.002,
        momentum: float = 0.9,
        lr_scheduler: Optional[str] = None,
        lr_factor: float = 0.99,
        lr_steps: Optional[List[int]] = None,
        epochs: int = 100,
        input_size: Tensor = None,
        initialization_method: str = "normal",
        **kwargs,
    ):
        """General interface for Deep Learning (DL) models."""
        super().__init__()
        self.save_hyperparameters(ignore=["loss", "optimizer"])
        self.loss = loss
        self.optimizer = optimizer
        self.check_supported_runmode(run_mode)
        self.run_mode = run_mode
        self.input_shape = input_shape
        self.lr = lr
        self.momentum = momentum
        self.lr_scheduler = lr_scheduler
        self.lr_factor = lr_factor
        self.lr_steps = lr_steps
        self.epochs = epochs
        self.input_size = input_size
        self.initialization_method = initialization_method
        self.scaler = None

    def on_fit_start(self):
        self.metrics = {
            step_name: {
                metric_name: (metric() if isinstance(metric, type) else metric)
                for metric_name, metric in self.set_metrics().items()
            }
            for step_name in ["train", "val", "test"]
        }
        return super().on_fit_start()

    def on_train_start(self):
        self.metrics = {
            step_name: {
                metric_name: (metric() if isinstance(metric, type) else metric)
                for metric_name, metric in self.set_metrics().items()
            }
            for step_name in ["train", "val", "test"]
        }
        return super().on_train_start()

    def finalize_step(self, step_prefix=""):
        try:
            for name, metric in self.metrics[step_prefix].items():
                try:
                    value = np.float32(metric.compute()) if isinstance(metric.compute(), np.float64) else metric.compute()
                    self.log_dict({f"{step_prefix}/{name}": value}, sync_dist=True)

                except (NotComputableError, ValueError) as e:
                    if step_prefix not in self._metrics_warning_printed:
                        self._metrics_warning_printed.add(step_prefix)
                        logging.warning(f"Metric for {step_prefix}/{name} not computable: {e}")

            for metric in self.metrics[step_prefix].values():
                metric.reset()
        except (NotComputableError, ValueError) as e:
            if step_prefix not in self._metrics_warning_printed:
                self._metrics_warning_printed.add(step_prefix)
                logging.warning(f"Metrics for {step_prefix} not computable")
                print(e)

            pass

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""

        if isinstance(self.optimizer, str):
            optimizer = create_optimizer(self.optimizer, self.lr, self.hparams.momentum)
        elif isinstance(self.optimizer, Optimizer):
            # Already set
            optimizer = self.optimizer
        else:
            optimizer = self.optimizer(self.parameters())

        if self.hparams.lr_scheduler is None or self.hparams.lr_scheduler == "":
            return optimizer
        scheduler = create_scheduler(
            self.hparams.lr_scheduler,
            optimizer,
            self.hparams.lr_factor,
            self.hparams.lr_steps,
            self.hparams.epochs,
        )
        optimizers = {"optimizer": optimizer, "lr_scheduler": scheduler}
        logging.info(f"Using: {optimizers}")
        return optimizers

    def on_test_epoch_start(self) -> None:
        self.metrics = {
            step_name: {metric_name: metric() for metric_name, metric in self.set_metrics().items()}
            for step_name in ["train", "val", "test"]
        }
        return super().on_test_epoch_start()

    def save_model(self, save_path, file_name, file_extension=".ckpt"):
        path = save_path / (file_name + file_extension)
        try:
            torch.save(self, path)
            logging.info(f"Model saved to {str(path.resolve())}.")
        except Exception as e:
            logging.error(f"Cannot save model to path {str(path.resolve())}: {e}.")


@gin.configurable("DLPredictionWrapper")
class DLPredictionWrapper(DLWrapper):
    """Interface for Deep Learning models."""

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(
        self,
        loss=CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        run_mode: RunMode = RunMode.classification,
        input_shape=None,
        lr: float = 0.002,
        momentum: float = 0.9,
        lr_scheduler: Optional[str] = None,
        lr_factor: float = 0.99,
        lr_steps: Optional[List[int]] = None,
        epochs: int = 100,
        input_size: Tensor = None,
        initialization_method: str = "normal",
        pytorch_forecasting: bool = False,
        explain: list = [],
        XAI_metric: list = [],
        **kwargs,
    ):
        super().__init__(
            loss=loss,
            optimizer=optimizer,
            run_mode=run_mode,
            input_shape=input_shape,
            lr=lr,
            momentum=momentum,
            lr_scheduler=lr_scheduler,
            lr_factor=lr_factor,
            lr_steps=lr_steps,
            epochs=epochs,
            input_size=input_size,
            initialization_method=initialization_method,
            kwargs=kwargs,
        )
        self.output_transform = None
        self.loss_weights = None
        self.pytorch_forecasting = pytorch_forecasting
        self.explain = explain
        self.XAI_metric = XAI_metric

    def set_weight(self, weight, dataset):
        """Set the weight for the loss function."""
        if isinstance(weight, list):
            weight = FloatTensor(weight).to(self.device)
        elif weight == "balanced":
            weight = FloatTensor(dataset.get_balance()).to(self.device)

        self.loss_weights = weight

    def set_metrics(self, *args):
        """Set the evaluation metrics for the prediction model."""

        def softmax_binary_output_transform(output):
            with torch.no_grad():
                y_pred, y = output
                y_pred = torch.softmax(y_pred, dim=1)
                return y_pred[:, -1], y

        def softmax_multi_output_transform(output):
            with torch.no_grad():
                y_pred, y = output
                y_pred = torch.softmax(y_pred, dim=1)
                return y_pred, y

        # Output transform is not applied for contrib metrics, so we do our own.
        if self.run_mode == RunMode.classification:
            # Binary classification

            if self.logit.out_features == 2:
                self.output_transform = softmax_binary_output_transform
                metrics = DLMetrics.BINARY_CLASSIFICATION_TORCHMETRICS

            else:
                # Multiclass classification
                self.output_transform = softmax_multi_output_transform
                metrics = DLMetrics.MULTICLASS_CLASSIFICATION
        # Regression
        elif self.run_mode == RunMode.regression:
            self.output_transform = lambda x: x
            metrics = DLMetrics.REGRESSION
        else:
            raise ValueError(f"Run mode {self.run_mode} not supported.")
        for key, value in metrics.items():
            # Torchmetrics metrics are not moved to the device by default
            if isinstance(value, torchmetrics.Metric):
                value.to(self.device)
        return metrics

    def step_fn(self, element, step_prefix=""):
        """Perform a step in the DL prediction model training loop.

        Args:
            element (object):
            step_prefix (str): Step type, by default: test, train, val.
        """

        if len(element) == 2:
            data, labels = element[0], (element[1]).to(self.device)
            if isinstance(data, list):
                for i in range(len(data)):
                    data[i] = data[i].float().to(self.device)
            else:
                data = data.float().to(self.device)
            mask = torch.ones_like(labels).bool()

        elif len(element) == 3:
            data, labels, mask = (
                element[0],
                element[1].to(self.device),
                element[2].to(self.device),
            )
            if isinstance(data, list):
                for i in range(len(data)):
                    data[i] = data[i].float().to(self.device)
            else:
                data = data.float().to(self.device)
        else:
            raise Exception("Loader should return either (data, label) or (data, label, mask)")
        out = self(data)

        # If aux_loss is present, it is returned as a tuple
        if len(out) == 2 and isinstance(out, tuple):
            out, aux_loss = out
        else:
            aux_loss = 0
        # Get prediction and target

        prediction = torch.masked_select(out, mask.unsqueeze(-1)).reshape(-1, out.shape[-1]).to(self.device)

        target = torch.masked_select(labels, mask).to(self.device)

        if prediction.shape[-1] > 1 and self.run_mode == RunMode.classification:
            # Classification task
            loss = self.loss(prediction, target.long(), weight=self.loss_weights.to(self.device)) + aux_loss
            # Returns torch.long because negative log likelihood loss
        elif self.run_mode == RunMode.regression:
            # Regression task
            loss = self.loss(prediction[:, 0], target.float()) + aux_loss
        else:
            raise ValueError(f"Run mode {self.run_mode} not yet supported. Please implement it.")
        transformed_output = self.output_transform((prediction, target))

        for key, value in self.metrics[step_prefix].items():
            if isinstance(value, torchmetrics.Metric):
                if key == "Binary_Fairness":
                    feature_names = key.feature_helper(self.trainer)
                    value.update(
                        transformed_output[0],
                        transformed_output[1],
                        data,
                        feature_names,
                    )
                else:
                    value.update(transformed_output[0], transformed_output[1])
            else:
                value.update(transformed_output)
        self.log(f"{step_prefix}/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss


@gin.configurable("DLPredictionPytorchForecastingWrapper")
class DLPredictionPytorchForecastingWrapper(DLPredictionWrapper):
    """Interface for Deep Learning models."""

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(
        self,
        loss=CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        run_mode: RunMode = RunMode.classification,
        input_shape=None,
        lr: float = 0.002,
        momentum: float = 0.9,
        lr_scheduler: Optional[str] = None,
        lr_factor: float = 0.99,
        lr_steps: Optional[List[int]] = None,
        epochs: int = 100,
        input_size: Tensor = None,
        initialization_method: str = "normal",
        pytorch_forecasting: bool = False,
        **kwargs,
    ):
        super().__init__(
            loss=loss,
            optimizer=optimizer,
            run_mode=run_mode,
            input_shape=input_shape,
            lr=lr,
            momentum=momentum,
            lr_scheduler=lr_scheduler,
            lr_factor=lr_factor,
            lr_steps=lr_steps,
            epochs=epochs,
            input_size=input_size,
            initialization_method=initialization_method,
            kwargs=kwargs,
        )

    def step_fn(self, element, step_prefix=""):
        """Perform a step in the DL prediction model training loop.

        Args:
            element (object):
            step_prefix (str): Step type, by default: test, train, val.
        """

        dic, labels = element[0], element[1][0]

        if isinstance(labels, list):
            labels = labels[-1]

        data = self.prep_data(dic)

        out = self(data)

        # If aux_loss is present, it is returned as a tuple
        if len(out) == 2 and isinstance(out, tuple):
            out, aux_loss = out
        else:
            aux_loss = 0
        # Get prediction and target

        prediction = out.to(self.device).squeeze(-1)

        target = labels.to(self.device)

        if prediction.shape[-1] > 1 and self.run_mode == RunMode.classification:
            # Classification task
            loss = self.loss(prediction, target.long(), weight=self.loss_weights.to(self.device)) + aux_loss
            # Returns torch.long because negative log likelihood loss
        elif self.run_mode == RunMode.regression:
            # Regression task

            loss = self.loss(prediction[:, 0], target.float()) + aux_loss
        else:
            raise ValueError(f"Run mode {self.run_mode} not yet supported. Please implement it.")
        transformed_output = self.output_transform((prediction, target))

        for key, value in self.metrics[step_prefix].items():
            if isinstance(value, torchmetrics.Metric):
                if key == "Binary_Fairness":
                    feature_names = self.metrics[step_prefix][key].feature_helper(self.trainer, step_prefix)
                    value.update(
                        transformed_output[0],
                        transformed_output[1].int(),
                        data,
                        feature_names,
                    )

                else:
                    value.update(transformed_output[0], transformed_output[1].int())
            else:
                value.update(transformed_output)
        self.log(f"{step_prefix}/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def prep_data_captum(self, x):
        """
        Prepares data to be fed into captum and generates baseline as well.

        Args:
            - x:Batch from dataloader
        Returns:
            - data:batch data in a tuple after being prepared
            - baselines:Basically zero tensors in the input
        """
        # captum requires gradient and float values

        data = (
            x["encoder_cat"].float().requires_grad_(),
            x["encoder_cont"].requires_grad_(),
            x["encoder_target"].float().requires_grad_(),
            x["encoder_lengths"].float().requires_grad_(),
            x["decoder_cat"].float().requires_grad_(),
            x["decoder_cont"].requires_grad_(),
            x["decoder_target"].float().requires_grad_(),
            x["decoder_lengths"].float().requires_grad_(),
            x["decoder_time_idx"].float().requires_grad_(),
            x["groups"].float().requires_grad_(),
            x["target_scale"].requires_grad_(),
        )
        baselines = (
            data[0].to(self.device),  # encoder_cat, no cat variables
            torch.zeros_like(data[1]).to(self.device),  # encoder_cont, set to zero
            torch.zeros_like(data[2]).to(self.device),  # encoder_target, set to zero
            data[3].to(self.device),  # encoder_lengths, leave unchanged
            data[4].to(self.device),  # decoder_cat, no cat variables
            torch.zeros_like(data[5]).to(self.device),  # decoder_cont, set to zero
            torch.zeros_like(data[6]).to(self.device),  # decoder_target, set to zero
            data[7].to(self.device),  # decoder_lengths, leave unchanged
            data[8].to(self.device),  # decoder_time_idx, unchanged
            data[9].to(self.device),  # groups, leave unchanged
            data[10].to(self.device),  # target_scale, leave unchanged
        )
        return data, baselines

    def plot_attributions(self, features_attrs, timestep_attrs, method_name, log_dir):
        """
        Plots the attribution values for features and timesteps.

        Args:
            - features_attrs: Array of feature attribution values.
            - timestep_attrs: Array of timestep attribution values.
            - method_name: Name of the attribution method.
            - log_dir: Directory to save the plots.
        Returns:
            Nothing
        """

        # Plot for feature attributions
        x_values = np.arange(1, len(features_attrs) + 1)
        plt.figure(figsize=(8, 6))
        plt.plot(
            x_values,
            features_attrs,
            marker="o",
            color="skyblue",
            linestyle="-",
            linewidth=2,
            markersize=8,
        )
        plt.xlabel("Feature")
        plt.ylabel("{} Attribution".format(method_name))
        plt.title("{} Attribution Values".format(method_name))
        plt.xticks(
            x_values,
            [
                "height",
                "weight",
                "age",
                "sex",
                "time_idx",
                "alb",
                "alp",
                "alt",
                "ast",
                "be",
                "bicar",
                "bili",
                "bili_dir",
                "bnd",
                "bun",
                "ca",
                "cai",
                "ck",
                "ckmb",
                "cl",
                "crea",
                "crp",
                "dbp",
                "fgn",
                "fio2",
                "glu",
                "hgb",
                "hr",
                "inr_pt",
                "k",
                "lact",
                "lymph",
                "map",
                "mch",
                "mchc",
                "mcv",
                "methb",
                "mg",
                "na",
                "neut",
                "o2sat",
                "pco2",
                "ph",
                "phos",
                "plt",
                "po2",
                "ptt",
                "resp",
                "sbp",
                "temp",
                "tnt",
                "urine",
                "wbc",
            ],
            rotation=90,
        )
        plt.tight_layout()
        plt.savefig(
            log_dir / "{}_attribution_features_plot.png".format(method_name),
            bbox_inches="tight",
        )

        # Plot for timestep attributions
        x_values = np.arange(1, len(timestep_attrs) + 1)
        plt.figure(figsize=(8, 6))
        plt.plot(
            x_values,
            timestep_attrs,
            marker="o",
            color="skyblue",
            linestyle="-",
            linewidth=2,
            markersize=8,
        )
        plt.xlabel("Time Step")
        plt.ylabel("{} Attribution".format(method_name))
        plt.title("{} Attribution Values".format(method_name))
        plt.xticks(x_values)
        plt.tight_layout()
        plt.savefig(log_dir / "{}_attribution_plot.png".format(method_name), bbox_inches="tight")

    def explantation(
        self,
        dataloader,
        method,
        log_dir=".",
        plot=False,
        XAI_metric=False,
        random_model=None,
        test_dataset=None,
        **kwargs,
    ):
        """
        Generic method to combine pytorchforecasting data loading , interpertations and captum to generate attributions

        Args:
            - dataloader: pytorchforecasting data loader
            - method: The explantation method chosen
            - log_dir= The directory to output the plots
            - plot= Determines if plots should be done or not
            - XAI_metric=Determines if XAI metrics should be calculated or not
        Returns:
            - all_attrs : Attribtuons of features per timesteps
            - features_attrs : Attribtuons of features averaged over timesteps
            - timestep_attrs : Attribtuons of timesteps averaged over features
            - f_ts_v_score: Faithfulness score for attribtuons of features per timesteps
            - f_ts_score: Faithfulness score for attribtuons of timesteps averaged over features
        """
        # Initialize lists to store attribution values for all instances
        all_attrs = []
        f_ts_score = []
        f_ts_v_score = []
        f_v_score = []
        r_score = []
        st_i_score = []
        st_o_score = []

        method_name = method if (method == "Random") or (method == "Attention") else (method.__name__)
        if (method_name == "Random") or (method_name == "Attention"):
            if method_name == "Attention":
                Interpertations = self.interpertations(dataloader=dataloader, log_dir=log_dir, plot=plot)
                timestep_attrs = Interpertations["attention"]
                features_attrs = Interpertations["static_variables"].tolist()
                features_attrs.extend(Interpertations["encoder_variables"].tolist())
                r_score = self.Data_Randomization(
                    x=None,
                    attribution=timestep_attrs,
                    explain_method=method,
                    random_model=random_model,
                    dataloader=dataloader,
                    method_name=method_name,
                )
                st_i_score, st_o_score = self.Relative_Stability(
                    x=None,
                    attribution=timestep_attrs,
                    explain_method=method,
                    method_name=method_name,
                    dataloader=dataloader,
                    **kwargs,
                )
            elif method_name == "Random":
                # Generate random attributions for baseline comparison
                all_attrs = np.random.normal(size=[64, 24, 53])
                features_attrs = all_attrs.mean(axis=(1))
                timestep_attrs = all_attrs.mean(axis=(2))
            if XAI_metric:
                for batch in dataloader:
                    for key, value in batch[0].items():
                        batch[0][key] = batch[0][key].to(self.device)
                    x = batch[0]

                    if method_name == "Random":
                        f_ts_v_score.append(
                            self.Faithfulness_Correlation(
                                x,
                                all_attrs,
                                pertrub="baseline",
                                feature_timestep=True,
                                subset_size=[4, 9],
                                nr_runs=100,
                            )
                        )
                        f_ts_score.append(
                            self.Faithfulness_Correlation(
                                x,
                                all_attrs,
                                pertrub="baseline",
                                time_step=True,
                                subset_size=4,
                                nr_runs=100,
                            )
                        )
                        f_v_score.append(
                            self.Faithfulness_Correlation(
                                x,
                                all_attrs,
                                pertrub="baseline",
                                feature=True,
                                subset_size=9,
                                nr_runs=100,
                            )
                        )

                        r_score.append(
                            self.Data_Randomization(
                                x,
                                attribution=all_attrs,
                                explain_method=method,
                                random_model=random_model,
                                method_name=method_name,
                            )
                        )
                        res1, res2 = self.Relative_Stability(
                            x,
                            all_attrs,
                            explain_method=method,
                            method_name=method_name,
                            dataloader=None,
                            **kwargs,
                        )
                        st_i_score.append(res1)
                        st_o_score.append(res2)
                    else:
                        f_ts_score.append(
                            self.Faithfulness_Correlation(
                                x,
                                timestep_attrs,
                                pertrub="baseline",
                                time_step=True,
                                subset_size=4,
                                nr_runs=100,
                            )
                        )
                        f_v_score.append(
                            self.Faithfulness_Correlation(
                                x,
                                features_attrs,
                                pertrub="baseline",
                                feature=True,
                                subset_size=9,
                                nr_runs=100,
                            )
                        )

            # Faithfulness score for attribtuons of features per timesteps
            f_ts_v_score = np.mean(f_ts_v_score)
            # Faithfulness score for attribtuons of timesteps averaged over features
            f_ts_score = np.mean(f_ts_score)
            f_v_score = np.mean(f_v_score)

            if method_name != "Attention":
                # r_score = (r_score - min_val) / (max_val - min_val)
                r_score = np.mean(r_score)
                st_i_score = np.max(st_i_score)
                st_o_score = np.max(st_o_score)
            return (
                all_attrs,
                features_attrs,
                timestep_attrs,
                f_ts_v_score,
                f_ts_score,
                f_v_score,
                r_score,
                st_i_score,
                st_o_score,
            )

        # Loop through the dataloader to compute attributions for all instances
        for batch in dataloader:
            for key, value in batch[0].items():
                batch[0][key] = batch[0][key].to(self.device)
            x = batch[0]

            data, baselines = self.prep_data_captum(x)

            # Initialize the explanation method
            explanation = (
                method(self.forward_captum, interpretable_model=SkLearnLasso(alpha=0.4))
                if method_name == "Lime"
                else method(self.forward_captum)
            )

            # Calculate attributions using the selected method
            if method is not captum.attr.Saliency:
                attr = explanation.attribute(data, baselines=baselines, **kwargs)
            else:
                attr = explanation.attribute(data, **kwargs)

            # Process and store the calculated attributions
            stacked_attr = (
                attr[1].cpu().detach().numpy()
                if method_name in ["Lime", "FeatureAblation"]
                else torch.stack(attr).cpu().detach().numpy()
            )
            if XAI_metric:
                f_ts_v_score.append(
                    self.Faithfulness_Correlation(
                        x,
                        stacked_attr,
                        pertrub="baseline",
                        feature_timestep=True,
                        subset_size=[4, 9],
                        nr_runs=100,
                    )
                )

                f_ts_score.append(
                    self.Faithfulness_Correlation(
                        x,
                        stacked_attr,
                        pertrub="baseline",
                        time_step=True,
                        subset_size=4,
                        nr_runs=100,
                    )
                )
                f_v_score.append(
                    self.Faithfulness_Correlation(
                        x,
                        stacked_attr,
                        pertrub="baseline",
                        feature=True,
                        subset_size=9,
                        nr_runs=100,
                    )
                )
                r_score.append(
                    self.Data_Randomization(
                        x,
                        attribution=stacked_attr,
                        explain_method=method,
                        random_model=random_model,
                        method_name=method_name,
                    )
                )

                res1, res2 = self.Relative_Stability(
                    x,
                    stacked_attr,
                    explain_method=method,
                    method_name=method_name,
                    dataloader=None,
                    **kwargs,
                )
                st_i_score.append(res1)
                st_o_score.append(res2)

            # aggregate over batch
            attr = np.mean(stacked_attr, axis=0)
            all_attrs.append(attr)
        # aggregate over all batches
        all_attrs = np.array(all_attrs).mean(axis=(0))
        # aggregate over all timesteps
        features_attrs = all_attrs.mean(axis=(0))
        # aggregate over all features
        timestep_attrs = all_attrs.mean(axis=(1))
        # Faithfulness score for attribtuons of features per timesteps
        f_ts_v_score = np.mean(f_ts_v_score)
        # Faithfulness score for attribtuons of timesteps averaged over features
        f_ts_score = np.mean(f_ts_score)
        # Faithfulness score for attribtuons of timesteps averaged over timesteps
        f_v_score = np.mean(f_v_score)

        # Random data score
        r_score = np.mean(r_score)
        st_i_score = np.max(st_i_score)
        st_o_score = np.max(st_o_score)

        if plot:
            log_dir_plots = log_dir / "plots"
            if not (log_dir_plots.exists()):
                log_dir_plots.mkdir(parents=True)
            # Plot attributions for features and timesteps

            self.plot_attributions(features_attrs, timestep_attrs, method_name, log_dir_plots)

        # Return computed attributions and metrics
        return (
            all_attrs,
            features_attrs,
            timestep_attrs,
            f_ts_v_score,
            f_ts_score,
            f_v_score,
            r_score,
            st_i_score,
            st_o_score,
        )
        # normalized_means = (means - means.min()) / (means.max() - means.min())

    def prep_data(self, x):
        """
        Prepares data for custom forward method

        Args:
            - x:Batch returned from dataloader
        Returns:
            data:Tuple consisting of the tensors of X in the format the forward method needs
        """
        data = (
            x["encoder_cat"],
            x["encoder_cont"],
            x["encoder_target"],
            x["encoder_lengths"],
            x["decoder_cat"],
            x["decoder_cont"],
            x["decoder_target"],
            x["decoder_lengths"],
            x["decoder_time_idx"],
            x["groups"],
            x["target_scale"],
        )
        return data

    def Faithfulness_Correlation(
        self,
        x,
        attribution,
        similarity_func=None,
        nr_runs=100,
        pertrub=None,
        subset_size=3,
        feature=False,
        time_step=False,
        feature_timestep=False,
    ):
        """
        Calculates faithfulness scores for captum attributions

        Args:
            - x:Batch input
            -attribution: attribution generated by captum,
            - similarity_func:function to determine similarity between sum of attributions and difference in prediction
            - nr_runs: How many times to repeat the experiment,
            - pertrub: What change to do to the input,
            - subset_size: The size of the subset of featrues to alter ,
            - feature: Determines if to calcualte faithfulness of feature attributions,
            - time_step: Determines if to calcualte faithfulness of timesteps attributions,
            - feature_timestep: Determines if to calcualte faithfulness of featrues per timesteps attributions,
        Returns:
            score: similarity score between sum of attributions and difference in prediction averaged over nr_runs

        Implementation of faithfulness correlation by Bhatt et al., 2020.

        The Faithfulness Correlation metric intend to capture an explanation's relative faithfulness
        (or 'fidelity') with respect to the model behaviour.

        Faithfulness correlation scores shows to what extent the predicted logits of each modified test point and
        the average explanation attribution for only the subset of features are (linearly) correlated, taking the
        average over multiple runs and test samples. The metric returns one float per input-attribution pair that
        ranges between -1 and 1, where higher scores are better.

        For each test sample, |S| features are randomly selected and replace them with baseline values (zero baseline
        or average of set). Thereafter, Pearson’s correlation coefficient between the predicted logits of each modified
        test point and the average explanation attribution for only the subset of features is calculated. Results is
        average over multiple runs and several test samples.
        This code is adapted from the quantus libray to suit our use case

        References:
            1) Umang Bhatt et al.: "Evaluating and aggregating feature-based model
            explanations." IJCAI (2020): 3016-3022.
            2)Hedström, Anna, et al. "Quantus: An explainable ai toolkit for
            responsible evaluation of neural network explanations and beyond."
            Journal of Machine Learning Research 24.34 (2023): 1-11.
        """

        def add_noise(x, indices, time_step, feature_timestep):
            noise = torch.randn_like(x["encoder_cont"])
            if time_step:
                idx0, idx1 = np.meshgrid(indices[0], indices[1], indexing="ij")

                with torch.no_grad():
                    x["encoder_cont"][idx0, idx1, :] += noise[idx0, idx1, :]

            elif feature:
                idx0, idx1 = np.meshgrid(indices[0], indices[1], indexing="ij")

                with torch.no_grad():
                    x["encoder_cont"][idx0, :, idx1] += noise[idx0, :, idx1]

            elif feature_timestep:
                idx0, idx1, idx2 = np.meshgrid(indices[0], indices[1], indices[2], indexing="ij")

                with torch.no_grad():
                    x["encoder_cont"][idx0, idx1, idx2] += noise[idx0, idx1, idx2]

        def apply_baseline(x, indices, time_step, feature_timestep):
            mask = torch.ones_like(x["encoder_cont"])
            if time_step:
                (
                    idx0,
                    idx1,
                ) = np.meshgrid(indices[0], indices[1], indexing="ij")

                mask[idx0, idx1, :] -= mask[idx0, idx1, :]
            elif feature:
                (
                    idx0,
                    idx1,
                ) = np.meshgrid(indices[0], indices[1], indexing="ij")

                mask[idx0, :, idx1] -= mask[idx0, :, idx1]

            elif feature_timestep:
                idx0, idx1, idx2 = np.meshgrid(indices[0], indices[1], indices[2], indexing="ij")

                mask[idx0, idx1, idx2] -= mask[idx0, idx1, idx2]

            with torch.no_grad():
                x["encoder_cont"] *= mask

        # Assuming 'attribution' is already a GPU tensor
        if not torch.is_tensor(attribution):
            attribution = torch.tensor(attribution).to(self.device)

        # Other initializations
        if similarity_func is None:
            similarity_func = correlation_spearman
        if pertrub is None:
            pertrub = "baseline"
        similarities = []

        # Assuming this is a method to prepare your data

        y_pred = self(self.prep_data(x)).detach()  # Keep on GPU
        pred_deltas = []
        att_sums = []

        for i_ix in range(nr_runs):
            if time_step:
                timesteps_idx = np.random.choice(24, subset_size, replace=False)
                patient_idx = np.random.choice(64, 1, replace=False)
                a_ix = [patient_idx, timesteps_idx]

            elif feature:
                feature_idx = np.random.choice(53, subset_size, replace=False)
                patient_idx = np.random.choice(64, 1, replace=False)
                a_ix = [patient_idx, feature_idx]
            elif feature_timestep:
                timesteps_idx = np.random.choice(24, subset_size[0], replace=False)
                feature_idx = np.random.choice(53, subset_size[1], replace=False)
                patient_idx = np.random.choice(64, 1, replace=False)
                a_ix = [patient_idx, timesteps_idx, feature_idx]

            # Apply perturbation
            if pertrub == "Noise":
                add_noise(x, a_ix, time_step, feature_timestep)
            elif pertrub == "baseline":
                apply_baseline(x, a_ix, time_step, feature_timestep)

            # Predict on perturbed input and calculate deltas
            y_pred_perturb = (self(self.prep_data(x))).detach()  # Keep on GPU

            if time_step:
                if attribution.size() == torch.Size([24]):
                    att_sums.append((attribution[timesteps_idx]).sum())
                else:
                    att_sums.append((attribution[patient_idx, :, :][:, timesteps_idx, :]).sum())
            elif feature:
                if len(attribution) == 53:
                    att_sums.append((attribution[feature_idx]).sum())
                else:
                    att_sums.append((attribution[patient_idx, :, :][:, :, feature_idx]).sum())
            elif feature_timestep:
                att_sums.append((attribution[patient_idx, :, :][:, timesteps_idx, :][:, :, feature_idx]).sum())

            pred_deltas.append((y_pred - y_pred_perturb)[patient_idx].item())
            # Convert to CPU for numpy operations

        pred_deltas_cpu = torch.tensor(pred_deltas).cpu().numpy()
        att_sums_cpu = torch.tensor(att_sums).cpu().numpy()

        similarities.append(similarity_func(pred_deltas_cpu, att_sums_cpu))

        score = np.nanmean(similarities)
        return score

    def Data_Randomization(
        self,
        x,
        attribution,
        explain_method,
        random_model,
        similarity_func=cosine,
        dataloader=None,
        method_name="",
        **kwargs,
    ):
        """

        Args:
            - x:Batch input
            -attribution: attribution
            - explain_method:function to generate explantations
            - random_model: Reference to model trained on random labels
            - similarity_func: Function to measure similiarity
            - dataloader:In case of using Attention as the explain method need to pass the dataloader instead of the batch ,
            - method_name: Name of the explantation

        Returns:
            score: similarity score between attributions of model trained on random data and model trained on real data

        Implementation of the Random Logit Metric by Sixt et al., 2020.

        The Random Logit Metric computes the distance between the original explanation and a reference explanation of
        a randomly chosen non-target class.
        This code is adapted from the quantus libray to suit our use case

        References:
            1) Leon Sixt et al.: "When Explanations Lie: Why Many Modified BP
            Attributions Fail." ICML (2020): 9046-9057.
            2)Hedström, Anna, et al. "Quantus: An explainable ai
            toolkit for responsible evaluation of neural network explanations and beyond."
              Journal of Machine Learning Research 24.34 (2023): 1-11.

        """

        if explain_method == "Attention":
            Attention_weights = random_model.interpertations(dataloader)
            attribution = attribution.cpu().numpy()
            min_val = np.min(attribution)
            max_val = np.max(attribution)

            attribution = (attribution - min_val) / (max_val - min_val)
            random_attr = Attention_weights["attention"].cpu().numpy()
            min_val = np.min(random_attr)
            max_val = np.max(random_attr)
            random_attr = (random_attr - min_val) / (max_val - min_val)
            score = similarity_func(random_attr, attribution)
        elif explain_method == "Random":
            score = similarity_func(np.random.normal(size=[64, 24, 53]).flatten(), attribution.flatten())
        else:
            data, baselines = self.prep_data_captum(x)

            explantation = explain_method(random_model.forward_captum)
            # Reformat attributions.
            if explain_method is not captum.attr.Saliency:
                attr = explantation.attribute(data, baselines=baselines, **kwargs)
            else:
                attr = explantation.attribute(data, **kwargs)

            # Process and store the calculated attributions
            random_attr = (
                attr[1].cpu().detach().numpy()
                if method_name in ["Lime", "FeatureAblation"]
                else torch.stack(attr).cpu().detach().numpy()
            )

            attribution = attribution.flatten()
            min_val = np.min(attribution)
            max_val = np.max(attribution)
            attribution = (attribution - min_val) / (max_val - min_val)
            random_attr = random_attr.flatten()
            min_val = np.min(random_attr)
            max_val = np.max(random_attr)
            random_attr = (random_attr - min_val) / (max_val - min_val)

            score = similarity_func(random_attr, attribution)
        return score

    def Relative_Stability(
        self,
        x,
        attribution,
        explain_method,
        method_name,
        dataloader=None,
        thershold=0.5,
        **kwargs,
    ):
        """
        Args:
               - x:Batch input
               -attribution: attribution
               - explain_method:function to generate explantations
               - method_name: Name of the explantation
               - dataloader:In case of using Attention as the explain method need to pass the dataloader instead of the batch ,


           Returns:
               RIS : relative distance between the explantation and the input
               ROS: relative distance between the explantation and the output


        References:
               1) `https://arxiv.org/pdf/2203.06877.pdf
               2)Hedström, Anna, et al. "Quantus: An explainable ai toolkit for responsible evaluation of neural network explanations and beyond." Journal of Machine Learning Research 24.34 (2023): 1-11.

        """

        def relative_stability_objective(x, xs, e_x, e_xs, eps_min=0.0001, input=False, device="cuda") -> torch.Tensor:
            """
            Computes relative input and output stabilities maximization objective
            as defined here :ref:`https://arxiv.org/pdf/2203.06877.pdf` by the authors.

            Args:

                x: Input tensor
                xs: perturbed tensor.
                e_x: Explanations for x.
                e_xs: Explanations for xs.
                eps_min:Value to avoid division by zero if needed
                input:Boolean to indicate if this is an input or an output
                device: the device to keep the tensors on

            Returns:

                ris_obj: Tensor
                    RIS maximization objective.
            """

            # Function to convert inputs to tensors if they are numpy arrays
            def to_tensor(input_array):
                if isinstance(input_array, np.ndarray):
                    return torch.tensor(input_array).to(device)
                return input_array.to(device)

            # Convert all inputs to tensors and move to GPU
            x, xs, e_x, e_xs = map(to_tensor, [x, xs, e_x, e_xs])

            if input:
                num_dim = x.ndim
            else:
                num_dim = e_x.ndim

            if num_dim == 3:

                def norm_function(arr):
                    return torch.norm(arr, dim=(-1, -2))

            elif num_dim == 2:

                def norm_function(arr):
                    return torch.norm(arr, dim=-1)

            else:

                def norm_function(arr):
                    return torch.norm(arr)

            nominator = (e_x - e_xs) / (e_x + (e_x == 0) * eps_min)
            nominator = norm_function(nominator)

            if input:
                denominator = x - xs
                denominator /= x + (x == 0) * eps_min
                denominator = norm_function(denominator)
                denominator += (denominator == 0) * eps_min
            else:
                denominator = torch.squeeze(x) - torch.squeeze(xs)
                denominator = torch.norm(denominator, dim=-1)
                denominator += (denominator == 0) * eps_min

            return nominator / denominator

        if not torch.is_tensor(attribution):
            attribution = torch.tensor(attribution).to(self.device)
        if explain_method == "Attention":
            y_pred = self.model.predict(dataloader)
            x_original = dataloader.dataset.data["reals"].clone()

            dataloader.dataset.add_noise()
            x_preturb = dataloader.dataset.data["reals"].clone()
            y_pred_preturb = self.model.predict(dataloader)
            Attention_weights = self.interpertations(dataloader)
            att_preturb = Attention_weights["attention"]
            # Calculate the absolute difference
            difference = torch.abs(y_pred_preturb - y_pred)

            # Find where the difference is less than or equal to a thershold
            close_indices = torch.nonzero(difference <= thershold).squeeze()
            RIS = relative_stability_objective(
                x_original[close_indices, :, :].detach(),
                x_preturb[close_indices, :, :].detach(),
                attribution,
                att_preturb,
                input=True,
            )

            ROS = relative_stability_objective(
                y_pred[close_indices],
                y_pred_preturb[close_indices],
                attribution,
                att_preturb,
                input=False,
            )

        else:
            y_pred = self(self.prep_data(x)).detach()
            x_original = x["encoder_cont"].detach().clone()

            with torch.no_grad():
                noise = torch.randn_like(x["encoder_cont"]) * 0.01
                x["encoder_cont"] += noise
            y_pred_preturb = self(self.prep_data(x)).detach()
            if explain_method == "Random":
                att_preturb = np.random.normal(size=[64, 24, 53])
                att_preturb = torch.tensor(att_preturb).to(self.device)
            else:
                data, baselines = self.prep_data_captum(x)

                explantation = explain_method(self.forward_captum)
                # Reformat attributions.
                if explain_method is not captum.attr.Saliency:
                    att_preturb = explantation.attribute(data, baselines=baselines, **kwargs)
                else:
                    att_preturb = explantation.attribute(data, **kwargs)

                # Process and store the calculated attributions
                att_preturb = (
                    att_preturb[1].detach()
                    if method_name in ["Lime", "FeatureAblation"]
                    else torch.stack(att_preturb).detach()
                )
            # Calculate the absolute difference
            difference = torch.abs(y_pred_preturb - y_pred)

            # Find where the difference is less than or equal to a thershold
            close_indices = torch.nonzero(difference <= thershold).squeeze()
            RIS = relative_stability_objective(
                x_original[close_indices, :, :].detach(),
                x["encoder_cont"][close_indices, :, :].detach(),
                attribution[close_indices, :, :],
                att_preturb[close_indices, :, :],
                input=True,
            )
            ROS = relative_stability_objective(
                y_pred[close_indices],
                y_pred_preturb[close_indices],
                attribution[close_indices, :, :],
                att_preturb[close_indices, :, :],
                input=False,
            )

        return np.max(RIS.cpu().numpy()).astype(np.float64), np.max(ROS.cpu().numpy()).astype(np.float64)


@gin.configurable("MLWrapper")
class MLWrapper(BaseModule, ABC):
    """Interface for prediction with traditional Scikit-learn-like Machine Learning models."""

    requires_backprop = False
    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(
        self,
        *args,
        run_mode=RunMode.classification,
        loss=log_loss,
        patience=10,
        mps=False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.scaler = None
        self.check_supported_runmode(run_mode)
        self.run_mode = run_mode
        self.loss = loss
        self.patience = patience
        self.mps = mps

    def set_metrics(self, labels):
        if self.run_mode == RunMode.classification:
            # Binary classification
            if len(np.unique(labels)) == 2:
                # if isinstance(self.model, lightgbm.basic.Booster):
                self.output_transform = lambda x: x[:, 1]
                self.label_transform = lambda x: x

                self.metrics = MLMetrics.BINARY_CLASSIFICATION
            # Multiclass classification
            else:
                # Todo: verify multiclass classification
                self.output_transform = lambda x: np.argmax(x, axis=-1)
                self.label_transform = lambda x: x
                self.metrics = MLMetrics.MULTICLASS_CLASSIFICATION

        # Regression
        else:
            if self.scaler is not None:  # We invert transform the labels and predictions if they were scaled.
                self.output_transform = lambda x: self.scaler.inverse_transform(x.reshape(-1, 1))
                self.label_transform = lambda x: self.scaler.inverse_transform(x.reshape(-1, 1))
            else:
                self.output_transform = lambda x: x
                self.label_transform = lambda x: x
            self.metrics = MLMetrics.REGRESSION

    def fit(self, train_dataset, val_dataset):
        """Fit the model to the training data."""
        train_rep, train_label = train_dataset.get_data_and_labels()
        val_rep, val_label = val_dataset.get_data_and_labels()

        self.set_metrics(train_label)

        if "class_weight" in self.model.get_params().keys():  # Set class weights
            self.model.set_params(class_weight=self.weight)

        val_loss = self.fit_model(train_rep, train_label, val_rep, val_label)

        train_pred = self.predict(train_rep)

        logging.debug(f"Model:{self.model}")
        self.log("train/loss", self.loss(train_label, train_pred), sync_dist=True)
        logging.debug(f"Train loss: {self.loss(train_label, train_pred)}")
        self.log("val/loss", val_loss, sync_dist=True)
        logging.debug(f"Val loss: {val_loss}")
        self.log_metrics(train_label, train_pred, "train")

    def fit_model(self, train_data, train_labels, val_data, val_labels):
        """Fit the model to the training data (default SKlearn syntax)"""
        self.model.fit(train_data, train_labels)
        val_loss = 0.0
        return val_loss

    def validation_step(self, val_dataset, _):
        val_rep, val_label = val_dataset.get_data_and_labels()
        val_rep, val_label = torch.from_numpy(val_rep).to(self.device), torch.from_numpy(val_label).to(self.device)
        self.set_metrics(val_label)

        val_pred = self.predict(val_rep)

        self.log_metrics("val/loss", self.loss(val_label, val_pred), sync_dist=True)
        logging.info(f"Val loss: {self.loss(val_label, val_pred)}")
        self.log_metrics(val_label, val_pred, "val")

    def test_step(self, dataset, _):
        test_rep, test_label = dataset
        test_rep, test_label = (
            test_rep.squeeze().cpu().numpy(),
            test_label.squeeze().cpu().numpy(),
        )
        self.set_metrics(test_label)
        test_pred = self.predict(test_rep)

        if self.mps:
            self.log(
                "test/loss",
                np.float32(self.loss(test_label, test_pred)),
                sync_dist=True,
            )
            self.log_metrics(np.float32(test_label), np.float32(test_pred), "test")
        else:
            self.log("test/loss", self.loss(test_label, test_pred), sync_dist=True)
            self.log_metrics(test_label, test_pred, "test")
        logging.debug(f"Test loss: {self.loss(test_label, test_pred)}")
        self.log_metrics(np.float32(test_label), np.float32(test_pred), "test")

    def predict(self, features):
        if self.run_mode == RunMode.regression:
            return self.model.predict(features)
        else:  # Classification: return probabilities
            return self.model.predict_proba(features)

    def log_metrics(self, label, pred, metric_type):
        """Log metrics to the PL logs."""

        self.log_dict(
            {
                # MPS dependent type casting
                f"{metric_type}/{name}": metric(self.label_transform(label), self.output_transform(pred))
                if not self.mps
                else metric(self.label_transform(label), self.output_transform(pred))
                # Fore very metric
                for name, metric in self.metrics.items()
                # Filter out metrics that return a tuple (e.g. precision_recall_curve)
                if not isinstance(
                    metric(self.label_transform(label), self.output_transform(pred)),
                    tuple,
                )
            },
            sync_dist=True,
        )

    def configure_optimizers(self):
        return None

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        del state["label_transform"]
        del state["output_transform"]
        return state

    def save_model(self, save_path, file_name, file_extension=".joblib"):
        path = save_path / (file_name + file_extension)
        try:
            dump(self.model, path)
            logging.info(f"Model saved to {str(path.resolve())}.")
        except Exception as e:
            logging.error(f"Cannot save model to path {str(path.resolve())}: {e}.")

    def set_model_args(self, model, *args, **kwargs):
        """Set hyperparameters of the model if they are supported by the model."""
        signature = inspect.signature(model.__init__).parameters
        possible_hps = list(signature.keys())
        # Get passed keyword arguments
        arguments = locals()["kwargs"]
        # Get valid hyperparameters
        hyperparams = {key: value for key, value in arguments.items() if key in possible_hps}
        logging.debug(f"Creating model with: {hyperparams}.")
        return model(**hyperparams)


@gin.configurable("ImputationWrapper")
class ImputationWrapper(DLWrapper):
    """Interface for imputation models."""

    requires_backprop = True
    _supported_run_modes = [RunMode.imputation]

    def __init__(
        self,
        loss: nn.modules.loss._Loss = MSELoss(),
        optimizer: Union[str, Optimizer] = "adam",
        run_mode: RunMode = RunMode.imputation,
        lr: float = 0.002,
        momentum: float = 0.9,
        lr_scheduler: Optional[str] = None,
        lr_factor: float = 0.99,
        lr_steps: Optional[List[int]] = None,
        input_size: Tensor = None,
        initialization_method: ImputationInit = ImputationInit.NORMAL,
        epochs=100,
        **kwargs: str,
    ) -> None:
        super().__init__(
            loss=loss,
            optimizer=optimizer,
            run_mode=run_mode,
            lr=lr,
            momentum=momentum,
            lr_scheduler=lr_scheduler,
            lr_factor=lr_factor,
            lr_steps=lr_steps,
            epochs=epochs,
            input_size=input_size,
            initialization_method=initialization_method,
            kwargs=kwargs,
        )
        self.check_supported_runmode(run_mode)
        self.run_mode = run_mode
        self.save_hyperparameters(ignore=["loss", "optimizer"])
        self.loss = loss
        self.optimizer = optimizer

    def set_metrics(self):
        return DLMetrics.IMPUTATION

    def init_weights(self, init_type="normal", gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
                if init_type == ImputationInit.NORMAL:
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == ImputationInit.XAVIER:
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == ImputationInit.KAIMING:
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_out")
                elif init_type == ImputationInit.ORTHOGONAL:
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError(f"Initialization method {init_type} is not implemented")
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif classname.find("BatchNorm2d") != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def on_fit_start(self) -> None:
        self.init_weights(self.hparams.initialization_method)
        for metrics in self.metrics.values():
            for metric in metrics.values():
                metric.reset()
        return super().on_fit_start()

    def step_fn(self, batch, step_prefix=""):
        amputated, amputation_mask, target, target_missingness = batch
        imputated = self(amputated, amputation_mask)
        amputated[amputation_mask > 0] = imputated[amputation_mask > 0]
        amputated[target_missingness > 0] = target[target_missingness > 0]

        loss = self.loss(amputated, target)
        self.log(f"{step_prefix}/loss", loss.item(), prog_bar=True)

        for metric in self.metrics[step_prefix].values():
            metric.update(
                (
                    torch.flatten(amputated.detach(), start_dim=1).clone(),
                    torch.flatten(target.detach(), start_dim=1).clone(),
                )
            )
        return loss

    def fit(self, train_dataset, val_dataset):
        raise NotImplementedError()

    def predict_step(self, data, amputation_mask=None):
        return self(data, amputation_mask)

    def predict(self, data):
        self.eval()
        data = data.to(self.device)
        data_missingness = torch.isnan(data).to(torch.float32)
        prediction = self.predict_step(data, data_missingness)
        data[data_missingness.bool()] = prediction[data_missingness.bool()]
        return data
