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
gin.config.external_configurable(nn.functional.nll_loss, module="torch.nn.functional")
gin.config.external_configurable(
    nn.functional.cross_entropy, module="torch.nn.functional"
)
gin.config.external_configurable(nn.functional.mse_loss, module="torch.nn.functional")

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
            raise ValueError(
                f"Runmode {runmode} not supported for {self.__class__.__name__}"
            )
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
            self.log_dict(
                {
                    f"{step_prefix}/{name}": (
                        np.float32(metric.compute())
                        if isinstance(metric.compute(), np.float64)
                        else metric.compute()
                    )
                    for name, metric in self.metrics[step_prefix].items()
                    if "_Curve" not in name
                },
                sync_dist=True,
            )
            for metric in self.metrics[step_prefix].values():
                metric.reset()
        except (NotComputableError, ValueError):
            if step_prefix not in self._metrics_warning_printed:
                self._metrics_warning_printed.add(step_prefix)
                logging.warning(f"Metrics for {step_prefix} not computable")
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
            step_name: {
                metric_name: metric()
                for metric_name, metric in self.set_metrics().items()
            }
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
                metrics = DLMetrics.BINARY_CLASSIFICATION
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
        """
        if isinstance(element[0], OrderedDict):  # check if the data loader is the one for the TFT nvidia implementation
            data, mask = element[0], element[1].to(self.device)

            for key, value in data.items():
                value = value.float().to(self.device)

                if value.shape[-1] == 1:
                    value = value.squeeze(-1)
                if value.dim() == 3:
                    value = value.permute(0, 2, 1)
                if key == "target":
                    labels = value.squeeze()
                data[key] = value
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
            raise Exception(
                "Loader should return either (data, label) or (data, label, mask)"
            )
        out = self(data)

        # If aux_loss is present, it is returned as a tuple
        if len(out) == 2 and isinstance(out, tuple):
            out, aux_loss = out
        else:
            aux_loss = 0
        # Get prediction and target

        prediction = (
            torch.masked_select(out, mask.unsqueeze(-1))
            .reshape(-1, out.shape[-1])
            .to(self.device)
        )

        target = torch.masked_select(labels, mask).to(self.device)

        if prediction.shape[-1] > 1 and self.run_mode == RunMode.classification:
            # Classification task
            loss = (
                self.loss(
                    prediction, target.long(), weight=self.loss_weights.to(self.device)
                )
                + aux_loss
            )
            # Returns torch.long because negative log likelihood loss
        elif self.run_mode == RunMode.regression:
            # Regression task
            loss = self.loss(prediction[:, 0], target.float()) + aux_loss
        else:
            raise ValueError(
                f"Run mode {self.run_mode} not yet supported. Please implement it."
            )
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
        self.log(
            f"{step_prefix}/loss", loss, on_step=False, on_epoch=True, sync_dist=True
        )
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
        data = (
            dic["encoder_cat"],
            dic["encoder_cont"],
            dic["encoder_target"],
            dic["encoder_lengths"],
            dic["decoder_cat"],
            dic["decoder_cont"],
            dic["decoder_target"],
            dic["decoder_lengths"],
            dic["decoder_time_idx"],
            dic["groups"],
            dic["target_scale"],
        )

        mask = torch.ones_like(labels).bool()

        out = self(data)

        # If aux_loss is present, it is returned as a tuple
        if len(out) == 2 and isinstance(out, tuple):
            out, aux_loss = out
        else:
            aux_loss = 0
        # Get prediction and target

        prediction = (
            torch.masked_select(out, mask.unsqueeze(-1))
            .reshape(-1, out.shape[-1])
            .to(self.device)
        )

        target = torch.masked_select(labels, mask).to(self.device)

        if prediction.shape[-1] > 1 and self.run_mode == RunMode.classification:
            # Classification task
            loss = (
                self.loss(
                    prediction, target.long(), weight=self.loss_weights.to(self.device)
                )
                + aux_loss
            )
            # Returns torch.long because negative log likelihood loss
        elif self.run_mode == RunMode.regression:
            # Regression task
            loss = self.loss(prediction[:, 0], target.float()) + aux_loss
        else:
            raise ValueError(
                f"Run mode {self.run_mode} not yet supported. Please implement it."
            )
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
        self.log(
            f"{step_prefix}/loss", loss, on_step=False, on_epoch=True, sync_dist=True
        )
        return loss

    def explantation_captum(self, test_loader, log_dir, method):
        # Initialize lists to store attribution values for all instances
        all_attrs = []

        # Loop through the test_loader to compute attributions for all instances
        for batch in test_loader:
            for key, value in batch[0].items():
                batch[0][key] = batch[0][key].to(self.device)
            x = batch[0]
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
            target = self(data)
            print(target.shape)
            baselines = (
                torch.zeros_like(data[0]).to(self.device),  # encoder_cat, set to zero
                torch.zeros_like(data[1]).to(self.device),  # encoder_cont, set to zero
                torch.zeros_like(data[2]).to(self.device),  # encoder_target, set to zero
                data[3].to(self.device),  # encoder_lengths, leave unchanged
                torch.zeros_like(data[4]).to(self.device),  # decoder_cat, set to zero
                torch.zeros_like(data[5]).to(self.device),  # decoder_cont, set to zero
                torch.zeros_like(data[6]).to(self.device),  # decoder_target, set to zero
                data[7].to(self.device),  # decoder_lengths, leave unchanged
                torch.zeros_like(data[8]).to(self.device),  # decoder_time_idx, set to zero
                data[9].to(self.device),  # groups, leave unchanged
                data[10].to(self.device),  # target_scale, leave unchanged
            )

            explantation = method(self.forward_captum)
            # Reformat attributions.
            attr, delta = explantation.attribute(
                data, target=target, return_convergence_delta=True, baselines=baselines, n_steps=20
            )
            # Convert attributions to numpy array and append to the list
            all_attrs.append(attr[0].cpu().detach().numpy())

        # Concatenate a‚ttribution values for all instances along the batch dimension
        all_attrs = np.concatenate(all_attrs, axis=0)
        means_feature = all_attrs.mean(axis=(0, 1))

        # Compute mean along the batch dimension
        means = all_attrs.mean(axis=(0, 2))
        # Normalize the means values to range [0, 1]
        normalized_means = (means - means.min()) / (means.max() - means.min())

        # Create x values (assuming you want a simple sequential x-axis)
        # Assuming you have 24 values
        x_values = np.arange(1, 57)
        # Plotting the featrue means
        plt.figure(figsize=(8, 6))
        plt.plot(
            x_values,
            means_feature,
            marker="o",
            color="skyblue",
            linestyle="-",
            linewidth=2,
            markersize=8,
        )
        plt.xlabel("Time Step")
        plt.ylabel("Normalized Attribution")
        plt.title("Attribution Values")
        plt.xticks(x_values)  # Set x-ticks to match the number of features
        plt.tight_layout()
        plt.savefig(log_dir / "attribution_features_plot.png", bbox_inches="tight")
        plt.figure(figsize=(8, 6))
        x_values = np.arange(1, 25)
        plt.plot(
            x_values,
            normalized_means,
            marker="o",
            color="skyblue",
            linestyle="-",
            linewidth=2,
            markersize=8,
        )
        plt.xlabel("Time Step")
        plt.ylabel("Normalized Attribution")
        plt.title("Attribution Values")
        plt.xticks(x_values)  # Set x-ticks to match the number of features
        plt.tight_layout()
        plt.savefig(log_dir / "attribution_plot.png", bbox_inches="tight")
        return means

    def Faithfulness_Correlation(self, test_loader, attribution, nr_runs=100, pertrub=None, subset_size=4):
        """
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
        2)Hedström, Anna, et al. "Quantus: An explainable ai toolkit for responsible evaluation of neural network explanations and beyond." Journal of Machine Learning Research 24.34 (2023): 1-11.
    """

        if torch.is_tensor(attribution):
            # Convert the tensor to a NumPy array
            example_numpy_array = attribution.cpu().detach().numpy()
        if pertrub == None:
            pertrub = "baseline"
        similarities = []
        for batch in test_loader:

            for key, value in batch[0].items():

                batch[0][key] = batch[0][key].to(self.device)
            x = batch[0]
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

            y_pred = self(data).detach().cpu().numpy()
            pred_deltas = []
            att_sums = []
            for i_ix in range(nr_runs):
                # Randomly mask by subset size.
                a_ix = np.random.choice(x["encoder_cont"].shape[1], subset_size, replace=False)

                # Move a_ix_tensor to the same device as mask

                if pertrub == "Noise":
                    # add normal noise to input
                    noise = torch.randn_like(x["encoder_cont"])

                    x["encoder_cont"][:, a_ix, :] += noise[:, a_ix, :]
                elif pertrub == "baseline":
                    # Create a mask tensor with zeros at specified time steps and ones everywhere else
                    # pytorch bug need to change to cpu for next step and then revert
                    mask = torch.ones_like(x["encoder_cont"]).cpu()

                    mask[:, a_ix, :] = 0
                    mask = mask.to(x["encoder_cont"].device)

                    x["encoder_cont"] = x["encoder_cont"] * mask

                # Predict on perturbed input x.
                y_pred_perturb = self(data).detach().cpu().numpy()

                pred_deltas.append((y_pred - y_pred_perturb).mean(axis=(0, 2)))

                # Sum attributions of the random subset.

                att_sums.append(np.sum(attribution[a_ix]))
            correlation_matrix = np.corrcoef(pred_deltas, att_sums, rowvar=False)

            # Get the correlation coefficient from the correlation matrix
            pearson_correlation = correlation_matrix[0, 1]
            similarities.append(pearson_correlation)
        return np.nanmean(similarities)

    def Data_Randomization(self, test_loader, attribution, explain_method, nr_runs=100, similarity=None):
        """
        Implementation of the Random Logit Metric by Sixt et al., 2020.

        The Random Logit Metric computes the distance between the original explanation and a reference explanation of
        a randomly chosen non-target class.
        This code is adapted from the quantus libray to suit our use case

        References:
            1) Leon Sixt et al.: "When Explanations Lie: Why Many Modified BP
            Attributions Fail." ICML (2020): 9046-9057.
            2)Hedström, Anna, et al. "Quantus: An explainable ai toolkit for responsible evaluation of neural network explanations and beyond." Journal of Machine Learning Research 24.34 (2023): 1-11.

        """
        a_perturbed = []
        for batch in test_loader:

            for key, value in batch[0].items():

                batch[0][key] = batch[0][key].to(self.device)
            x = batch[0]
            y = batch[1][0]
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
            y_off = np.array(
                [
                    np.random.choice(
                        [y_ for y_ in list(np.arange(0, self.num_classes)) if y_ != y]
                    )
                ]
            )
            baselines = (
                torch.zeros_like(data[0]).to(self.device),  # encoder_cat, set to zero
                torch.zeros_like(data[1]).to(self.device),  # encoder_cont, set to zero
                torch.zeros_like(data[2]).to(self.device),  # encoder_target, set to zero
                data[3].to(self.device),  # encoder_lengths, leave unchanged
                torch.zeros_like(data[4]).to(self.device),  # decoder_cat, set to zero
                torch.zeros_like(data[5]).to(self.device),  # decoder_cont, set to zero
                torch.zeros_like(data[6]).to(self.device),  # decoder_target, set to zero
                data[7].to(self.device),  # decoder_lengths, leave unchanged
                torch.zeros_like(data[8]).to(self.device),  # decoder_time_idx, set to zero
                data[9].to(self.device),  # groups, leave unchanged
                data[10].to(self.device),  # target_scale, leave unchanged
            )

            explantation = explain_method(self.forward_captum)
            # Reformat attributions.
            attr, delta = explantation.attribute(
                data, target=y_off, return_convergence_delta=True, baselines=baselines, n_steps=20
            )
            # Convert attributions to numpy array and append to the list
            a_perturbed.append(attr[0].cpu().detach().numpy())
        a_perturbed = np.concatenate(a_perturbed, axis=0)
        a_perturbed = a_perturbed.mean(axis=(0, 1))

        # Compute mean along the batch dimension
        a_perturbed = a_perturbed.mean(axis=(0, 2))
        # Normalize the means values to range [0, 1]
        normalized_a_perturbed = (a_perturbed - a_perturbed.min()) / (a_perturbed.max() - a_perturbed.min())
        correlation_matrix = np.corrcoef(normalized_a_perturbed, attribution, rowvar=False)

        # Get the correlation coefficient from the correlation matrix
        pearson_correlation = correlation_matrix[0, 1]
        return pearson_correlation


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
            if (
                self.scaler is not None
            ):  # We invert transform the labels and predictions if they were scaled.
                self.output_transform = lambda x: self.scaler.inverse_transform(
                    x.reshape(-1, 1)
                )
                self.label_transform = lambda x: self.scaler.inverse_transform(
                    x.reshape(-1, 1)
                )
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
        val_rep, val_label = torch.from_numpy(val_rep).to(
            self.device
        ), torch.from_numpy(val_label).to(self.device)
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
                f"{metric_type}/{name}": metric(
                    self.label_transform(label), self.output_transform(pred)
                )
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
        hyperparams = {
            key: value for key, value in arguments.items() if key in possible_hps
        }
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
            if hasattr(m, "weight") and (
                classname.find("Conv") != -1 or classname.find("Linear") != -1
            ):
                if init_type == ImputationInit.NORMAL:
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == ImputationInit.XAVIER:
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == ImputationInit.KAIMING:
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_out")
                elif init_type == ImputationInit.ORTHOGONAL:
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError(
                        f"Initialization method {init_type} is not implemented"
                    )
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
