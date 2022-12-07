from torchmetrics import MeanSquaredError, MeanAbsoluteError, Accuracy
from typing import List, Optional, Union
from torch.nn import Module, MSELoss, CrossEntropyLoss
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
import inspect
import logging
import os
import pickle

import gin
import lightgbm
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
)

import torch
# from ignite.contrib.metrics import AveragePrecision, ROC_AUC, PrecisionRecallCurve, RocCurve

from torchmetrics import AveragePrecision, AUROC, PrecisionRecallCurve, ROC

# from ignite.metrics import MeanAbsoluteError, Accuracy
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import joblib

from icu_benchmarks.models.utils import save_model, load_model_state, create_optimizer, create_scheduler
from icu_benchmarks.models.metrics import BalancedAccuracy, MAE, CalibrationCurve
from icu_benchmarks.models.encoders import LSTMNet

from pytorch_lightning import LightningModule

gin.config.external_configurable(torch.nn.functional.nll_loss, module="torch.nn.functional")
gin.config.external_configurable(torch.nn.functional.cross_entropy, module="torch.nn.functional")
gin.config.external_configurable(torch.nn.functional.mse_loss, module="torch.nn.functional")

gin.config.external_configurable(lightgbm.LGBMClassifier, module="lightgbm")
gin.config.external_configurable(lightgbm.LGBMRegressor, module="lightgbm")
gin.config.external_configurable(LogisticRegression)


@gin.configurable("DLWrapper")
class DLWrapper(LightningModule):
    def __init__(
        self, loss=CrossEntropyLoss(), optimizer=torch.optim.Adam, loss_weights=None
    ):
        self.save_hyperparameters(ignore=["loss", "optimizer"])
        self.loss = loss
        self.optimizer = optimizer
        self.scaler = None

    def set_logdir(self, logdir):
        self.logdir = logdir

    def set_scaler(self, scaler):
        self.scaler = scaler

    def set_metrics(self):
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

        # output transform is not applied for contrib metrics so we do our own.
        if self.encoder.logit.out_features == 2:
            self.output_transform = softmax_binary_output_transform
            self.metrics = {
                "PR": AveragePrecision(),
                "AUC": AUROC(),
                "PR_Curve": PrecisionRecallCurve(),
                "ROC_Curve": ROC(),
                "Calibration_Curve": CalibrationCurve(),
            }

        elif self.encoder.logit.out_features == 1:
            self.output_transform = lambda x: x
            if self.scaler is not None:
                self.metrics = {"MAE": MAE(invert_transform=self.scaler.inverse_transform)}
            else:
                self.metrics = {"MAE": MeanAbsoluteError()}

        else:
            self.output_transform = softmax_multi_output_transform
            self.metrics = {"Accuracy": Accuracy(), "BalancedAccuracy": BalancedAccuracy()}

    def training_step(self, element, element_idx):

        if len(element) == 2:
            data, labels = element[0], element[1].to(self.device)
            if isinstance(data, list):
                for i in range(len(data)):
                    data[i] = data[i].float().to(self.device)
            else:
                data = data.float().to(self.device)
            mask = torch.ones_like(labels).bool()

        elif len(element) == 3:
            data, labels, mask = element[0], element[1].to(self.device), element[2].to(self.device)
            if isinstance(data, list):
                for i in range(len(data)):
                    data[i] = data[i].float().to(self.device)
            else:
                data = data.float().to(self.device)
        else:
            raise Exception("Loader should return either (data, label) or (data, label, mask)")
        out = self.encoder(data)
        if len(out) == 2 and isinstance(out, tuple):
            out, aux_loss = out
        else:
            aux_loss = 0
        out_flat = torch.masked_select(out, mask.unsqueeze(-1)).reshape(-1, out.shape[-1])
        label_flat = torch.masked_select(labels, mask)
        if out_flat.shape[-1] > 1:
            loss = self.loss(out_flat, label_flat.long(), weight=self.hparams.loss_weight) + aux_loss  # torch.long because NLL
        else:
            loss = self.loss(out_flat[:, 0], label_flat.float()) + aux_loss  # Regression task

        transformed_output = self.output_transform((out_flat, label_flat))
        for metric in self.metrics:
            metric.update(transformed_output)
        self.log("train/loss", on_step=False, on_epoch=True)

        return loss

    def on_train_epoch_end(self) -> None:
        self.log_dict({name: metric.compute() for name, metric in self.metrics.items()})
        for metric in self.metrics:
            metric.reset()
        return super().on_train_epoch_end()

    def validation_step(self, batch, batch_idx):
        ...

    @gin.configurable(module="DLWrapper")
    def train(
        self,
        train_dataset,
        val_dataset,
        weight,
        epochs=1000,
        batch_size=64,
        patience=10,
        min_delta=1e-4,
        save_weights=True,
    ):

        self.set_metrics()
        metrics = self.metrics

        torch.autograd.set_detect_anomaly(True)  # Check for any nans in gradients

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.n_worker,
            pin_memory=self.pin_memory,
            prefetch_factor=2,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.n_worker,
            pin_memory=self.pin_memory,
            prefetch_factor=2,
        )

        if isinstance(weight, list):
            weight = torch.FloatTensor(weight).to(self.device)
        elif weight == "balanced":
            weight = torch.FloatTensor(train_dataset.get_balance()).to(self.device)

        best_loss = float("inf")
        epoch_no_improvement = 0
        train_writer = SummaryWriter(os.path.join(self.logdir, "tensorboard", "train"))
        val_writer = SummaryWriter(os.path.join(self.logdir, "tensorboard", "val"))

        for epoch in range(epochs):
            # Train step
            train_loss, train_metric_results = self._do_training(train_loader, weight, metrics)

            # Validation step
            val_loss, val_metric_results = self.evaluate(val_loader, metrics, weight)

            # Early stopping
            if val_loss <= best_loss - min_delta:
                best_metrics = val_metric_results
                epoch_no_improvement = 0
                if save_weights:
                    self.save_weights(epoch, os.path.join(self.logdir, "model.torch"))
                best_loss = val_loss
                logging.info("Validation loss improved to {:.4f} ".format(val_loss))
            else:
                epoch_no_improvement += 1
                logging.info("No improvement on loss for {} epochs".format(epoch_no_improvement))
            if epoch_no_improvement >= patience:
                logging.info("No improvement on loss for more than {} epochs. We stop training".format(patience))
                break

            # Logging
            train_string = "Train Epoch:{}"
            train_values = [epoch + 1]
            for name, value in train_metric_results.items():
                if name.split("_")[-1] != "Curve":
                    train_string += ", " + name + ":{:.4f}"
                    train_values.append(value)
                    train_writer.add_scalar(name, value, epoch)
            train_writer.add_scalar("Loss", train_loss, epoch)

            val_string = "Val Epoch:{}"
            val_values = [epoch + 1]
            for name, value in val_metric_results.items():
                if name.split("_")[-1] != "Curve":
                    val_string += ", " + name + ":{:.4f}"
                    val_values.append(value)
                    val_writer.add_scalar(name, value, epoch)
            val_writer.add_scalar("Loss", val_loss, epoch)

            logging.info(train_string.format(*train_values))
            logging.info(val_string.format(*val_values))

        with open(os.path.join(self.logdir, "val_metrics.pkl"), "wb") as f:
            best_metrics["loss"] = best_loss
            pickle.dump(best_metrics, f)

        self.load_weights(os.path.join(self.logdir, "model.torch"))  # We load back the best iteration

    def test(self, dataset, weight):
        self.set_metrics()
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.n_worker, pin_memory=self.pin_memory)
        if isinstance(weight, list):
            weight = torch.FloatTensor(weight).to(self.device)
        test_loss, test_metrics = self.evaluate(test_loader, self.metrics, weight)

        with open(os.path.join(self.logdir, "test_metrics.pkl"), "wb") as f:
            test_metrics["loss"] = test_loss
            pickle.dump(test_metrics, f)
        for key, value in test_metrics.items():
            if isinstance(value, float):
                logging.info("Test {} :  {}".format(key, value))

    def evaluate(self, eval_loader, metrics, weight):
        self.encoder.eval()
        eval_loss = []

        with torch.no_grad():
            for v, elem in enumerate(eval_loader):
                loss, preds, target = self.step_fn(elem, weight)
                eval_loss.append(loss)
                for name, metric in metrics.items():
                    metric.update(self.output_transform((preds, target)))

            eval_metric_results = {}
            for name, metric in metrics.items():
                eval_metric_results[name] = metric.compute()
                metric.reset()
        eval_loss = float(sum(eval_loss) / (v + 1))
        return eval_loss, eval_metric_results

    def save_weights(self, epoch, save_path):
        save_model(self.encoder, self.optimizer, epoch, save_path)

    def load_weights(self, load_path):
        load_model_state(load_path, self.encoder, optimizer=self.optimizer)

@gin.configurable("MLWrapper")
class MLWrapper(object):
    def __init__(self, model=lightgbm.LGBMClassifier):
        self.model = model
        self.scaler = None

    def set_logdir(self, logdir):
        self.logdir = logdir

    def set_metrics(self, labels):
        if len(np.unique(labels)) == 2:
            if isinstance(self.model, lightgbm.basic.Booster):
                self.output_transform = lambda x: x
            else:
                self.output_transform = lambda x: x[:, 1]
            self.label_transform = lambda x: x

            self.metrics = {"PR": average_precision_score, "AUC": roc_auc_score}

        elif np.all(labels[:10].astype(int) == labels[:10]):
            self.output_transform = lambda x: np.argmax(x, axis=-1)
            self.label_transform = lambda x: x
            self.metrics = {"Accuracy": accuracy_score, "BalancedAccuracy": balanced_accuracy_score}

        else:
            if self.scaler is not None:  # We invert transform the labels and predictions if they were scaled.
                self.output_transform = lambda x: self.scaler.inverse_transform(x.reshape(-1, 1))
                self.label_transform = lambda x: self.scaler.inverse_transform(x.reshape(-1, 1))
            else:
                self.output_transform = lambda x: x
                self.label_transform = lambda x: x
            self.metrics = {"MAE": mean_absolute_error}

    def set_scaler(self, scaler):
        self.scaler = scaler

    @gin.configurable(module="MLWrapper")
    def train(self, train_dataset, val_dataset, weight, patience=10, save_weights=True):

        train_rep, train_label = train_dataset.get_data_and_labels()
        val_rep, val_label = val_dataset.get_data_and_labels()
        self.set_metrics(train_label)
        metrics = self.metrics

        if "class_weight" in self.model.get_params().keys():  # Set class weights
            self.model.set_params(class_weight=weight)

        if "eval_set" in inspect.getfullargspec(self.model.fit).args:  # This is lightgbm
            self.model.set_params(random_state=np.random.get_state()[1][0])
            self.model.fit(
                train_rep,
                train_label,
                eval_set=(val_rep, val_label),
                callbacks=[
                    lightgbm.early_stopping(patience, verbose=False),
                    lightgbm.log_evaluation(period=-1, show_stdv=False),
                ],
            )
            val_loss = list(self.model.best_score_["valid_0"].values())[0]
            model_type = "lgbm"
        else:
            model_type = "sklearn"
            self.model.fit(train_rep, train_label)
            val_loss = 0.0

        if "MAE" in self.metrics.keys():
            val_pred = self.model.predict(val_rep)
            train_pred = self.model.predict(train_rep)
        else:
            val_pred = self.model.predict_proba(val_rep)
            train_pred = self.model.predict_proba(train_rep)

        train_metric_results = {}
        train_string = ""
        train_values = []
        val_string = "Val Results: " + "loss" + ":{:.4f}"
        val_values = [val_loss]
        val_metric_results = {"loss": val_loss}
        for name, metric in metrics.items():
            train_metric_results[name] = metric(self.label_transform(train_label), self.output_transform(train_pred))
            val_metric_results[name] = metric(self.label_transform(val_label), self.output_transform(val_pred))
            train_string += "Train Results: " if len(train_string) == 0 else ", "
            train_string += name + ":{:.4f}"
            val_string += ", " + name + ":{:.4f}"
            train_values.append(train_metric_results[name])
            val_values.append(val_metric_results[name])
        logging.info(train_string.format(*train_values))
        logging.info(val_string.format(*val_values))

        if save_weights:
            if model_type == "lgbm":
                self.save_weights(save_path=os.path.join(self.logdir, "model.txt"), model_type=model_type)
            else:
                self.save_weights(save_path=os.path.join(self.logdir, "model.joblib"), model_type=model_type)

        with open(os.path.join(self.logdir, "val_metrics.pkl"), "wb") as f:
            pickle.dump(val_metric_results, f)

    def test(self, dataset, weight):
        test_rep, test_label = dataset.get_data_and_labels()
        self.set_metrics(test_label)
        if "MAE" in self.metrics.keys() or isinstance(self.model, lightgbm.basic.Booster):  # If we reload a LGBM classifier
            test_pred = self.model.predict(test_rep)
        else:
            test_pred = self.model.predict_proba(test_rep)
        test_string = ""
        test_values = []
        test_metric_results = {}
        for name, metric in self.metrics.items():
            test_metric_results[name] = metric(self.label_transform(test_label), self.output_transform(test_pred))
            test_string += "Test Results: " if len(test_string) == 0 else ", "
            test_string += name + ":{:.4f}"
            test_values.append(test_metric_results[name])

        logging.info(test_string.format(*test_values))
        with open(os.path.join(self.logdir, "test_metrics.pkl"), "wb") as f:
            pickle.dump(test_metric_results, f)

    def save_weights(self, save_path, model_type="lgbm"):
        if model_type == "lgbm":
            self.model.booster_.save_model(save_path)
        else:
            joblib.dump(self.model, save_path)

    def load_weights(self, load_path):
        if load_path.suffix == ".txt":
            self.model = lightgbm.Booster(model_file=load_path)
        else:
            with open(load_path, "rb") as f:
                self.model = joblib.load(f)


@gin.configurable("ImputationWrapper")
class ImputationWrapper(LightningModule):

    needs_training = True
    needs_fit = False

    def __init__(
        self,
        loss: _Loss = MSELoss(),
        optimizer: Union[str, Optimizer] = "adam",
        lr: float = 0.002,
        momentum: float = 0.9,
        lr_scheduler: Optional[str] = None,
        lr_factor: float = 0.99,
        lr_steps: Optional[List[int]] = None,
        epochs: int = 100,
        input_size: torch.Tensor = None,
        initialization_method: str = "normal",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["loss", "optimizer"])
        self.loss = loss
        self.optimizer = optimizer

        self.metrics = {
            "rmse": MeanSquaredError(squared=False),
            "mae": MeanAbsoluteError(),
        }

    def init_weights(self, init_type="normal", gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
                if init_type == "normal":
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == "xavier":
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == "kaiming":
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_out")
                elif init_type == "orthogonal":
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
                if hasattr(m, "bias") and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif classname.find("BatchNorm2d") != -1:
                torch.nn.init.normal_(m.weight.data, 1.0, gain)
                torch.nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def on_fit_start(self) -> None:
        self.metrics = {metric_name: metric.to(self.device) for metric_name, metric in self.metrics.items()}
        self.init_weights(self.hparams.initialization_method)
        return super().on_fit_start()

    def fit(self, input_data) -> None:
        raise NotImplementedError()

    def forward(self, amputated, amputation_mask) -> torch.Tensor:
        raise NotImplementedError()

    def training_step(self, batch):
        amputated, amputation_mask, target = batch
        imputated = self(amputated, amputation_mask)

        loss = self.loss(imputated, target)
        self.log("train/loss", loss.item(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        amputated, amputation_mask, target = batch
        imputated = self(amputated, amputation_mask)

        loss = self.loss(imputated, target)
        self.log("val/loss", loss.item(), prog_bar=True)

        for metric in self.metrics.values():
            metric.update(imputated, target)

    def on_validation_epoch_end(self) -> None:
        self.log_dict({f"val/{metric_name}": metric.compute() for metric_name, metric in self.metrics.items()})
        for metric in self.metrics.values():
            metric.reset()

    def on_test_epoch_start(self) -> None:
        self.metrics = {metric_name: metric.to(self.device) for metric_name, metric in self.metrics.items()}
        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx):

        amputated, amputation_mask, target = batch
        imputated = self(amputated, amputation_mask)

        loss = self.loss(imputated, target)
        self.log("test/loss", loss.item())

        for metric in self.metrics.values():
            metric.update(imputated, target)

    def on_test_epoch_end(self) -> None:
        self.log_dict({f"test/{metric_name}": metric.compute() for metric_name, metric in self.metrics.items()})
        for metric in self.metrics.values():
            metric.reset()

    def configure_optimizers(self):
        if isinstance(self.optimizer, str):
            optimizer = create_optimizer(self.optimizer, self, self.hparams.lr, self.hparams.momentum)
        else:
            optimizer = self.optimizer(self.parameters())
        scheduler = create_scheduler(
            self.hparams.lr_scheduler, optimizer, self.hparams.lr_factor, self.hparams.lr_steps, self.hparams.epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
