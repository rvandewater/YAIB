import logging
from typing import Dict, Any
from typing import List, Optional, Union
from torch.nn import MSELoss, CrossEntropyLoss
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
import inspect
import gin
import lightgbm
import numpy as np

from sklearn.metrics import mean_absolute_error

import torch
from ignite.exceptions import NotComputableError

from icu_benchmarks.models.constants import ImputationInit
from icu_benchmarks.models.utils import create_optimizer, create_scheduler

from pytorch_lightning import LightningModule

from icu_benchmarks.models.constants import MLMetrics, DLMetrics
from icu_benchmarks.models.metrics import MAE

gin.config.external_configurable(torch.nn.functional.nll_loss, module="torch.nn.functional")
gin.config.external_configurable(torch.nn.functional.cross_entropy, module="torch.nn.functional")
gin.config.external_configurable(torch.nn.functional.mse_loss, module="torch.nn.functional")


@gin.configurable("BaseModule")
class BaseModule(LightningModule):
    needs_training = False
    needs_fit = False

    weight = None
    metrics = {}
    trained_columns = None

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


@gin.configurable("DLWrapper")
class DLWrapper(BaseModule):
    needs_training = True
    needs_fit = False
    _metrics_warning_printed = set()

    def __init__(
        self,
        loss=CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        input_shape=None,
        lr: float = 0.002,
        momentum: float = 0.9,
        lr_scheduler: Optional[str] = None,
        lr_factor: float = 0.99,
        lr_steps: Optional[List[int]] = None,
        epochs: int = 100,
        input_size: torch.Tensor = None,
        initialization_method: str = "normal",
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["loss", "optimizer"])
        self.loss = loss
        self.optimizer = optimizer
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

    def finalize_step(self, step_prefix=""):
        try:
            self.log_dict(
                {
                    f"{step_prefix}/{name}": metric.compute()
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
        if isinstance(self.optimizer, str):
            optimizer = create_optimizer(self.optimizer, self, self.hparams.lr, self.hparams.momentum)
        else:
            optimizer = self.optimizer(self.parameters())

        if self.hparams.lr_scheduler is None:
            return optimizer
        scheduler = create_scheduler(
            self.hparams.lr_scheduler, optimizer, self.hparams.lr_factor, self.hparams.lr_steps, self.hparams.epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_test_epoch_start(self) -> None:
        self.metrics = {
            step_name: {metric_name: metric() for metric_name, metric in self.set_metrics().items()}
            for step_name in ["train", "val", "test"]
        }
        return super().on_test_epoch_start()


@gin.configurable("DLClassificationWrapper")
class DLClassificationWrapper(DLWrapper):
    def set_weight(self, weight, dataset):
        if isinstance(weight, list):
            weight = torch.FloatTensor(weight).to(self.device)
        elif weight == "balanced":
            weight = torch.FloatTensor(dataset.get_balance()).to(self.device)
        self.loss_weights = weight

    def set_metrics(self, *args):
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

        # Binary classification
        # output transform is not applied for contrib metrics so we do our own.
        if self.logit.out_features == 2:
            self.output_transform = softmax_binary_output_transform
            metrics = DLMetrics.BINARY_CLASSIFICATION

        # Regression
        elif self.logit.out_features == 1:
            self.output_transform = lambda x: x
            if self.scaler is not None:
                metrics = {"MAE": MAE(invert_transform=self.scaler.inverse_transform)}
            else:
                metrics = DLMetrics.REGRESSION

        # Multiclass classification
        else:
            self.output_transform = softmax_multi_output_transform
            metrics = DLMetrics.MULTICLASS_CLASSIFICATION
        return metrics

    def step_fn(self, element, step_prefix=""):
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
        out = self(data)
        if len(out) == 2 and isinstance(out, tuple):
            out, aux_loss = out
        else:
            aux_loss = 0
        prediction = torch.masked_select(out, mask.unsqueeze(-1)).reshape(-1, out.shape[-1]).to(self.device)
        target = torch.masked_select(labels, mask).to(self.device)
        if prediction.shape[-1] > 1:
            loss = (
                self.loss(prediction, target.long(), weight=self.loss_weights.to(self.device)) + aux_loss
            )  # torch.long because NLL
        else:
            loss = self.loss(prediction[:, 0], target.float()) + aux_loss  # Regression task

        transformed_output = self.output_transform((prediction, target))
        for metric in self.metrics[step_prefix].values():
            metric.update(transformed_output)
        self.log(f"{step_prefix}/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss


@gin.configurable("MLClassificationWrapper")
class MLClassificationWrapper(BaseModule):
    needs_training = False
    needs_fit = True

    def __init__(self, *args, model=None, patience=10, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.scaler = None

    def set_metrics(self, labels):
        # Binary classification
        if len(np.unique(labels)) == 2:
            if isinstance(self.model, lightgbm.basic.Booster):
                self.output_transform = lambda x: x
            else:
                self.output_transform = lambda x: x[:, 1]
            self.label_transform = lambda x: x

            self.metrics = MLMetrics.BINARY_CLASSIFICATION

        # Multiclass classification
        elif np.all(labels[:10].astype(int) == labels[:10]):
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
            self.metrics = {"MAE": mean_absolute_error}

    def fit(self, train_dataset, val_dataset):
        train_rep, train_label = train_dataset.get_data_and_labels()
        val_rep, val_label = val_dataset.get_data_and_labels()
        train_rep, train_label = torch.from_numpy(train_rep).to(self.device), torch.from_numpy(train_label).to(self.device)
        val_rep, val_label = torch.from_numpy(val_rep).to(self.device), torch.from_numpy(val_label).to(self.device)
        self.set_metrics(train_label)

        if "class_weight" in self.model.get_params().keys():  # Set class weights
            self.model.set_params(class_weight=self.weight)

        if "eval_set" in inspect.getfullargspec(self.model.fit).args:  # This is lightgbm
            self.model.set_params(random_state=np.random.get_state()[1][0])

            self.model.fit(
                train_rep.cpu().numpy(),
                train_label.cpu().numpy(),
                eval_set=(val_rep.cpu().numpy(), val_label.cpu().numpy()),
                callbacks=[
                    lightgbm.early_stopping(self.hparams.patience, verbose=False),
                    lightgbm.log_evaluation(period=-1, show_stdv=False),
                ],
            )
            val_loss = list(self.model.best_score_["valid_0"].values())[0]
        else:
            val_loss = 0.0
            self.model.fit(train_rep, train_label)

        if "MAE" in self.metrics.keys():
            train_pred = self.model.predict(train_rep)
        else:
            train_pred = self.model.predict_proba(train_rep)

        self.log("train/loss", 0.0, sync_dist=True)
        self.log("val/loss", val_loss, sync_dist=True)
        self.log_dict(
            {
                f"train/{name}": metric(self.label_transform(train_label), self.output_transform(train_pred))
                for name, metric in self.metrics.items()
                if "_Curve" not in name
            },
            sync_dist=True,
        )

    def validation_step(self, val_dataset, _):
        val_rep, val_label = val_dataset.get_data_and_labels()
        val_rep, val_label = torch.from_numpy(val_rep).to(self.device), torch.from_numpy(val_label).to(self.device)
        self.set_metrics(val_label)

        if "MAE" in self.metrics.keys():
            val_pred = self.model.predict(val_rep)
        else:
            val_pred = self.model.predict_proba(val_rep)

        self.log_dict(
            {
                f"val/{name}": metric(self.label_transform(val_label), self.output_transform(val_pred))
                for name, metric in self.metrics.items()
                if "_Curve" not in name
            },
            sync_dist=True,
        )

    def test_step(self, dataset, _):
        test_rep, test_label = dataset
        test_rep, test_label = test_rep.squeeze().cpu().numpy(), test_label.squeeze().cpu().numpy()
        # test_rep, test_label = torch.from_numpy(test_rep).to(self.device), torch.from_numpy(test_label).to(self.device)
        self.set_metrics(test_label)
        if "MAE" in self.metrics.keys() or isinstance(self.model, lightgbm.basic.Booster):  # If we reload a LGBM classifier
            test_pred = self.model.predict(test_rep)
        else:
            test_pred = self.model.predict_proba(test_rep)

        self.log("test/loss", 0.0, sync_dist=True)
        self.log_dict(
            {
                f"test/{name}": metric(self.label_transform(test_label), self.output_transform(test_pred))
                for name, metric in self.metrics.items()
                if "_Curve" not in name
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


@gin.configurable("ImputationWrapper")
class ImputationWrapper(DLWrapper):
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
        input_size: torch.Tensor = None,
        initialization_method: ImputationInit = ImputationInit.NORMAL,
        **kwargs: str,
    ) -> None:
        super().__init__()
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
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == ImputationInit.XAVIER:
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == ImputationInit.KAIMING:
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_out")
                elif init_type == ImputationInit.ORTHOGONAL:
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError(f"Initialization method {init_type} is not implemented")
                if hasattr(m, "bias") and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif classname.find("BatchNorm2d") != -1:
                torch.nn.init.normal_(m.weight.data, 1.0, gain)
                torch.nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def on_fit_start(self) -> None:
        self.init_weights(self.hparams.initialization_method)
        for metrics in self.metrics.values():
            for metric in metrics.values():
                metric.reset()
        logging.info("IMPUTATION METRICS RESET.")
        return super().on_fit_start()

    def step_fn(self, batch, step_prefix=""):
        amputated, amputation_mask, target = batch
        imputated = self(amputated, amputation_mask)
        amputated[amputation_mask > 0] = imputated[amputation_mask > 0]

        loss = self.loss(amputated, target)
        self.log(f"{step_prefix}/loss", loss.item(), prog_bar=True)

        for metric in self.metrics[step_prefix].values():
            metric.update(
                (torch.flatten(amputated.detach(), start_dim=1).clone(), torch.flatten(target.detach(), start_dim=1).clone())
            )
        return loss

    def fit(self, train_dataset, val_dataset):
        raise NotImplementedError()

    def predict_step(self, data, amputation_mask=None):
        return self(data, amputation_mask)

    def predict(self, data):
        self.eval()
        data = data.to(self.device)
        data_missingness = torch.isnan(data)
        prediction = self.predict_step(data, data_missingness)
        data[data_missingness] = prediction[data_missingness]
        return data
