import inspect
import json
import logging
from pathlib import Path

import gin
import joblib
import lightgbm
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from icu_benchmarks.models.metric_constants import MLMetrics, DLMetrics
from icu_benchmarks.models.encoders import LSTMNet
from icu_benchmarks.models.metrics import MAE
from icu_benchmarks.models.utils import save_model, load_model_state, log_table_row, JsonNumpyEncoder

gin.config.external_configurable(torch.nn.functional.nll_loss, module="torch.nn.functional")
gin.config.external_configurable(torch.nn.functional.cross_entropy, module="torch.nn.functional")
gin.config.external_configurable(torch.nn.functional.mse_loss, module="torch.nn.functional")

gin.config.external_configurable(lightgbm.LGBMClassifier, module="lightgbm")
gin.config.external_configurable(lightgbm.LGBMRegressor, module="lightgbm")
gin.config.external_configurable(LogisticRegression)


def pick_device_config(hint=None):
    if (hint == "cuda" or hint is None) and torch.cuda.is_available():
        device = torch.device("cuda:0")
        pin_memory = True
        n_worker = 1
    elif (hint == "mps" or hint is None) and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        pin_memory = True
        n_worker = 1
    else:
        device = torch.device("cpu")
        pin_memory = False
        n_worker = 8
    return device, pin_memory, n_worker


@gin.configurable("DLWrapper")
class DLWrapper(object):
    def __init__(self, encoder=LSTMNet, loss=torch.nn.functional.cross_entropy, optimizer_fn=torch.optim.Adam, device=None):
        device, pin_memory, n_worker = pick_device_config(device)

        self.device = device
        logging.info(f"Model will be trained using {device}")
        self.pin_memory = pin_memory
        self.n_worker = n_worker

        self.model = encoder
        self.model.to(device)
        self.loss = loss
        self.optimizer = optimizer_fn(self.model.parameters())
        self.scaler = None

    def set_log_dir(self, log_dir: Path):
        self.log_dir = log_dir

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

        # Binary classification
        # output transform is not applied for contrib metrics so we do our own.
        if self.model.logit.out_features == 2:
            self.output_transform = softmax_binary_output_transform
            self.metrics = DLMetrics.BINARY_CLASSIFICATION

        # Regression
        elif self.model.logit.out_features == 1:
            self.output_transform = lambda x: x
            if self.scaler is not None:
                self.metrics = {"MAE": MAE(invert_transform=self.scaler.inverse_transform)}
            else:
                self.metrics = DLMetrics.REGRESSION

        # Multiclass classification
        else:
            self.output_transform = softmax_multi_output_transform
            self.metrics = DLMetrics.MULTICLASS_CLASSIFICATION

    def step_fn(self, element, loss_weight=None):

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
        out = self.model(data)
        if len(out) == 2 and isinstance(out, tuple):
            out, aux_loss = out
        else:
            aux_loss = 0
        out_flat = torch.masked_select(out, mask.unsqueeze(-1)).reshape(-1, out.shape[-1])
        label_flat = torch.masked_select(labels, mask)
        if out_flat.shape[-1] > 1:
            loss = self.loss(out_flat, label_flat.long(), weight=loss_weight) + aux_loss  # torch.long because NLL
        else:
            loss = self.loss(out_flat[:, 0], label_flat.float()) + aux_loss  # Regression task

        return loss, out_flat, label_flat

    def _do_training(self, train_loader, weight, metrics):
        # Training epoch
        self.model.train()
        agg_train_loss = 0
        for elem in tqdm(train_loader, leave=False):
            loss, preds, target = self.step_fn(elem, weight)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            agg_train_loss += loss
            for name, metric in metrics.items():
                metric.update(self.output_transform((preds, target)))

        train_metric_results = {}
        for name, metric in metrics.items():
            train_metric_results[name] = metric.compute()
            metric.reset()
        train_loss = float(agg_train_loss / len(train_loader))
        return train_loss, train_metric_results

    @gin.configurable(module="DLWrapper")
    def train(
        self,
        train_dataset,
        val_dataset,
        weight,
        seed,
        epochs=1000,
        batch_size=64,
        patience=10,
        min_delta=1e-4,
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
        train_writer = SummaryWriter(self.log_dir / "tensorboard" / "train")
        val_writer = SummaryWriter(self.log_dir / "tensorboard" / "val")

        table_header = ["EPOCH", "SPLIT", "METRICS", "COMMENT"]
        widths = [5, 5, 25, 50]
        log_table_row(table_header, widths=widths)
        disable_tqdm = logging.getLogger().isEnabledFor(logging.INFO)
        for epoch in trange(epochs, leave=False, disable=disable_tqdm):
            # Train step
            train_loss, train_metric_results = self._do_training(train_loader, weight, metrics)

            # Validation step
            val_loss, val_metric_results = self.evaluate(val_loader, metrics, weight)

            # Early stopping
            if val_loss <= best_loss - min_delta:
                best_metrics = val_metric_results
                epoch_no_improvement = 0
                self.save_weights(epoch, self.log_dir / "model.torch")
                best_loss = val_loss
                comment = "Validation loss improved to {:.4f} ".format(val_loss)
            else:
                epoch_no_improvement += 1
                comment = "No improvement on loss for {} epochs".format(epoch_no_improvement)
            if epoch_no_improvement >= patience:
                logging.info("No improvement on loss for more than {} epochs. We stop training".format(patience))
                break

            # Logging
            test_metric_strings = []
            for name, value in train_metric_results.items():
                if isinstance(value, np.float):
                    test_metric_strings.append(f"{name}: {value:.4f}")
                    train_writer.add_scalar(name, value, epoch)
            train_writer.add_scalar("Loss", train_loss, epoch)

            val_metric_strings = []
            for name, value in val_metric_results.items():
                if isinstance(value, np.float):
                    val_metric_strings.append(f"{name}: {value:.4f}")
                    val_writer.add_scalar(name, value, epoch)
            val_writer.add_scalar("Loss", val_loss, epoch)

            log_table_row([epoch, "Train", ", ".join(test_metric_strings), ""], widths=widths)
            log_table_row([epoch, "Val", ", ".join(val_metric_strings), comment], widths=widths)

        best_metrics["loss"] = best_loss

        with open(self.log_dir / "best_metrics.json", "w") as f:
            json.dump(best_metrics, f, cls=JsonNumpyEncoder)

        self.load_weights(self.log_dir / "model.torch")  # We load back the best iteration

    def test(self, dataset, weight, seed):
        self.set_metrics()
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.n_worker, pin_memory=self.pin_memory)
        if isinstance(weight, list):
            weight = torch.FloatTensor(weight).to(self.device)
        test_loss, test_metrics = self.evaluate(test_loader, self.metrics, weight)

        test_metrics["loss"] = test_loss
        with open(self.log_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, cls=JsonNumpyEncoder)

        for key, value in test_metrics.items():
            if isinstance(value, float):
                logging.info("Test {}: {}".format(key, value))

        return test_loss

    def evaluate(self, eval_loader, metrics, weight):
        self.model.eval()
        agg_eval_loss = 0

        with torch.no_grad():
            for elem in eval_loader:
                loss, preds, target = self.step_fn(elem, weight)
                agg_eval_loss += loss
                for name, metric in metrics.items():
                    metric.update(self.output_transform((preds, target)))

            eval_metric_results = {}
            for name, metric in metrics.items():
                eval_metric_results[name] = metric.compute()
                metric.reset()
        eval_loss = float(agg_eval_loss / len(eval_loader))
        return eval_loss, eval_metric_results

    def save_weights(self, epoch, save_path):
        save_model(self.model, self.optimizer, epoch, save_path)

    def load_weights(self, load_path):
        load_model_state(load_path, self.model, optimizer=self.optimizer)

    def predict(self, dataset, weight, seed):
        self.set_metrics()
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=self.n_worker, pin_memory=self.pin_memory)
        if isinstance(weight, list):
            weight = torch.FloatTensor(weight).to(self.device)

        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for elem in loader:
                _, preds, _ = self.step_fn(elem, weight)
                all_preds += preds.cpu().numpy().tolist()
        all_preds = np.array(all_preds)
        print(all_preds)

        return all_preds

    def calculate_metrics(self: object, predictions: np.ndarray, labels: np.ndarray):
        metric_results = {}
        print(predictions)
        predictions = torch.from_numpy(predictions)
        for name, metric in self.metrics.items():
            metric.update(self.output_transform((predictions, labels)))
            value = metric.compute()
            metric_results[name] = value
            # Only log float values
            if isinstance(value, np.float):
                logging.info("Test {}: {}".format(name, value))
        return metric_results


@gin.configurable("MLWrapper")
class MLWrapper(object):
    def __init__(self, model=lightgbm.LGBMClassifier):
        self.model = model
        self.scaler = None

    def set_log_dir(self, log_dir: Path):
        self.log_dir = log_dir

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
            self.metrics = MLMetrics.REGRESSION

    def set_scaler(self, scaler):
        self.scaler = scaler

    @gin.configurable(module="MLWrapper")
    def train(self, train_dataset, val_dataset, weight, seed, patience=10):
        train_rep, train_label = train_dataset.get_data_and_labels()
        val_rep, val_label = val_dataset.get_data_and_labels()
        self.set_metrics(train_label)
        metrics = self.metrics

        if "class_weight" in self.model.get_params().keys():  # Set class weights
            self.model.set_params(class_weight=weight)

        if "eval_set" in inspect.getfullargspec(self.model.fit).args:  # This is lightgbm
            model_type = "lgbm"
            self.model.set_params(random_state=seed)
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
        val_string = "Val Results: loss: {:.4f}"
        val_values = [val_loss]
        val_metric_results = {"loss": val_loss}
        for name, metric in metrics.items():
            train_metric_results[name] = metric(self.label_transform(train_label), self.output_transform(train_pred))
            val_metric_results[name] = metric(self.label_transform(val_label), self.output_transform(val_pred))
            if isinstance(train_metric_results[name], np.float):
                train_string += "Train Results: " if len(train_string) == 0 else ", "
                train_string += name + ":{:.4f}"
                val_string += ", " + name + ":{:.4f}"
                train_values.append(train_metric_results[name])
                val_values.append(val_metric_results[name])
        logging.info(train_string.format(*train_values))
        logging.info(val_string.format(*val_values))

        model_file = "model.txt" if model_type == "lgbm" else "model.joblib"
        self.save_weights(save_path=(self.log_dir / model_file), model_type=model_type)
        with open(self.log_dir / "val_metrics.json", "w") as f:
            json.dump(val_metric_results, f, cls=JsonNumpyEncoder)

    def test(self, dataset, weight, seed):
        test_rep, test_label = dataset.get_data_and_labels()
        self.set_metrics(test_label)
        if "MAE" in self.metrics.keys() or isinstance(self.model, lightgbm.basic.Booster):  # If we reload a LGBM classifier
            test_pred = self.model.predict(test_rep)
        else:
            test_pred = self.model.predict_proba(test_rep)

        test_metric_results = {}
        for name, metric in self.metrics.items():
            value = metric(self.label_transform(test_label), self.output_transform(test_pred))
            test_metric_results[name] = value
            # Only log float values
            if isinstance(value, np.float):
                logging.info("Test {}: {}".format(name, value))

        with open(self.log_dir / "test_metrics.json", "w") as f:
            json.dump(test_metric_results, f, cls=JsonNumpyEncoder)

        return log_loss(test_label, test_pred)

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

    def predict(self, dataset, weight, seed):
        test_rep, _ = dataset.get_data_and_labels()
        if isinstance(self.model, lightgbm.basic.Booster):  # If we reload a LGBM classifier
            return self.model.predict(test_rep)
        else:
            return self.model.predict_proba(test_rep)

    def calculate_metrics(self: object, predictions: np.ndarray, labels: np.ndarray):
        metric_results = {}
        for name, metric in self.metrics.items():
            value = metric(self.label_transform(labels), predictions)
            metric_results[name] = value
            # Only log float values
            if isinstance(value, np.float):
                logging.info("Test {}: {}".format(name, value))
        return metric_results
