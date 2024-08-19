import inspect
import logging

import gin
import numpy as np

from icu_benchmarks.contants import RunMode
from icu_benchmarks.models.wrappers import MLWrapper
import xgboost as xgb
from xgboost.callback import EarlyStopping, LearningRateScheduler
from wandb.integration.xgboost import wandb_callback as wandb_xgb
import wandb
from statistics import mean
from optuna.integration import XGBoostPruningCallback
import shap

@gin.configurable
class XGBClassifier(MLWrapper):
    _supported_run_modes = [RunMode.classification]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(xgb.XGBClassifier, device="cpu",*args, **kwargs)
        super().__init__(*args, **kwargs)

    def predict(self, features):
        """Predicts labels for the given features."""
        return self.model.predict_proba(features)

    def fit_model(self, train_data, train_labels, val_data, val_labels):
        """Fit the model to the training data (default SKlearn syntax)"""
        callbacks = [EarlyStopping(self.hparams.patience)]

        if wandb.run is not None:
            callbacks.append(wandb_xgb())
        self.model.fit(train_data, train_labels, eval_set=[(val_data, val_labels)], verbose=False)
        self.explainer = shap.TreeExplainer(self.model)
        self.train_shap_values = self.explainer(train_data)
        # shap.summary_plot(shap_values, X_test, feature_names=features)
        # logging.info(self.model.get_booster().get_score(importance_type='weight'))
        # self.log_dict(self.model.get_booster().get_score(importance_type='weight'))
        # Return the first metric we use for validation
        eval_score = mean(next(iter(self.model.evals_result_["validation_0"].values())))
        return eval_score #, callbacks=callbacks)

    def test_step(self, dataset, _):
        test_rep, test_label = dataset
        test_rep, test_label = test_rep.squeeze().cpu().numpy(), test_label.squeeze().cpu().numpy()
        self.set_metrics(test_label)
        test_pred = self.predict(test_rep)
        if self.explainer is not None:
            self.test_shap_values = self.explainer(test_rep)
            # logging.debug(f"Shap values: {self.test_shap_values}")
            # self.log("test/shap_values", self.test_shap_values, sync_dist=True)
        if self.mps:
            self.log("test/loss", np.float32(self.loss(test_label, test_pred)), sync_dist=True)
            self.log_metrics(np.float32(test_label), np.float32(test_pred), "test")
        else:
            self.log("test/loss", self.loss(test_label, test_pred), sync_dist=True)
            self.log_metrics(test_label, test_pred, "test")
        logging.debug(f"Test loss: {self.loss(test_label, test_pred)}")

    def set_model_args(self, model, *args, **kwargs):
        """XGBoost signature does not include the hyperparams so we need to pass them manually."""
        signature = inspect.signature(model.__init__).parameters
        possible_hps = list(signature.keys())
        # Get passed keyword arguments
        arguments = locals()["kwargs"]
        # Get valid hyperparameters
        hyperparams = arguments
        logging.debug(f"Creating model with: {hyperparams}.")
        return model(**hyperparams)

    def get_feature_importance(self):
        return self.model.feature_importances_