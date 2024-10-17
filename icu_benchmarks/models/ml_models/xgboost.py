import inspect
import logging
import os
from statistics import mean

import gin
import numpy as np
import shap
import wandb
import xgboost as xgb
from xgboost.callback import EarlyStopping
from wandb.integration.xgboost import wandb_callback as wandb_xgb

from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.wrappers import MLWrapper


# Uncomment if needed in the future
# from optuna.integration import XGBoostPruningCallback


@gin.configurable
class XGBClassifier(MLWrapper):
    _supported_run_modes = [RunMode.classification]
    _explain_values = False

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(xgb.XGBClassifier, device="cpu", *args, **kwargs)
        super().__init__(*args, **kwargs)

    def predict(self, features):
        """
        Predicts class probabilities for the given features.

        Args:
            features: Input features for prediction.

        Returns:
            numpy.ndarray: Predicted probabilities for each class.
        """
        return self.model.predict_proba(features)

    def fit_model(self, train_data, train_labels, val_data, val_labels):
        """Fit the model to the training data (default SKlearn syntax)"""
        callbacks = [EarlyStopping(self.hparams.patience)]

        if wandb.run is not None:
            callbacks.append(wandb_xgb())
        logging.info(f"train_data: {train_data.shape}, train_labels: {train_labels.shape}")
        logging.info(train_labels)
        self.model.fit(train_data, train_labels, eval_set=[(val_data, val_labels)], verbose=False)
        if self._explain_values:
            self.explainer = shap.TreeExplainer(self.model)
            self.train_shap_values = self.explainer(train_data)
        # shap.summary_plot(shap_values, X_test, feature_names=features)
        # logging.info(self.model.get_booster().get_score(importance_type='weight'))
        # self.log_dict(self.model.get_booster().get_score(importance_type='weight'))
        # Return the first metric we use for validation
        eval_score = mean(next(iter(self.model.evals_result_["validation_0"].values())))
        return eval_score  # , callbacks=callbacks)

    def test_step(self, dataset, _):
        # Pred indicators indicates the row id, and time in hours
        test_rep, test_label, pred_indicators = dataset
        test_rep, test_label, pred_indicators = (
            test_rep.squeeze().cpu().numpy(),
            test_label.squeeze().cpu().numpy(),
            pred_indicators.squeeze().cpu().numpy(),
        )
        self.set_metrics(test_label)
        test_pred = self.predict(test_rep)
        if len(pred_indicators.shape) > 1 and len(test_pred.shape) > 1 and pred_indicators.shape[1] == test_pred.shape[1]:
            pred_indicators = np.hstack((pred_indicators, test_label.reshape(-1, 1)))
            pred_indicators = np.hstack((pred_indicators, test_pred))
            # Save as: id, time (hours), ground truth, prediction 0, prediction 1
            np.savetxt(os.path.join(self.logger.save_dir, "pred_indicators.csv"), pred_indicators, delimiter=",")
            logging.debug(f"Saved row indicators to {os.path.join(self.logger.save_dir, f'row_indicators.csv')}")
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
        valid_params = signature.keys()

        # Filter out invalid arguments
        valid_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        logging.debug(f"Creating model with: {valid_kwargs}.")
        return model(**valid_kwargs)

    def get_feature_importance(self):
        if not hasattr(self.model, "feature_importances_"):
            raise ValueError("Model has not been fit yet. Call fit_model() before getting feature importances.")
        return self.model.feature_importances_
