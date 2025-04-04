import inspect
import logging
from statistics import mean

import gin
import numpy as np
import shap
import wandb
import xgboost as xgb
from xgboost.callback import EarlyStopping
from wandb.integration.xgboost import wandb_callback as wandb_xgb
from sklearn.metrics import log_loss

from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.wrappers import MLWrapper


# Uncomment if needed in the future
# from optuna.integration import XGBoostPruningCallback


@gin.configurable
class XGBClassifier(MLWrapper):
    _supported_run_modes = [RunMode.classification]
    _explain_values = False

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(
            xgb.XGBClassifier, *args, **kwargs, eval_metric=log_loss, device="cpu", missing="inf", verbosity=0
        )
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
        self.model.fit(train_data, train_labels, eval_set=[(val_data, val_labels)], verbose=0)
        # if self.explain_features:
        self.explainer = shap.TreeExplainer(
            self.model, train_data, feature_perturbation="interventional", model_output="probability"
        )
        if self.explain_features:
            logging.info("Explaining features")
            self.train_shap_values = self.explainer.shap_values(train_data)
        # shap.summary_plot(shap_values, X_test, feature_names=features)
        # logging.info(self.model.get_booster().get_score(importance_type='weight'))
        # self.log_dict(self.model.get_booster().get_score(importance_type='weight'))
        # Return the first metric we use for validation
        eval_score = mean(next(iter(self.model.evals_result_["validation_0"].values())))
        return eval_score  # , callbacks=callbacks)

    def set_model_args(self, model, *args, **kwargs):
        """XGBoost signature does not include the hyperparams so we need to pass them manually."""
        signature = inspect.signature(model.__init__).parameters
        valid_params = signature.keys()

        # Filter out invalid arguments
        valid_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        logging.debug(f"Creating model with: {valid_kwargs}.")
        return model(**valid_kwargs)

    def _explain_model(self, reps, labels):
        if not hasattr(self.model, "feature_importances_"):
            raise ValueError("Model has not been fit yet. Call fit_model() before getting feature importances.")
        # feature_importances = self.model.feature_importances_
        shap_values = self.explainer.shap_values(reps)
        # feature_importances = np.abs(shap_values).mean(axis=1)
        return shap_values

    # def explainer(self, reps):
    #     if not hasattr(self.model, "feature_importances_"):
    #         raise ValueError("Model has not been fit yet. Call fit_model() before getting feature importances.")
    #     return self.model.feature_importances_
