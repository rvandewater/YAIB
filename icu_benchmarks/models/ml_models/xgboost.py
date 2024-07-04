import inspect
import logging

import gin
from icu_benchmarks.contants import RunMode
from icu_benchmarks.models.wrappers import MLWrapper
import xgboost as xgb
from xgboost.callback import EarlyStopping, LearningRateScheduler
from wandb.integration.xgboost import wandb_callback as wandb_xgb
import wandb
from optuna.integration import XGBoostPruningCallback
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
        self.model.fit(train_data, train_labels, eval_set=[(val_data, val_labels)], callbacks=callbacks)


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