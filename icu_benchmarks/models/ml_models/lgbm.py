import gin
import lightgbm as lgbm
import numpy as np
import shap
import sys
from io import StringIO
import os
import wandb
from wandb.integration.lightgbm import wandb_callback as wandb_lgbm

from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.wrappers import MLWrapper


class LGBMWrapper(MLWrapper):
    def fit_model(self, train_data, train_labels, val_data, val_labels):
        """Fitting function for LGBM models."""
        self.model.set_params(random_state=np.random.get_state()[1][0])
        # callbacks = [lgbm.early_stopping(self.hparams.patience, verbose=False)]
        callbacks = [
            lgbm.log_evaluation(period=0),  # Disable default logging
        ]
        # callbacks = []
        # Redirect stderr and stdout to suppress warnings
        stderr_backup = sys.stderr
        stdout_backup = sys.stdout
        sys.stderr = StringIO()
        sys.stdout = StringIO()
        try:
            self.model = self.model.fit(
                train_data,
                train_labels,
                eval_set=(val_data, val_labels),
                callbacks=callbacks,
            )
        finally:
            sys.stderr = stderr_backup
            # to surpress:  [Warning] No further splits with positive gain, best gain: -inf with the current config
            sys.stdout = stdout_backup
        reduce_samples = False
        if reduce_samples:
            n_samples = min(1000, len(train_data))
            indices = np.random.choice(len(train_data), size=n_samples, replace=False)
        else:
            indices = np.arange(len(train_data))
        background_sample = train_data[indices]
        self.explainer = shap.TreeExplainer(
            self.model, background_sample, feature_perturbation="interventional", model_output="probability"
        )
        val_loss = list(self.model.best_score_["valid_0"].values())[0]
        return val_loss


@gin.configurable
class LGBMClassifier(LGBMWrapper):
    _supported_run_modes = [RunMode.classification]

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("verbosity", -1)
        self.model = self.set_model_args(lgbm.LGBMClassifier, *args, **kwargs, verbose=-1)
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

    def _explain_model(self, reps, labels):
        if not hasattr(self.model, "feature_importances_"):
                raise ValueError("Model has not been fit yet. Call fit_model() before getting feature importances.")
        # feature_importances = self.model.feature_importances_
        shap_values = self.explainer.shap_values(reps, labels)
        # feature_importances = np.abs(shap_values).mean(axis=1)
        return shap_values


@gin.configurable
class LGBMRegressor(LGBMWrapper):
    _supported_run_modes = [RunMode.regression]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(lgbm.LGBMRegressor, *args, **kwargs)
        super().__init__(*args, **kwargs)
