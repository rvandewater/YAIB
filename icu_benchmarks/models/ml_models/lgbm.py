import gin
import lightgbm as lgbm
import numpy as np
import wandb
from wandb.integration.lightgbm import wandb_callback as wandb_lgbm

from icu_benchmarks.contants import RunMode
from icu_benchmarks.models.wrappers import MLWrapper


class LGBMWrapper(MLWrapper):
    def fit_model(self, train_data, train_labels, val_data, val_labels):
        """Fitting function for LGBM models."""
        self.model.set_params(random_state=np.random.get_state()[1][0])
        callbacks = [lgbm.early_stopping(self.hparams.patience, verbose=True), lgbm.log_evaluation(period=-1)]

        if wandb.run is not None:
            callbacks.append(wandb_lgbm())

        self.model = self.model.fit(
            train_data,
            train_labels,
            eval_set=(val_data, val_labels),
            callbacks=callbacks,
        )
        val_loss = list(self.model.best_score_["valid_0"].values())[0]
        return val_loss


@gin.configurable
class LGBMClassifier(LGBMWrapper):
    _supported_run_modes = [RunMode.classification]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(lgbm.LGBMClassifier, *args, **kwargs)
        super().__init__(*args, **kwargs)

    def predict(self, features):
        """Predicts labels for the given features."""
        return self.model.predict_proba(features)


@gin.configurable
class LGBMRegressor(LGBMWrapper):
    _supported_run_modes = [RunMode.regression]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(lgbm.LGBMRegressor, *args, **kwargs)
        super().__init__(*args, **kwargs)
