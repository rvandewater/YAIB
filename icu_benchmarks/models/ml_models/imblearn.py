import logging

from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier
from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.wrappers import MLWrapper
import gin
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import RUSBoostClassifier, BalancedRandomForestClassifier, EasyEnsembleClassifier
import xgboost as xgb

@gin.configurable
class BRFClassifier(MLWrapper):
    _supported_run_modes = [RunMode.classification]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(BalancedRandomForestClassifier, *args, **kwargs)
        super().__init__(*args, **kwargs)


@gin.configurable
class XGBEnsembleClassifier(MLWrapper):
    _supported_run_modes = [RunMode.classification]
    individual_model = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=5000,
        max_depth=10,
        scale_pos_weight=30,
        min_child_weight=1,
        max_delta_step=3,
        colsample_bytree=0.25,
        gamma=0.9,
        reg_lambda=0.1,
        reg_alpha=100,
        random_state=42,
        eval_metric='logloss'
    )
    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(EasyEnsembleClassifier, *args, **kwargs, estimator=self.individual_model)
        super().__init__(*args, **kwargs)

    def fit_model(self, train_data, train_labels, val_data, val_labels):
        import xgboost as xgb
        from sklearn.metrics import log_loss

        # Try XGBoost first with provided config
        try:
            self.model.fit(train_data, train_labels,)

            val_pred_proba = self.model.predict_proba(val_data)
            val_loss = log_loss(val_labels, val_pred_proba)
            logging.info(f"XGBoost model trained successfully. Validation loss: {val_loss:.4f}")
            return val_loss

        except Exception as e:
            logging.warning(f"XGBoost failed: {e}")

        return val_loss
