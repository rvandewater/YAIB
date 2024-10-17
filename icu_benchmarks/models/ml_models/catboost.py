import gin
import catboost as cb
from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.wrappers import MLWrapper


@gin.configurable
class CBClassifier(MLWrapper):
    _supported_run_modes = [RunMode.classification]

    def __init__(self, *args, **kwargs):
        # self.model = self.set_model_args(cb.CatBoostClassifier, task_type="GPU"
        # if not kwargs['cpu'] else "CPU", *args, **kwargs)
        self.model = self.set_model_args(cb.CatBoostClassifier, task_type="CPU", *args, **kwargs)
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
