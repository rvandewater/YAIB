from imblearn.ensemble import BalancedRandomForestClassifier
from icu_benchmarks.contants import RunMode
from icu_benchmarks.models.wrappers import MLWrapper
import gin

@gin.configurable
class BalancedRFClassifier(MLWrapper):
    _supported_run_modes = [RunMode.classification]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(BalancedRandomForestClassifier, *args, **kwargs)
        super().__init__(*args, **kwargs)