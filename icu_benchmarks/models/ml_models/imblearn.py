from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier
from icu_benchmarks.contants import RunMode
from icu_benchmarks.models.wrappers import MLWrapper
import gin


@gin.configurable
class BRFClassifier(MLWrapper):
    _supported_run_modes = [RunMode.classification]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(BalancedRandomForestClassifier, *args, **kwargs)
        super().__init__(*args, **kwargs)


@gin.configurable
class RUSBClassifier(MLWrapper):
    _supported_run_modes = [RunMode.classification]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(RUSBoostClassifier, *args, **kwargs)
        super().__init__(*args, **kwargs)
