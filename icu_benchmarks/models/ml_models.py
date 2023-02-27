import gin
import lightgbm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as sklearn_LogisticRegression

from icu_benchmarks.models.wrappers import MLClassificationWrapper


@gin.configurable
class LGBMClassifier(MLClassificationWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.model_args()

    @gin.configurable(module="LGBMClassifier")
    def model_args(self, *args, **kwargs):
        return lightgbm.LGBMClassifier(*args, **kwargs)


@gin.configurable
class LGBMRegressor(MLClassificationWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.model_args()

    @gin.configurable(module="LGBMRegressor")
    def model_args(self, *args, **kwargs):
        return lightgbm.LGBMRegressor(*args, **kwargs)


@gin.configurable
class LogisticRegression(MLClassificationWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.model_args()

    @gin.configurable(module="LogisticRegression")
    def model_args(self, *args, **kwargs):
        return sklearn_LogisticRegression(*args, **kwargs)


@gin.configurable
class RFClassifier(MLClassificationWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.model_args()

    @gin.configurable(module="RFClassifier")
    def model_args(self, *args, **kwargs):
        return RandomForestClassifier(*args, **kwargs)
