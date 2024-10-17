import gin
from sklearn import linear_model, ensemble, svm, neural_network
from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.wrappers import MLWrapper


@gin.configurable
class LogisticRegression(MLWrapper):
    __supported_run_modes = [RunMode.classification]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(linear_model.LogisticRegression, *args, **kwargs)
        super().__init__(*args, **kwargs)


@gin.configurable()
class LinearRegression(MLWrapper):
    _supported_run_modes = [RunMode.regression]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(linear_model.LinearRegression, *args, **kwargs)
        super().__init__(*args, **kwargs)


@gin.configurable()
class ElasticNet(MLWrapper):
    _supported_run_modes = [RunMode.regression]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(linear_model.ElasticNet, *args, **kwargs)
        super().__init__(*args, **kwargs)


@gin.configurable
class RFClassifier(MLWrapper):
    _supported_run_modes = [RunMode.classification]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(ensemble.RandomForestClassifier, *args, **kwargs)
        super().__init__(*args, **kwargs)


@gin.configurable
class SVMClassifier(MLWrapper):
    _supported_run_modes = [RunMode.classification]

    def __init__(self, *args, **kwargs):
        self.model = self.model_args(svm.SVC, *args, **kwargs)
        super().__init__(*args, **kwargs)


@gin.configurable
class SVMRegressor(MLWrapper):
    _supported_run_modes = [RunMode.regression]

    def __init__(self, *args, **kwargs):
        self.model = self.model_args(svm.SVR, *args, **kwargs)
        super().__init__(*args, **kwargs)


@gin.configurable
class PerceptronClassifier(MLWrapper):
    _supported_run_modes = [RunMode.classification]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(neural_network.MLPClassifier, *args, **kwargs)
        super().__init__(*args, **kwargs)


@gin.configurable
class MLPClassifier(MLWrapper):
    _supported_run_modes = [RunMode.classification]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(neural_network.MLPClassifier, *args, **kwargs)
        super().__init__(*args, **kwargs)


class MLPRegressor(MLWrapper):
    _supported_run_modes = [RunMode.regression]

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(neural_network.MLPRegressor, *args, **kwargs)
        super().__init__(*args, **kwargs)
