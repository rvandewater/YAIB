import inspect
import logging

import gin
import lightgbm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as LR, Perceptron
from sklearn import svm
from icu_benchmarks.models.wrappers import MLWrapper
from sklearn.neural_network import MLPClassifier, MLPRegressor
from icu_benchmarks.contants import RunMode


@gin.configurable
class LGBMClassifier(MLWrapper):
    supported_run_modes = [RunMode.classification]
    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(*args, **kwargs)
        super().__init__(*args, **kwargs)
        # self.model = self.model_args()

    @gin.configurable(module="LGBMClassifier")
    def set_model_args(self, *args, **kwargs):
        val = inspect.signature(lightgbm.LGBMClassifier.__init__).parameters
        arguments = locals()["kwargs"]
        possible_hps = list(val.keys())
        hyperparams = {key: value for key, value in arguments.items() if key in possible_hps}
        logging.info(hyperparams)
        return lightgbm.LGBMClassifier(**hyperparams)

@gin.configurable
class LGBMRegressorWrapper(MLWrapper):
    supported_run_modes = [RunMode.regression]
    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(*args, **kwargs)
        super().__init__(*args, **kwargs)
        # self.model = self.set_model_args(*args, **kwargs)

    # @gin.configurable(module="LGBMRegressor")
    def set_model_args(self, *args, **kwargs):
        val = inspect.signature(lightgbm.LGBMRegressor.__init__).parameters
        arguments = locals()["kwargs"]
        possible_hps = list(val.keys())
        hyperparams = {key: value for key, value in arguments.items() if key in possible_hps}
        logging.info(hyperparams)
        return lightgbm.LGBMRegressor(**hyperparams)

# Scikit-learn models
@gin.configurable
class LogisticRegression(MLWrapper):
    supported_run_modes = [RunMode.classification]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.model_args()

    @gin.configurable(module="LogisticRegression")
    def model_args(self, *args, **kwargs):
        return LR(*args, **kwargs)


@gin.configurable
class RFClassifier(MLWrapper):
    supported_run_modes = [RunMode.classification]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.model_args()

    @gin.configurable(module="RFClassifier")
    def model_args(self, *args, **kwargs):
        return RandomForestClassifier(*args, **kwargs)


@gin.configurable
class SVMClassifier(MLWrapper):
    supported_run_modes = [RunMode.classification]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.model_args()

    @gin.configurable(module="SVMClassifier")
    def model_args(self, *args, **kwargs):
        return svm.SVC(*args, **kwargs)

@gin.configurable
class SVMRegressor(MLWrapper):
    supported_run_modes = [RunMode.regression]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.model_args()

    @gin.configurable(module="SVMRegressor")
    def model_args(self, *args, **kwargs):
        return svm.SVR(*args, **kwargs)

@gin.configurable
class PerceptronClassifier(MLWrapper):
    supported_run_modes = [RunMode.classification]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.model_args()

    @gin.configurable(module="PerceptronClassifier")
    def model_args(self, *args, **kwargs):
        return Perceptron(*args, **kwargs)


@gin.configurable
class MLPClassifier(MLWrapper):
    supported_run_modes = [RunMode.classification]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.model_args()

    @gin.configurable(module="MLPClassifier")
    def model_args(self, *args, **kwargs):
        return MLPClassifier(*args, **kwargs)

class MLPRegressor(MLWrapper):
    supported_run_modes = [RunMode.regression]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.model_args()

    @gin.configurable(module="MLPRegressor")
    def model_args(self, *args, **kwargs):
        return MLPRegressor(*args, **kwargs)
