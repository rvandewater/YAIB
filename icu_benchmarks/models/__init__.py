from typing import Union

from icu_benchmarks.models.dl_models.rnn import GRUNet, LSTMNet, RNNet
from icu_benchmarks.models.dl_models.tcn import TemporalConvNet
from icu_benchmarks.models.dl_models.transformer import BaseTransformer, LocalTransformer, Transformer
from icu_benchmarks.models.ml_models.catboost import CBClassifier
from icu_benchmarks.models.ml_models.imblearn import BRFClassifier, RUSBClassifier
from icu_benchmarks.models.ml_models.lgbm import LGBMClassifier, LGBMRegressor
from icu_benchmarks.models.ml_models.sklearn import (
    ElasticNet,
    LinearRegression,
    LogisticRegression,
    MLPClassifier,
    MLPRegressor,
    PerceptronClassifier,
    RFClassifier,
    SVMClassifier,
    SVMRegressor,
)
from icu_benchmarks.models.ml_models.xgboost import XGBClassifier

DLModel = Union[
    GRUNet,
    RNNet,
    LSTMNet,
    TemporalConvNet,
    BaseTransformer,
    Transformer,
    LocalTransformer,
]
MLModelClassifier = Union[
    XGBClassifier,
    LGBMClassifier,
    RUSBClassifier,
    BRFClassifier,
    CBClassifier,
    LogisticRegression,
    SVMClassifier,
    PerceptronClassifier,
    MLPClassifier,
    RFClassifier,
]
MLModelRegression = Union[
    MLPRegressor,
    ElasticNet,
    LinearRegression,
    SVMRegressor,
    LGBMRegressor,
]

__all__ = [
    "GRUNet",
    "RNNet",
    "LSTMNet",
    "TemporalConvNet",
    "BaseTransformer",
    "Transformer",
    "LocalTransformer",
    "CBClassifier",
    "RUSBClassifier",
    "BRFClassifier",
    "LGBMClassifier",
    "LGBMRegressor",
    "XGBClassifier",
    "LogisticRegression",
    "LinearRegression",
    "ElasticNet",
    "RFClassifier",
    "SVMClassifier",
    "SVMRegressor",
    "MLPRegressor",
    "MLPClassifier",
    "PerceptronClassifier",
]
