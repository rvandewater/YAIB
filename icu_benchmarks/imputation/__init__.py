from .baselines import *
from .mlp import *
from .np import *
from .rnn import *
from .simple_diffusion import *

name_mapping = {
    "NP": NPImputation,
    "KNN": KNNImputation,
    "MICE": MICEImputation,
    "Mean": MeanImputation,
    "Median": MedianImputation,
    "Zero": ZeroImputation,
    "MostFrequent": MostFrequentImputation,
    "MissForest": MissForestImputation,
    "GAIN": GAINImputation,
    "BRITS": BRITSImputation,
    "SAITS": SAITSImputation,
    "Attention": AttentionImputation,
    "SimpleDiffusion": Simple_Diffusion_Model,
    "BRNN": BRNNImputation,
    "RNN": RNNImputation,
    "MLP": MLPImputation,
}