import torch
from pandas import DataFrame
from hyperimpute.plugins.imputers import Imputers
from sklearn.impute import KNNImputer, SimpleImputer
from icu_benchmarks.data.loader import ImputationDataset
from icu_benchmarks.models.wrappers import ImputationWrapper
import gin


@gin.configurable("KNN")
class KNNImputation(ImputationWrapper):
    
    needs_training = False
    needs_fit = True
    
    def __init__(self, *args, n_neighbors=2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.imputer = KNNImputer(n_neighbors=n_neighbors)

    def fit(self, data: ImputationDataset):
        self.imputer.fit(data.amputated_values.values)
    
    def forward(self, amputated_values, amputation_mask):
        debatched_values = amputated_values.reshape((-1, amputated_values.shape[-1]))
        debatched_values = debatched_values.to("cpu")
        output = torch.Tensor(self.imputer.transform(debatched_values)).to(amputated_values.device)
        
        output = output.reshape(amputated_values.shape)
        return output

@gin.configurable("Mean")
class MeanImputation(ImputationWrapper):
    
    needs_training = False
    needs_fit = True
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.imputer = SimpleImputer(strategy="mean")
    
    def fit(self, data: ImputationDataset):
        self.imputer.fit(data.amputated_values.values)
    
    def forward(self, amputated_values, amputation_mask):
        debatched_values = amputated_values.reshape((-1, amputated_values.shape[-1]))
        debatched_values = debatched_values.to("cpu")
        output = torch.Tensor(self.imputer.transform(debatched_values)).to(amputated_values.device)
        
        output = output.reshape(amputated_values.shape)
        return output

@gin.configurable("Median")
class MedianImputation(ImputationWrapper):
    
    needs_training = False
    needs_fit = True
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.imputer = SimpleImputer(strategy="median")
    
    def fit(self, data: ImputationDataset):
        self.imputer.fit(data.amputated_values.values)
    
    def forward(self, amputated_values, amputation_mask):
        debatched_values = amputated_values.reshape((-1, amputated_values.shape[-1]))
        debatched_values = debatched_values.to("cpu")
        output = torch.Tensor(self.imputer.transform(debatched_values)).to(amputated_values.device)
        
        output = output.reshape(amputated_values.shape)
        return output


@gin.configurable("Zero")
class ZeroImputation(ImputationWrapper):
    
    needs_training = False
    needs_fit = True
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    
    def fit(self, data: ImputationDataset):
        self.imputer.fit(data.amputated_values.values)
    
    def forward(self, amputated_values, amputation_mask):
        debatched_values = amputated_values.reshape((-1, amputated_values.shape[-1]))
        debatched_values = debatched_values.to("cpu")
        output = torch.Tensor(self.imputer.transform(debatched_values)).to(amputated_values.device)
        
        output = output.reshape(amputated_values.shape)
        return output



@gin.configurable("MostFrequent")
class MostFrequentImputation(ImputationWrapper):
    
    needs_training = False
    needs_fit = True
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.imputer = SimpleImputer(strategy="most_frequent")
    
    def fit(self, data: ImputationDataset):
        self.imputer.fit(data.amputated_values.values)
    
    def forward(self, amputated_values, amputation_mask):
        debatched_values = amputated_values.reshape((-1, amputated_values.shape[-1]))
        debatched_values = debatched_values.to("cpu")
        output = torch.Tensor(self.imputer.transform(debatched_values)).to(amputated_values.device)
        
        output = output.reshape(amputated_values.shape)
        return output

@gin.configurable("MissForest")
class MissForestImputation(ImputationWrapper):
    
    needs_training = False
    needs_fit = True
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.imputer = Imputers().get("sklearn_missforest")
    
    def fit(self, data: ImputationDataset):
        self.imputer._model.fit(data.amputated_values.values)
    
    def forward(self, amputated_values, amputation_mask):
        debatched_values = amputated_values.reshape((-1, amputated_values.shape[-1]))
        debatched_values = debatched_values.to("cpu")
        output = torch.Tensor(self.imputer._model.transform(debatched_values)).to(amputated_values.device)
        
        output = output.reshape(amputated_values.shape)
        return output
    
@gin.configurable("GAIN")
class GAINImputation(ImputationWrapper):
    
    needs_training = False
    needs_fit = True
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.imputer = Imputers().get("gain")
    
    def fit(self, data: ImputationDataset):
        self.imputer._model.fit(torch.Tensor(data.amputated_values.values))
    
    def forward(self, amputated_values, amputation_mask):
        debatched_values = amputated_values.reshape((-1, amputated_values.shape[-1]))
        debatched_values = debatched_values.to("cpu")
        output = torch.Tensor(self.imputer._model.transform(debatched_values)).to(amputated_values.device)
        
        output = output.reshape(amputated_values.shape)
        return output