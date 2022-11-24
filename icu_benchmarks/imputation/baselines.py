from hyperimpute.plugins.imputers import Imputers
from sklearn.impute import KNNImputer
from icu_benchmarks.models.wrappers import ImputationWrapper
import gin


@gin.configurable("KNNImputation")
class KNNImputation(ImputationWrapper):
    
    needs_training = False
    needs_fit = True
    
    def __init__(self, *args, n_neighbors=10, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.imputer = KNNImputer(n_neighbors)
    
    def fit(self, data):
        self.imputer.fit(data)
    
    def forward(self, amputated_values, amputation_mask):
        return self.imputer.transform(amputated_values)

@gin.configurable("MeanImputation")
class MeanImputation(ImputationWrapper):
    
    needs_training = False
    needs_fit = True
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.imputer = Imputers().get("mean")
    
    def fit(self, data):
        self.imputer.fit(data)
    
    def forward(self, amputated_values, amputation_mask):
        return self.imputer.transform(amputated_values)

@gin.configurable("MedianImputation")
class MedianImputation(ImputationWrapper):
    
    needs_training = False
    needs_fit = True
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.imputer = Imputers().get("median")
    
    def fit(self, data):
        self.imputer.fit(data)
    
    def forward(self, amputated_values, amputation_mask):
        return self.imputer.transform(amputated_values)



@gin.configurable("MostFrequentImputation")
class MostFrequentImputation(ImputationWrapper):
    
    needs_training = False
    needs_fit = True
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.imputer = Imputers().get("most_frequent")
    
    def fit(self, data):
        self.imputer.fit(data)
    
    def forward(self, amputated_values, amputation_mask):
        return self.imputer.transform(amputated_values)

@gin.configurable("MissForestImputation")
class MissForestImputation(ImputationWrapper):
    
    needs_training = False
    needs_fit = True
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.imputer = Imputers().get("missforest")
    
    def fit(self, data):
        self.imputer.fit(data)
    
    def forward(self, amputated_values, amputation_mask):
        return self.imputer.transform(amputated_values)
    
@gin.configurable("GAINImputation")
class GAINImputation(ImputationWrapper):
    
    needs_training = False
    needs_fit = True
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.imputer = Imputers().get("gain")
    
    def fit(self, data):
        self.imputer.fit(data)
    
    def forward(self, amputated_values, amputation_mask):
        return self.imputer.transform(amputated_values)
