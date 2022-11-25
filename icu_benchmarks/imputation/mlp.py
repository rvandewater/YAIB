from icu_benchmarks.models.wrappers import ImputationWrapper
from torch.nn import Linear, ReLU, BatchNorm1d, Sequential, Sigmoid, Flatten
import torch
import gin


@gin.configurable("MLPImputation")
class MLPImputation(ImputationWrapper):
    
    needs_training = True
    needs_fit = False
    
    def __init__(self, *args, input_size, num_hidden_layers=3, hidden_layer_size=10, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.model = [
            Linear(input_size[1] * 2, hidden_layer_size),
            ReLU(),
            BatchNorm1d(),
        ]
        for _ in range(num_hidden_layers):
            self.model += [Linear(hidden_layer_size, hidden_layer_size), ReLU(), BatchNorm1d()]
        self.model += [Linear(hidden_layer_size, input_size[1]), Sigmoid()]
        
        self.model = Sequential(*self.model)
    
    def forward(self, amputated, amputation_mask):
        model_input = torch.cat((amputated, amputation_mask), dim=1)
        return self.model(model_input)

    
