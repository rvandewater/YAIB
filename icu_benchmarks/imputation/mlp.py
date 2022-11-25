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
            Flatten(),
            Linear(input_size[1] * 2 * input_size[2], hidden_layer_size),
            ReLU(),
            BatchNorm1d(hidden_layer_size),
        ]
        for _ in range(num_hidden_layers):
            self.model += [Linear(hidden_layer_size, hidden_layer_size), ReLU(), BatchNorm1d(hidden_layer_size)]
        self.model += [Linear(hidden_layer_size, input_size[1] * input_size[2]), Sigmoid()]

        self.model = Sequential(*self.model)
    
    def forward(self, amputated, amputation_mask):
        amputated = torch.nan_to_num(amputated, nan=0.0)
        model_input = torch.cat((amputated, amputation_mask), dim=1)
        
        output = self.model(model_input)
        output = output.reshape(amputated.shape)
        
        return output