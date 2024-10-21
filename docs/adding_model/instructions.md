# Adding new models to YAIB
## Example
We refer to the page [adding a new model](https://github.com/rvandewater/YAIB/wiki/Adding-a-new-model) for detailed instructions on adding new models.
We allow prediction models to be easily added and integrated into a Pytorch Lightning module. This
incorporates advanced logging and debugging capabilities, as well as
built-in parallelism. Our interface derives from the [`BaseModule`](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html).

Adding a model consists of three steps:
1. Add a model through the existing `MLPredictionWrapper` or `DLPredictionWrapper`.
2. Add a GIN config file to bind hyperparameters.
3. Execute YAIB using a simple command.

This folder contains everything you need to add a model to YAIB.
Putting the `RNN.gin` file in `configs/prediction_models` and the `rnn.py` file into icu_benchmarks/models allows you to run the model fully.

``` 
icu-benchmarks train \
    -d demo_data/mortality24/mimic_demo \ # Insert cohort dataset here
    -n mimic_demo \
    -t BinaryClassification \ # Insert task name here
    -tn Mortality24 \
    --log-dir ../yaib_logs/ \
    -m RNN \ # Insert model here
    -s 2222 \
    -l ../yaib_logs/ \
    --tune
``` 
# Adding more models
## Regular ML
For standard Scikit-Learn type models (e.g., LGBM), one can
simply wrap `MLPredictionWrapper` the function with minimal code
overhead. Many ML (and some DL) models can be incorporated this way, requiring minimal code additions. See below.

``` {#code:ml-model-definition frame="single" style="pycharm" caption="\\textit{Example ML model definition}" label="code:ml-model-definition" columns="fullflexible"}
@gin.configurable
class RFClassifier(MLWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.model_args()

    @gin.configurable(module="RFClassifier")
    def model_args(self, *args, **kwargs):
        return RandomForestClassifier(*args, **kwargs)
```
## Adding DL models
It is relatively straightforward to add new Pytorch models to YAIB. We first provide a standard RNN-model which needs no extra components. Then, we show the implementation of the Temporal Fusion Transformer model.

### Standard RNN-model
The definition of dl models can be done by creating a subclass from the
`DLPredictionWrapper`, inherits the standard methods needed for
training dl learning models. Pytorch Lightning significantly reduces the code
overhead.


``` {#code:dl-model-definition frame="single" style="pycharm" caption="\\textit{Example DL model definition}" label="code:dl-model-definition" columns="fullflexible"}
@gin.configurable
class RNNet(DLPredictionWrapper):
    """Torch standard RNN model"""

    def __init__(self, input_size, hidden_dim, layer_dim, num_classes, *args, **kwargs):
        super().__init__(
            input_size=input_size, hidden_dim=hidden_dim, layer_dim=layer_dim, num_classes=num_classes, *args, **kwargs
        )
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_size[2], hidden_dim, layer_dim, batch_first=True)
        self.logit = nn.Linear(hidden_dim, num_classes)

    def init_hidden(self, x):
        h0 = x.new_zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return h0

    def forward(self, x):
        h0 = self.init_hidden(x)
        out, hn = self.rnn(x, h0)
        pred = self.logit(out)
        return pred
```
### Adding a SOTA model: Temporal Fusion Transformer
There are two main questions when you want to add a more complex model:

* _Do you want to manually define the model or use an existing library?_ This might require adapting the `DLPredictionWrapper`.
* _Does the model expect the data to be in a certain format?_ This might require adapting the `PredictionDataset`.

By adapting, we mean creating a new subclass that inherits most functionality to avoid code duplication, is future-proof, and follows good coding practices.

First, you can add modules to `models/layers.py` to use them for your model.
``` {#code:building blocks frame="single" style="pycharm" caption="\\textit{Example building block}" label="code: layers" columns="fullflexible"}
class StaticCovariateEncoder(nn.Module):
    """
    Network to produce 4 context vectors to enrich static variables
    Variable selection Network --> GRNs
    """

    def __init__(self, num_static_vars, hidden, dropout):
        super().__init__()
        self.vsn = VariableSelectionNetwork(hidden, dropout, num_static_vars)
        self.context_grns = nn.ModuleList([GRN(hidden, hidden, dropout=dropout) for _ in range(4)])

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        variable_ctx, sparse_weights = self.vsn(x)

        # Context vectors:
        # variable selection context
        # enrichment context
        # state_c context
        # state_h context
        cs, ce, ch, cc = [m(variable_ctx) for m in self.context_grns]

        return cs, ce, ch, cc
```
Note that we can create modules out of modules as well.

### Adapting the `DLPredictionWrapper`
The next step is to use the building blocks defined in layers.py or modules from an existing library to add to the model in `models/dl_models.py`. In this In this case, we use the Pytorch-forecasting library (https://github.com/jdb78/pytorch-forecasting):

``` {#code:dl-model-definition frame="single" style="pycharm" caption="\\textit{Example DL model definition}" label="code:dl-model-definition" columns="fullflexible"}
class TFTpytorch(DLPredictionWrapper):

    supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(self, dataset, hidden, dropout, n_heads, dropout_att, lr, optimizer, num_classes, *args, **kwargs):
        super().__init__(lr=lr, optimizer=optimizer, *args, **kwargs)
        self.model = TemporalFusionTransformer.from_dataset(
            dataset=dataset)
        self.logit = nn.Linear(7, num_classes)

   
    def forward(self, x):
        out = self.model(x)
        pred = self.logit(out["prediction"])
        return pred
```

### Adapting the `PredictionDataset`
Some models require an adjusted dataloader to facilitate, for example, explainability methods. In this case, changes need to be made to the `data/loader.py` file to ensure the data loader returns the data in the correct format.
This can be done by creating a class that inherits from PredictionDataset and editing the get_item method.
``` {#code:dataset frame="single" style="pycharm" caption="\\textit{Example custom dataset definition}" label="code: dataset" columns="fullflexible"}
@gin.configurable("PredictionDatasetTFT")
class PredictionDatasetTFT(PredictionDataset):
 def __init__(self, *args, ram_cache: bool = True, **kwargs):
        super().__init__(*args, ram_cache=True, **kwargs)

def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Function to sample from the data split of choice. Used for TFT.
        The data needs to be given to the model in the following order 
        [static categorical, static continuous,known categorical,known continuous, observed categorical, observed continuous,target,id]
```
Then, you must check `models/wrapper.py`, particularly the step_fn method, to ensure the data is correctly transferred to the device.

## Adding the model config GIN file
To define hyperparameters for each model in a standardized manner, we use GIN-config. We need to specify a GIN file to bind the parameters to train and optimize this model from a choice of hyperparameters. Note that we can use modifiers for the optimizer (e.g, Adam optimizer) and ranges that we can specify in rounded brackets "()". Square brackets, "[]",  result in a random choice where the variable is uniformly sampled. 
``` 
# Hyperparameters for TFT model.

# Common settings for DL models
include "configs/prediction_models/common/DLCommon.gin"

# Optimizer params
train_common.model = @TFT

optimizer/hyperparameter.class_to_tune = @Adam
optimizer/hyperparameter.weight_decay = 1e-6
optimizer/hyperparameter.lr = (1e-5, 3e-4)

# Encoder params
model/hyperparameter.class_to_tune = @TFT
model/hyperparameter.encoder_length = 24
model/hyperparameter.hidden = 256
model/hyperparameter.num_classes = %NUM_CLASSES
model/hyperparameter.dropout = (0.0, 0.4)
model/hyperparameter.dropout_att = (0.0, 0.4)
model/hyperparameter.n_heads =4
model/hyperparameter.example_length=25
``` 
## Training the model
After these steps, your model should be trainable with the following command:

``` 
icu-benchmarks train \
    -d demo_data/mortality24/mimic_demo \ # Insert cohort dataset here
    -n mimic_demo \
    -t BinaryClassification \ # Insert task name here
    -tn Mortality24 \
    --log-dir ../yaib_logs/ \
    -m TFT \ # Insert model here
    -s 2222 \
    -l ../yaib_logs/ \
    --tune
``` 
