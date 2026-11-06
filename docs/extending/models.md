# Adding models

A prediction model in YAIB is a class that wraps the model and a gin config that selects it and declares its
hyperparameters. Adding one takes three steps:

1. Write the model class, based on `MLWrapper` (classical models) or `DLPredictionWrapper` (PyTorch models).
2. Add a config `configs/prediction_models/<Name>.gin`.
3. Run it with `-m <Name>`.

The wrappers handle training, early stopping, evaluation, metrics, logging and saving, so the model class only defines
the model itself. Deep learning models are [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) modules.

## Classical models

Wrap any scikit-learn-style estimator with `MLWrapper`. `set_model_args` creates the estimator with the hyperparameters
gin binds to the class, keeping only those the estimator accepts:

```python title="icu_benchmarks/models/ml_models/sklearn.py"
import gin
import shap
from sklearn import ensemble

from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.wrappers import MLWrapper


@gin.configurable
class RFClassifier(MLWrapper):
    _supported_run_modes = [RunMode.classification]
    explainer_cls = shap.TreeExplainer  # optional: enables --explain-features

    def __init__(self, *args, **kwargs):
        self.model = self.set_model_args(ensemble.RandomForestClassifier, *args, **kwargs)
        super().__init__(*args, **kwargs)
```

| Attribute              | Meaning                                                                                                                                          |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `_supported_run_modes` | `RunMode.classification` and/or `RunMode.regression`; other tasks are rejected.                                                                  |
| `explainer_cls`        | A SHAP explainer class, to support [feature explanations](../user-guide/feature-explanations.md). Deep learning models are explained without it. |

Override `fit_model` for estimators that need special fitting, such as early stopping on the validation set; see
`LGBMWrapper` in `icu_benchmarks/models/ml_models/lgbm.py`.

## Deep learning models

Subclass `DLPredictionWrapper` and define the layers and `forward`. The input `x` has the shape
`(batch, time steps, features)`; `input_size` is that shape, so `input_size[2]` is the number of features. `forward`
returns one prediction per time step, `(batch, time steps, num_classes)`; the wrapper selects the labeled steps and
computes the loss. This is the RNN model in `icu_benchmarks/models/dl_models/rnn.py`:

```python title="icu_benchmarks/models/dl_models/rnn.py"
import gin
from torch import nn as nn

from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.wrappers import DLPredictionWrapper


@gin.configurable
class RNNet(DLPredictionWrapper):
    """Torch standard RNN model"""

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(self, input_size, hidden_dim, layer_dim, num_classes, *args, **kwargs):
        super().__init__(
            *args,
            input_size=input_size,
            hidden_dim=hidden_dim,
            layer_dim=layer_dim,
            num_classes=num_classes,
            **kwargs,
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

Two conventions matter:

- Pass the constructor arguments on to `super().__init__` so that Lightning saves them with the checkpoint.
- Name the final output layer `self.logit`. For classification, the wrapper reads `self.logit.out_features` to choose
    between the binary and the multiclass metrics.

**Reusable layers** are in `icu_benchmarks/models/dl_models/layers.py` (attention blocks, transformer blocks and more).

**Models that need the missingness pattern**, such as GRU-D and BAT, subclass `ObservationMaskWrapper`. YAIB's
preprocessing adds a `MissingIndicator_<name>` column for every variable; the wrapper pairs each value with its indicator,
so `self.observed_mask(x)` tells the model which values were measured and which were filled in.

**Models that need another input format** can use their own dataset class: subclass `PredictionPolarsDataset` in
`icu_benchmarks/data/loader.py`, override `__getitem__`, and check that the wrapper's `step_fn` moves every part of the
batch to the device.

## The config file

The config selects the model and declares its hyperparameters and search space. This is `RNN.gin`:

```python title="configs/prediction_models/RNN.gin"
--8<--
configs/prediction_models/RNN.gin
--8<--
```

- Include `common/DLCommon.gin` (deep learning) or `common/MLCommon.gin` (classical) for the shared settings and the
    tuning budget.
- Bind the model with `train_common.model = @<YourClass>`.
- Declare hyperparameters in the `model` scope (and `optimizer` for deep learning models): a single value is fixed, a
    tuple `(low, high)` or `(low, high, "log")` is a range, and a list is a set of choices. See
    [Hyperparameter tuning](../user-guide/hyperparameter-tuning.md).
- The common configs import the built-in model modules so that gin knows their classes. If your class lives in a new
    module, import it at the top of your config, e.g. `import icu_benchmarks.models.dl_models.my_model`.

## Train the model

The config's file name is the model name:

```bash
uv run icu-benchmarks train \
    -d demo_data/mortality24/mimic_demo \
    -n mimic_demo \
    -t BinaryClassification \
    -tn Mortality24 \
    -m RNN \
    -s 2222 \
    -l ../yaib_logs/ \
    --tune
```

Before opening a pull request, add your model to the parametrized model tests in `tests/test_dl_models.py` or
`tests/test_ml_models.py`; see [Contributing](../development/contributing.md).
