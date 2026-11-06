# Adding imputation models

An imputation model subclasses `ImputationWrapper` from `icu_benchmarks.models.wrappers`, a PyTorch Lightning module.
There are two kinds:

- **Trained models** (`requires_backprop = True`, the default) learn by gradient descent, like a deep learning model.
- **Fitted models** (`requires_backprop = False`) receive the complete training data once in `fit`, like the
    scikit-learn baselines.

## The model class

`forward` receives the amputated values (masked values are missing) and the amputation mask, and returns the imputed
values with the same shape:

```python title="icu_benchmarks/imputation/my_imputer.py"
import gin
import torch

from icu_benchmarks.models.wrappers import ImputationWrapper


@gin.configurable("MyImputer")
class MyImputer(ImputationWrapper):
    # False: fit() receives the training data once; True: trained by gradient descent
    requires_backprop = False

    def __init__(self, *args, my_arg=10, **kwargs):
        super().__init__(*args, my_arg=my_arg, **kwargs)
        self.my_arg = my_arg

    def fit(self, train_dataset, val_dataset):
        # Only for requires_backprop = False: fit on the training data
        ...

    def forward(self, amputated_values, amputation_mask):
        imputed_values = torch.nan_to_num(amputated_values)  # replace with your imputation
        return imputed_values
```

For a fitted model, see `KNNImputation` in `icu_benchmarks/imputation/baselines.py`, whose `fit` calls scikit-learn's
`KNNImputer` on `train_dataset.amputated_values`. For a trained model, see `MLPImputation` in
`icu_benchmarks/imputation/mlp.py`.

## The config file

Add `configs/imputation_models/<Name>.gin`. The file name is the model name for `-m`, and the class's gin name should
match it:

```python title="configs/imputation_models/MyImputer.gin"
import gin.torch.external_configurables
import icu_benchmarks.models.wrappers
import icu_benchmarks.models.dl_models
import icu_benchmarks.models.utils
import icu_benchmarks.data.split_process_data
# The module that defines your class
import icu_benchmarks.imputation.my_imputer

train_common.model = @MyImputer

MyImputer.my_arg = 20
```

Trained models also set the training loop and optimizer, as in `MLP.gin`:

```python title="configs/imputation_models/MLP.gin"
--8<--
configs/imputation_models/MLP.gin
--8<--
```

## Train and use it

Train it with the `DatasetImputation` task, then pass its checkpoint to a prediction task with `-imp`; see
[Imputation](../user-guide/imputation.md). Add your model to `MODELS` in `tests/test_imputation_models.py`, which runs
the training, evaluation and prediction steps of every imputation model.
