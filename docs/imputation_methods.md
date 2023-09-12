
# Adding new Imputation Models

To add another imputation model, you have to create a class that inherits from `ImputationWrapper` in `icu_benchmarks.models.wrappers`. Your model class should look like this:

```python
from icu_benchmarks.models.wrappers import ImputationWrapper
import gin


@gin.configurable("newmethod")
class New_Method(ImputationWrapper):
    # adjust this accordingly
    # if true, the method is trained iteratively (like a deep learning model). 
    # If false it receives the complete training data to perform a fit on
    requires_backprop = False  

    def __init__(self, *args, model_arg1, model_arg2, **kwargs):
        super().__init__(*args, **kwargs)
        # define your new model here
        self.model = ...

    # the following method has to be implemented for all methods
    def forward(self, amputated_values, amputation_mask):
        imputated_values = amputated_values
        ...
        return imputated_values

    # implement this, if needs_fit is true, otherwise you can leave it out.
    # this method receives the complete input training data to perform a fit on.
    def fit(self, train_data):
        ...
```

You also need to create a gin configuration file in the `configs/imputation` directory, 
named `newmethod.gin` after the name that was entered into the `gin.configurable` decorator call.

Your `.gin` file should look like this:

```python
import gin.torch.external_configurables
import icu_benchmarks.models.wrappers
import icu_benchmarks.models.dl_models
import icu_benchmarks.models.utils
import icu_benchmarks.data.split_process_data
# import here the file you created your New_Method class in
import icu_benchmarks.imputation.new_model

# Train params
train_common.model =


@newmethod  # change this into the name of the gin configuration file

# here you can set some training parameters


train_common.epochs = 1000
train_common.batch_size = 64
train_common.patience = 10
train_common.min_delta = 1e-4
train_common.use_wandb = True

ImputationWrapper.optimizer =


@Adam


ImputationWrapper.lr_scheduler = "cosine"

# Optimizer params
Adam.lr = 3e-4
Adam.weight_decay = 1e-6

# here you can set the model parameters you want to configure
newmethod.model_arg1 = 20
newmethod.model_arg2 = 15
```

You can find further configurations in the `Dataset_Imputation.gin` file in the `configs/tasks/` directory.
To start a training of an imputation method with the newly created imputation method, use the following command:

```bash
python run.py train -d path/to/preprocessed/data/files -n dataset_name -t Dataset_Imputation -m newmethod
```

For the dataset path please enter the path to the directory where the preprocessed `dyn.parquet`, `outc.parquet` and `sta.parquet` are stored. The `dataset_name` is only for logging purposes and breaks nothing if not set correctly. Keep in mind to use the name of the `.gin` config file created for the imputation method as model name for the `-m` parameter.

For reference for a deep learning based imputation method you can take a look at how the `MLPImputation` method is implemented in `icu_benchmarks/imputation/mlp.py` with its `MLP.gin` configuration file. For reference regarding methods with `needs_fit=True`, take a look at the `icu_benchmarks/imputation/baselines.py` file with several baseline implementations and their corresponding config files in `configs/imputation/`.
