# Yet Another ICU Benchmark
This project aims to provide a unified interface for multiple common ICU prediction endpoints for common ICU datasets. 
We aim to support the following datasets: 
- Amsterdam UMC Database
- HiRID
- MIMIC III/IV
- eICU

For installation details, please check the [legacy readme](README_old.md). 

This file contains documentation on the structure of the project. This is subject to change as we adapt it.
## Directories
A short description of the folders:
- configs: this folder contains subdirectories with GIN configuration files, that specify details about the benchmark tasks
- docs: legacy documents
- files: folder with contents:
  - dataset_stats: some sample data in parquet files (?)
  - fake_data: generated data to demonstrate HiRID benchmark
  - pretrained_weights: weights that have been pre-trained on HiRID data
- icu_benchmarks: top-level package, contains the following:
  - common: package that contains common constants, dataset class, processing code
  - data: package that contains the main preprocessing code, also contains pytorch dataloader
  - endpoints: package that contains detailed endpoint generation code
  - imputation: imputation methods
  - labels: label generation
  - models: main package for the defined models
  - preprocessing: preprocessing package code
  - synthetic_data: package for generating synthetic data
- preprocessing: (?)
- run_scripts: lots of shell scripts for previous paper experiments (?)
- tests: testing package

## Libraries
We currently use the following libraries:
- [Pytorch](https://pytorch.org/) 
    - An open source machine learning framework for 
- [Pytorch Ignite](https://github.com/pytorch/ignite)
    - Library for training and evaluating neural networks in Pytorch
- [GIN](https://github.com/google/gin-config)
    - Gin provides a lightweight configuration framework for Python
- [Pathos](https://pathos.readthedocs.io/en/latest/)
  - Parallel computing framework, used for preprocessing

## Installation

You can either install this repo using conda or using the setup.py. As the hyperimpute package currently depends on a package with faulty install procedures (`geomloss`), the you have to install the package like this:
```bash
pip install torch numpy && pip install -e .
```

# CLI Commands

## Preprocess and Train
The following command will start training on a prepared HiRID dataset for sequential Mortality prediction with an LGBM Classifier: 
```
python run.py train \
    -d ../data/mortality_seq/hirid \
    -n hirid \
    -t Mortality_At24Hours \
    -m LGBMClassifier \
    -hp LGBMClassifier.subsample='RS([0.33,0.66])' LGBMClassifier.colsample_bytree=0.66
```
> `RS([...])` is the syntax for invoking random search on a list of hyperparameters, both in configs and the command line.

> Run with `PYTORCH_ENABLE_MPS_FALLBACK=1` on Macs with Metal Performance Shaders

> Please note that, for Windows based systems, paths need to be formatted differently, e.g: ` r"\..\data\mortality_seq\hirid"`.
> Additionally, the next line character (\\)  needs to be replaced by (^) (Command Prompt) or (`) (Powershell) respectively.
## Evaluate
It is possible to evaluate a model trained on another dataset. In this case, the source dataset is HiRID and the target is MIMIC-IV:
```
python run.py evaluate \
    -d ../data/mortality_seq/miiv \
    -n miiv \
    -t Mortality_At24Hours \
    -m LGBMClassifier \
    -sn hirid \
    --source ../data/mortality_seq/hirid/logs/hirid/Mortality_At24Hours/LGBMClassifier/2022-11-10T22-52-52/seed_1111
```

## Run Tests
```
python -m pytest ./tests/recipes
coverage run -m pytest ./tests/recipes
# then use either of the following
coverage report
coverage html
```

## Autoformat and lint
For development purposes, we use the `Black` package to autoformat our code and a `Flake8` Linting/CI check:
```
black . -l 127
flake8 . --count --max-complexity=14 --max-line-length=127 --statistics
```

## Adding new Imputation Models

To add another imputation model, you have to create a class that inherits from `ImputationWrapper` in `icu_benchmarks.models.wrappers`. Your model class should look like this:

```python
from icu_benchmarks.models.wrappers import ImputationWrapper
import gin

@gin.configurable("newmethod")
class New_Method(ImputationWrapper):

  # adjust this accordingly
  needs_training = False # if true, the method is trained iteratively (like a deep learning model)
  needs_fit = True # if true, it receives the complete training data to perform a fit on

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
import icu_benchmarks.models.encoders
import icu_benchmarks.models.utils
import icu_benchmarks.data.preprocess
# import here the file you created your New_Method class in
import icu_benchmarks.imputation.new_model

preprocess.use_features = False

# Train params
train_imputation_method.model = @newmethod # change this into the name of the gin configuration file
train_imputation_method.do_test = True

# here you can set some training parameters
train_imputation_method.epochs = 1000
train_imputation_method.batch_size = 64
train_imputation_method.patience = 10
train_imputation_method.min_delta = 1e-4
train_imputation_method.wandb = True

ImputationWrapper.optimizer = @Adam
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