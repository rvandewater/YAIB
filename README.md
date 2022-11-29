# Yet Another ICU Benchmark
This project aims to provide a unified interface for multiple common ICU prediction endpoints for common ICU datasets. 
We support the following datasets: 
- Amsterdam UMC Database
- HiRID
- MIMIC III/IV
- eICU

We refer to the `PyICU` or `RICU` package for generating cohorts and labels in order to execute a task. 
# Installation

```
conda env update -f <environment.yml|environment_mps.yml>
conda activate yaib
pip install -e .
```

> Use `environment.yml` on x86 hardware and `environment_mps.yml` on Macs with Metal Performance Shaders.

> Note that the last command installs the package called `icu-benchmarks`.

# Use with CLI Commands
## Preprocess and Train
The following command will start training on a prepared HiRID dataset for sequential Mortality prediction with an LGBM Classifier: 
```
icu-benchmarks train \
    -d ../data/mortality_seq/hirid \
    -n hirid \
    -t Mortality_At24Hours \
    -m LGBMClassifier \
    -hp LGBMClassifier.subsample=1.0 model/random_search.num_leaves='[20,40,60]' \
    -c \
    -s 1111 2222 3333 4444 5555
```
> Run with `PYTORCH_ENABLE_MPS_FALLBACK=1` on Macs with Metal Performance Shaders.

> Please note that, for Windows based systems, paths need to be formatted differently, e.g: ` r"\..\data\mortality_seq\hirid"`.
> Additionally, the next line character (\\)  needs to be replaced by (^) (Command Prompt) or (`) (Powershell) respectively.

### Random Search in Configs
To understand how a parameter can be searched via random search, let's look at the following example configuration:
```
...
# Optimizer params
Adam.weight_decay = 1e-6
optimizer/random_search.class_to_configure = @Adam
optimizer/random_search.lr = [3e-4, 1e-4, 3e-5, 1e-5]

# Encoder params
LSTMNet.input_dim = %EMB
LSTMNet.num_classes = %NUM_CLASSES
model/random_search.class_to_configure = @LSTMNet
model/random_search.hidden_dim = [32, 64, 128, 256]
model/random_search.layer_dim = [1, 2, 3]

run_random_searches.scopes = ["model", "optimizer"]
```
`run_random_searches.scopes` defines the scopes that the random search runs in (the strings in front of the slashes in the lines above).
Each scope represents a class which will get bindings with randomly searched parameters.
In this example, we have the two scopes `model` and `optimizer`.
For each scope a `class_to_configure` needs to be set to the class it represents, in this case `LSTMNet` and `Adam` respectively.
Then, we can add whichever parameter we want to the classes like this: `model/random_search.<param> = ['list', 'of', 'possible', 'values']`.
The scopes take care of adding the parameters only to the pertinent classes, whereas the `random_search()` function actually randomly choses a value
and binds it to the gin configuration.

If we run `experiments` and want to overwrite the model configuration, this can be done easily:
```
include "configs/tasks/Mortality_At24Hours.gin"
include "configs/models/LSTM.gin"

Adam.lr = 1e-4

model/random_search.hidden_dim = [100, 200]
```
This configuration for example overwrites the `lr` parameter of `Adam` with a concrete value,
while it only specifies a different search space for `hidden_dim` of `LSTMNet` to run the random search on.

The same holds true for the command line. Setting the following flag would achieve the same result (make sure to only have spaces between parameters):
```
-hp Adam.lr=1e-4 model/random_search.hidden_dim='[100,200]'
```

### Output Structure
```
log_dir/
├── dataset1/
│   ├── task1/
│   │   ├── model1/
│   │   │   ├── YYYY-MM-DDTHH-MM-SS (run1)/
│   │   │   │   ├── HYPER_PARAMS
│   │   │   │   ├── seed1/
│   │   │   │   │   ├── model
│   │   │   │   │   ├── train_config.gin
│   │   │   │   │   └── metrics
│   │   │   │   └── seed2/
│   │   │   │       └── ...
│   │   │   ├── YYYY-MM-DDTHH-MM-SS (run2)/
│   │   │   │   └── ...
│   │   └── model2/
│   │       └── ...
│   └── task2/
│       └── ...
└── dataset2/
    └── ...
```

## Evaluate
It is possible to evaluate a model trained on another dataset. In this case, the source dataset is HiRID and the target is MIMIC-IV:
```
icu-benchmarks evaluate \
    -d ../data/mortality_seq/miiv \
    -n miiv \
    -t Mortality_At24Hours \
    -m LGBMClassifier \
    -sn hirid \
    --source-dir ../data/mortality_seq/hirid/logs/hirid/Mortality_At24Hours/LGBMClassifier/2022-11-10T22-52-52/seed_1111 \
    -c \
    -s 1111 2222 3333 4444 5555
```

### Output Structure
The benchmark generates an output structure that takes into account multiple aspects of the training and evaluation 
specifications:
<pre>
log_dir/
├── dataset1/
│   ├── task1/
│   │   ├── model1/
│   │   │   ├── YYYY-MM-DDTHH-MM-SS (run1)/
│   │   │   │   ├── HYPER_PARAMS
│   │   │   │   ├── seed1/
│   │   │   │   │   ├── model
│   │   │   │   │   ├── train_config.gin
│   │   │   │   │   └── metrics
│   │   │   │   └── seed2/
│   │   │   │       └── ...
│   │   │   ├── YYYY-MM-DDTHH-MM-SS (run2)/
│   │   │   │   └── ...
<b>│   │   │   ├── from_dataset2/
│   │   │   │   ├── YYYY-MM-DDTHH-MM-SS (run1)/
│   │   │   │   │   ├── seed1/
│   │   │   │   │   │   ├── train_config.gin
│   │   │   │   │   │   └── metrics
│   │   │   │   │   └── seed2/
│   │   │   │   │       └── ...
│   │   │   │   └── YYYY-MM-DDTHH-MM-SS (run2)/
│   │   │   │       └── ...
│   │   │   └── from_dataset3/
│   │   │       └── ...</b>
│   │   └── model2/
│   │       └── ...
│   └── task2/
│       └── ...
└── dataset2/
    └── ...
</pre>

# Development
## Directories
Note: redo this for the first release

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
We currently use the following libraries for development:
- [Pytorch](https://pytorch.org/) 
    - An open source machine learning framework for 
- [Pytorch Ignite](https://github.com/pytorch/ignite)
    - Library for training and evaluating neural networks in Pytorch
- [GIN](https://github.com/google/gin-config)
    - Gin provides a lightweight configuration framework for Python

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
