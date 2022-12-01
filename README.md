![YAIB logo](docs/figures/yaib_logo.png)

# Yet Another ICU Benchmark

[![CI](https://github.com/rvandewater/YAIB/actions/workflows/ci.yml/badge.svg?branch=development)](https://github.com/rvandewater/YAIB/actions/workflows/ci.yml)

Yet another ICU benchmark (YAIB) aims to
This package provides a framework for doing clinical machine learning experiments.
We support the following datasets out of the box:

| Dataset                 | MIMIC-III / IV                                                                                                                            | eICU-CRD                                             | HiRID                                                 | AUMCdb         |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------|-------------------------------------------------------|----------------|
| Admissions              | 40k / 50k                                                                                                                                 | 200k                                                 | 33k                                                   | 23k            |
| Frequency (time-series) | 1 hour                                                                                                                                    | 5 minutes                                            | 2 / 5 minutes                                         | up to 1 minute |
| Origin                  | USA                                                                                                                                       | USA                                                  | Switzerland                                           | Netherlands    |

The benchmark is designed for operating on preprocessed parquet files. We refer to the PyICU (in development)
or [ricu package](https://github.com/eth-mds/ricu) for generating these parquet files for particular cohorts and endpoints.

We provide several common tasks for clinical prediction:

| No  | Task Theme                | Temporality        | Type                                | 
|-----|---------------------------|--------------------|-------------------------------------|
| 1   | ICU Mortality             | Hourly (after 24H) | Sequential Classification           |
| 2   | Acute Kidney Injury (AKI) | Hourly             | Sequence to Sequence Classification |
| 3   | Sepsis                    | Hourly             | Sequence to Sequence Classification |
| 4   | Circulatory Failure       | 5 Minutes          | Sequence to Sequence Classification |
| 5   | Length of Stay (LoS)      | Hourly             | Sequence to Sequence Regression     | 

Please refer to [cohort definitions]() for further information.

## Paper

If you use this code in your research, please cite the following publication:

```
```

This paper can also be found on arxiv: TBD

# Installation

```
conda env update -f <environment.yml|environment_mps.yml>
conda activate yaib
pip install -e .
```

> Use `environment.yml` on x86 hardware and `environment_mps.yml` on Macs with Metal Performance Shaders.

> Note that the last command installs the package called `icu-benchmarks`.

# Usage

## Getting the Datasets

HiRID, eICU, and MIMIC IV can be accessed through [PhysioNet](https://physionet.org/). A guide to this process can be
found [here](https://eicu-crd.mit.edu/gettingstarted/access/).
AUMCdb can be accessed through a seperate access [procedure](https://github.com/AmsterdamUMC/AmsterdamUMCdb). We do not have
involvement in the access procedure and can not answer to any requests for data access.
## Extracting cohorts
TODO
## Preprocess and Train

The following command will start training on a prepared HiRID dataset for sequential Mortality prediction with an LGBM
Classifier:

```
icu-benchmarks train \
    -d ../data/mortality_seq/hirid \
    -n hirid \
    -t Mortality_At24Hours \
    -m LGBMClassifier \
    -hp LGBMClassifier.subsample=1.0 model/random_search.num_leaves=[20,40,60] \
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

`run_random_searches.scopes` defines the scopes that the random search runs in (the strings in front of the slashes in the
lines above).
Each scope represents a class which will get bindings with randomly searched parameters.
In this example, we have the two scopes `model` and `optimizer`.
For each scope a `class_to_configure` needs to be set to the class it represents, in this case `LSTMNet` and `Adam`
respectively.
We can add whichever parameter we want to the classes following this syntax:

```
run_random_searches.scopes = ["<scope>", ...]
<scope>/random_search.class_to_configure = @<SomeClass>
<scope>/random_search.<param> = ['list', 'of', 'possible', 'values']
```

The scopes take care of adding the parameters only to the pertinent classes, whereas the `random_search()` function actually
randomly choses a value
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

The same holds true for the command line. Setting the following flag would achieve the same result (make sure to only have
spaces between parameters):

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

It is possible to evaluate a model trained on another dataset. In this case, the source dataset is HiRID and the target is
MIMIC-IV:

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

## Metrics

Several metrics are defined for this benchmark:

- Binary Classification: Because our tasks are all highly imbalanced, we use both ROC and PR Area Under the Curve
  using [sklearn.metrics.roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
  and [sklearn.metrics.average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score)
- Regression : The Mean Absolute Error (MAE) is used
  with [sklearn.metrics.mean_absolute_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)

# Development

YAIB is in active development. The following sections could be relevant for adding new code to our repository

## Libraries

The following libraries are important to the operation of YAIB:

- [Pandas](https://github.com/pandas-dev/pandas): Popular data structure framework.
- [ReciPys](https://github.com/rvandewater/recipys): A modular preprocessing package for Pandas dataframes.
- [Pytorch](https://pytorch.org/): An open source machine learning framework for deep learning applications.
- [Pytorch Ignite](https://github.com/pytorch/ignite): Library for training and evaluating neural networks in Pytorch.
- [Cuda Toolkit](https://developer.nvidia.com/cuda-toolkit): GPU acceleration used for deep learning models.
- [Scikit-learn](https://github.com/scikit-learn/scikit-learn): Machine learning library.
- [LightGBM](https://github.com/microsoft/LightGBM): Gradient boosting framework.
- [GIN](https://github.com/google/gin-config): Provides a lightweight configuration framework for Python.

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

# Acknowledgements

We do not own any of the datasets used in this benchmark. This project uses heavily adapted components of
the [HiRID benchmark](https://github.com/ratschlab/HIRID-ICU-Benchmark/).