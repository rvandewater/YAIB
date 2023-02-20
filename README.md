![YAIB logo](docs/figures/yaib_logo.png)

# Yet Another ICU Benchmark

[![CI](https://github.com/rvandewater/YAIB/actions/workflows/ci.yml/badge.svg?branch=development)](https://github.com/rvandewater/YAIB/actions/workflows/ci.yml)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Platform](https://img.shields.io/badge/platform-linux--64%20|%20win--64%20|%20osx--64-lightgrey)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

[//]: # (TODO: add coverage once we have some tests )

Yet another ICU benchmark (YAIB) provides a framework for doing clinical machine learning experiments on (ICU) EHR data.
We support the following datasets out of the box:

| Dataset                 | [MIMIC-III](https://physionet.org/content/mimiciii/) / [IV](https://physionet.org/content/mimiciv/) | [eICU-CRD](https://physionet.org/content/eicu-crd/) | [HiRID](https://physionet.org/content/hirid/1.1.1/) | [AUMCdb](https://doi.org/10.17026/dans-22u-f8vd) |
|-------------------------|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------|--------------------------------------------------|
| Admissions              | 40k / 50k                                                                                           | 200k                                                | 33k                                                 | 23k                                              |
| Frequency (time-series) | 1 hour                                                                                              | 5 minutes                                           | 2 / 5 minutes                                       | up to 1 minute                                   |
| Origin                  | USA                                                                                                 | USA                                                 | Switzerland                                         | Netherlands                                      |

The benchmark is designed for operating on preprocessed parquet files. We refer to the PyICU (in development)
or [ricu package](https://github.com/eth-mds/ricu) for generating these parquet files for particular cohorts and endpoints.

We provide several common tasks for clinical prediction:

| No  | Task Theme                | Temporality        | Type                                | 
|-----|---------------------------|--------------------|-------------------------------------|
| 1   | ICU Mortality             | Hourly (after 24H) | Sequential Classification           |
| 2   | Acute Kidney Injury (AKI) | Hourly (within 6H) | Sequence to Sequence Classification |
| 3   | Sepsis                    | Hourly (within 6H) | Sequence to Sequence Classification |

[//]: # (| 4   | Circulatory Failure       | 5 Minutes          | Sequence to Sequence Classification |)

[//]: # (| 5   | Length of Stay &#40;LoS&#41;      | Hourly             | Sequence to Sequence Regression     | )


Please refer to [cohort definitions]() for further information.

## Paper

If you use this code in your research, please cite the following publication:

```
```

This paper can also be found on arxiv: TBD

# Installation

YAIB can be installed using conda or pip. Below you will find the three CLI commands to install YAIB using conda.
The

The first command will install an environment based on Python 3.10 (currently).

```
conda env update -f <environment.yml|environment_mps.yml>
```

> Use `environment.yml` on x86 hardware and `environment_mps.yml` on Macs with Metal Performance Shaders.

We then activate the environment and install a package called `icu-benchmarks`, after which YAIB should be operational.

```
conda activate yaib
pip install -e .
```

If you want to install the icu-benchmarks package with pip, execute the command below:

```
pip install torch numpy && pip install -e .
```

If you are on a Mac with Metal Performance Shader, install the package with the following command:

```
pip install torch numpy && pip install -e .[mps]
```

# Usage

## Getting the Datasets

HiRID, eICU, and MIMIC IV can be accessed through [PhysioNet](https://physionet.org/). A guide to this process can be
found [here](https://eicu-crd.mit.edu/gettingstarted/access/).
AUMCdb can be accessed through a separate access [procedure](https://github.com/AmsterdamUMC/AmsterdamUMCdb). We do not have
involvement in the access procedure and can not answer to any requests for data access.

## Data Conversion

Since the datasets were created independently of each other, they do not share the same data structure or data identifiers. In
order to make them interoperable, use the preprocessing utilities
provided by the [ricu package](https://github.com/eth-mds/ricu).
Ricu pre-defines a large number of clinical concepts and how to load them from a given dataset, providing a common interface to
the data, that is used in this
benchmark.

### Extracting cohorts

TODO

# Data

YAIB expects data generated by [pyicu](https://github.com/prockenschaub/pyicu), a
rewritten [ricu](https://github.com/prockenschaub/ricu-package) for Python.

## Demo data

In the folder `demo_data` we provide processed publicly available demo datasets from eICU and MIMIC with the necessary lables
for `Akute Kidney Injury`, `Mortality at 24h` and `Sepsis`.

# Use with CLI Commands

## Preprocess and Train

The following command will run training and evaluation on the MIMIC demo dataset for Mortality prediction at 24h with the
LGBMClassifier.
Child samples are reduced due to the small amount of training data.

```
icu-benchmarks train \
    -d demo_data/mortality24/mimic_demo \
    -n mimic_demo \
    -t BinaryClassification \
    -tn Mortality24 \
    -m LGBMClassifier \
    -hp LGBMClassifier.min_child_samples=10 \
    --generate_cache
    --load_cache \
    --seed 2222 \
    -l ../yaib_logs/ \
    --tune
```

> For a list of available flags, run `icu-benchmarks train -h`.

> Run with `PYTORCH_ENABLE_MPS_FALLBACK=1` on Macs with Metal Performance Shaders.

[//]: # (> Please note that, for Windows based systems, paths need to be formatted differently, e.g: ` r"\..\data\mortality_seq\hirid"`.)
> For Windows based systems, the next line character (\\)  needs to be replaced by (^) (Command Prompt) or (`) (Powershell)
> respectively.

### Hyperparameter Tuning

To understand how a parameter can be automatically tuned via bayesian optimization, let's look at the following example
configuration:

```
...
# Optimizer params
optimizer/hyperparameter.class_to_tune = @Adam
optimizer/hyperparameter.weight_decay = 1e-6
optimizer/hyperparameter.lr = (1e-5, 3e-4)

# Encoder params
model/hyperparameter.class_to_tune = @LSTMNet
model/hyperparameter.num_classes = %NUM_CLASSES
model/hyperparameter.hidden_dim = (32, 256)
model/hyperparameter.layer_dim = (1, 3)

tune_hyperparameters.scopes = ["model", "optimizer"]  # defines the scopes that the random search runs in
tune_hyperparameters.n_initial_points = 5  # defines random points to initilaize gaussian process
tune_hyperparameters.n_calls = 30  # numbe rof iterations to find best set of hyperparameters
tune_hyperparameters.folds_to_tune_on = 2  # number of folds to use to evaluate set of hyperparameters
```

In this example, we have the two scopes `model` and `optimizer`, the scopes take care of adding the parameters only to the
pertinent classes.
For each scope a `class_to_tune` needs to be set to the class it represents, in this case `LSTMNet` and `Adam`
respectively.
We can add whichever parameter we want to the classes following this syntax:

```
tune_hyperparameters.scopes = ["<scope>", ...]
<scope>/hyperparameter.class_to_tune = @<SomeClass>
<scope>/hyperparameter.<param> = ['list', 'of', 'possible', 'values']
```

If we run `experiments` and want to overwrite the model configuration, this can be done easily:

```
include "configs/tasks/Mortality_At24Hours.gin"
include "configs/models/LSTM.gin"

optimizer/hyperparameter.lr = 1e-4

model/hyperparameter.hidden_dim = [100, 200]
```

This configuration for example overwrites the `lr` parameter of `Adam` with a concrete value,
while it only specifies a different search space for `hidden_dim` of `LSTMNet` to run the random search on.

The same holds true for the command line. Setting the following flag would achieve the same result (make sure to only have
spaces between parameters):

```
-hp optimizer/hyperparameter.lr=1e-4 model/hyperparameter.hidden_dim='[100,200]'
```

There is an implicit hierarchy, independent of where the parameters are added (`model.gin`, `experiment.gin` or CLI `-hp`):

```
LSTM.hidden_dim = 8                         # always takes precedence
model/hyperparameter.hidden_dim = 6         # second most important
model/hyperparameter.hidden_dim = (4, 6)    # only evaluated if the others aren't found in gin configs and CLI
```

The hierarchy CLI `-hp` > `experiment.gin` > `model.gin` is only important for bindings on the same "level" from above.

## Evaluate

It is possible to evaluate a model trained on another dataset. In this case, the source dataset is the demo data from MIMIC and
the target is the eICU demo:

```
icu-benchmarks evaluate \
    -d demo_data/mortality24/eicu_demo \
    -n eicu_demo \
    -t BinaryClassification \
    -tn Mortality24 \
    -m LGBMClassifier \
    -c \
    -s 2222 \
    -l ../yaib_logs \
    -sn mimic \
    --source-dir ../yaib_logs/mimic_demo/Mortality24/LGBMClassifier/2022-12-12T15-24-46/fold_0
```

## Imputation

Below is an example call for training an imputation model
```
icu-benchmarks train \
    -d demo_data/mortality24/mimic_demo \
    -n mimic_demo \
    -t DatasetImputation \
    -m Mean \
    -lc -gc \
    -s 2222 \
    -l ../yaib_logs/ 
```

For more details on how to implement new imputation methods, visit [this document](docs/adding_new_imputation_methods.md).

## Hyperparameter Optimization using Weights and Biases Sweeps

[This sweep file](wandb_sweep.yaml) shows an example on how to run a hyperparameter sweep with W&B. The general structure of the YAML should look like this:
``` yaml
program: icu-benchmarks
command:
  - ${env}
  - ${program}
  - "train"
# .... other program parameters ....
  - "--wandb-sweep"
method: grid
parameters:
  # gin config parameter name:
    # values: [a, b, etc...]
  # example:
  ImputationDataset.mask_method:
    values: ["MCAR", "MAR", "MNAR"]
```

You can then create a sweep with 
``` bash
wandb sweep path/to/sweep_file.yaml
```
which will give you a sweep id.

and start an agent to perform the optimization using the following command:
``` bash
wandb agent YOUR_SWEEP_ID
```

## Training a Classification model using a pretrained imputation model

Below is an example call to train a classification model using a pretrained imputation model:
``` bash
icu-benchmarks train \
    -d demo_data/mortality24/mimic_demo \
    -n mimic_demo \
    -t BinaryClassificationPretrainedImputation \
    -tn Mortality24 \
    -m LGBMClassifier \
    -hp LGBMClassifier.min_child_samples=10 \
    -c \
    -s 2222 \
    -l ../yaib_logs/ \
    --use_pretrained_imputation path/to/pretrained/imputation_model.ckpt
    --tune
```

Where `path/to/pretrained/imputation_model.ckpt` is the path to the `model.ckpt` created by training an imputation model with our framework.

## Metrics

Several metrics are defined for this benchmark:

- Binary Classification: Because our tasks are all highly imbalanced, we use both ROC and PR Area Under the Curve
  using [sklearn.metrics.roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
  and [sklearn.metrics.average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score)
- Regression : The Mean Absolute Error (MAE) is used
  with [sklearn.metrics.mean_absolute_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)

## Models

We provide several existing machine learning models that are commonly used for multivariate time-series data.
`pytorch` is used for the deep learning models, `lightgbm` for the boosted tree approaches, and `sklearn` for the logistic
regression model and metrics.
The benchmark provides the following built-in models:

- [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic+regression):
  Standard regression approach.
- [LightGBM](https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf): Efficient gradient
  boosting trees.
- [Long Short-term Memory (LSTM)](https://ieeexplore.ieee.org/document/818041): The most commonly used type of Recurrent Neural
  Networks for long sequences.
- [Gated Recurrent Unit (GRU)](https://arxiv.org/abs/1406.1078) : A extension to LSTM which showed improvement over them in the
  context of polyphonic music modeling and speech signal modeling ([paper](https://arxiv.org/abs/1412.3555)).
- [Temporal Convolutional Networks (TCN)](https://arxiv.org/pdf/1803.01271 ): 1D convolution approach to sequence data. By
  using dilated convolution to extend the receptive field of the network it has shown great performance on long-term
  dependencies.
- [Transformers](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf): The most common Attention
  based approach.

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

We do not own any of the datasets used in this benchmark. This project uses adapted components of
the [HiRID benchmark](https://github.com/ratschlab/HIRID-ICU-Benchmark/). We thank the authors for providing this codebase.

# License

This source code is released under the MIT license, included [here](LICENSE).
