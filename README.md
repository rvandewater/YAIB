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

# CLI Commands

## Setup

```
conda env update -f <environment.yml|environment_mps.yml>
conda activate yaib
pip install -e .
```

> Use `environment.yml` on Intel hardware, `environment_mps.yml` on Macs with Metal Performance Shaders

## Preprocess and Train
The following command will start training on a prepared HiRID dataset for sequential Mortality prediction with an LGBM Classifier: 
```
icu-benchmarks train \
    -d ../data/mortality_seq/hirid \
    -n hirid \
    -t Mortality_At24Hours \
    -m LGBMClassifier \
    -hp LGBMClassifier.subsample='RS([0.33,0.66])' LGBMClassifier.colsample_bytree=0.66 \
    -c \
    -s 1111 2222 3333 4444 5555
```
> `RS([...])` is the syntax for invoking random search on a list of hyperparameters, both in configs and the command line.

> Run with `PYTORCH_ENABLE_MPS_FALLBACK=1` on Macs with Metal Performance Shaders

> Please note that, for Windows based systems, paths need to be formatted differently, e.g: ` r"\..\data\mortality_seq\hirid"`.
> Additionally, the next line character (\\)  needs to be replaced by (^) (Command Prompt) or (`) (Powershell) respectively.

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
<pre><b>│   │   │   ├── from_dataset2/
│   │   │   │   ├── YYYY-MM-DDTHH-MM-SS (run1)/
│   │   │   │   │   ├── seed1/
│   │   │   │   │   │   ├── train_config.gin
│   │   │   │   │   │   └── metrics
│   │   │   │   │   └── seed2/
│   │   │   │   │       └── ...
│   │   │   │   └── YYYY-MM-DDTHH-MM-SS (run2)/
│   │   │   │       └── ...
│   │   │   └── from_dataset3/
│   │   │       └── ...</pre></b>
│   │   └── model2/
│   │       └── ...
│   └── task2/
│       └── ...
└── dataset2/
    └── ...
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
