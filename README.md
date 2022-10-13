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

## Preprocess and Train

```
python -m icu_benchmarks.run train \
                            --data-dir ../data/ricu/mimic \
                            -c configs/tasks/Classification/LogisticRegression.gin \
                            -l logs/benchmark_exp/LogisticRegression/ \
                            -t Dynamic_CircFailure_12Hours\
                            -o True \
                            --penalty 'l2' \
                            --c_parameter 0.01 \
                            -sd 1111 2222 3333 4444 5555 6666 7777 8888 9999 0000

python -m icu_benchmarks.run train \
                            --data-dir ../data/ricu/mimic \
                            -c configs/tasks/Classification/LGBM.gin \
                            -l logs/ricu/random_search/24_binary/LGBM/run \
                            -t Mortality_At24Hours \
                            -rs True\
                            -sd 1111 2222 3333 \
                            --depth 3 4 5 6 7 \
                            --loss-weight balanced None \
                            --subsample-feat 0.33 0.66 1.00 \
                            --subsample-data 0.33 0.66 1.00
```

## Run Tests

```
python -m pytest ./tests/recipes
coverage run -m pytest ./tests/recipes
# then use either of the following
coverage report
coverage html
```

## Autoformat
```
black . -l 127
```
