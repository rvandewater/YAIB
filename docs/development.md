# Development

YAIB is in active development. The following sections could be relevant for adding new code to our repository

## Libraries

The following libraries are important to the operation of YAIB:

- [Pandas](https://github.com/pandas-dev/pandas): Popular data structure framework.
- [ReciPys](https://github.com/rvandewater/recipys): A modular preprocessing package for Pandas dataframes.
- [Pytorch](https://pytorch.org/): An open source machine learning framework for deep learning applications.
- [Pytorch Lightning](https://www.pytorchlightning.ai/): A lightweight Pytorch wrapper for AI research.
- [Pytorch Ignite](https://github.com/pytorch/ignite): Library for training and evaluating neural networks in Pytorch.
- [Cuda Toolkit](https://developer.nvidia.com/cuda-toolkit): GPU acceleration used for deep learning models.
- [Scikit-learn](https://github.com/scikit-learn/scikit-learn): Machine learning library.
- [Scikit-optimize](https://scikit-optimize.github.io/stable/): Used for Bayesian optimization.
- [LightGBM](https://github.com/microsoft/LightGBM): Gradient boosting framework.
- [GIN](https://github.com/google/gin-config): Provides a lightweight configuration framework for Python.
- [Wandb](https://wandb.ai/): A tool for visualizing and tracking machine learning experiments.
- [Pytest](https://docs.pytest.org/en/stable/): A testing framework for Python.
### Imputation
- [HyperImpute](https://github.com/vanderschaarlab/hyperimpute): Imputation library for MissForest and GAIN.
- [PyPOTS](https://github.com/WenjieDu/PyPOTS): Imputation library.
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
