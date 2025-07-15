Development
===========

YAIB is in active development. The following sections could be relevant for adding new code to our repository.

Libraries
---------

The following libraries are important to the operation of YAIB:

Core Dependencies
~~~~~~~~~~~~~~~~~

- `Pandas <https://github.com/pandas-dev/pandas>`_: Popular data structure framework.
- `ReciPys <https://github.com/rvandewater/recipys>`_: A modular preprocessing package for Pandas dataframes.
- `Pytorch <https://pytorch.org/>`_: An open source machine learning framework for deep learning applications.
- `Pytorch Lightning <https://www.pytorchlightning.ai/>`_: A lightweight Pytorch wrapper for AI research.
- `Pytorch Ignite <https://github.com/pytorch/ignite>`_: Library for training and evaluating neural networks in Pytorch.
- `Cuda Toolkit <https://developer.nvidia.com/cuda-toolkit>`_: GPU acceleration used for deep learning models.
- `Scikit-learn <https://github.com/scikit-learn/scikit-learn>`_: Machine learning library.
- `Scikit-optimize <https://scikit-optimize.github.io/stable/>`_: Used for Bayesian optimization.
- `LightGBM <https://github.com/microsoft/LightGBM>`_: Gradient boosting framework.
- `GIN <https://github.com/google/gin-config>`_: Provides a lightweight configuration framework for Python.
- `Wandb <https://wandb.ai/>`_: A tool for visualizing and tracking machine learning experiments.
- `Pytest <https://docs.pytest.org/en/stable/>`_: A testing framework for Python.

Imputation Libraries
~~~~~~~~~~~~~~~~~~~~

- `HyperImpute <https://github.com/vanderschaarlab/hyperimpute>`_: Imputation library for MissForest and GAIN.
- `PyPOTS <https://github.com/WenjieDu/PyPOTS>`_: Imputation library.

Running Tests
-------------

To run the test suite:

.. code-block:: bash

   python -m pytest ./tests/recipes
   coverage run -m pytest ./tests/recipes

   # then use either of the following
   coverage report
   coverage html

Code Formatting and Linting
----------------------------

For development purposes, we use the ``Black`` package to autoformat our code and a ``Flake8`` linting/CI check:

.. code-block:: bash

   black . -l 127
   flake8 . --count --max-complexity=14 --max-line-length=127 --statistics