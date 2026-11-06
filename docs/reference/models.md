# Training and model wrappers

`train_common` trains and evaluates one fold. Every model subclasses one of the wrappers: `MLWrapper` for classical models, `DLPredictionWrapper` for PyTorch Lightning models and `ImputationWrapper` for imputation models.

::: icu_benchmarks.models.train

::: icu_benchmarks.models.wrappers

::: icu_benchmarks.models.constants

::: icu_benchmarks.models.custom_metrics

::: icu_benchmarks.models.utils
