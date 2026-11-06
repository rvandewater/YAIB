# API reference

This reference is generated from the docstrings in the `icu_benchmarks` package, which use the Google style.

| Module                                            | Contents                                                                                                   |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| [Entry point and run utilities](run.md)           | The `icu-benchmarks` CLI, argument parsing, run directories, result aggregation, repeated cross-validation |
| [Data](data.md)                                   | Splitting and preprocessing (`preprocess_data`), preprocessors, datasets, target scaling, pooling          |
| [Training and model wrappers](models.md)          | `train_common`, `load_model`, the ML/DL/imputation wrappers, metrics                                       |
| [Deep learning models](dl-models.md)              | RNN, LSTM, GRU, GRU-D, TCN, Transformer, Bi-Axial Transformer, Mamba                                       |
| [Classical machine learning models](ml-models.md) | Logistic regression, elastic net, random forest, SVM, LightGBM, XGBoost, CatBoost, imbalanced-learn        |
| [Imputation models](imputation.md)                | Baselines, MLP, RNN, neural processes, CSDI, diffusion and SSSD models, amputation                         |
| [Hyperparameter tuning](tuning.md)                | The Optuna tuner and its gin helpers                                                                       |

Most classes and functions are gin configurables: any of their arguments can be set in a `.gin` file or with `-hp` on the
command line, as described in [Configuration](../user-guide/configuration.md).
