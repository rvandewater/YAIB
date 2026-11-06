# Preprocessing

Preprocessing turns the raw cohort into model inputs. It runs once per fold, after the stays are split, and is fitted on
the training split only, so no information from the validation or test stays leaks into it. The steps are
[ReciPys](https://github.com/rvandewater/ReciPys) recipes.

## The default pipelines

The task configs select a preprocessor with `preprocess.preprocessor`:

| Gin name                           | Class                              | Used by                                                               |
| ---------------------------------- | ---------------------------------- | --------------------------------------------------------------------- |
| `base_classification_preprocessor` | `PolarsClassificationPreprocessor` | `BinaryClassification.gin`                                            |
| `base_regression_preprocessor`     | `PolarsRegressionPreprocessor`     | `Regression.gin` (adds [target scaling](../user-guide/regression.md)) |
| `base_imputation_preprocessor`     | `PandasImputationPreprocessor`     | `DatasetImputation.gin`                                               |

The classification and regression pipelines apply these steps:

**Static variables**

1. Add a missingness indicator for every variable.
2. Scale the numeric variables (with `scaling=True`) and fill their missing values with 0, the training mean.
3. Fill missing categorical values with the most frequent one and encode the categories as integers (unseen categories
    become -1).

**Dynamic variables**

1. Scale the variables (with `scaling=True`).
2. Add a `MissingIndicator_<name>` column that is 1 where a value was not measured.
3. With `generate_features=True` (the default for classical models), add features over the stay so far: the count of
    measurements, and the minimum, maximum and mean of each variable.
4. With a [pretrained imputation model](../user-guide/imputation.md#use-a-trained-imputation-model) (`-imp`), impute
    the missing values with it.
5. Fill the remaining missing values forward in time, then with 0 (the training mean) before the first measurement.

Arguments of the preprocessors, settable with gin:

| Argument              | Default                                                                  | Meaning                                                             |
| --------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------- |
| `generate_features`   | `True` for ML, `False` for DL (set in `MLCommon.gin` and `DLCommon.gin`) | Add the summary features over time.                                 |
| `scaling`             | `True`                                                                   | Scale the features.                                                 |
| `use_static_features` | `True`                                                                   | Use the static variables.                                           |
| `vars_to_exclude`     | `None`                                                                   | Variables that get no missingness indicators or generated features. |

For example, `-hp 'base_classification_preprocessor.scaling=False'` turns off feature scaling.

## Writing a custom preprocessor

A preprocessor subclasses `Preprocessor` and implements two methods:

- `apply(data, vars)` takes the split data, `data[split][segment]` for the `train`, `val` and `test` splits and the
    `STATIC`, `DYNAMIC` and `OUTCOME` segments, and returns it preprocessed, with the features in a `FEATURES` segment.
- `to_cache_string()` returns a string that identifies the preprocessor and its settings; it is part of the
    [cache](../user-guide/running-experiments.md#caching-the-preprocessed-data) key.

The simplest way is to subclass a default preprocessor and change one part:

```python title="my_preprocessor.py"
import gin

from icu_benchmarks.data.preprocessor import PolarsClassificationPreprocessor


@gin.configurable("my_preprocessor")
class CustomPreprocessor(PolarsClassificationPreprocessor):
    def __init__(self, *args, clip_quantile: float = 0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_quantile = clip_quantile

    def _process_dynamic(self, data, vars):
        # change the dynamic steps here
        return super()._process_dynamic(data, vars)

    def to_cache_string(self):
        return super().to_cache_string() + f"_clip_{self.clip_quantile}"
```

Pass the file with `-p`:

```bash
uv run icu-benchmarks train ... -p my_preprocessor.py -hp my_preprocessor.clip_quantile=0.95
```

- The file must define a class named `CustomPreprocessor`; YAIB raises an error otherwise.
- YAIB imports the file before it parses the configs, so the configs and `-hp` can bind the preprocessor's gin
    parameters, and it selects the preprocessor after the configs, so the task config's default does not replace it.
- The run's `train_config.gin` records the file path. Evaluation, fine-tuning and `--samples` runs that read that config
    load the same preprocessor; keep the file at its path.

To make a preprocessor the default for a task instead, register it with gin and bind it in the task config:
`preprocess.preprocessor = @my_preprocessor`.
