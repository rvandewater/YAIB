# Running experiments

An experiment in YAIB has three parts, each chosen on the command line:

| Part        | Flag                        | Where it is defined                                                                                       |
| ----------- | --------------------------- | --------------------------------------------------------------------------------------------------------- |
| The dataset | `-d` (folder), `-n` (name)  | A folder with `dyn.parquet`, `sta.parquet` and `outc.parquet`                                             |
| The task    | `-t` (config), `-tn` (name) | `configs/tasks/<task>.gin`: run mode, variables, preprocessing, cross-validation                          |
| The model   | `-m`                        | `configs/prediction_models/<model>.gin`: the model, its training settings and hyperparameter search space |

The configs are [gin](https://github.com/google/gin-config) files. They bind values to the arguments of YAIB's classes
and functions; see [Configuration](configuration.md).

## Example: an LSTM for mortality prediction

```bash
uv run icu-benchmarks train \
    -d demo_data/mortality24/mimic_demo \
    -n mimic_demo \
    -t BinaryClassification \
    -tn Mortality24 \
    -m LSTM \
    --seed 2222 \
    -l ../yaib_logs/ \
    --tune
```

### The task config

`-t BinaryClassification` loads `configs/tasks/BinaryClassification.gin`:

```python title="configs/tasks/BinaryClassification.gin"
--8<--
configs/tasks/BinaryClassification.gin
--8<--
```

It sets the run mode, weights the classes for the imbalanced labels, selects the loss for deep learning models and the
preprocessor, and includes three shared files:

- `PredictionTaskVariables.gin` names the three data files and the variables and their roles:

    ```python title="configs/tasks/common/PredictionTaskVariables.gin (shortened)"
    preprocess.file_names = {
        "DYNAMIC": "dyn.parquet",
        "OUTCOME": "outc.parquet",
        "STATIC": "sta.parquet",
    }

    vars = {
        "GROUP": "stay_id",  # one stay per group
        "SEQUENCE": "time",  # the time axis of the dynamic data
        "LABEL": "label",  # the outcome to predict
        "DYNAMIC": ["alb", "alp", "alt", ..., "urine", "wbc"],
        "STATIC": ["age", "sex", "height", "weight"],
    }
    ```

- `CrossValidation.gin` sets 5 repetitions × 5 folds (`execute_repeated_cv.cv_repetitions` and `cv_folds`).

- `Dataloader.gin` passes the variables to the datasets.

`configs/tasks/Regression.gin` is the same for regression; `DatasetImputation.gin` trains
[imputation models](imputation.md).

### The model config

`-m LSTM` loads `configs/prediction_models/LSTM.gin`:

```python title="configs/prediction_models/LSTM.gin"
--8<--
configs/prediction_models/LSTM.gin
--8<--
```

`DLCommon.gin` holds the settings shared by all deep learning models: no feature generation in preprocessing, the Adam
optimizer, up to 50 epochs with early stopping after 10 epochs without improvement, batch size 64, and the tuning
budget (`DLTuning.gin`). Classical models include `MLCommon.gin` instead, which turns on feature generation (summary
statistics over time) for the dynamic variables.

The `hyperparameter` bindings define the search space: a tuple `(low, high)` is a range, a list is a set of choices, and
a single value is fixed. See [Hyperparameter tuning](hyperparameter-tuning.md).

## What a run does

1. **Configure:** parse the task and model configs, then apply `-hp` overrides.
2. **Tune** (with `--tune`): search the hyperparameters with Optuna on the first folds, then bind the best ones.
3. **Cross-validate:** for each repetition and fold, split the stays into train, validation and test sets, preprocess
    them (fitting the preprocessing on the training split only), train the model, and evaluate it on the test split.
4. **Aggregate:** collect the metrics of all folds, with their mean and confidence interval.

Results go to `<log-dir>/<name>/<task-name>/<model>/<timestamp>/`; the [Quickstart](../getting-started/quickstart.md#look-at-the-results)
shows the files.

## Caching the preprocessed data

Preprocessing the full datasets takes time. With `-gc` (`--generate_cache`) YAIB saves the preprocessed splits next to
the data, in `<data-dir>/cache/`, and with `-lc` (`--load_cache`) it reuses them. The cache key includes the seed,
fold, task settings and preprocessing configuration, so a changed configuration does not load a stale cache. Use both
flags together to create the cache on the first run and reuse it afterwards.

## Useful options

| Option                         | Effect                                                                                                                                                              |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-hp <binding> ...`            | Override any gin binding, e.g. `-hp train_common.epochs=10`.                                                                                                        |
| `--cpu`                        | Train on the CPU even if a GPU is available.                                                                                                                        |
| `-db` / `--debug`              | Load only a small part of the data, for quick tests.                                                                                                                |
| `--reproducible`               | Make PyTorch deterministic.                                                                                                                                         |
| `--complete-train`             | Train on all the data, without a test split.                                                                                                                        |
| `-mo DYNAMIC` / `--modalities` | Train on a subset of the modalities (keys of `modality_mapping`).                                                                                                   |
| `--label <name>`               | Choose one label when the task config lists several.                                                                                                                |
| `-pl` / `--plot`               | Save common plots of the results.                                                                                                                                   |
| `--explain-features`           | Save feature attributions: SHAP for tree-based models, Integrated Gradients and more for deep learning models; see [Feature explanations](feature-explanations.md). |

The [CLI reference](cli.md) lists every option.
