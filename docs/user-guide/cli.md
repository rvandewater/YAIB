# CLI reference

```text
icu-benchmarks [train | evaluate] -d DATA_DIR -t TASK [options]
```

`train` is the default command. `evaluate` is the same as `--eval`: it evaluates the model in `--source-dir` without
training. Run `uv run icu-benchmarks train -h` for the help of your installed version.

## Experiment

| Option                             | Description                                                                                                                                     |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `-d`, `--data-dir DATA_DIR`        | **Required.** Folder with the cohort's parquet files.                                                                                           |
| `-t`, `--task TASK`                | **Required.** Task config in `configs/tasks/`, e.g. `BinaryClassification`, `Regression`, `DatasetImputation`.                                  |
| `-n`, `--name NAME`                | Name of the (target) dataset, used for the log folder.                                                                                          |
| `-tn`, `--task-name TASK_NAME`     | Name of the task, used for the log folder, e.g. `Mortality24`.                                                                                  |
| `-m`, `--model MODEL`              | Model config in `configs/prediction_models/`, e.g. `LGBMClassifier`, `GRU`; in `configs/imputation_models/` for imputation tasks.               |
| `-e`, `--experiment EXPERIMENT`    | Experiment config in `configs/experiments/<EXPERIMENT>.gin`, used instead of the task and model configs.                                        |
| `-l`, `--log-dir LOG_DIR`          | Log directory for results and model weights.                                                                                                    |
| `-s`, `--seed SEED`                | Random seed for preprocessing, tuning and training.                                                                                             |
| `-hp`, `--hyperparams HP [HP ...]` | Gin bindings that override the configs, e.g. `-hp train_common.epochs=10 'LGBMClassifier.max_depth=5'`. Pass all bindings after a single `-hp`. |
| `--label LABEL`                    | Label to use when the task config lists several.                                                                                                |
| `-mo`, `--modalities MO [MO ...]`  | Modalities to use, as keys of the task's `modality_mapping` (e.g. `DYNAMIC`, `STATIC`); overrides the selection of the configs.                 |

## Data and preprocessing

| Option                                 | Description                                                                                                                                  |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `-gc`, `--generate_cache`              | Save the preprocessed data to `<data-dir>/cache/`.                                                                                           |
| `-lc`, `--load_cache`                  | Load the preprocessed data from the cache if it exists.                                                                                      |
| `-p`, `--preprocessor PATH`            | Python file with a `CustomPreprocessor` class to use instead of the task's preprocessor; see [Preprocessing](../extending/preprocessing.md). |
| `-imp`, `--pretrained-imputation PATH` | Checkpoint of a trained imputation model to impute the data with; see [Imputation](imputation.md).                                           |
| `-db`, `--debug`                       | Load only part of the data, for quick tests.                                                                                                 |
| `--complete-train`                     | Train on all the data and skip testing.                                                                                                      |

## Training and tuning

| Option                 | Description                                                                                                                                                                                          |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--tune`               | Tune the hyperparameters before training; see [Hyperparameter tuning](hyperparameter-tuning.md).                                                                                                     |
| `--hp-checkpoint PATH` | An earlier run's `hyperparameter_tuning_logs.db`, relative to `<log-dir>/<name>/<task-name>/<model>/` (or absolute). With `--tune`, tuning continues from it; without, training uses its best trial. |
| `--cpu`                | Use the CPU even if a GPU is available.                                                                                                                                                              |
| `--reproducible`       | Make PyTorch deterministic.                                                                                                                                                                          |
| `-wd`, `--wandb-sweep` | Log to Weights & Biases and read the sweep's parameters; see [Weights & Biases](weights-and-biases.md).                                                                                              |

## Evaluation and transfer

| Option                      | Description                                                                                   |
| --------------------------- | --------------------------------------------------------------------------------------------- |
| `--eval`                    | Only evaluate the model from `--source-dir`, skip training.                                   |
| `-ft`, `--fine-tune N`      | Load the model from `--source-dir` and fine-tune it on `N` training stays of the target data. |
| `-sa`, `--samples N`        | Train with `N` training stays; with `--source-dir`, reuse that run's configuration.           |
| `-sn`, `--source-name NAME` | Name of the source dataset, used for the log folder.                                          |
| `--source-dir PATH`         | Fold folder of the source run with `train_config.gin` and the model weights.                  |

See [Evaluation and transfer](evaluation-and-transfer.md).

## Output

| Option               | Description                                                                                                                                                                                                                   |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-v`, `--verbose`    | Log debug messages.                                                                                                                                                                                                           |
| `-pl`, `--plot`      | Save common plots.                                                                                                                                                                                                            |
| `--explain-features` | Save feature attributions of the test set: SHAP values for tree-based models, attributions and their quality metrics for deep learning models (needs `--extra explain`); see [Feature explanations](feature-explanations.md). |

Boolean flags also have a `--no-` form, e.g. `--no-tune`.
