# Configuration

YAIB is configured with [gin](https://github.com/google/gin-config). Most functions and classes are gin
*configurables*, and a `.gin` file binds values to their arguments:

```python
train_common.epochs = 50                  # an argument of the function train_common
LGBMClassifier.max_depth = 5              # an argument of the class LGBMClassifier
train_common.model = @GRUNet              # @ refers to a configurable
NUM_CLASSES = 2                           # a macro ...
model/hyperparameter.num_classes = %NUM_CLASSES   # ... used with %
include "configs/prediction_models/common/DLCommon.gin"
```

The [API reference](../reference/index.md) documents the arguments of every configurable.

## Config folders

| Folder                              | Contents                                                                                                                  | Selected with                    |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| `configs/tasks/`                    | Task configs: run mode, variables, preprocessor, cross-validation                                                         | `-t`                             |
| `configs/tasks/common/`             | Shared parts included by the task configs                                                                                 |                                  |
| `configs/prediction_models/`        | One config per prediction model                                                                                           | `-m`                             |
| `configs/prediction_models/common/` | Shared settings and tuning budgets for ML (`MLCommon.gin`, `MLTuning.gin`) and DL (`DLCommon.gin`, `DLTuning.gin`) models |                                  |
| `configs/imputation_models/`        | One config per imputation model                                                                                           | `-m` with `-t DatasetImputation` |
| `configs/experiments/`              | Your own experiment configs (create the folder)                                                                           | `-e`                             |

YAIB reads the configs relative to the current directory, so run it from the repository root.

## Overriding settings

To change a setting for one run, pass gin bindings with `-hp`. Put all bindings after one `-hp`, separated by spaces, and
quote strings:

```bash
uv run icu-benchmarks train ... -m GRU \
    -hp train_common.epochs=10 train_common.batch_size=128 'base_classification_preprocessor.scaling=False'
```

`-hp` bindings are applied after the config files, so they win over them.

## Experiment configs

For a combination you run often, write an experiment config that includes a task and a model config and changes what
you need:

```python title="configs/experiments/GRU_short.gin"
include "configs/tasks/BinaryClassification.gin"
include "configs/prediction_models/GRU.gin"

train_common.epochs = 20
execute_repeated_cv.cv_repetitions = 2
```

```bash
uv run icu-benchmarks train -d demo_data/mortality24/mimic_demo -n mimic_demo -t BinaryClassification -m GRU -e GRU_short
```

With `-e`, YAIB parses only the experiment config (and the task config for the run mode), and the run is logged under
`<log-dir>/<name>/<experiment>/`.

## The saved configuration

Every fold writes the full configuration it ran with to `train_config.gin`, including the tuned hyperparameters. To
evaluate or fine-tune a model, YAIB reads this file back from `--source-dir`, so a model is always rebuilt with the
settings it was trained with; see [Evaluation and transfer](evaluation-and-transfer.md).

## Common settings

| Binding                                              | Default                   | Meaning                                                  |
| ---------------------------------------------------- | ------------------------- | -------------------------------------------------------- |
| `execute_repeated_cv.cv_repetitions`                 | `5`                       | Number of repetitions (different test sets)              |
| `execute_repeated_cv.cv_folds`                       | `5`                       | Number of train/validation folds per repetition          |
| `train_common.epochs`                                | `50` (DL)                 | Maximum number of epochs                                 |
| `train_common.patience`                              | `10` (DL)                 | Epochs without improvement before early stopping         |
| `train_common.batch_size`                            | `64` (DL)                 | Batch size                                               |
| `train_common.weight`                                | `"balanced"`              | Class weights for imbalanced labels                      |
| `base_classification_preprocessor.generate_features` | `True` (ML), `False` (DL) | Add summary features over time for the dynamic variables |
| `base_classification_preprocessor.scaling`           | `True`                    | Scale the features                                       |
| `tune_hyperparameters.n_calls`                       | `30`                      | Number of tuning trials                                  |

The same preprocessor arguments exist for `base_regression_preprocessor`.
