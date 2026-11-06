# Adding tasks

YAIB provides five clinical prediction tasks by default:

| No  | Task                      | Frequency                  | Type                  |
| --- | ------------------------- | -------------------------- | --------------------- |
| 1   | ICU mortality             | Once per stay (after 24 h) | Binary classification |
| 2   | Acute kidney injury (AKI) | Hourly (within 6 h)        | Binary classification |
| 3   | Sepsis                    | Hourly (within 6 h)        | Binary classification |
| 4   | Kidney function (KF)      | Once per stay              | Regression            |
| 5   | Length of stay (LoS)      | Hourly (within 7 days)     | Regression            |

A task has two parts: the **cohort** (which stays, which variables, which label), defined in
[YAIB-cohorts](https://github.com/rvandewater/YAIB-cohorts), and the **task config** in YAIB, which says how to read
and model it.

## Changing the cohort

Cohorts are built to be adapted. For example, predicting the ordinal KDIGO stage instead of the presence of AKI only
requires changing the outcome variable from `aki` to `kdigo`. If the general setup stays the same (covariates,
exclusion criteria, prediction times) but a new outcome should be predicted, define the outcome as a
[clinical concept](clinical-concepts.md) in ricu and use it as the outcome variable. If the covariates, exclusion
criteria or prediction times change, the existing helper functions can be reused; the existing task definitions in
YAIB-cohorts are examples.

## Writing a task config

The task configs in `configs/tasks/` cover binary classification (`BinaryClassification.gin`), regression
(`Regression.gin`) and imputation (`DatasetImputation.gin`). A new cohort in the standard format works with them
directly. For a cohort with other files, columns or settings, copy one and change it:

```python title="configs/tasks/MyTask.gin"
include "configs/tasks/common/Imports.gin"
include "configs/tasks/common/CrossValidation.gin"

Run.mode = "Classification"          # "Classification", "Regression" or "Imputation"
NUM_CLASSES = 2
HORIZON = 24
train_common.weight = "balanced"
train_common.ram_cache = True

DLPredictionWrapper.loss = @cross_entropy

preprocess.file_names = {
    "DYNAMIC": "dyn.parquet",
    "OUTCOME": "outc.parquet",
    "STATIC": "sta.parquet",
}
vars = {
    "GROUP": "stay_id",
    "SEQUENCE": "time",
    "LABEL": "my_outcome",
    "DYNAMIC": ["hr", "map", "resp", "o2sat", "temp"],
    "STATIC": ["age", "sex"],
}
preprocess.preprocessor = @base_classification_preprocessor
preprocess.vars = %vars
preprocess.use_static = True

include "configs/tasks/common/Dataloader.gin"
```

Then run it with `-t MyTask`. The file name is the task name.

Variants of the defaults show smaller changes: `BinaryClassificationNoStatic.gin` and `BinaryClassificationNoDynamic.gin`
train without the static or the dynamic variables. You can also select variable groups at run time with `--modalities`
and a `modality_mapping`, as in `PredictionTaskVariables.gin`.

For multiclass classification, set `NUM_CLASSES` to the number of classes; the metrics switch to the multiclass set
(accuracy, balanced accuracy, AUC).
