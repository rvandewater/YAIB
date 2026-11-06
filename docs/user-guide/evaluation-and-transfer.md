# Evaluation and transfer

Every training run already evaluates each fold on its test split. This page covers the other cases: testing a trained
model on another dataset, fine-tuning it, and training with fewer samples.

All three load a source run through `--source-dir`, which points to one **fold folder** of that run, for example
`../yaib_logs/mimic_demo/Mortality24/LGBMClassifier/2026-11-06T14-08-10/repetition_0/fold_0`. YAIB reads the fold's
`train_config.gin`, so the model is rebuilt with the settings it was trained with, and loads its weights:
`model.ckpt` (or `last.ckpt`) for deep learning models, `model.joblib` (or `last.joblib`) for classical ones.

## Evaluate on another dataset

Train on the MIMIC-III demo cohort, then test that model on the eICU demo cohort without training:

```bash
uv run icu-benchmarks evaluate \
    -d demo_data/mortality24/eicu_demo \
    -n eicu_demo \
    -t BinaryClassification \
    -tn Mortality24 \
    -m LGBMClassifier \
    -s 2222 \
    -l ../yaib_logs \
    -sn mimic_demo \
    --source-dir ../yaib_logs/mimic_demo/Mortality24/LGBMClassifier/<timestamp>/repetition_0/fold_0
```

`evaluate` is the same as `train --eval`. The model and task must match the source run. The results are written to
`<log-dir>/<name>/<task-name>/<model>/_from_<source-name>/<timestamp>/`.

## Fine-tune on the target dataset

Replace `--eval` with `-ft N` to load the source model and continue training it on `N` stays of the target data
(training and validation stays together; the rest of the cohort is used for testing):

```bash
uv run icu-benchmarks train \
    -d demo_data/mortality24/eicu_demo -n eicu_demo \
    -t BinaryClassification -tn Mortality24 -m GRU \
    -l ../yaib_logs -sn mimic_demo \
    --source-dir ../yaib_logs/mimic_demo/Mortality24/GRU/<timestamp>/repetition_0/fold_0 \
    -ft 500
```

The results are written to `.../_from_<source-name>/fine_tune_<N>/<timestamp>/`.

## Train with fewer samples

`-sa N` (`--samples`) trains with `N` stays (training and validation together), to study how performance depends on
the amount of data.
With `--source-dir`, the run reuses the source run's configuration (including its tuned hyperparameters) instead of the
task and model configs, and is written to `.../samples_<N>/<timestamp>/`.

## Train on all data

`--complete-train` trains on all stays of the cohort, with only a validation split and no test split, for example to
build a final model for deployment after the evaluation is done.

## Pretrained models

Pretrained models for the full datasets are available in the [YAIB-models](https://github.com/rvandewater/YAIB-models)
repository. Point `--source-dir` at one of their fold folders to evaluate them on your data.

!!! note "Regression models"

    Regression runs save the target scaling with the model. Evaluation and fine-tuning reuse the source run's scaling
    and refuse a missing or mismatched `target_transform.json`; see [Regression](regression.md).

!!! warning "Trusted files only"

    Model files (`.joblib`, `.ckpt`) and caches are Python pickles. Only load files you trust.
