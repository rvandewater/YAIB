# Quickstart

This page trains a first model on the demo data included in the repository. It takes about a minute on a laptop.

## The demo data

The authors of MIMIC-III and eICU have published small demo datasets on PhysioNet:
the [MIMIC-III Clinical Database Demo](https://physionet.org/content/mimiciii-demo/1.4/) and the
[eICU Collaborative Research Database Demo](https://physionet.org/content/eicu-crd-demo/2.0.1/). They are released under
the [Open Data Commons Open Database License v1.0](https://opendatacommons.org/licenses/odbl/1-0/) and can be used
without a credentialing procedure. We created demo cohorts **solely from these datasets** for each supported task. To
the best of our knowledge, this complies with the license and the dataset authors' instructions. Use of the demo
cohorts is only permitted under that license.

!!! warning "Human subject research data"

    We strongly recommend completing a human subject research training to make sure you handle this data properly.

The cohorts are in `demo_data/<task>/<dataset>`, for example `demo_data/mortality24/mimic_demo`:

| Folder            | Task                               | Task config (`-t`)     |
| ----------------- | ---------------------------------- | ---------------------- |
| `mortality24`     | ICU mortality after 24 hours       | `BinaryClassification` |
| `aki`             | Acute kidney injury within 6 hours | `BinaryClassification` |
| `sepsis`          | Sepsis within 6 hours              | `BinaryClassification` |
| `kidney_function` | Kidney function on day 2           | `Regression`           |
| `los`             | Remaining length of stay           | `Regression`           |

Each folder holds `dyn.parquet` (hourly time series), `sta.parquet` (static variables) and `outc.parquet` (labels), and
each task has an `eicu_demo` and a `mimic_demo` cohort.

## Train a model

Train an XGBoost model for mortality prediction on the MIMIC-III demo cohort, with hyperparameter tuning:

```bash
uv run icu-benchmarks train \
    -d demo_data/mortality24/mimic_demo \
    -n mimic_demo \
    -t BinaryClassification \
    -tn Mortality24 \
    -m XGBClassifier \
    --seed 2222 \
    -l ../yaib_logs/ \
    --tune
```

| Flag     | Meaning                                             |
| -------- | --------------------------------------------------- |
| `-d`     | The cohort folder.                                  |
| `-n`     | The dataset name, used for the log folder.          |
| `-t`     | The task config in `configs/tasks/`.                |
| `-tn`    | The task name, used for the log folder.             |
| `-m`     | The model config in `configs/prediction_models/`.   |
| `--seed` | The random seed for splitting, tuning and training. |
| `-l`     | The log directory.                                  |
| `--tune` | Tune the hyperparameters before training.           |

All flags are listed in the [CLI reference](../user-guide/cli.md).

To try a deep learning model, replace `-m XGBClassifier` with `-m GRU` (or `LSTM`, `TCN`, `Transformer`, `Mamba`, ...).
Deep models tune and train for longer.

## Look at the results

YAIB runs 5 × 5 repeated cross-validation by default: 5 repetitions with different test sets, each with 5 train and
validation folds. The run above writes:

```text
../yaib_logs/mimic_demo/Mortality24/XGBClassifier/<timestamp>/
├── accumulated_test_metrics.json     # "avg", "std" and "CI_0.95" of each metric over all folds
├── aggregated_test_metrics.json      # the test metrics of every fold, by repetition and fold
├── hyperparameter_tuning_logs.db     # Optuna study (with --tune)
├── optimization_history.png          # tuning progress (with --tune)
└── repetition_0/
    └── fold_0/
        ├── test_metrics.json         # loss, AUC, PR (AUPRC) and the confusion matrix of this fold
        ├── train_config.gin          # the full configuration of this fold
        ├── durations.json
        └── last.joblib               # the trained model (model.ckpt for deep learning models)
```

In `accumulated_test_metrics.json`, `avg` is the mean over all folds, `std` is the population standard deviation divided
by √n (a standard error of the mean), and `CI_0.95` is a 95% t-interval. To track runs in a dashboard, see
[Weights & Biases](../user-guide/weights-and-biases.md).

## Next steps

- [Running experiments](../user-guide/running-experiments.md) explains how the dataset, task and model configs fit
    together.
- [Datasets and access](datasets.md) and [Generating cohorts](cohorts.md) describe how to run YAIB on the full datasets.
- [Evaluation and transfer](../user-guide/evaluation-and-transfer.md) shows how to test a trained model on another
    dataset.
