# Imputation

ICU time series have many missing values. By default, the preprocessors fill them with simple rules (and add missingness
indicators). YAIB can also train dedicated imputation models and use a trained one to impute the data of a prediction
task.

Several imputation models need the optional dependencies:

```bash
uv sync --extra imputation
```

## Train an imputation model

Imputation models use the `DatasetImputation` task and a model config from `configs/imputation_models/`:

```bash
uv run icu-benchmarks train \
    -d demo_data/mortality24/mimic_demo \
    -n mimic_demo \
    -t DatasetImputation \
    -m Mean \
    -s 2222 \
    -l ../yaib_logs/
```

With an imputation task (`Run.mode = "Imputation"`), `-m` names a config in `configs/imputation_models/`. The task needs
no outcome file or label: the stays to split come from the static file, and only the dynamic variables in `vars` are
imputed. By default, the preprocessor drops the stays that have a missing value in any of these variables
(`base_imputation_preprocessor.filter_missing_values`), so the model is trained and evaluated on fully observed stays.
On the MIMIC demo, this keeps 15 of the 99 stays.

To train, YAIB masks part of the observed values (amputation) and learns to reconstruct them. The amputation is set in
the dataset:

| Binding                                               | Default  | Meaning                                                               |
| ----------------------------------------------------- | -------- | --------------------------------------------------------------------- |
| `ImputationPandasDataset.mask_method`                 | `"MCAR"` | Missingness mechanism: `MCAR`, `MAR` or `MNAR`                        |
| `ImputationPandasDataset.mask_proportion`             | `0.3`    | Proportion of values to mask                                          |
| `ImputationPandasDataset.mask_observation_proportion` | `0.3`    | Proportion of the observed values used by the MAR and MNAR mechanisms |

Imputation is evaluated with the RMSE, MAE and Jensen-Shannon divergence of the reconstructed values.

## Available models

| Kind                | Models (`-m`)                                                                                            |
| ------------------- | -------------------------------------------------------------------------------------------------------- |
| Simple baselines    | `Mean`, `Median`, `Zero`, `MostFrequent`                                                                 |
| scikit-learn        | `KNN`, `MICE`, `ICE`                                                                                     |
| hyperimpute (extra) | `MissForest`, `GAIN`, `Miwae`, `Miracle`, `Sinkhorn`, `SoftImpute`                                       |
| PyPOTS (extra)      | `BRITS`, `SAITS`, `Attention`                                                                            |
| Neural networks     | `MLP`, `RNN`, `BRNN`, `NP` (neural process)                                                              |
| Diffusion           | `CSDI`, `Diffusion`, `Simple_Diffusion`, `DiffWave`, `SSSDS4`, `SSSDSA` (the SSSD models need the extra) |

## Use a trained imputation model

Pass the checkpoint of an imputation run to a prediction task with `-imp` (`--pretrained-imputation`):

```bash
uv run icu-benchmarks train \
    -d demo_data/mortality24/mimic_demo -n mimic_demo \
    -t BinaryClassification -tn Mortality24 \
    -m LGBMClassifier \
    -l ../yaib_logs/ \
    -imp ../yaib_logs/mimic_demo/DatasetImputation/MLP/<timestamp>/repetition_0/fold_0/model.ckpt
```

Use the fold's `model.ckpt` (the best epoch) for trained models, or `last.ckpt` for models that are only fitted, such
as the baselines. The preprocessor then fills the missing values of the dynamic variables with the imputation model
before training the prediction model. The checkpoint is loaded with full unpickling, so only use files you trust.

!!! bug "Known issue"

    `-imp` currently has no effect. The checkpoint is loaded, but `preprocess_data` passes it only to the deprecated
    Pandas classification preprocessor, which the current pipeline cannot run
    ([#214](https://github.com/rvandewater/YAIB/issues/214)). The default preprocessors fill the missing values with
    their usual rules.

To add your own imputation model, see [Adding imputation models](../extending/imputation-models.md).
