# Feature explanations

With `--explain-features`, YAIB explains the test predictions of each fold, to show which features drive a model's
predictions:

- **Tree-based models** get [SHAP](https://shap.readthedocs.io/) values.
- **Deep learning models** get attributions computed with [Captum](https://captum.ai/), for every labeled hour, together
    with metrics of how faithful these attributions are.

Explanations are not computed during tuning trials, which would be slow and unused. YAIB checks before tuning starts that
the model supports them, so an unsupported model fails immediately instead of after a long run.

## Tree-based models: SHAP values

```bash
uv run icu-benchmarks train \
    -d demo_data/mortality24/mimic_demo -n mimic_demo \
    -t BinaryClassification -tn Mortality24 \
    -m LGBMClassifier \
    -l ../yaib_logs/ \
    --explain-features
```

SHAP values are computed with `shap.TreeExplainer`, so they are available for `LGBMClassifier`, `LGBMRegressor`,
`XGBClassifier`, `CBClassifier`, `RFClassifier` and `BRFClassifier`: the models whose class sets `explainer_cls`.
Linear models, SVMs and `RUSBClassifier` do not have an explainer yet.

Each fold writes `test_shap_values.parquet`: one row per test sample and one column per model input (the
`trained_columns`, including generated features). For regression models, the values are in native target units and the
base values are saved as `test_shap_base_values.npy`. After all folds, YAIB concatenates the folds' values into
`aggregated_shap_values.parquet` in the run folder.

## Deep learning models: attributions and their quality

Install the optional Captum dependency first:

```bash
uv sync --extra explain
```

```bash
uv run icu-benchmarks train \
    -d demo_data/aki/eicu_demo -n eicu_demo \
    -t BinaryClassification -tn AKI \
    -m GRU \
    -l ../yaib_logs/ \
    --explain-features
```

This works for all deep learning prediction models (`GRU`, `LSTM`, `RNN`, `GRUD`, `TCN`, `Transformer`,
`LocalTransformer`, `BAT`, `Mamba`) on binary classification and regression tasks.

### What is explained

- **Every labeled hour.** For hourly tasks such as AKI and sepsis, each hour with a label is explained separately; for
    once-per-stay tasks such as mortality, the one labeled hour. A sample of `max_stays` test stays (100 by default) is
    explained, to bound the cost.
- **The model output at that hour:** the log-odds of the positive class for binary classification, the predicted value
    for regression (attributions are converted to native target units).
- **Variables, not columns.** A variable is a value column together with its `MissingIndicator_<name>` column. Its
    attribution is summed over both columns and over all hours up to the explained one. Removing a variable means "not
    measured": its value is set to 0 (the training mean, after standardisation) and its missingness indicator to 1, for its
    whole history.

### Methods

| Method                | What it computes                                                                                                                                                                                                  |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `IntegratedGradients` | [Integrated Gradients](https://arxiv.org/abs/1703.01365) from the "not measured" input to the actual input. The attributions sum to the output difference; the mean deviation is reported as `convergence_delta`. |
| `FeatureAblation`     | The output drop when a variable is removed (occlusion of its whole history).                                                                                                                                      |
| `Random`              | Random attributions: the reference that the other methods should beat on every metric.                                                                                                                            |

### Quality metrics

| Metric         | Meaning                                                                                                                                                                                                                      | Better              |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| `faithfulness` | Correlation, over random variable subsets, between a subset's total attribution and the output drop when it is removed ([Bhatt et al., 2020](https://arxiv.org/abs/2005.00631)). The same subsets are used for every method. | Higher (Random ≈ 0) |
| `deletion`     | Mean output drop when the 1, 2, ..., 10 highest-attributed variables are removed (area over the perturbation curve).                                                                                                         | Higher              |
| `sparsity`     | Gini index of the absolute attributions: whether a few variables carry the explanation.                                                                                                                                      | Context-dependent   |

Faithfulness and deletion tell you whether the attributions reflect what the model does, not whether the model is right:
judge them against the `Random` reference and together with the model's test performance.

### Output

Each fold writes:

| File                              | Contents                                                                                              |
| --------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `test_attributions.parquet`       | One row per method and explained hour: the stay ID, the hour, the method and one column per variable. |
| `test_feature_importance.parquet` | Per method and variable, the mean absolute and the mean attribution.                                  |
| `test_explanation_metrics.json`   | The metrics, keyed `explain/<method>/<metric>`, averaged over the explained hours.                    |

The metrics are averaged over the folds into `accumulated_test_metrics.json` like the other test metrics, and the
feature importances of all folds are collected in `aggregated_feature_importance.parquet`.

### Settings

The `explain_predictions` configurable controls the methods, metrics and cost:

| Binding                                        | Default                                                | Meaning                                                        |
| ---------------------------------------------- | ------------------------------------------------------ | -------------------------------------------------------------- |
| `explain_predictions.methods`                  | `("IntegratedGradients", "FeatureAblation", "Random")` | Attribution methods                                            |
| `explain_predictions.metrics`                  | `("faithfulness", "deletion", "sparsity")`             | Quality metrics                                                |
| `explain_predictions.max_stays`                | `100`                                                  | Test stays to explain; every labeled hour of each is explained |
| `explain_predictions.ig_steps`                 | `32`                                                   | Integration steps of Integrated Gradients                      |
| `explain_predictions.faithfulness_subsets`     | `20`                                                   | Random variable subsets per stay                               |
| `explain_predictions.faithfulness_subset_size` | `0.1`                                                  | Share of the variables in each subset                          |
| `explain_predictions.deletion_steps`           | `10`                                                   | Top variables removed for the deletion metric                  |
| `explain_predictions.batch_size`               | `64`                                                   | Inputs per forward pass                                        |

The cost grows with the number of explained hours, so hourly tasks with long stays are the most expensive: on the
AKI demo cohort, 30 stays (about 1,300 labeled hours) take about 10 seconds on a laptop CPU. Lower `max_stays` or
`ig_steps` for large cohorts, for example `-hp explain_predictions.max_stays=50`.

## Enabling it in a config

Instead of the flag, you can bind `train_common.explain_features = True` in a model or experiment config. The flag
overrides the binding: `--explain-features` turns explanations on and `--no-explain-features` turns them off.
