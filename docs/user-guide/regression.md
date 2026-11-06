# Regression

Use `-t Regression` for the regression tasks, kidney function and length of stay:

```bash
uv run icu-benchmarks train \
    -d demo_data/los/mimic_demo -n mimic_demo \
    -t Regression -tn LoS \
    -m LGBMRegressor \
    -l ../yaib_logs/ --tune
```

Models that support regression include `LGBMRegressor`, `ElasticNet`, `SVMRegressor` and the deep learning models.

## Target scaling

By default, targets are min/max scaled using only the training split of each fold. The validation and test splits reuse
that transform without clipping. Feature scaling is independent of target scaling. Targets must use consistent units
across datasets.

To change the target scaling, bind `base_regression_preprocessor.target_scaling`:

```bash
# No target scaling
-hp 'base_regression_preprocessor.target_scaling="none"'

# Fixed bounds, to reproduce a prescribed normalization
-hp 'base_regression_preprocessor.target_scaling="fixed"' \
    'base_regression_preprocessor.outcome_min=0' \
    'base_regression_preprocessor.outcome_max=168'
```

A complete legacy pair of bounds (`outcome_min` and `outcome_max` without `target_scaling`) still selects fixed scaling,
with a deprecation warning.

## Metrics

| Metric                      | Units                         |
| --------------------------- | ----------------------------- |
| `MAE_native`, `RMSE_native` | Original target units         |
| `MSE_native`                | Squared original target units |
| `R2`                        | Unitless                      |

`train/loss`, `val/loss` and `test/loss` stay in the scaled model space, which is what optimization and early stopping
use. Hyperparameter tuning scores trials with the native-unit MSE instead.

Do not compare the native metrics with scaled metrics from older YAIB versions, and do not average errors across
endpoints with different clinical units.

## Saved files

Every regression fold saves `target_transform.json` next to its model. Keep them together: evaluation and fine-tuning
reuse the source transform, and refuse a missing or mismatched one.

- Classical regression models are saved as a `.joblib` dictionary with `model`, `target_transform` and
    `trained_columns`.
- Deep learning checkpoints embed the transform.
- Saved predictions and SHAP values are in native units; SHAP base values are saved separately as
    `*_shap_base_values.npy`.

Regression models trained with older YAIB versions must be retrained, and their preprocessing caches are invalidated.

## Custom integrations

`preprocess_data()` returns `(data, target_transform)`; pass both to `train_common()`. For other run modes the transform
is `None`. A cache hit restores the data and the fitted transform together. YAIB assumes that the input files do not
change within a cached run.
