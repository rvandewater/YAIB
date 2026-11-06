# Feature explanations

YAIB can save [SHAP](https://shap.readthedocs.io/) values for the test set of each fold, to show which features drive a
model's predictions.

```bash
uv run icu-benchmarks train \
    -d demo_data/mortality24/mimic_demo -n mimic_demo \
    -t BinaryClassification -tn Mortality24 \
    -m LGBMClassifier \
    -l ../yaib_logs/ \
    --explain-features
```

## Supported models

SHAP values are computed with `shap.TreeExplainer`, so they are available for the tree-based models:
`LGBMClassifier`, `LGBMRegressor`, `XGBClassifier`, `CBClassifier`, `RFClassifier` and `BRFClassifier`. A model
supports explanations when its class sets `explainer_cls`. Linear models, SVMs, `RUSBClassifier` and the deep learning
models do not have an explainer yet.

YAIB checks this before tuning starts: requesting explanations for a model without an explainer fails immediately
instead of after a long run.

## Output

Each fold writes `test_shap_values.parquet`: one row per test sample and one column per model input (the
`trained_columns`, including generated features). For regression models, the values are in native target units and the
base values are saved as `test_shap_base_values.npy`.

After all folds, YAIB concatenates the folds' values into `aggregated_shap_values.parquet` in the run folder.

SHAP values are not computed during tuning trials, which would be slow and unused.

## Enabling it in a config

Instead of the flag, you can bind `train_common.explain_features = True` in a model or experiment config. The flag
overrides the binding: `--explain-features` turns explanations on and `--no-explain-features` turns them off.
