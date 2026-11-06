# Adding evaluation metrics

The metrics are defined in `icu_benchmarks/models/constants.py`, separately for classical models (`MLMetrics`, using
scikit-learn functions) and deep learning models (`DLMetrics`, using [PyTorch Ignite](https://pytorch.org/ignite/) and
[TorchMetrics](https://lightning.ai/docs/torchmetrics/)). Each class has one dictionary per kind of task:

| Dictionary                      | Used for                           |
| ------------------------------- | ---------------------------------- |
| `BINARY_CLASSIFICATION`         | Two classes                        |
| `MULTICLASS_CLASSIFICATION`     | More than two classes              |
| `REGRESSION`                    | Regression, in native target units |
| `IMPUTATION` (`DLMetrics` only) | Imputation models                  |

The current definitions:

```python title="icu_benchmarks/models/constants.py"
--8<--
icu_benchmarks/models/constants.py
--8<--
```

## Adding a metric

To add a metric, add an entry to the dictionaries of the tasks it applies to. The key is the name under which the
metric is logged and saved in `test_metrics.json`.

- For **classical models**, the value is a function `metric(y_true, y_pred)`, as in scikit-learn.
- For **deep learning models**, the value is an Ignite metric class or a TorchMetrics metric instance.

Custom metrics go in `icu_benchmarks/models/custom_metrics.py`. For example, the Jensen-Shannon divergence used for
imputation is an Ignite `EpochMetric` that wraps SciPy's `jensenshannon`:

```python title="icu_benchmarks/models/custom_metrics.py"
class JSD(EpochMetric):
    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        check_compute_fn: bool = False,
    ) -> None:
        super(JSD, self).__init__(
            lambda x, y: JSD_fn(x, y),
            output_transform=output_transform,
            check_compute_fn=check_compute_fn,
        )

        def JSD_fn(y_preds: torch.Tensor, y_targets: torch.Tensor):
            return jensenshannon(abs(y_preds).flatten(), abs(y_targets).flatten()) ** 2
```

It is then listed in `DLMetrics.IMPUTATION` as `"jsd": JSD`.

Only metrics that return a single number are logged and averaged across folds. Curve metrics (`PR_Curve`, `RO_Curve`,
`Calibration_Curve`) are currently skipped when logging. For classical models, the confusion matrix is saved as one
count per cell, `class_<true>_pred_<predicted>`.

## Fairness metrics

`custom_metrics.py` also contains `BinaryFairnessWrapper`, which computes TorchMetrics' group fairness metrics
(for example demographic parity) for a group variable such as `sex`. It is defined in
`DLMetrics.BINARY_CLASSIFICATION_TORCHMETRICS`, which the wrappers do not use by default.
