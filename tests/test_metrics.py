"""Custom metrics, metric loggers and the small model utilities must compute what they report."""

import json
import logging
from datetime import timedelta

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torchmetrics.classification import BinaryAccuracy

from icu_benchmarks.models.constants import DLMetrics, MLMetrics
from icu_benchmarks.models.custom_metrics import (
    JSD,
    MAE,
    BalancedAccuracy,
    TorchMetricsWrapper,
    accuracy,
    confusion_matrix,
)
from icu_benchmarks.models.utils import (
    JSONMetricsLogger,
    JsonResultLoggingEncoder,
    create_optimizer,
    create_scheduler,
    get_smoothed_labels,
    log_table_row,
    scorer_wrapper,
)


OUTPUT = torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.15, 0.05], [0.3, 0.3, 0.4], [0.2, 0.5, 0.3]])
TARGET = torch.tensor([1, 1, 2, 0])


def test_top1_accuracy():
    assert accuracy(OUTPUT, TARGET)[0].item() == pytest.approx(50.0)


@pytest.mark.xfail(strict=True, raises=RuntimeError, reason="accuracy calls view() on a non-contiguous slice for k > 1")
def test_topk_accuracy():
    assert accuracy(OUTPUT, TARGET, topk=(2,))[0].item() == pytest.approx(75.0)


def test_confusion_matrix_rounds_probabilities():
    y_true = np.array([0, 0, 1, 1])
    counts = confusion_matrix(y_true, np.array([0.2, 0.7, 0.9, 0.6]))
    assert counts == {"class_0_pred_0": 1, "class_0_pred_1": 1, "class_1_pred_0": 0, "class_1_pred_1": 2}
    assert sum(confusion_matrix(y_true, np.array([0.2, 0.7, 0.9, 0.6]), normalize=True).values()) == pytest.approx(1)


def test_mae_inverts_targets_before_scoring():
    metric = MAE(invert_transform=lambda x: x * 10)
    metric.update((torch.tensor([0.1, 0.5]), torch.tensor([0.2, 0.2])))
    assert metric.compute() == pytest.approx(2.0)


def test_jsd_is_zero_for_identical_and_positive_for_different_distributions():
    same, different = JSD(), JSD()
    same.update((torch.tensor([0.2, 0.8]), torch.tensor([0.2, 0.8])))
    different.update((torch.tensor([0.9, 0.1]), torch.tensor([0.1, 0.9])))
    assert same.compute() == pytest.approx(0, abs=1e-6)
    assert 0 < different.compute() <= 1


@pytest.mark.xfail(
    strict=True,
    reason="BalancedAccuracy passes the missing attribute self.balanced_accuracy_compute to EpochMetric, which ignite "
    "turns into a lazy MetricsLambda, so compute() returns that object instead of a score",
)
def test_balanced_accuracy():
    metric = BalancedAccuracy()
    metric.update((torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]]), torch.tensor([0, 1, 1])))
    assert metric.compute() == pytest.approx(1.0)


def test_torchmetrics_wrapper_takes_output_tuples():
    metric = TorchMetricsWrapper(BinaryAccuracy())
    metric.update((torch.tensor([0.9, 0.2, 0.7]), torch.tensor([1, 0, 0])))
    assert metric.compute().item() == pytest.approx(2 / 3)
    metric.reset()
    metric.update((torch.tensor([0.9]), torch.tensor([1])))
    assert metric.compute().item() == pytest.approx(1)


def test_metric_tables_score_binary_predictions():
    y_true, y_score = np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])
    assert MLMetrics.BINARY_CLASSIFICATION["AUC"](y_true, y_score) == pytest.approx(0.75)
    assert set(MLMetrics.REGRESSION) == {"MAE_native", "MSE_native", "R2", "RMSE_native"}
    assert set(DLMetrics.REGRESSION) == {"MAE_native", "MSE_native"}
    assert {"AUC", "PR"} <= set(DLMetrics.BINARY_CLASSIFICATION)


def test_scorer_wrapper_flattens_binary_probabilities():
    y_true = np.array([0, 1, 1, 0])
    probabilities = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.9, 0.1]])
    assert scorer_wrapper(roc_auc_score)(y_true, probabilities) == pytest.approx(roc_auc_score(y_true, [0, 1, 0, 0]))
    assert scorer_wrapper()(y_true, probabilities[:, 1]) == pytest.approx(
        average_precision_score(y_true, [0.2, 0.7, 0.4, 0.1])
    )


def test_create_optimizer_and_scheduler():
    model = torch.nn.Linear(2, 1)
    assert isinstance(create_optimizer("Adam", model, 0.1), torch.optim.Adam)
    sgd = create_optimizer("sgd", model, 0.1, momentum=0.5)
    assert isinstance(sgd, torch.optim.SGD) and sgd.defaults["momentum"] == 0.5
    assert isinstance(create_optimizer("radam", model, 0.1), torch.optim.RAdam)
    with pytest.raises(ValueError):
        create_optimizer("lbfgs", model, 0.1)
    assert create_scheduler(None, sgd, 0.5, None, 10) is None
    assert isinstance(create_scheduler("step", sgd, 0.5, [2], 10), torch.optim.lr_scheduler.MultiStepLR)
    assert isinstance(create_scheduler("exponential", sgd, 0.5, None, 10), torch.optim.lr_scheduler.ExponentialLR)
    assert isinstance(create_scheduler("cosine", sgd, 0.5, None, 10), torch.optim.lr_scheduler.CosineAnnealingLR)
    with pytest.raises(ValueError, match="no lr steps"):
        create_scheduler("step", sgd, 0.5, [], 10)
    with pytest.raises(ValueError):
        create_scheduler("cyclic", sgd, 0.5, None, 10)


def test_json_encoder_serializes_numpy_torch_and_durations():
    payload = {"i": np.int64(3), "f": np.float32(0.5), "a": np.arange(2), "t": torch.tensor([1.0]), "d": timedelta(seconds=1)}
    assert json.loads(json.dumps(payload, cls=JsonResultLoggingEncoder)) == {
        "i": 3,
        "f": 0.5,
        "a": [0, 1],
        "t": [1.0],
        "d": "0:00:01",
    }
    with pytest.raises(TypeError):
        json.dumps({"x": object()}, cls=JsonResultLoggingEncoder)


def test_json_metrics_logger_merges_metrics_per_stage(tmp_path):
    logger = JSONMetricsLogger(tmp_path / "metrics")
    logger.log_metrics({"train/loss": 1.0, "val/AUC": 0.6})
    logger.log_metrics({"train/loss": 0.5, "train/AUC": np.float64(0.7)})
    assert json.loads((tmp_path / "metrics" / "train_metrics.json").read_text()) == {"loss": 0.5, "AUC": 0.7}
    assert json.loads((tmp_path / "metrics" / "val_metrics.json").read_text()) == {"AUC": 0.6}
    assert not (tmp_path / "metrics" / "test_metrics.json").exists()
    assert logger.name == "json_metrics_logger" and logger.version == "0.1"


def test_log_table_row_pads_and_truncates(caplog):
    with caplog.at_level(logging.INFO):
        log_table_row(["abcdef", 1], header=["abc", "value"])
        log_table_row(["x"], widths=[3], highlight=True)
    assert caplog.messages[0] == "abc | 1    "
    assert caplog.messages[1] == "\x1b[31;32mx  \x1b[0m"


def step_smoothing(dte, h_true, h_min, h_max, delta_h, gamma):
    return float(dte <= h_true)


@pytest.mark.parametrize(
    "label,event,expected",
    [
        ([0, 0, 0, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 0, 0, 0]),
        ([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]),
        # Positive at the end of the stay without an event: the event is assumed h_true hours after the label onset
        ([0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 1, 1]),
    ],
)
def test_smoothed_labels_mark_hours_before_event(label, event, expected):
    smoothed = get_smoothed_labels(
        np.array(label, dtype=float), np.array(event, dtype=float), smoothing_fn=step_smoothing, h_true=2, h_min=0, h_max=4
    )
    np.testing.assert_array_equal(smoothed, expected)
