"""DL metrics must be logged for multiclass tasks, and one uncomputable metric must not drop the others (#245)."""

import json

import gin
import numpy as np
import polars as pl
import pytest
import torch
from sklearn.metrics import balanced_accuracy_score
from torch.nn.functional import cross_entropy

from icu_benchmarks.data.constants import DataSegment as Segment, DataSplit as Split
from icu_benchmarks.models.custom_metrics import BalancedAccuracy
from icu_benchmarks.models.dl_models.rnn import GRUNet
from icu_benchmarks.models.train import train_common

VARS = {"GROUP": "stay_id", "SEQUENCE": "time", "LABEL": "label", "DYNAMIC": ["x"], "STATIC": []}


def test_balanced_accuracy_computes_a_float():
    metric = BalancedAccuracy()
    y_pred = torch.tensor([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6], [0.7, 0.2, 0.1]])
    y_true = torch.tensor([0, 1, 2, 1])
    metric.update((y_pred, y_true))
    assert metric.compute() == pytest.approx(balanced_accuracy_score(y_true, y_pred.argmax(1)))


def splits(num_classes, test_labels=None):
    rng = np.random.default_rng(0)
    data = {}
    for split in (Split.train, Split.val, Split.test):
        labels = rng.integers(0, num_classes, 40) if test_labels is None or split != Split.test else test_labels
        data[split] = {
            Segment.features: pl.DataFrame({"stay_id": range(40), "time": [0] * 40, "x": labels + rng.normal(size=40)}),
            Segment.outcome: pl.DataFrame({"stay_id": range(40), "label": labels}),
        }
    return data


def train_gru(data, num_classes, log_dir):
    gin.clear_config()
    gin.bind_parameter("PredictionPolarsDataset.vars", VARS)
    gin.bind_parameter("DLPredictionWrapper.loss", cross_entropy)
    for name, value in {"input_size": [1, 1, 1], "hidden_dim": 4, "layer_dim": 1, "num_classes": num_classes}.items():
        gin.bind_parameter(f"GRUNet.{name}", value)
    try:
        train_common(data, log_dir=log_dir, model=GRUNet, weight="balanced", epochs=1, cpu=True, num_workers=0)
    finally:
        gin.clear_config()
    return json.loads((log_dir / "test_metrics.json").read_text())


def test_multiclass_metrics_are_logged(tmp_path):
    metrics = train_gru(splits(3), 3, tmp_path)
    assert {"Accuracy", "BalancedAccuracy"} <= metrics.keys()
    assert 0 <= metrics["BalancedAccuracy"] <= 1


def test_uncomputable_metric_skips_only_itself(tmp_path):
    # AUC is undefined when the test split has a single class; PR is still logged
    metrics = train_gru(splits(2, test_labels=np.zeros(40, dtype=int)), 2, tmp_path)
    assert "AUC" not in metrics
    assert "PR" in metrics
