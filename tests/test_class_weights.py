"""Balanced class weights must belong to the right class (#243), and DL classification must train without them (#246)."""

import gin
import numpy as np
import polars as pl
import pytest
from torch.nn.functional import cross_entropy

from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.constants import DataSegment as Segment, DataSplit as Split
from icu_benchmarks.data.loader import PredictionPolarsDataset
from icu_benchmarks.models.dl_models.rnn import GRUNet
from icu_benchmarks.models.train import train_common

VARS = {"GROUP": "stay_id", "SEQUENCE": "time", "LABEL": "label", "DYNAMIC": ["x"], "STATIC": []}


def dataset(labels):
    n = len(labels)
    data = {
        Split.train: {
            Segment.features: pl.DataFrame({"stay_id": range(n), "time": [0] * n, "x": np.zeros(n)}),
            Segment.outcome: pl.DataFrame({"stay_id": range(n), "label": labels}),
        }
    }
    return PredictionPolarsDataset(data, split=Split.train, vars=VARS, ram_cache=False)


@pytest.mark.parametrize("seed", range(8))
def test_minority_class_gets_larger_weight(seed):
    labels = np.random.default_rng(seed).permutation([1.0] * 10 + [0.0] * 90)
    assert dataset(labels).get_balance() == pytest.approx([100 / 180, 100 / 20])


def test_minority_class_zero():
    assert dataset([0] * 10 + [1] * 90).get_balance() == pytest.approx([100 / 20, 100 / 180])


def test_missing_labels_are_not_a_class():
    assert dataset([0.0, 1.0, 1.0, None, float("nan")]).get_balance() == pytest.approx([1.5, 0.75])


def test_labels_must_be_class_indices():
    with pytest.raises(ValueError, match="labels 0..K-1"):
        dataset([1, 2, 2]).get_balance()


def splits():
    rng = np.random.default_rng(0)
    data = {}
    for split in (Split.train, Split.val, Split.test):
        x = rng.normal(size=40)
        data[split] = {
            Segment.features: pl.DataFrame({"stay_id": range(40), "time": [0] * 40, "x": x}),
            Segment.outcome: pl.DataFrame({"stay_id": range(40), "label": (x > 0).astype(int)}),
        }
    return data


def gru(mode, num_classes):
    return GRUNet(input_size=[1, 1, 1], hidden_dim=4, layer_dim=1, num_classes=num_classes, run_mode=mode)


def test_regression_ignores_class_weights():
    model = gru(RunMode.regression, 1)
    model.set_weight("balanced", dataset([0.3, 12.5, 7.0]))
    assert model.loss_weights is None


def test_dl_classification_trains_without_class_weights(tmp_path):
    gin.clear_config()
    gin.bind_parameter("PredictionPolarsDataset.vars", VARS)
    gin.bind_parameter("DLPredictionWrapper.loss", cross_entropy)
    for name, value in {"input_size": [1, 1, 1], "hidden_dim": 4, "layer_dim": 1, "num_classes": 2}.items():
        gin.bind_parameter(f"GRUNet.{name}", value)
    try:
        # weight defaults to "", which used to crash the classification loss
        score = train_common(splits(), log_dir=tmp_path, model=GRUNet, epochs=1, cpu=True, num_workers=0)
        assert np.isfinite(score)
    finally:
        gin.clear_config()
