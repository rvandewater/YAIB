"""DL models must be tested with the weights of the best validation epoch, not the last one (#244)."""

import json

import gin
import gin.torch.external_configurables  # noqa: F401  registers torch.optim.Adam
import numpy as np
import polars as pl
import pytest
import torch
from torch.nn.functional import cross_entropy

from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.constants import DataSegment as Segment, DataSplit as Split
from icu_benchmarks.models.dl_models.rnn import GRUNet
from icu_benchmarks.models.train import train_common

VARS = {"GROUP": "stay_id", "SEQUENCE": "time", "LABEL": "label", "DYNAMIC": ["x"], "STATIC": []}
STAYS, HOURS, EPOCHS = 64, 5, 15


def random_label_splits():
    """Labels are noise, so a large GRU overfits and its validation loss rises after the first epochs."""
    rng = np.random.default_rng(0)
    features = {"stay_id": np.repeat(np.arange(STAYS), HOURS), "time": np.tile(np.arange(HOURS), STAYS)}
    return {
        split: {
            Segment.features: pl.DataFrame({**features, "x": rng.normal(size=STAYS * HOURS)}),
            Segment.outcome: pl.DataFrame({"stay_id": np.arange(STAYS), "label": rng.integers(0, 2, STAYS).astype(float)}),
        }
        for split in (Split.train, Split.val, Split.test)
    }


@pytest.fixture
def gru_config():
    gin.clear_config()
    gin.bind_parameter("PredictionPolarsDataset.vars", VARS)
    gin.bind_parameter("DLPredictionWrapper.loss", cross_entropy)
    # As in the DL configs, the learning rate is bound on the optimizer
    gin.bind_parameter("torch.optim.Adam.lr", 1e-2)
    for name, value in {"input_size": [1, HOURS, 1], "hidden_dim": 64, "layer_dim": 2, "num_classes": 2}.items():
        gin.bind_parameter(f"GRUNet.{name}", value)
    yield
    gin.clear_config()


def test_dl_tests_best_validation_checkpoint(gru_config, tmp_path):
    optimizer = gin.get_configurable("torch.optim.Adam")
    kwargs = dict(optimizer=optimizer, model=GRUNet, mode=RunMode.classification, weight="balanced", batch_size=8, cpu=True)
    train_common(random_label_splits(), log_dir=tmp_path, epochs=EPOCHS, patience=100, num_workers=0, **kwargs)

    best = torch.load(tmp_path / "model.ckpt", weights_only=False)
    callback = next(state for key, state in best["callbacks"].items() if "ModelCheckpoint" in key)
    assert callback["monitor"] == "val/loss"
    # Validation loss rises while overfitting, so the best epoch is not the last one
    assert best["epoch"] < EPOCHS - 1

    # Evaluating model.ckpt reproduces the training run's test predictions, so that run tested the best weights.
    # Compare the AUC: the balanced loss weights can come out in a different order per run until #243 is fixed.
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    train_common(
        random_label_splits(),
        log_dir=eval_dir,
        source_dir=tmp_path,
        eval_only=True,
        load_weights=True,
        num_workers=0,
        **kwargs,
    )
    original = json.loads((tmp_path / "test_metrics.json").read_text())
    replay = json.loads((eval_dir / "test_metrics.json").read_text())
    assert replay["AUC"] == pytest.approx(original["AUC"])
