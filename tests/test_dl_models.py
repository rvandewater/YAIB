"""The DL models must predict per time step without looking ahead, and train, test and reload through train_common."""

import json

import gin
import numpy as np
import polars as pl
import pytest
import torch

from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.constants import DataSegment as Segment, DataSplit as Split
from icu_benchmarks.models.dl_models.layers import SelfAttention, SparseBlock, parallel_recomb
from icu_benchmarks.models.dl_models.rnn import GRUNet, LSTMNet, RNNet
from icu_benchmarks.models.dl_models.tcn import TemporalConvNet
from icu_benchmarks.models.dl_models.transformer import LocalTransformer, Transformer
from icu_benchmarks.models.train import load_model, train_common

BATCH, STEPS, FEATURES, CLASSES = 3, 6, 4, 2
INPUT_SIZE = (BATCH, STEPS, FEATURES)
VARS = {"GROUP": "stay_id", "SEQUENCE": "time", "LABEL": "label", "DYNAMIC": ["x", "noise"], "STATIC": []}

MODELS = {
    "rnn": lambda: RNNet(INPUT_SIZE, hidden_dim=5, layer_dim=1, num_classes=CLASSES),
    "lstm": lambda: LSTMNet(INPUT_SIZE, hidden_dim=5, layer_dim=2, num_classes=CLASSES),
    "gru": lambda: GRUNet(INPUT_SIZE, hidden_dim=5, layer_dim=1, num_classes=CLASSES),
    "tcn": lambda: TemporalConvNet(INPUT_SIZE, num_channels=[5, 5], num_classes=CLASSES, kernel_size=2),
    "tcn_depth_from_length": lambda: TemporalConvNet(INPUT_SIZE, num_channels=5, num_classes=CLASSES, max_seq_length=STEPS),
    "transformer": lambda: Transformer(INPUT_SIZE, hidden=7, heads=2, ff_hidden_mult=2, depth=1, num_classes=CLASSES),
    "local_transformer": lambda: LocalTransformer(
        INPUT_SIZE, hidden=8, heads=2, ff_hidden_mult=2, depth=2, num_classes=CLASSES, local_context=2
    ),
}


@pytest.fixture(autouse=True)
def clean_gin():
    gin.clear_config()
    yield
    gin.clear_config()


@pytest.mark.parametrize("name", MODELS)
def test_forward_is_causal(name):
    torch.manual_seed(0)
    model = MODELS[name]().eval()
    x = torch.randn(INPUT_SIZE)
    out = model(x)
    assert out.shape == (BATCH, STEPS, CLASSES) and torch.isfinite(out).all()
    future = x.clone()
    future[:, 3:] += 10
    # Predictions up to hour 2 must not change when later hours change
    torch.testing.assert_close(model(future)[:, :3], out[:, :3])
    assert not torch.allclose(model(future)[:, 3:], out[:, 3:])


def test_tcn_int_channels_require_sequence_length():
    with pytest.raises(Exception, match="maximum sequence length"):
        TemporalConvNet(INPUT_SIZE, num_channels=5, num_classes=CLASSES)


def test_transformer_rejects_local_context():
    with pytest.raises(ValueError, match="LocalTransformer"):
        Transformer(INPUT_SIZE, hidden=4, heads=1, ff_hidden_mult=1, depth=1, num_classes=CLASSES, local_context=2)


def test_attention_masks():
    t = torch.arange(1, 5, dtype=torch.float).reshape(1, -1)
    causal = parallel_recomb(t.clone(), t.clone(), "all")[0]
    torch.testing.assert_close(causal, torch.tril(torch.ones(4, 4)))
    local = parallel_recomb(t.clone(), t.clone(), "local", local_context=1)[0]
    torch.testing.assert_close(local, torch.tril(torch.ones(4, 4)) - torch.tril(torch.ones(4, 4), diagonal=-2))
    strided = parallel_recomb(t.clone(), t.clone(), "strided", local_context=2)[0]
    assert strided[3].tolist() == [0, 1, 0, 1]


@pytest.mark.parametrize("aggregation", ["union", "split"])
def test_sparse_attention_keeps_shape(aggregation):
    torch.manual_seed(0)
    x = torch.randn(2, 5, 4)
    attention = SelfAttention(4, 3, heads=2, att_type=["strided", "local"], local_context=2, mask_aggregation=aggregation)
    assert attention(x).shape == x.shape
    assert SparseBlock(4, 3, heads=2, ff_hidden_mult=2, local_context=2)(x).shape == x.shape


def test_configure_optimizers_with_scheduler():
    model = GRUNet(INPUT_SIZE, hidden_dim=3, layer_dim=1, num_classes=CLASSES, lr_scheduler="cosine", epochs=4)
    configured = model.configure_optimizers()
    assert isinstance(configured["optimizer"], torch.optim.Adam)
    assert isinstance(configured["lr_scheduler"], torch.optim.lr_scheduler.CosineAnnealingLR)
    assert isinstance(
        GRUNet(INPUT_SIZE, hidden_dim=3, layer_dim=1, num_classes=CLASSES).configure_optimizers(), torch.optim.Adam
    )


def test_imputation_run_mode_is_rejected():
    with pytest.raises(ValueError, match="not supported"):
        GRUNet(INPUT_SIZE, hidden_dim=3, layer_dim=1, num_classes=CLASSES, run_mode=RunMode.imputation)


def classification_splits(stays=16, hours=4):
    """Per-stay labels that depend on the last value of x."""
    rng = np.random.default_rng(0)
    data = {}
    for split in (Split.train, Split.val, Split.test):
        x = rng.normal(size=stays * hours)
        features = pl.DataFrame(
            {
                "stay_id": np.repeat(np.arange(stays), hours),
                "time": np.tile(np.arange(hours), stays),
                "x": x,
                "noise": rng.normal(size=stays * hours),
            }
        )
        label = (x[hours - 1 :: hours] > 0).astype(int)
        label[:2] = [0, 1]  # Both classes in every split
        data[split] = {
            Segment.features: features,
            Segment.outcome: pl.DataFrame({"stay_id": np.arange(stays), "label": label}),
        }
    return data


MODEL_PARAMS = {
    GRUNet: {"hidden_dim": 4, "layer_dim": 1},
    TemporalConvNet: {"num_channels": [4]},
    Transformer: {"hidden": 4, "heads": 1, "ff_hidden_mult": 1, "depth": 1},
}


def train(model_cls, log_dir, **kwargs):
    gin.bind_parameter("PredictionPolarsDataset.vars", VARS)
    gin.bind_parameter("DLPredictionWrapper.loss", torch.nn.functional.cross_entropy)
    for key, value in {**MODEL_PARAMS[model_cls], "num_classes": 2}.items():
        gin.bind_parameter(f"{model_cls.__name__}.{key}", value)
    kwargs = dict(model=model_cls, mode=RunMode.classification, epochs=1, batch_size=4, cpu=True, num_workers=0, **kwargs)
    return train_common(classification_splits(), log_dir=log_dir, **kwargs)


@pytest.mark.parametrize("model_cls", MODEL_PARAMS)
def test_train_common_one_epoch(model_cls, tmp_path):
    loss = train(model_cls, tmp_path, weight="balanced")
    assert np.isfinite(loss) and loss > 0
    metrics = json.loads((tmp_path / "test_metrics.json").read_text())
    assert metrics["loss"] == pytest.approx(loss) and 0 <= metrics["AUC"] <= 1 and 0 <= metrics["PR"] <= 1
    assert "loss" in json.loads((tmp_path / "train_metrics.json").read_text())
    assert "loss" in json.loads((tmp_path / "val_metrics.json").read_text())
    assert {"model.ckpt", "last.ckpt"} <= {p.name for p in tmp_path.iterdir()}
    assert f"{model_cls.__name__}.num_classes = 2" in (tmp_path / "train_config.gin").read_text()


@pytest.mark.parametrize(
    "model_cls",
    [
        GRUNet,
        TemporalConvNet,
        pytest.param(
            Transformer,
            marks=pytest.mark.xfail(
                strict=True,
                raises=TypeError,
                reason="#259: BaseTransformer does not pass input_size to the wrapper, so it is missing from the checkpoint",
            ),
        ),
    ],
)
def test_reload_reproduces_test_predictions(model_cls, tmp_path):
    train(model_cls, tmp_path, weight="balanced")
    loaded = load_model(model_cls, tmp_path, mode=RunMode.classification)
    assert isinstance(loaded, model_cls) and loaded.trained_columns == ["noise", "x"]
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    train(model_cls, eval_dir, source_dir=tmp_path, eval_only=True, load_weights=True)
    # The AUC depends on the predictions only (the balanced loss weights are not stable, see get_balance)
    original = json.loads((tmp_path / "test_metrics.json").read_text())
    evaluated = json.loads((eval_dir / "test_metrics.json").read_text())
    assert evaluated["AUC"] == pytest.approx(original["AUC"])


def test_load_model_without_weights_fails(tmp_path):
    with pytest.raises(Exception, match="No weights"):
        load_model(GRUNet, tmp_path, mode=RunMode.classification)
    with pytest.raises(Exception, match="No weights"):
        load_model(GRUNet, tmp_path / "missing", mode=RunMode.classification)
    with pytest.raises(ValueError, match="valid run mode"):
        load_model(GRUNet, tmp_path, mode="Clustering")
