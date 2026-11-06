"""The Mamba model's scan must match the state-space recurrence, and the fused mamba-ssm path must get the same inputs."""

from pathlib import Path

import gin
import pytest
import torch
import torch.nn.functional as F

import icu_benchmarks.models.dl_models.mamba as mamba
import icu_benchmarks.run  # noqa: F401 (registers the Run configurable of the task configs)
from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.dl_models.mamba import Mamba, MambaMixer, selective_scan

ROOT = Path(__file__).parent.parent


@pytest.fixture(autouse=True)
def clean_gin():
    gin.clear_config()
    yield
    gin.clear_config()


def naive_scan(u, delta, A, B, C, D, z=None, delta_bias=None, delta_softplus=False):
    """Element-by-element recurrence in mamba-ssm's (batch, channels, time) layout, as selective_scan_fn takes it."""
    if delta_bias is not None:
        delta = delta + delta_bias[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    batch, channels, steps = u.shape
    y = torch.zeros_like(u)
    for b in range(batch):
        for c in range(channels):
            h = torch.zeros(A.shape[1], dtype=u.dtype)
            for t in range(steps):
                h = torch.exp(delta[b, c, t] * A[c]) * h + delta[b, c, t] * B[b, :, t] * u[b, c, t]
                y[b, c, t] = (h * C[b, :, t]).sum() + D[c] * u[b, c, t]
    return y * F.silu(z) if z is not None else y


def scan_inputs(batch=2, steps=5, channels=3, state=4):
    torch.manual_seed(0)
    u = torch.randn(batch, steps, channels, dtype=torch.float64)
    delta = F.softplus(torch.randn(batch, steps, channels, dtype=torch.float64))
    A = -torch.rand(channels, state, dtype=torch.float64) - 0.1
    B, C = torch.randn(batch, steps, state, dtype=torch.float64), torch.randn(batch, steps, state, dtype=torch.float64)
    return u, delta, A, B, C, torch.randn(channels, dtype=torch.float64)


def test_scan_matches_recurrence():
    u, delta, A, B, C, D = scan_inputs()
    expected = naive_scan(u.transpose(1, 2), delta.transpose(1, 2), A, B.transpose(1, 2), C.transpose(1, 2), D)
    torch.testing.assert_close(selective_scan(u, delta, A, B, C, D), expected.transpose(1, 2))


def test_fused_path_gets_the_same_inputs(monkeypatch):
    # mamba-ssm needs CUDA, so stand in for its kernel with the naive recurrence and compare against the PyTorch path
    torch.manual_seed(0)
    mixer = MambaMixer(d_model=6, d_state=4, d_conv=3).double()
    x = torch.randn(2, 5, 6, dtype=torch.float64)
    expected = mixer(x)
    monkeypatch.setattr(mamba, "selective_scan_fn", naive_scan)
    monkeypatch.setattr(mamba, "use_fused_scan", lambda u: True)
    torch.testing.assert_close(mixer(x), expected)


def test_initialization_follows_the_reference():
    torch.manual_seed(0)
    mixer = MambaMixer(d_model=32, d_state=8, dt_min=0.001, dt_max=0.1)
    assert mixer.d_inner == 64 and mixer.dt_rank == 2
    torch.testing.assert_close(-torch.exp(mixer.A_log[0]), -torch.arange(1.0, 9.0))
    dt = F.softplus(mixer.dt_proj.bias)
    assert (dt >= 0.001 - 1e-6).all() and (dt <= 0.1 + 1e-6).all()
    deep, shallow = Mamba((1, 4, 3), hidden=8, depth=4, num_classes=2), Mamba((1, 4, 3), hidden=8, depth=1, num_classes=2)
    assert deep.blocks[0].mixer.out_proj.weight.abs().max() <= shallow.blocks[0].mixer.out_proj.weight.abs().max()


@pytest.mark.parametrize(("mode", "num_classes"), [(RunMode.classification, 3), (RunMode.regression, 1)])
def test_backward(mode, num_classes):
    torch.manual_seed(0)
    model = Mamba((2, 7, 3), hidden=8, depth=2, num_classes=num_classes, dropout=0.1, run_mode=mode)
    out = model(torch.randn(2, 7, 3))
    assert out.shape == (2, 7, num_classes)
    out.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    # load_from_checkpoint takes the input size from the checkpoint and the architecture from the gin config
    assert model.hparams["input_size"] == (2, 7, 3)


def test_config_parses(monkeypatch):
    # The configs include each other relative to the repository
    monkeypatch.chdir(ROOT)
    gin.parse_config_files_and_bindings(
        ["configs/prediction_models/Mamba.gin", "configs/tasks/BinaryClassification.gin"], [], finalize_config=False
    )
    assert gin.query_parameter("train_common.model").selector.endswith("Mamba")
    assert gin.query_parameter("model/hyperparameter.hidden") == (32, 256, "log")
    assert gin.query_parameter("model/hyperparameter.d_state") == 16
