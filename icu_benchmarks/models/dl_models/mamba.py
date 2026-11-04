"""Mamba selective state-space model for YAIB's hourly time series.

Mamba (Gu and Dao, 2023, https://arxiv.org/abs/2312.00752) replaces attention with a selective state-space layer whose
state update depends on the input, so it scales linearly with sequence length. EHRMamba (Fallahpour et al., 2024,
https://arxiv.org/abs/2405.14567) showed that this backbone works well for EHR prediction. This module contains the Mamba
backbone only, adapted to the inputs every YAIB deep learning model receives, so its results are not comparable to
those reported for EHRMamba:

- **Input.** EHRMamba reads a sequence of tokenized medical codes and events with concept, type, time, age and visit
  embeddings. YAIB gives each stay a dense hourly grid of features (with missingness indicators), so a linear layer
  embeds each hour, as in YAIB's Transformer. No positional or time embedding is added: the recurrence is ordered by
  construction, and the hourly grid makes the time between steps constant.
- **Training.** EHRMamba is pretrained on all of MIMIC-IV and then fine-tuned with multitask prompted fine-tuning. Here
  the model is trained from scratch for each task, like the other YAIB deep learning models.
- **Output.** EHRMamba classifies a whole sequence. YAIB predicts at every hour, so the model returns one prediction
  per time step. Mamba is causal (the convolution is left-padded and the scan runs forward in time), so the
  prediction at hour t only depends on hours up to t and per-stay tasks use the prediction at the last observed hour.
  YAIB pads sequences at the end, so padding never affects the predictions for real hours.
- **Size.** EHRMamba uses 32 layers with a width of 768 on sequences of up to 2048 tokens. YAIB's sequences are at most
  a few hundred hours, so the default configuration is much smaller (see configs/prediction_models/Mamba.gin).

The layer itself follows the reference implementation (https://github.com/state-spaces/mamba, ``Mamba`` and
``MixerModel``): input projection with a gate, causal depthwise convolution, input-dependent step size and B/C
projections, S4D-real initialization of A, the inverse-softplus initialization of the step-size bias, pre-norm residual
blocks with RMSNorm, and output projections rescaled by 1/sqrt(depth). The scan runs in plain PyTorch, so the model
trains on CPU, MPS and CUDA without extra packages. If ``mamba-ssm`` is installed (``pip install mamba-ssm``, which
needs an NVIDIA GPU) and the model runs on CUDA, its fused selective-scan kernel is used instead; the results are the
same up to floating-point differences.
"""

import math

import gin
import torch
import torch.nn.functional as F
from torch import nn

from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.wrappers import DLPredictionWrapper

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:  # mamba-ssm is optional and only works on NVIDIA GPUs
    selective_scan_fn = None


def selective_scan(u, delta, A, B, C, D):
    """Run the selective state-space recurrence in plain PyTorch.

    h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * u_t and y_t = C_t . h_t + D * u_t, for every channel.

    Args:
        u: Input of shape (batch, time, channels).
        delta: Positive step sizes of shape (batch, time, channels).
        A: State matrix (diagonal, negative) of shape (channels, state).
        B: Input matrices of shape (batch, time, state).
        C: Output matrices of shape (batch, time, state).
        D: Skip connection of shape (channels,).

    Returns:
        Output of shape (batch, time, channels).
    """
    batch, _, channels = u.shape
    # (batch, channels, state) per time step. Unbinding once instead of indexing every step keeps the backward pass
    # linear in the sequence length: each index would allocate a gradient the size of the whole sequence.
    decay = torch.exp(delta.unsqueeze(-1) * A).unbind(1)
    update = ((delta * u).unsqueeze(-1) * B.unsqueeze(2)).unbind(1)
    state = u.new_zeros(batch, channels, A.shape[1])
    states = []
    for step_decay, step_update in zip(decay, update):
        state = torch.addcmul(step_update, step_decay, state)
        states.append(state)
    return torch.einsum("btcs,bts->btc", torch.stack(states, dim=1), C) + u * D


def use_fused_scan(u):
    """Whether to use the fused CUDA kernel from mamba-ssm for this input."""
    return selective_scan_fn is not None and u.is_cuda


class MambaMixer(nn.Module):
    """A Mamba layer: gated selective state-space model with a causal depthwise convolution."""

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
    ):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Step sizes start log-uniform in [dt_min, dt_max]; the bias is their inverse softplus
        dt_init_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=dt_init_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

        # S4D-real initialization: A = -(1, 2, ..., d_state) for every channel, stored as log(-A) to keep A negative
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x):
        steps = x.shape[1]
        x, gate = self.in_proj(x).chunk(2, dim=-1)
        # Left padding plus truncation keeps the convolution causal
        x = F.silu(self.conv1d(x.transpose(1, 2))[..., :steps].transpose(1, 2))
        dt, B, C = torch.split(self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())
        if use_fused_scan(x):
            y = selective_scan_fn(
                x.transpose(1, 2).contiguous(),
                (dt @ self.dt_proj.weight.t()).transpose(1, 2).contiguous(),
                A,
                B.transpose(1, 2).contiguous(),
                C.transpose(1, 2).contiguous(),
                self.D.float(),
                z=gate.transpose(1, 2).contiguous(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            ).transpose(1, 2)
        else:
            y = selective_scan(x, F.softplus(self.dt_proj(dt)), A, B, C, self.D) * F.silu(gate)
        return self.out_proj(y)


class MambaBlock(nn.Module):
    """Pre-norm residual block around a Mamba layer."""

    def __init__(self, d_model, dropout=0.0, **mixer_kwargs):
        super().__init__()
        self.norm = nn.RMSNorm(d_model, eps=1e-5)
        self.mixer = MambaMixer(d_model, **mixer_kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.mixer(self.norm(x)))


@gin.configurable
class Mamba(DLPredictionWrapper):
    """Mamba selective state-space model with one prediction per time step.

    Args:
        input_size: Shape of an input batch (batch, time, features).
        hidden: Width of the model (d_model).
        depth: Number of Mamba blocks.
        num_classes: Number of outputs per time step.
        d_state: Size of the state per channel.
        d_conv: Kernel size of the causal convolution.
        expand: Expansion factor of the inner channels (d_inner = expand * hidden).
        dropout: Dropout on the embedded input and on each block's output.
    """

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(
        self,
        input_size,
        hidden,
        depth,
        num_classes,
        *args,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(
            *args,
            input_size=input_size,
            hidden=hidden,
            depth=depth,
            num_classes=num_classes,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            **kwargs,
        )
        self.input_embedding = nn.Linear(input_size[2], hidden)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[MambaBlock(hidden, dropout=dropout, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(depth)]
        )
        self.norm = nn.RMSNorm(hidden, eps=1e-5)
        self.logit = nn.Linear(hidden, num_classes)
        # As in the reference MixerModel: keep the residual stream's variance from growing with depth
        for block in self.blocks:
            with torch.no_grad():
                block.mixer.out_proj.weight /= math.sqrt(depth)

    def forward(self, x):
        x = self.embedding_dropout(self.input_embedding(x))
        x = self.norm(self.blocks(x))
        return self.logit(x)
