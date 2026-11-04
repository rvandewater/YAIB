"""Bi-Axial Transformer (BAT): attention along both the variable axis and the time axis of the hourly grid.

BAT (DeVries et al., Bi-Axial Transformers: Addressing the Increasing Complexity of EHR Classification, 2025,
https://arxiv.org/abs/2508.12418) turns every (variable, hour) cell of a stay into a token made of the value, whether it
was measured, a learned embedding of the variable and an encoding of the time. Two streams of these tokens attend along
both axes in opposite orders: one attends across the variables within each hour and then across the hours of each
variable, the other across the hours first. Both streams use the same attention layers for each axis, are pooled over
the variables and merged by a nonlinear layer. Jagd et al. (https://arxiv.org/abs/2509.19885) pretrained BAT on YAIB.

Ported from the YAIB fork by Katja Jagd and Rachael DeVries (https://github.com/Katja-Jagd/YAIB, branch ``katja``,
``icu_benchmarks/models/dl_models/bat.py``, MIT License). The fork's BAT encoder comes from the BAT authors' reference
code, and its causal ("autoregressive") encoder, which this module ports, was added by Rachael DeVries. The time encoding
is SeFT's positional encoding (Horn et al., https://github.com/BorgwardtLab/Set_Functions_for_Time_Series,
``seft/models/transformer.py``, BSD 3-Clause License); its license notice is reproduced at the end of this docstring.

How the port fits YAIB:

- **Inputs.** The variables ("sensors") are the input columns that are not missingness indicators, static features
  included. Each token gets the value and the observation flag read from the ``MissingIndicator_<name>`` column (see
  ``ObservationMaskWrapper``); variables without an indicator count as measured. YAIB's data is on a regular hourly grid,
  so the time of a token is its hour in the stay. Static features are variables like the others instead of a separate
  embedding that is concatenated after pooling, as in the fork.
- **Output.** Attention across hours is causal and every other step works within an hour, so the prediction for hour t
  only depends on hours up to t. The model predicts at every hour, and per-stay tasks use the prediction at the last
  observed hour, which has attended to the whole stay. YAIB pads stays at the end, so padding never affects the
  predictions for real hours and no padding mask is needed.
- **Not ported.** The bidirectional encoder of the BAT paper, which max-pools over all variables and hours, is not
  included: it can only serve per-stay tasks, and YAIB does not pass a padding mask to ``forward``, so it would have to
  guess the padding from all-zero rows. The fork used it for per-stay tasks and the causal encoder for hourly tasks.
  Also left out: the pooling and input ablation options, attention-map outputs, the separate prediction heads (the
  ``logit`` layer replaces them), the forecasting head and self-supervised pretraining (``SSL_BAT``).
- **Attention layers.** YAIB's ``TransformerBlock`` (post-norm, ReLU feed-forward) replaces the pre-norm blocks of the
  x-transformers package, which avoids a new dependency. Hyperparameters follow YAIB's Transformer: ``depth`` is the
  fork's ``layers`` and ``dropout_att`` its ``attn_dropout``.
- **Fixes to the fork.** The padding mask of the time attention (``use_mask``) was repeated in the wrong order, so each
  row of a batch got the mask of another stay; it is no longer needed. An odd ``value_embed_size`` broke the time
  encoding; it is rounded up to an even size. The placeholder ``logit`` layer, which only existed to select the metrics,
  is now the output layer.

License notice of SeFT (https://github.com/BorgwardtLab/Set_Functions_for_Time_Series), BSD 3-Clause License:

    Copyright 2020 Max Horn

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    1. Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its
       contributors may be used to endorse or promote products derived from
       this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AD CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
    TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import gin
import torch
from torch import nn

from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.dl_models.layers import TransformerBlock
from icu_benchmarks.models.wrappers import ObservationMaskWrapper


def time_encoding(steps: int, dim: int, max_time: float, device=None) -> torch.Tensor:
    """SeFT's sinusoidal encoding of the times 0, ..., steps - 1, with timescales from 1 to max_time.

    Returns:
        Tensor of shape (steps, dim): sines in the first half and cosines in the second half of the last axis.
    """
    timescales = max_time ** torch.linspace(0, 1, dim // 2, device=device)
    scaled = torch.arange(steps, device=device, dtype=torch.float)[:, None] / timescales
    return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)


@gin.configurable
class BAT(ObservationMaskWrapper):
    """Causal Bi-Axial Transformer with one prediction per hour; see the module docstring.

    Args:
        input_size: Shape (batch, time, columns) of the input.
        num_classes: Number of classes, or 1 for regression.
        value_embed_size: Size of the embedding of a value and its observation flag (rounded up to an even number). The
            token size adds an embedding of the variable, of size 1.6 * sqrt(input columns) (the fork used the number
            of variables, which is only known once the column names are set).
        depth: Number of attention blocks per axis.
        heads: Number of attention heads.
        ff_hidden_mult: Width of the feed-forward layers, as a multiple of the token size.
        dropout: Dropout in the attention blocks.
        dropout_att: Dropout on the attention outputs.
        max_time: Largest timescale of the time encoding, in hours.
    """

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(
        self,
        input_size,
        num_classes,
        value_embed_size=8,
        depth=1,
        heads=1,
        ff_hidden_mult=4,
        dropout=0.0,
        dropout_att=0.0,
        max_time=500,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            input_size=input_size,
            num_classes=num_classes,
            value_embed_size=value_embed_size,
            depth=depth,
            heads=heads,
            ff_hidden_mult=ff_hidden_mult,
            dropout=dropout,
            dropout_att=dropout_att,
            max_time=max_time,
            **kwargs,
        )
        num_columns = input_size[2]
        sensor_embed_size = min(600, round(1.6 * num_columns**0.5))
        # The time encoding needs an even token size
        sensor_embed_size += sensor_embed_size % 2
        value_embed_size += value_embed_size % 2
        emb = value_embed_size + sensor_embed_size
        self.max_time = max_time
        self.value_embedding = nn.Linear(2, value_embed_size)
        # One embedding per input column; the rows of the indicator columns stay unused
        self.sensor_embedding = nn.Embedding(num_columns, sensor_embed_size)

        def blocks(causal):
            return nn.Sequential(
                *[
                    TransformerBlock(emb, emb, heads, ff_hidden_mult, dropout=dropout, mask=causal, dropout_att=dropout_att)
                    for _ in range(depth)
                ]
            )

        # Across the variables within an hour, and causally across the hours of a variable
        self.sensor_attention = blocks(causal=False)
        self.time_attention = blocks(causal=True)
        self.merge = nn.Linear(2 * emb, 2 * emb)
        self.logit = nn.Linear(2 * emb, num_classes)

    def attend_across_sensors(self, tokens):
        batch, steps, sensors, emb = tokens.shape
        return self.sensor_attention(tokens.reshape(batch * steps, sensors, emb)).reshape(tokens.shape)

    def attend_across_hours(self, tokens):
        batch, steps, sensors, emb = tokens.shape
        by_sensor = tokens.transpose(1, 2).reshape(batch * sensors, steps, emb)
        return self.time_attention(by_sensor).reshape(batch, sensors, steps, emb).transpose(1, 2)

    def forward(self, x):
        sensors = ~self.is_indicator
        values, observed = x[..., sensors], self.observed_mask(x)[..., sensors]
        batch, steps, _ = values.shape
        # One token per (hour, variable): value and observation flag, variable embedding, time encoding
        tokens = torch.cat(
            [
                self.value_embedding(torch.stack([values, observed], dim=-1)),
                self.sensor_embedding.weight[sensors].expand(batch, steps, -1, -1),
            ],
            dim=-1,
        )
        tokens = tokens + time_encoding(steps, tokens.shape[-1], self.max_time, x.device)[:, None]
        # The two streams attend along both axes in opposite orders, with shared layers per axis
        sensors_first = self.attend_across_hours(self.attend_across_sensors(tokens))
        hours_first = self.attend_across_sensors(self.attend_across_hours(tokens))
        merged = torch.relu(self.merge(torch.cat([sensors_first.mean(dim=2), hours_first.mean(dim=2)], dim=-1)))
        return self.logit(merged)
