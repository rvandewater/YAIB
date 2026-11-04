"""GRU-D: a GRU that decays inputs and hidden state with the time since each variable was last observed.

GRU-D (Che et al., Recurrent Neural Networks for Multivariate Time Series with Missing Values, Scientific Reports 8,
6085, 2018, https://doi.org/10.1038/s41598-018-24271-9) feeds a GRU the observation mask m_t and decays both the
imputed inputs and the hidden state with the time delta_t since each variable was last observed (equations 2 and 10 to
16 of the paper):

    gamma_x = exp(-max(0, W_gamma_x delta_t + b_gamma_x))      (W_gamma_x diagonal)
    gamma_h = exp(-max(0, W_gamma_h delta_t + b_gamma_h))
    x_hat_t = m_t x_t + (1 - m_t) (gamma_x x_last + (1 - gamma_x) x_mean)
    h_hat   = gamma_h h_{t-1}
    z_t, r_t = sigmoid(W x_hat_t + U h_hat + V m_t + b)        (one set of weights per gate)
    h_t     = (1 - z_t) h_hat + z_t tanh(W x_hat_t + U (r_t h_hat) + V m_t + b)

Ported from the YAIB fork by Katja Jagd and Rachael DeVries (https://github.com/Katja-Jagd/YAIB, branch ``katja``,
``icu_benchmarks/models/dl_models/grud.py``, MIT License), which was used for BAT pretraining on YAIB (Jagd et al.,
https://arxiv.org/abs/2509.19885) and as a baseline for the Bi-Axial Transformer (DeVries et al.,
https://arxiv.org/abs/2508.12418). The fork's cell is a PyTorch translation of the TensorFlow GRU-D in SeFT (Horn et al.,
https://github.com/BorgwardtLab/Set_Functions_for_Time_Series, ``seft/models/gru_d.py``, BSD 3-Clause License), which
adapts the Keras implementation by the first author of GRU-D (https://github.com/PeterChe1990/GRU-D,
``nn_utils/grud_layers.py``, MIT License). Both license notices are reproduced at the end of this docstring.

How the port fits YAIB:

- **Inputs.** The model takes the same hourly feature matrix as YAIB's GRU. The observation mask m_t comes from the
  ``MissingIndicator_<name>`` columns of the preprocessing recipe (see ``ObservationMaskWrapper``); columns without an
  indicator, including the indicators themselves, count as always observed. YAIB's data is on a regular hourly grid, so
  delta_t counts hours since the last measurement. The features are standardised, so the empirical mean x_mean is 0,
  which is also the value the recipe fills in before the first measurement. Static features are inputs at every hour,
  as for the other YAIB models, instead of initialising the hidden state as in the fork and SeFT.
- **Output.** The model predicts at every hour from the hidden state h_t, so it is causal by construction. Per-stay
  tasks use the prediction at the last observed hour, which is the fork's ``pooling="hidden"``.
- **Fixes to the fork.** The decay parameters were initialised to exactly zero, where relu has no gradient, so the
  decays never trained; their biases now start slightly positive (decay rates close to 1, as before). The gate biases
  were never initialised and were not used when the mask was fed to the gates (the Keras reference always adds them).
  Recurrent and mask dropout were never applied because of an inverted condition. The new state interpolates with the
  decayed state h_hat, as in equation 16 of the paper (the Keras reference used the undecayed h_{t-1}). The cell is a
  plain module instead of TorchScript, so the model can be pickled with ``save_model``.
- **Left out.** The fork's options for ablations and other tasks: zero, forward or raw imputation instead of the
  input decay, masking decay, indicator-only or value-only inputs, mean or max pooling over time, the static encoder,
  self-supervised pretraining (``SSL_GRUD``) and GRU-D as an imputation model.

License notices of the upstream implementations:

SeFT (https://github.com/BorgwardtLab/Set_Functions_for_Time_Series), BSD 3-Clause License:

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

GRU-D Keras implementation (https://github.com/PeterChe1990/GRU-D), MIT License:

    Copyright (c) 2018 Zhengping Che

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of
    the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
    THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

import gin
import torch
import torch.nn.functional as F
from torch import nn

from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.wrappers import ObservationMaskWrapper

# Initial bias of the decay rates: gamma = exp(-0.01) is close to 1 (no decay), but relu passes gradients
DECAY_BIAS_INIT = 0.01


def decay_rate(pre_activation: torch.Tensor) -> torch.Tensor:
    """Decay rate exp(-max(0, x)) in (0, 1], monotonically decreasing in the time since the last observation."""
    return torch.exp(-torch.relu(pre_activation))


def variational_dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """One dropout mask per sequence, shared by all time steps (Gal and Ghahramani, 2016), as in the Keras GRU-D."""
    if not training or p == 0:
        return x
    shape = (x.shape[0], 1, x.shape[-1]) if x.dim() == 3 else x.shape
    return x * F.dropout(x.new_ones(shape), p=p, training=True)


class GRUDLayer(nn.Module):
    """GRU-D recurrence over a whole sequence.

    Args:
        input_dim: Number of input columns.
        hidden_dim: Size of the hidden state.
        dropout: Dropout on the inputs and the mask, with one mask per sequence.
        recurrent_dropout: Dropout on the hidden state in the recurrent connections, with one mask per sequence.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0, recurrent_dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        # Input decay: one rate per column (diagonal W_gamma_x)
        self.input_decay_weight = nn.Parameter(torch.empty(input_dim))
        self.input_decay_bias = nn.Parameter(torch.empty(input_dim))
        # Hidden decay: a full matrix W_gamma_h
        self.hidden_decay = nn.Linear(input_dim, hidden_dim)
        # W x_hat + b and V m for the update gate, the reset gate and the candidate state, in this order
        self.input_gates = nn.Linear(input_dim, 3 * hidden_dim)
        self.mask_gates = nn.Linear(input_dim, 3 * hidden_dim, bias=False)
        # U h_hat for the update and reset gates, and U (r * h_hat) for the candidate state
        self.recurrent_gates = nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)
        self.recurrent_candidate = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialise as the Keras reference: Glorot-uniform kernels, orthogonal recurrent kernels, zero gate biases.

        The decays start without decay, as in the reference, but with a small positive bias instead of zero: relu has no
        gradient at zero, so all-zero decay parameters would never train.
        """
        for gates in (self.input_gates, self.mask_gates):
            for kernel in gates.weight.split(self.hidden_dim):
                nn.init.xavier_uniform_(kernel)
        for kernel in (*self.recurrent_gates.weight.split(self.hidden_dim), self.recurrent_candidate.weight):
            nn.init.orthogonal_(kernel)
        nn.init.zeros_(self.input_gates.bias)
        for weight, bias in ((self.input_decay_weight, self.input_decay_bias), self.hidden_decay.parameters()):
            nn.init.zeros_(weight)
            nn.init.constant_(bias, DECAY_BIAS_INIT)

    def forward(self, x: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        """Run GRU-D over the sequence.

        Args:
            x: Inputs of shape (batch, time, columns). Values that were not observed are ignored, except as a fallback
                for the first hours before a column was measured, where the empirical mean 0 is used.
            observed: Observation mask of the same shape, 1 where x was measured and 0 where it was imputed.

        Returns:
            Hidden states of shape (batch, time, hidden_dim).
        """
        batch, steps, _ = x.shape
        # delta_t: hours since each column was last observed before t (equation 2), and that last observed value
        deltas, last_values = [], []
        delta, last = torch.zeros_like(x[:, 0]), torch.zeros_like(x[:, 0])
        for t in range(steps):
            deltas.append(delta)
            last_values.append(last)
            delta = 1 + (1 - observed[:, t]) * delta
            last = observed[:, t] * x[:, t] + (1 - observed[:, t]) * last
        delta, last = torch.stack(deltas, dim=1), torch.stack(last_values, dim=1)

        # Everything except the recurrence depends on the inputs only, so it is computed for all time steps at once
        gamma_x = decay_rate(delta * self.input_decay_weight + self.input_decay_bias)
        x_hat = observed * x + (1 - observed) * gamma_x * last
        gamma_h = decay_rate(self.hidden_decay(delta))
        x_hat = variational_dropout(x_hat, self.dropout, self.training)
        mask = variational_dropout(observed, self.dropout, self.training)
        gate_inputs = self.input_gates(x_hat) + self.mask_gates(mask)
        gate_inputs, candidate_inputs = gate_inputs.split([2 * self.hidden_dim, self.hidden_dim], dim=-1)

        h = x.new_zeros(batch, self.hidden_dim)
        keep = variational_dropout(x.new_ones(batch, self.hidden_dim), self.recurrent_dropout, self.training)
        outputs = []
        for t in range(steps):
            h = gamma_h[:, t] * h
            update, reset = torch.sigmoid(gate_inputs[:, t] + self.recurrent_gates(h * keep)).chunk(2, dim=-1)
            candidate = torch.tanh(candidate_inputs[:, t] + self.recurrent_candidate(reset * h * keep))
            h = (1 - update) * h + update * candidate
            outputs.append(h)
        return torch.stack(outputs, dim=1)


@gin.configurable
class GRUD(ObservationMaskWrapper):
    """GRU-D (Che et al., 2018) with one prediction per hour; see the module docstring.

    Args:
        input_size: Shape (batch, time, columns) of the input.
        hidden_dim: Size of the hidden state.
        num_classes: Number of classes, or 1 for regression.
        dropout: Dropout on the inputs and the observation mask, with one mask per sequence.
        recurrent_dropout: Dropout on the hidden state in the recurrent connections, with one mask per sequence.
    """

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(self, input_size, hidden_dim, num_classes, dropout=0.0, recurrent_dropout=0.0, *args, **kwargs):
        super().__init__(
            *args,
            input_size=input_size,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            **kwargs,
        )
        self.grud = GRUDLayer(input_size[2], hidden_dim, dropout=dropout, recurrent_dropout=recurrent_dropout)
        self.logit = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.logit(self.grud(x, self.observed_mask(x)))
