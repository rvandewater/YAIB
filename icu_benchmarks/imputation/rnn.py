from icu_benchmarks.models.wrappers import ImputationWrapper
import torch
import torch.nn as nn
from torch.autograd import Variable
import gin


# Adapted from https://github.com/Graph-Machine-Learning-Group/grin/blob/main/lib/nn/models/rnn_imputers.py
@gin.configurable("RNN")
class RNNImputation(ImputationWrapper):
    """Imputation model with Gated Recurrent Units (GRU) or Long-Short Term Memory Network (LSTM). Defaults to GRU."""

    requires_backprop = True

    def __init__(self, *args, input_size, hidden_size=64, state_init="zero", cell="gru", **kwargs) -> None:
        super().__init__(*args, input_size=input_size, hidden_size=hidden_size, state_init=state_init, cell=cell, **kwargs)
        self.input_size = input_size
        self.n_features = input_size[2]
        self.hidden_size = hidden_size
        self.state_init = state_init
        self.cell = cell

        if cell == "gru":
            cell = nn.GRUCell
        elif cell == "lstm":
            cell = nn.LSTMCell
        else:
            raise NotImplementedError(f'"{cell}" cell not implemented.')

        self.rnn = cell(self.n_features, self.hidden_size)
        self.fn = nn.Linear(self.hidden_size, self.n_features)

    def init_hidden_state(self, x):
        if self.state_init == "zero":
            return torch.zeros((x.size(0), self.hidden_size), device=x.device, dtype=x.dtype)
        if self.state_init == "noise":
            return torch.randn(x.size(0), self.hidden_size, device=x.device, dtype=x.dtype)

    def forward(self, amputated, amputation_mask, return_hidden=False):
        steps = amputated.size(1)
        amputated = torch.where(amputation_mask.bool(), torch.zeros_like(amputated), amputated)
        h = self.init_hidden_state(amputated)
        c = self.init_hidden_state(amputated)

        output = self.fn(h)
        hs = [h]
        preds = [output]
        for s in range(steps - 1):
            x_t = torch.where(amputation_mask[:, s].bool(), output, amputated[:, s])
            if self.cell == "gru":
                h = self.rnn(x_t, h)
            elif self.cell == "lstm":
                h, c = self.rnn(x_t, (h, c))
            output = self.fn(h)
            hs.append(h)
            preds.append(output)

        output = torch.stack(preds, 1)
        h = torch.stack(hs, 1)

        if return_hidden:
            return output, h
        return output


@gin.configurable("BRNN")
class BRNNImputation(ImputationWrapper):
    """Imputation model with Bidirectional Gated Recurrent Units (GRU) or Long-Short Term Memory Network (LSTM). Defaults to
    GRU."""

    requires_backprop = True

    def __init__(self, *args, input_size, hidden_size=64, state_init="zero", dropout=0.0, cell="gru", **kwargs) -> None:
        super().__init__(
            *args, input_size=input_size, hidden_size=hidden_size, state_init=state_init, dropout=dropout, cell=cell, **kwargs
        )
        self.hidden_size = hidden_size
        self.fwd_rnn = RNNImputation(input_size=input_size, hidden_size=hidden_size, state_init=state_init, cell=cell)
        self.bwd_rnn = RNNImputation(input_size=input_size, hidden_size=hidden_size, state_init=state_init, cell=cell)
        self.dropout = nn.Dropout(dropout)
        self.fn = nn.Linear(2 * hidden_size, input_size[2])

    def forward(self, amputated, amputation_mask):
        _, h_fwd = self.fwd_rnn(amputated, amputation_mask, return_hidden=True)
        _, h_bwd = self.bwd_rnn(self.reverse_tensor(amputated, 1), self.reverse_tensor(amputation_mask, 1), return_hidden=True)
        h_bwd = self.reverse_tensor(h_bwd, 1)

        h = self.dropout(torch.cat([h_fwd, h_bwd], -1))
        output = self.fn(h)

        return output

    @staticmethod
    def reverse_tensor(tensor=None, axis=-1):
        if tensor is None:
            return None
        if tensor.dim() <= 1:
            return tensor
        indices = range(tensor.size()[axis])[::-1]
        indices = Variable(torch.LongTensor(indices), requires_grad=False).to(tensor.device)
        return tensor.index_select(axis, indices)
