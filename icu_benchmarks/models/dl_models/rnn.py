import gin
from torch import nn as nn

from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.wrappers import DLPredictionWrapper


@gin.configurable
class RNNet(DLPredictionWrapper):
    """Torch standard RNN model"""

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(self, input_size, hidden_dim, layer_dim, num_classes, *args, **kwargs):
        super().__init__(
            *args, input_size=input_size, hidden_dim=hidden_dim, layer_dim=layer_dim, num_classes=num_classes, **kwargs
        )
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_size[2], hidden_dim, layer_dim, batch_first=True)
        self.logit = nn.Linear(hidden_dim, num_classes)

    def init_hidden(self, x):
        h0 = x.new_zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return h0

    def forward(self, x):
        h0 = self.init_hidden(x)
        out, hn = self.rnn(x, h0)
        pred = self.logit(out)
        return pred


@gin.configurable
class LSTMNet(DLPredictionWrapper):
    """Torch standard LSTM model."""

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(self, input_size, hidden_dim, layer_dim, num_classes, *args, **kwargs):
        super().__init__(
            *args, input_size=input_size, hidden_dim=hidden_dim, layer_dim=layer_dim, num_classes=num_classes, **kwargs
        )
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_size[2], hidden_dim, layer_dim, batch_first=True)
        self.logit = nn.Linear(hidden_dim, num_classes)

    def init_hidden(self, x):
        h0 = x.new_zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = x.new_zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t for t in (h0, c0)]

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, h = self.rnn(x, (h0, c0))
        pred = self.logit(out)
        return pred


@gin.configurable
class GRUNet(DLPredictionWrapper):
    """Torch standard GRU model."""

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(self, input_size, hidden_dim, layer_dim, num_classes, *args, **kwargs):
        super().__init__(
            *args, input_size=input_size, hidden_dim=hidden_dim, layer_dim=layer_dim, num_classes=num_classes, **kwargs
        )
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.GRU(input_size[2], hidden_dim, layer_dim, batch_first=True)
        self.logit = nn.Linear(hidden_dim, num_classes)

    def init_hidden(self, x):
        h0 = x.new_zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return h0

    def forward(self, x):
        h0 = self.init_hidden(x)
        out, hn = self.rnn(x, h0)
        pred = self.logit(out)
        return pred
