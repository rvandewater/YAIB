import gin
from numbers import Integral
import numpy as np
import torch.nn as nn
import lightgbm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from icu_benchmarks.models.layers import TransformerBlock, LocalBlock, TemporalBlock, PositionalEncoding
from icu_benchmarks.models.wrappers import DLClassificationWrapper, MLClassificationWrapper


@gin.configurable
class LGBMClassifier(MLClassificationWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.model_args()

    @gin.configurable(module="LGBMClassifier")
    def model_args(self, *args, **kwargs):
        return lightgbm.LGBMClassifier(*args, **kwargs)


@gin.configurable
class LGBMRegressor(MLClassificationWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.model_args()

    @gin.configurable(module="LGBMRegressor")
    def model_args(self, *args, **kwargs):
        return lightgbm.LGBMRegressor(*args, **kwargs)


@gin.configurable
class LogisticRegression(MLClassificationWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.model_args()

    @gin.configurable(module="LogisticRegression")
    def model_args(self, *args, **kwargs):
        return LogisticRegression(*args, **kwargs)


@gin.configurable
class RFClassifier(MLClassificationWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.model_args()

    @gin.configurable(module="RFClassifier")
    def model_args(self, *args, **kwargs):
        return RandomForestClassifier(*args, **kwargs)

@gin.configurable
class RNNet(DLClassificationWrapper):
    """Torch standard RNN model"""
    def __init__(self, input_size, hidden_dim, layer_dim, num_classes, *args, **kwargs):
        super().__init__(
            input_size=input_size, hidden_dim=hidden_dim, layer_dim=layer_dim, num_classes=num_classes, *args, **kwargs
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
class LSTMNet(DLClassificationWrapper):
    """Torch standard LSTM model."""
    def __init__(self, input_size, hidden_dim, layer_dim, num_classes, *args, **kwargs):
        super().__init__(
            input_size=input_size, hidden_dim=hidden_dim, layer_dim=layer_dim, num_classes=num_classes, *args, **kwargs
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
class GRUNet(DLClassificationWrapper):
    """Torch standard GRU model."""
    def __init__(self, input_size, hidden_dim, layer_dim, num_classes, *args, **kwargs):
        super().__init__(
            input_size=input_size, hidden_dim=hidden_dim, layer_dim=layer_dim, num_classes=num_classes, *args, **kwargs
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


@gin.configurable
class Transformer(DLClassificationWrapper):
    """Transformer model as defined by the HiRID-Benchmark (https://github.com/ratschlab/HIRID-ICU-Benchmark)."""
    def __init__(
        self,
        input_size,
        hidden,
        heads,
        ff_hidden_mult,
        depth,
        num_classes,
        *args,
        dropout=0.0,
        l1_reg=0,
        pos_encoding=True,
        dropout_att=0.0,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            hidden=hidden,
            heads=heads,
            ff_hidden_mult=ff_hidden_mult,
            depth=depth,
            num_classes=num_classes,
            *args,
            dropout=dropout,
            l1_reg=l1_reg,
            pos_encoding=pos_encoding,
            dropout_att=dropout_att,
            **kwargs,
        )
        hidden = hidden if hidden % 2 == 0 else hidden + 1  # Make sure hidden is even
        self.input_embedding = nn.Linear(input_size[2], hidden)  # This acts as a time-distributed layer by defaults
        if pos_encoding:
            self.pos_encoder = PositionalEncoding(hidden)
        else:
            self.pos_encoder = None

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(
                    emb=hidden,
                    hidden=hidden,
                    heads=heads,
                    mask=True,
                    ff_hidden_mult=ff_hidden_mult,
                    dropout=dropout,
                    dropout_att=dropout_att,
                )
            )

        self.tblocks = nn.Sequential(*tblocks)
        self.logit = nn.Linear(hidden, num_classes)
        self.l1_reg = l1_reg

    def forward(self, x):
        x = self.input_embedding(x)
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        x = self.tblocks(x)
        pred = self.logit(x)

        return pred


@gin.configurable
class LocalTransformer(DLClassificationWrapper):
    def __init__(
        self,
        input_size,
        hidden,
        heads,
        ff_hidden_mult,
        depth,
        num_classes,
        *args,
        dropout=0.0,
        l1_reg=0,
        pos_encoding=True,
        local_context=1,
        dropout_att=0.0,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            hidden=hidden,
            heads=heads,
            ff_hidden_mult=ff_hidden_mult,
            depth=depth,
            num_classes=num_classes,
            *args,
            dropout=dropout,
            l1_reg=l1_reg,
            pos_encoding=pos_encoding,
            local_context=local_context,
            dropout_att=dropout_att,
            **kwargs,
        )

        hidden = hidden if hidden % 2 == 0 else hidden + 1  # Make sure hidden is even
        self.input_embedding = nn.Linear(input_size[2], hidden)  # This acts as a time-distributed layer by defaults
        if pos_encoding:
            self.pos_encoder = PositionalEncoding(hidden)
        else:
            self.pos_encoder = None

        tblocks = []
        for i in range(depth):
            tblocks.append(
                LocalBlock(
                    emb=hidden,
                    hidden=hidden,
                    heads=heads,
                    mask=True,
                    ff_hidden_mult=ff_hidden_mult,
                    local_context=local_context,
                    dropout=dropout,
                    dropout_att=dropout_att,
                )
            )

        self.tblocks = nn.Sequential(*tblocks)
        self.logit = nn.Linear(hidden, num_classes)
        self.l1_reg = l1_reg

    def forward(self, x):
        x = self.input_embedding(x)
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        x = self.tblocks(x)
        pred = self.logit(x)

        return pred


@gin.configurable
class TemporalConvNet(DLClassificationWrapper):
    """Temporal Convolutional Network. Adapted from TCN original paper https://github.com/locuslab/TCN"""

    def __init__(self, input_size, num_channels, num_classes, *args, max_seq_length=0, kernel_size=2, dropout=0.0, **kwargs):
        super().__init__(
            input_size=input_size,
            num_channels=num_channels,
            num_classes=num_classes,
            *args,
            max_seq_length=max_seq_length,
            kernel_size=kernel_size,
            dropout=dropout,
            **kwargs,
        )
        layers = []

        # We compute automatically the depth based on the desired seq_length.
        if isinstance(num_channels, Integral) and max_seq_length:
            num_channels = [num_channels] * int(np.ceil(np.log(max_seq_length / 2) / np.log(kernel_size)))
        elif isinstance(num_channels, Integral) and not max_seq_length:
            raise Exception("a maximum sequence length needs to be provided if num_channels is int")

        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = input_size[2] if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)
        self.logit = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Permute to channel first
        o = self.network(x)
        o = o.permute(0, 2, 1)  # Permute to channel last
        pred = self.logit(o)
        return pred
