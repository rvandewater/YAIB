import gin
import numpy as np
import torch
import torch.nn as nn
from icu_benchmarks.models.layers import TransformerBlock, LocalBlock, parrallel_recomb,\
    TemporalBlock, SparseBlock, PositionalEncoding


@gin.configurable('LSTM')
class LSTMNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, num_classes):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.logit = nn.Linear(hidden_dim, num_classes)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.to(self.device) for t in (h0, c0)]

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, h = self.rnn(x, (h0, c0))
        pred = self.logit(out)
        return pred


@gin.configurable('GRU')
class GRUNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, num_classes):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.logit = nn.Linear(hidden_dim, num_classes)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(self.device)
        return h0

    def forward(self, x):
        h0 = self.init_hidden(x)
        out, hn = self.rnn(x, h0)
        pred = self.logit(out)

        return pred


@gin.configurable('Transformer')
class Transformer(nn.Module):
    def __init__(self, emb, hidden, heads, ff_hidden_mult, depth, num_classes, dropout=0.0, l1_reg=0,
                 pos_encoding=True, dropout_att=0.0):
        super().__init__()

        self.input_embedding = nn.Linear(emb, hidden)  # This acts as a time-distributed layer by defaults
        if pos_encoding:
            self.pos_encoder = PositionalEncoding(hidden)
        else:
            self.pos_encoder = None

        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(emb=hidden, hidden=hidden, heads=heads, mask=True,
                                            ff_hidden_mult=ff_hidden_mult,
                                            dropout=dropout, dropout_att=dropout_att))

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


@gin.configurable('LocalTransformer')
class LocalTransformer(nn.Module):
    def __init__(self, emb, hidden, heads, ff_hidden_mult, depth, num_classes, dropout=0.0, l1_reg=0,
                 pos_encoding=True, local_context=1, dropout_att=0.0):
        super().__init__()

        self.input_embedding = nn.Linear(emb, hidden)  # This acts as a time-distributed layer by defaults
        if pos_encoding:
            self.pos_encoder = PositionalEncoding(hidden)
        else:
            self.pos_encoder = None

        tblocks = []
        for i in range(depth):
            tblocks.append(LocalBlock(emb=hidden, hidden=hidden, heads=heads, mask=True,
                                      ff_hidden_mult=ff_hidden_mult, local_context=local_context,
                                      dropout=dropout, dropout_att=dropout_att))

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


@gin.configurable('NaiveSparseTransformer')
class NaiveSparseTransformer(nn.Module):
    def __init__(self, emb, hidden, heads, ff_hidden_mult, depth, num_classes, dropout=0.0, l1_reg=0,
                 mask_aggregation='union', local_context=3, pos_encoding=True, dropout_att=0.0):
        super().__init__()
        self.input_embedding = nn.Linear(emb, hidden)  # This acts as a time-distributed layer by defaults

        tblocks = []
        for i in range(depth):
            tblocks.append(SparseBlock(emb=hidden, hidden=hidden, heads=heads, mask=True,
                                       ff_hidden_mult=ff_hidden_mult, dropout=dropout,
                                       mask_aggregation=mask_aggregation, local_context=local_context,
                                       dropout_att=dropout_att))
        if pos_encoding:
            self.pos_encoder = PositionalEncoding(hidden)
        else:
            self.pos_encoder = None

        self.tblocks = nn.Sequential(*tblocks)
        self.logit = nn.Linear(hidden, num_classes)
        self.l1_reg = l1_reg

    def forward(self, x):
        x = self.input_embedding(x)
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        x = self.tblocks(x)
        pred = self.logit(x)
        w_input = list(self.input_embedding.parameters())[0]
        l1_norm_input = torch.torch.norm(w_input, 1)

        return pred, l1_norm_input * self.l1_reg


# From TCN original paper https://github.com/locuslab/TCN
@gin.configurable('TCN')
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, num_classes,
                 max_seq_length=0, kernel_size=2, dropout=0.0):
        super(TemporalConvNet, self).__init__()
        layers = []

        # We compute automatically the depth based on the desired seq_length.
        if isinstance(num_channels, int) and max_seq_length:
            num_channels = [num_channels] * int(np.ceil(np.log(max_seq_length / 2) / np.log(kernel_size)))
        elif isinstance(num_channels, int) and not max_seq_length:
            raise Exception('a maximum sequence length needs to be provided if num_channels is int')

        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.logit = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Permute to channel first
        o = self.network(x)
        o = o.permute(0, 2, 1)  # Permute to channel last
        pred = self.logit(o)
        return pred
