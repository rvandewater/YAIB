import gin
from numbers import Integral
import numpy as np
import torch.nn as nn
from icu_benchmarks.contants import RunMode
from icu_benchmarks.models.layers import TransformerBlock, LocalBlock, TemporalBlock, PositionalEncoding,LazyEmbedding,StaticCovariateEncoder,TFTBack
from typing import Dict
from icu_benchmarks.models.wrappers import DLPredictionWrapper
from torch import Tensor,cat,jit,FloatTensor
from pytorch_forecasting import TemporalFusionTransformer,TimeSeriesDataSet

from pytorch_forecasting.metrics import  QuantileLoss

@gin.configurable
class RNNet(DLPredictionWrapper):
    """Torch standard RNN model"""

    _supported_run_modes = [RunMode.classification, RunMode.regression]

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
class LSTMNet(DLPredictionWrapper):
    """Torch standard LSTM model."""

    _supported_run_modes = [RunMode.classification, RunMode.regression]

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
class GRUNet(DLPredictionWrapper):
    """Torch standard GRU model."""

    _supported_run_modes = [RunMode.classification, RunMode.regression]

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
class Transformer(DLPredictionWrapper):
    """Transformer model as defined by the HiRID-Benchmark (https://github.com/ratschlab/HIRID-ICU-Benchmark)."""

    _supported_run_modes = [RunMode.classification, RunMode.regression]

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
class LocalTransformer(DLPredictionWrapper):
    _supported_run_modes = [RunMode.classification, RunMode.regression]

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
class TemporalConvNet(DLPredictionWrapper):
    """Temporal Convolutional Network. Adapted from TCN original paper https://github.com/locuslab/TCN"""

    _supported_run_modes = [RunMode.classification, RunMode.regression]

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

@gin.configurable
class TFT(DLPredictionWrapper):
    """ 
    Implementation of https://arxiv.org/abs/1912.09363 
    """


    _supported_run_modes = [RunMode.classification, RunMode.regression]
    def __init__(self,num_classes, encoder_length,hidden,dropout,
                 n_heads,dropout_att,example_length,*args,quantiles=[0.1, 0.5, 0.9],static_categorical_inp_size=[2],temporal_known_categorical_inp_size=[],
    temporal_observed_categorical_inp_size=[48],static_continuous_inp_size=3,temporal_known_continuous_inp_size=0,
    temporal_observed_continuous_inp_size=48,temporal_target_size=1,**kwargs):
        #derived variables
        num_static_vars=len(static_categorical_inp_size)+static_continuous_inp_size
        num_future_vars=len(temporal_known_categorical_inp_size)+temporal_known_continuous_inp_size
        num_historic_vars=sum([num_future_vars,
                                      temporal_observed_continuous_inp_size,
                                      temporal_target_size,
                                      len(temporal_observed_categorical_inp_size),
                                      ])
        
        super().__init__(num_classes=num_classes, encoder_length=encoder_length,hidden=hidden,
                 n_heads=n_heads,dropout_att=dropout_att,example_length=example_length,quantiles=quantiles,
                 num_static_vars=num_static_vars,num_future_vars=num_future_vars,num_historic_vars=num_historic_vars,*args,static_categorical_inp_size=1,temporal_known_categorical_inp_size=0,
                temporal_observed_categorical_inp_size=48,static_continuous_inp_size=3,temporal_known_continuous_inp_size=0,
                temporal_observed_continuous_inp_size=48,temporal_target_size=1,**kwargs)

        
        
        self.encoder_length = encoder_length #this determines from how distant past we want to use data from

        self.embedding = LazyEmbedding(static_categorical_inp_size,temporal_known_categorical_inp_size,
                temporal_observed_categorical_inp_size,static_continuous_inp_size,temporal_known_continuous_inp_size,
                temporal_observed_continuous_inp_size,temporal_target_size,hidden)
        
        self.static_encoder = StaticCovariateEncoder(num_static_vars,hidden,dropout)
        self.TFTpart2 = TFTBack(encoder_length,num_historic_vars,hidden,dropout,num_future_vars,
                n_heads,dropout_att,example_length,quantiles)
        self.logit = nn.Linear(len(quantiles), num_classes)

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        
       
        s_inp, t_known_inp, t_observed_inp, t_observed_tgt = self.embedding(x)
        # Static context
        cs, ce, ch, cc = self.static_encoder(s_inp)
        ch, cc = ch.unsqueeze(0), cc.unsqueeze(0) #lstm initial states
        # Temporal input
        
        _historical_inputs = []
        
        # Check for t_observed_inp
        if t_observed_inp is not None:
            _historical_inputs.append(t_observed_inp[:, :self.encoder_length, :])

        # Check for t_known_inp
        if t_known_inp is not None:
            _historical_inputs.append(t_known_inp[:, :self.encoder_length, :])
        # Check for t_observed_tgt
        if t_observed_tgt is not None:
            _historical_inputs.append(t_observed_tgt[:, :self.encoder_length, :])
        historical_inputs = cat(_historical_inputs, dim=-2)
        future_inputs= Tensor()
        if t_known_inp is not None:
            future_inputs = t_known_inp[:, self.encoder_length:]
        
        
        o=self.TFTpart2(historical_inputs, cs, ch, cc, ce, future_inputs.to(historical_inputs.device))
        pred = self.logit(o)
        return pred

@gin.configurable
class TFTpytorch(TemporalFusionTransformer,DLPredictionWrapper):
    """
    Implementation of https://arxiv.org/abs/1912.09363 
    """
    supported_run_modes = [RunMode.classification, RunMode.regression]
    def __init__(self,dataset,hidden,dropout,
                 n_heads,dropout_att,lr,optimizer,num_classes,*args,**kwargs):
        
        DLPredictionWrapper.__init__(self,lr=lr,optimizer=optimizer,*args,**kwargs)        
        
        TemporalFusionTransformer.__init__(self)

        self.model=TemporalFusionTransformer.from_dataset(dataset=dataset,hidden_size=hidden,dropout=dropout,
                 attention_head_size=n_heads,learning_rate=lr,optimizer=optimizer,loss=QuantileLoss(),hidden_continuous_size=hidden)
        self.hparams.update(self.model.hparams)

        #for key in model.hparams.keys():
        # self.hparams[key]=model.hparams[key]
       # del self.hparams["input_shape"]
       # del self.hparams["input_size"]
        #print(self.hparams)
       # print(self.m)
        self.logit = nn.Linear(7, num_classes)
    def set_weight(self, weight, dataset):
        """
        Set the weight for the loss function
        """
        if isinstance(weight, list):
            weight = FloatTensor(weight)
        elif weight == "balanced":
            weight = FloatTensor(dataset.get_balance())
        self.loss_weights = weight
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        log, out = self.step(x, y, batch_idx)
        print("finally here")
        log.update(self.create_log(x, y, out, batch_idx))
        self.testing_step_outputs.append(log)
        self.log(f"test/loss", log, on_step=False, on_epoch=True, sync_dist=True)

        return log


"""
@gin.configurable
class TFTpytorch(TemporalFusionTransformer,DLPredictionWrapper):
    
    Implementation of https://arxiv.org/abs/1912.09363 
    
    supported_run_modes = [RunMode.classification, RunMode.regression]
    def __init__(self,dataset,hidden,dropout,
                 n_heads,dropout_att,lr,optimizer,num_classes,*args,**kwargs):
        
        DLPredictionWrapper.__init__(self,lr=lr,optimizer=optimizer,*args,**kwargs)        
        self.model=TemporalFusionTransformer.from_dataset(dataset)
        
    
        
    def set_weight(self, weight, dataset):
        
        Set the weight for the loss function
        
        if isinstance(weight, list):
            weight = FloatTensor(weight)
        elif weight == "balanced":
            weight = FloatTensor(dataset.get_balance())
        self.loss_weights = weight
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        log, out = self.model.step(x, y, batch_idx)
        log.update(self.model.create_log(x, y, out, batch_idx))
        self.testing_step_outputs.append(log)
        self.model.log(f"test/loss", log, on_step=False, on_epoch=True, sync_dist=True)

        return log
    def forward(self, x: Dict[str,Tensor]) -> Dict[str, Tensor]:
        return self.model.forward(x)
    
"""