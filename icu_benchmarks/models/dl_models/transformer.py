import gin
from torch import nn as nn

from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.dl_models.layers import PositionalEncoding, TransformerBlock, LocalBlock
from icu_benchmarks.models.wrappers import DLPredictionWrapper


class BaseTransformer(DLPredictionWrapper):
    _supported_run_modes = [RunMode.classification, RunMode.regression]
    """Refactored Transformer model as defined by the HiRID-Benchmark (https://github.com/ratschlab/HIRID-ICU-Benchmark)."""

    def __init__(
            self,
            block_class,
            input_size,
            hidden,
            heads,
            ff_hidden_mult,
            depth,
            num_classes,
            dropout=0.0,
            l1_reg=0,
            pos_encoding=True,
            dropout_att=0.0,
            local_context=None,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if local_context is not None and self._get_name() == "Transformer":
            raise ValueError("Local context is only supported for LocalTransformer")
        hidden = hidden if hidden % 2 == 0 else hidden + 1  # Make sure hidden is even
        self.input_embedding = nn.Linear(input_size[2], hidden)  # This acts as a time-distributed layer by defaults
        if pos_encoding:
            self.pos_encoder = PositionalEncoding(hidden)
        else:
            self.pos_encoder = None

        t_blocks = []
        for _ in range(depth):
            t_blocks.append(
                block_class(
                    emb=hidden,
                    hidden=hidden,
                    heads=heads,
                    mask=True,
                    ff_hidden_mult=ff_hidden_mult,
                    dropout=dropout,
                    dropout_att=dropout_att,
                    **({"local_context": local_context} if local_context is not None else {}),
                )
            )

        self.t_blocks = nn.Sequential(*t_blocks)
        self.logit = nn.Linear(hidden, num_classes)
        self.l1_reg = l1_reg

    def forward(self, x):
        x = self.input_embedding(x)
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        x = self.t_blocks(x)
        pred = self.logit(x)
        return pred


@gin.configurable
class Transformer(BaseTransformer):
    """Transformer model."""

    def __init__(self, *kwargs, **args):
        super().__init__(TransformerBlock, *kwargs, **args)


@gin.configurable
class LocalTransformer(BaseTransformer):
    """Transformer model with local context."""

    def __init__(self, *kwargs, **args):
        super().__init__(LocalBlock, **args, *kwargs)
