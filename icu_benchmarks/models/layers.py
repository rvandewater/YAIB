import gin
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch import Tensor
from torch.nn.parameter import UninitializedParameter
from typing import Dict, Tuple, Optional
from torch.nn import LayerNorm


@gin.configurable("masking")
def parallel_recomb(q_t, kv_t, att_type="all", local_context=3, bin_size=None):
    """Return mask of attention matrix (ts_q, ts_kv)"""
    with torch.no_grad():
        q_t[q_t == -1.0] = float("inf")  # We want padded to attend to everyone to avoid any nan.
        kv_t[kv_t == -1.0] = float("inf")  # We want no one to attend the padded values

        if bin_size is not None:  # General case where we use unaligned timesteps.
            q_t = q_t / bin_size
            starts_q = q_t[:, 0:1].clone()  # Needed because of Memory allocation issue
            q_t -= starts_q
            kv_t = kv_t / bin_size
            starts_kv = kv_t[:, 0:1].clone()  # Needed because of Memory allocation issue
            kv_t -= starts_kv

        bs, ts_q = q_t.size()
        _, ts_kv = kv_t.size()
        q_t_rep = q_t.view(bs, ts_q, 1).repeat(1, 1, ts_kv)
        kv_t_rep = kv_t.view(bs, 1, ts_kv).repeat(1, ts_q, 1)
        diff_mask = (q_t_rep - kv_t_rep).to(q_t_rep.device)
        if att_type == "all":
            return (diff_mask >= 0).float()
        if att_type == "local":
            return ((diff_mask >= 0) * (diff_mask <= local_context) + (diff_mask == float("inf"))).float()
        if att_type == "strided":
            return ((diff_mask >= 0) * (torch.floor(diff_mask) % local_context == 0) + (diff_mask == float("inf"))).float()


class PositionalEncoding(nn.Module):
    """Positional Encoding, mostly from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""

    def __init__(self, emb, max_len=3000):
        super().__init__()
        pe = torch.zeros(max_len, emb)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb, 2).float() * (-math.log(10000.0) / emb))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        bs, n, emb = x.size()
        return x + self.pe[:, :n, :]


class SelfAttention(nn.Module):
    """Multi Head Attention block from Attention is All You Need (https://arxiv.org/abs/1706.03762). Input has shape
    (batch_size, n_timestamps, emb)."""

    def __init__(
        self, emb, hidden, heads=8, mask=True, att_type="all", local_context=None, mask_aggregation="union", dropout_att=0.0
    ):
        """Initialize the Multi Head Block.
        Args:
            emb: Dimension of the input vector.
            hidden: Hidden dimension of query, key, value matrices.
            heads: Number of heads.
            mask: Whether to mask the attention matrix."""
        super().__init__()

        self.emb = emb
        self.heads = heads
        self.hidden = hidden
        self.mask = mask
        self.drop_att = nn.Dropout(dropout_att)

        # Sparse transformer specific params
        self.att_type = att_type
        self.local_context = local_context
        self.mask_aggregation = mask_aggregation

        # Query, keys and value matrices
        self.w_keys = nn.Linear(emb, hidden * heads, bias=False)
        self.w_queries = nn.Linear(emb, hidden * heads, bias=False)
        self.w_values = nn.Linear(emb, hidden * heads, bias=False)

        # Output linear function
        self.unifyheads = nn.Linear(heads * hidden, emb)

    def forward(self, x):
        """
        Args:
            x: Input data tensor with shape (batch_size, n_timestamps, emb)
        Returns:
            Self attention tensor with shape (batch_size, n_timestamps, emb)
        """
        # bs - batch_size, n - vectors number, emb - embedding dimensionality
        bs, n, emb = x.size()
        h = self.heads
        hidden = self.hidden

        keys = self.w_keys(x).view(bs, n, h, hidden)
        queries = self.w_queries(x).view(bs, n, h, hidden)
        values = self.w_values(x).view(bs, n, h, hidden)

        # fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(bs * h, n, hidden)
        queries = queries.transpose(1, 2).contiguous().view(bs * h, n, hidden)
        values = values.transpose(1, 2).contiguous().view(bs * h, n, hidden)

        # dive on the square oot of dimensionality
        queries = queries / (hidden ** (1 / 2))
        keys = keys / (hidden ** (1 / 2))

        # dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        if self.mask:  # We deal with different masking and recombination types here
            if isinstance(self.att_type, list):  # Local and sparse attention
                if self.mask_aggregation == "union":
                    mask_tensor = 0
                    for att_type in self.att_type:
                        mask_tensor += parallel_recomb(
                            torch.arange(1, n + 1, dtype=torch.float, device=dot.device).reshape(1, -1),
                            torch.arange(1, n + 1, dtype=torch.float, device=dot.device).reshape(1, -1),
                            att_type,
                            self.local_context,
                        )[0]
                    mask_tensor = torch.clamp(mask_tensor, 0, 1)
                    dot = torch.where(mask_tensor.bool(), dot, torch.tensor(float("-inf")).to(dot.device)).view(bs * h, n, n)

                elif self.mask_aggregation == "split":
                    dot_list = list(torch.split(dot, dot.shape[0] // len(self.att_type), dim=0))
                    for i, att_type in enumerate(self.att_type):
                        mask_tensor = parallel_recomb(
                            torch.arange(1, n + 1, dtype=torch.float, device=dot.device).reshape(1, -1),
                            torch.arange(1, n + 1, dtype=torch.float, device=dot.device).reshape(1, -1),
                            att_type,
                            self.local_context,
                        )[0]

                        dot_list[i] = torch.where(
                            mask_tensor.bool(), dot_list[i], torch.tensor(float("-inf")).to(dot.device)
                        ).view(*dot_list[i].shape)
                    dot = torch.cat(dot_list, dim=0)
            else:  # Full causal masking
                mask_tensor = parallel_recomb(
                    torch.arange(1, n + 1, dtype=torch.float, device=dot.device).reshape(1, -1),
                    torch.arange(1, n + 1, dtype=torch.float, device=dot.device).reshape(1, -1),
                    self.att_type,
                    self.local_context,
                )[0]
                dot = torch.where(mask_tensor.bool(), dot, torch.tensor(float("-inf")).to(dot.device)).view(bs * h, n, n)

        # dot now has row-wise self-attention probabilities
        dot = F.softmax(dot, dim=2)

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(bs, h, n, hidden)

        # apply the dropout
        out = self.drop_att(out)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(bs, n, h * hidden)
        return self.unifyheads(out)


class SparseBlock(nn.Module):
    def __init__(
        self,
        emb,
        hidden,
        heads,
        ff_hidden_mult,
        dropout=0.0,
        mask=True,
        mask_aggregation="union",
        local_context=3,
        dropout_att=0.0,
    ):
        super().__init__()

        self.attention = SelfAttention(
            emb,
            hidden,
            heads=heads,
            mask=mask,
            mask_aggregation=mask_aggregation,
            local_context=local_context,
            att_type=["strided", "local"],
            dropout_att=dropout_att,
        )
        self.mask = mask
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(nn.Linear(emb, ff_hidden_mult * emb), nn.ReLU(), nn.Linear(ff_hidden_mult * emb, emb))

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention.forward(x)
        x = self.norm1(attended + x)
        x = self.drop(x)
        fedforward = self.ff(x)
        x = self.norm2(fedforward + x)

        x = self.drop(x)

        return x


class LocalBlock(nn.Module):
    def __init__(self, emb, hidden, heads, ff_hidden_mult, dropout=0.0, mask=True, local_context=3, dropout_att=0.0):
        super().__init__()

        self.attention = SelfAttention(
            emb,
            hidden,
            heads=heads,
            mask=mask,
            mask_aggregation=None,
            local_context=local_context,
            att_type="local",
            dropout_att=dropout_att,
        )
        self.mask = mask
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(nn.Linear(emb, ff_hidden_mult * emb), nn.ReLU(), nn.Linear(ff_hidden_mult * emb, emb))

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention.forward(x)
        x = self.norm1(attended + x)
        x = self.drop(x)
        fedforward = self.ff(x)
        x = self.norm2(fedforward + x)

        x = self.drop(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, emb, hidden, heads, ff_hidden_mult, dropout=0.0, mask=True, dropout_att=0.0):
        super().__init__()

        self.attention = SelfAttention(emb, hidden, heads=heads, mask=mask, dropout_att=dropout_att)
        self.mask = mask
        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(nn.Linear(emb, ff_hidden_mult * emb), nn.ReLU(), nn.Linear(ff_hidden_mult * emb, emb))

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention.forward(x)
        x = self.norm1(attended + x)
        x = self.drop(x)
        fedforward = self.ff(x)
        x = self.norm2(fedforward + x)

        x = self.drop(x)

        return x


class Chomp1d(nn.Module):
    """From TCN original paper https://github.com/locuslab/TCN"""

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation), dim=None
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation), dim=None
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class MaybeLayerNorm(nn.Module):
    """
    Implements layer normalization or identity function depending on output_size
    """

    def __init__(self, output_size, hidden, eps):
        super().__init__()
        if output_size and output_size == 1:
            self.ln = nn.Identity()
        else:
            self.ln = LayerNorm(output_size if output_size else hidden, eps=eps)

    def forward(self, x):
        return self.ln(x)


class GLU(nn.Module):
    """
    Gated Linear Unit consists of a linear layer followed by a GLU where input is split in half along dim to form a and b
    GLU(a,b)=a ⊗ σ(b)where σ is signmoid activation and ⊗ is element-wise product
    """

    def __init__(self, hidden, output_size):
        super().__init__()
        self.lin = nn.Linear(hidden, output_size * 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin(x)
        x = F.glu(x)
        return x


class GRN(nn.Module):
    """
    Gated Residual Network consists of a maybe normalization layer -->linear --> ELU -->linear-->GLU
     in addition to a residual connection
    """

    def __init__(
        self,
        input_size,
        hidden,
        output_size=None,
        context_hidden=None,
        dropout=0.0,
    ):
        super().__init__()

        self.layer_norm = MaybeLayerNorm(output_size, hidden, eps=1e-3)

        self.lin_a = nn.Linear(input_size, hidden)

        if context_hidden is not None:
            self.lin_c = nn.Linear(context_hidden, hidden, bias=False)
        else:
            self.lin_c = nn.Identity()
        self.lin_i = nn.Linear(hidden, hidden)
        self.glu = GLU(hidden, output_size if output_size else hidden)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(input_size, output_size) if output_size else None

    def forward(self, a: Tensor, c: Optional[Tensor] = None):
        x = self.lin_a(a)

        if c is not None:
            x = x + self.lin_c(c).unsqueeze(1)
        x = F.elu(x)
        x = self.lin_i(x)
        x = self.dropout(x)
        x = self.glu(x)
        y = a if self.out_proj is None else self.out_proj(a)
        x = x + y

        return self.layer_norm(x)


# @torch.jit.script #Currently broken with autocast
def fused_pointwise_linear_v1(x, a, b):
    out = torch.mul(x.unsqueeze(-1), a)
    out = out + b
    return out


# @torch.jit.script
def fused_pointwise_linear_v2(x, a, b):
    out = x.unsqueeze(3) * a
    out = out + b
    return out


class TFTEmbedding(nn.Module):
    def __init__(
        self,
        static_categorical_inp_size,
        temporal_known_categorical_inp_size,
        temporal_observed_categorical_inp_size,
        static_continuous_inp_size,
        temporal_known_continuous_inp_size,
        temporal_observed_continuous_inp_size,
        temporal_target_size,
        hidden,
        initialize_cont_params=False,
    ):
        # initialize_cont_params=False prevents form initializing parameters inside this class
        # so they can be lazily initialized in LazyEmbedding module
        super().__init__()
        # these are basically number of varaibales that falls under each category
        self.s_cat_inp_size = static_categorical_inp_size
        self.t_cat_k_inp_size = temporal_known_categorical_inp_size
        self.t_cat_o_inp_size = temporal_observed_categorical_inp_size
        self.s_cont_inp_size = static_continuous_inp_size
        self.t_cont_k_inp_size = temporal_known_continuous_inp_size
        self.t_cont_o_inp_size = temporal_observed_continuous_inp_size
        self.t_tgt_size = temporal_target_size

        self.hidden = hidden

        # There are 7 types of input:
        # 1. Static categorical
        # 2. Static continuous
        # 3. Temporal known a priori categorical
        # 4. Temporal known a priori continuous
        # 5. Temporal observed categorical
        # 6. Temporal observed continuous
        # 7. Temporal observed targets (time series obseved so far)

        self.s_cat_embed = (
            nn.ModuleList([nn.Embedding(n, self.hidden) for n in self.s_cat_inp_size]) if self.s_cat_inp_size else None
        )
        self.t_cat_k_embed = (
            nn.ModuleList([nn.Embedding(n, self.hidden) for n in self.t_cat_k_inp_size]) if self.t_cat_k_inp_size else None
        )
        self.t_cat_o_embed = (
            nn.ModuleList([nn.Embedding(n, self.hidden) for n in self.t_cat_o_inp_size]) if self.t_cat_o_inp_size else None
        )

        if initialize_cont_params:
            self.s_cont_embedding_vectors = (
                nn.Parameter(torch.Tensor(self.s_cont_inp_size, self.hidden)) if self.s_cont_inp_size else None
            )
            self.t_cont_k_embedding_vectors = (
                nn.Parameter(torch.Tensor(self.t_cont_k_inp_size, self.hidden)) if self.t_cont_k_inp_size else None
            )
            self.t_cont_o_embedding_vectors = (
                nn.Parameter(torch.Tensor(self.t_cont_o_inp_size, self.hidden)) if self.t_cont_o_inp_size else None
            )
            self.t_tgt_embedding_vectors = nn.Parameter(torch.Tensor(self.t_tgt_size, self.hidden))
            self.s_cont_embedding_bias = (
                nn.Parameter(torch.zeros(self.s_cont_inp_size, self.hidden)) if self.s_cont_inp_size else None
            )
            self.t_cont_k_embedding_bias = (
                nn.Parameter(torch.zeros(self.t_cont_k_inp_size, self.hidden)) if self.t_cont_k_inp_size else None
            )
            self.t_cont_o_embedding_bias = (
                nn.Parameter(torch.zeros(self.t_cont_o_inp_size, self.hidden)) if self.t_cont_o_inp_size else None
            )
            self.t_tgt_embedding_bias = nn.Parameter(torch.zeros(self.t_tgt_size, self.hidden))

            self.reset_parameters()

    def reset_parameters(self):
        """'
        embeddings are initilitized using xavier's method and biases are initlitized with zeros
        """
        if self.s_cont_embedding_vectors is not None:
            torch.nn.init.xavier_normal_(self.s_cont_embedding_vectors)
            torch.nn.init.zeros_(self.s_cont_embedding_bias)
        if self.t_cont_k_embedding_vectors is not None:
            torch.nn.init.xavier_normal_(self.t_cont_k_embedding_vectors)
            torch.nn.init.zeros_(self.t_cont_k_embedding_bias)
        if self.t_cont_o_embedding_vectors is not None:
            torch.nn.init.xavier_normal_(self.t_cont_o_embedding_vectors)
            torch.nn.init.zeros_(self.t_cont_o_embedding_bias)
        if self.t_tgt_embedding_vectors is not None:
            torch.nn.init.xavier_normal_(self.t_tgt_embedding_vectors)
            torch.nn.init.zeros_(self.t_tgt_embedding_bias)
        if self.s_cat_embed is not None:
            for module in self.s_cat_embed:
                module.reset_parameters()
        if self.t_cat_k_embed is not None:
            for module in self.t_cat_k_embed:
                module.reset_parameters()
        if self.t_cat_o_embed is not None:
            for module in self.t_cat_o_embed:
                module.reset_parameters()

    def _apply_embedding(
        self,
        cat: Optional[Tensor],
        cont: Optional[Tensor],
        cat_emb: Optional[nn.ModuleList],
        cont_emb: Tensor,
        cont_bias: Tensor,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        e_cat = (
            torch.stack([embed(cat[..., i].int()) for i, embed in enumerate(cat_emb)], dim=-2)
            if (cat is not None) and (cat.size()[1] > 0)
            else None
        )
        if (cont is not None) and (cont.size()[1] > 0):
            # the line below is equivalent to following einsums
            # e_cont = torch.einsum('btf,fh->bthf', cont, cont_emb)
            # e_cont = torch.einsum('bf,fh->bhf', cont, cont_emb)
            e_cont = torch.mul(cont.unsqueeze(-1), cont_emb)
            e_cont = e_cont + cont_bias
        # e_cont = fused_pointwise_linear_v1(cont, cont_emb, cont_bias)
        else:
            e_cont = None

        if e_cat is not None and e_cont is not None:
            return torch.cat([e_cat, e_cont], dim=-2)
        elif e_cat is not None:
            return e_cat
        elif e_cont is not None:
            return e_cont
        else:
            return None

    def forward(self, x: Dict[str, Tensor]):
        # temporal/static categorical/continuous known/observed input
        s_cat_inp = x.get("s_cat", None)
        s_cont_inp = x.get("s_cont", None)
        t_cat_k_inp = x.get("k_cat", None)
        t_cont_k_inp = x.get("k_cont", None)
        t_cat_o_inp = x.get("o_cat", None)
        t_cont_o_inp = x.get("o_cont", None)
        t_tgt_obs = x["target"]  # Has to be present
        # Static inputs are expected to be equal for all timesteps
        # For memory efficiency there is no assert statement

        s_cat_inp = s_cat_inp[:, 0, :] if s_cat_inp is not None else None
        s_cont_inp = s_cont_inp[:, 0, :] if s_cont_inp is not None else None

        s_inp = self._apply_embedding(
            s_cat_inp, s_cont_inp, self.s_cat_embed, self.s_cont_embedding_vectors, self.s_cont_embedding_bias
        )

        t_known_inp = self._apply_embedding(
            t_cat_k_inp, t_cont_k_inp, self.t_cat_k_embed, self.t_cont_k_embedding_vectors, self.t_cont_k_embedding_bias
        )

        t_observed_inp = self._apply_embedding(
            t_cat_o_inp, t_cont_o_inp, self.t_cat_o_embed, self.t_cont_o_embedding_vectors, self.t_cont_o_embedding_bias
        )

        # Temporal observed targets
        t_observed_tgt = torch.matmul(t_tgt_obs.unsqueeze(3).unsqueeze(4), self.t_tgt_embedding_vectors.unsqueeze(1)).squeeze(
            3
        )
        t_observed_tgt = t_observed_tgt + self.t_tgt_embedding_bias

        return s_inp, t_known_inp, t_observed_inp, t_observed_tgt


class LazyEmbedding(nn.modules.lazy.LazyModuleMixin, TFTEmbedding):
    cls_to_become = TFTEmbedding

    def __init__(
        self,
        static_categorical_inp_size,
        temporal_known_categorical_inp_size,
        temporal_observed_categorical_inp_size,
        static_continuous_inp_size,
        temporal_known_continuous_inp_size,
        temporal_observed_continuous_inp_size,
        temporal_target_size,
        hidden,
    ):
        super().__init__(
            static_categorical_inp_size,
            temporal_known_categorical_inp_size,
            temporal_observed_categorical_inp_size,
            static_continuous_inp_size,
            temporal_known_continuous_inp_size,
            temporal_observed_continuous_inp_size,
            temporal_target_size,
            hidden,
            initialize_cont_params=False,
        )
        if static_continuous_inp_size:
            self.s_cont_embedding_vectors = UninitializedParameter()
            self.s_cont_embedding_bias = UninitializedParameter()
        else:
            self.s_cont_embedding_vectors = None
            self.s_cont_embedding_bias = None
        if temporal_known_continuous_inp_size:
            self.t_cont_k_embedding_vectors = UninitializedParameter()
            self.t_cont_k_embedding_bias = UninitializedParameter()
        else:
            self.t_cont_k_embedding_vectors = None
            self.t_cont_k_embedding_bias = None

        if temporal_observed_continuous_inp_size:
            self.t_cont_o_embedding_vectors = UninitializedParameter()
            self.t_cont_o_embedding_bias = UninitializedParameter()
        else:
            self.t_cont_o_embedding_vectors = None
            self.t_cont_o_embedding_bias = None
        self.t_tgt_embedding_vectors = UninitializedParameter()
        self.t_tgt_embedding_bias = UninitializedParameter()

    def initialize_parameters(self, x):
        if self.has_uninitialized_params():
            s_cont_inp = x.get("s_cont", None)
            t_cont_k_inp = x.get("k_cont", None)
            t_cont_o_inp = x.get("o_cont", None)
            t_tgt_obs = x["target"]  # Has to be present
            if (s_cont_inp is not None) and (s_cont_inp.size()[1] > 0):
                self.s_cont_embedding_vectors.materialize((s_cont_inp.shape[-1], self.hidden))
                self.s_cont_embedding_bias.materialize((s_cont_inp.shape[-1], self.hidden))

            if (t_cont_k_inp is not None) and (t_cont_k_inp.size()[1] > 0):
                self.t_cont_k_embedding_vectors.materialize((t_cont_k_inp.shape[-1], self.hidden))
                self.t_cont_k_embedding_bias.materialize((t_cont_k_inp.shape[-1], self.hidden))

            if (t_cont_o_inp) is not None and (t_cont_o_inp.size()[1] > 0):
                self.t_cont_o_embedding_vectors.materialize((t_cont_o_inp.shape[-1], self.hidden))
                self.t_cont_o_embedding_bias.materialize((t_cont_o_inp.shape[-1], self.hidden))

            self.t_tgt_embedding_vectors.materialize((t_tgt_obs.shape[-1], self.hidden))
            self.t_tgt_embedding_bias.materialize((t_tgt_obs.shape[-1], self.hidden))

            self.reset_parameters()


class VariableSelectionNetwork(nn.Module):
    """
    Learns to select important netowrks consists of GRNs with one GRN for variable weights
    and the others for input embedding
    """

    def __init__(self, hidden, dropout, num_inputs):
        super().__init__()
        self.hidden = hidden
        self.joint_grn = GRN(hidden * num_inputs, hidden, output_size=num_inputs, context_hidden=hidden)
        self.var_grns = nn.ModuleList([GRN(hidden, hidden, dropout=dropout) for _ in range(num_inputs)])

    def forward(self, x: Tensor, context: Optional[Tensor] = None):
        if x.numel() == 0:  # Check if x is an empty tensor
            batch_size = context.size(0) if context is not None else 1
            variable_ctx = torch.zeros(batch_size, 1, self.hidden, device=x.device)
            sparse_weights = torch.ones(batch_size, 1, self.hidden, device=x.device)
            return variable_ctx, sparse_weights

        Xi = torch.flatten(x, start_dim=-2)
        grn_outputs = self.joint_grn(Xi, c=context)
        sparse_weights = F.softmax(grn_outputs, dim=-1)
        transformed_embed_list = [m(x[..., i, :]) for i, m in enumerate(self.var_grns)]
        transformed_embed = torch.stack(transformed_embed_list, dim=-1)
        variable_ctx = torch.matmul(transformed_embed, sparse_weights.unsqueeze(-1)).squeeze(-1)

        return variable_ctx, sparse_weights


class StaticCovariateEncoder(nn.Module):
    """
    Network to produce 4 contexts vectors to enrich static variables
    Vriable selection Network --> GRNs
    """

    def __init__(self, num_static_vars, hidden, dropout):
        super().__init__()
        self.vsn = VariableSelectionNetwork(hidden, dropout, num_static_vars)
        self.context_grns = nn.ModuleList([GRN(hidden, hidden, dropout=dropout) for _ in range(4)])

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        variable_ctx, sparse_weights = self.vsn(x)

        # Context vectors:
        # variable selection context
        # enrichment context
        # state_c context
        # state_h context
        cs, ce, ch, cc = [m(variable_ctx) for m in self.context_grns]

        return cs, ce, ch, cc


class InterpretableMultiHeadAttention(nn.Module):
    """
    Multi-head attention different as it outputs the attention_probability and it combines the attention weights instead of
    concating them different from the one implemented already in YAIB
    """

    def __init__(self, n_head, hidden, dropout_att, dropout, example_length):
        super().__init__()
        self.n_head = n_head
        assert hidden % n_head == 0
        self.d_head = hidden // n_head
        self.qkv_linears = nn.Linear(hidden, (2 * n_head + 1) * self.d_head, bias=False)
        self.out_proj = nn.Linear(self.d_head, hidden, bias=False)
        self.dropout_att = nn.Dropout(dropout_att)
        self.out_dropout = nn.Dropout(dropout)
        self.scale = self.d_head**-0.5
        self.register_buffer("_mask", torch.triu(torch.full((example_length, example_length), float("-inf")), 1).unsqueeze(0))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        bs, t, h_size = x.shape
        qkv = self.qkv_linears(x)
        q, k, v = qkv.split((self.n_head * self.d_head, self.n_head * self.d_head, self.d_head), dim=-1)
        q = q.view(bs, t, self.n_head, self.d_head)
        k = k.view(bs, t, self.n_head, self.d_head)
        v = v.view(bs, t, self.d_head)

        # attn_score = torch.einsum('bind,bjnd->bnij', q, k)
        attn_score = torch.matmul(q.permute((0, 2, 1, 3)), k.permute((0, 2, 3, 1)))
        attn_score.mul_(self.scale)
        attn_score = attn_score + self._mask

        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.dropout_att(attn_prob)

        # attn_vec = torch.einsum('bnij,bjd->bnid', attn_prob, v)
        attn_vec = torch.matmul(attn_prob, v.unsqueeze(1))
        m_attn_vec = torch.mean(attn_vec, dim=1)
        out = self.out_proj(m_attn_vec)
        out = self.out_dropout(out)

        return out, attn_prob


class TFTBack(nn.Module):
    """
    Big part of TFT architecture consists of static enrichment followed by mutli-head self-attention then
    position wise feed forward followed by a gate and a dense layer
    GRNs-->multi-head attention-->GRNs-->GLU-->Linear-->output
    """

    def __init__(
        self,
        encoder_length,
        num_historic_vars,
        hidden,
        dropout,
        num_future_vars,
        n_head,
        dropout_att,
        example_length,
        quantiles,
    ):
        super().__init__()

        self.encoder_length = encoder_length
        self.history_vsn = VariableSelectionNetwork(hidden, dropout, num_historic_vars)
        self.history_encoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.future_vsn = VariableSelectionNetwork(hidden, dropout, num_future_vars)
        self.future_encoder = nn.LSTM(hidden, hidden, batch_first=True)

        self.input_gate = GLU(hidden, hidden)
        self.input_gate_ln = LayerNorm(hidden, eps=1e-3)

        self.enrichment_grn = GRN(hidden, hidden, context_hidden=hidden, dropout=dropout)
        self.attention = InterpretableMultiHeadAttention(n_head, hidden, dropout_att, dropout, example_length)
        self.attention_gate = GLU(hidden, hidden)
        self.attention_ln = LayerNorm(hidden, eps=1e-3)

        self.positionwise_grn = GRN(hidden, hidden, dropout=dropout)

        self.decoder_gate = GLU(hidden, hidden)
        self.decoder_ln = LayerNorm(hidden, eps=1e-3)

        self.quantile_proj = nn.Linear(hidden, len(quantiles))

    def forward(self, historical_inputs, cs, ch, cc, ce, future_inputs):
        historical_features, _ = self.history_vsn(historical_inputs, cs)
        history, state = self.history_encoder(historical_features, (ch, cc))
        future_features, _ = self.future_vsn(future_inputs, cs)

        future, _ = self.future_encoder(future_features, state)

        # skip connection
        input_embedding = torch.cat([historical_features, future_features], dim=1)
        temporal_features = torch.cat([history, future], dim=1)
        temporal_features = self.input_gate(temporal_features)
        temporal_features = temporal_features + input_embedding
        temporal_features = self.input_gate_ln(temporal_features)

        # Static enrichment
        enriched = self.enrichment_grn(temporal_features, c=ce)

        # Temporal self attention
        x, attn_prob = self.attention(enriched)

        # Don't compute historical quantiles
        x = x[:, self.encoder_length:, :]
        temporal_features = temporal_features[:, self.encoder_length:, :]
        enriched = enriched[:, self.encoder_length:, :]

        x = self.attention_gate(x)
        x = x + enriched
        x = self.attention_ln(x)

        # Position-wise feed-forward
        x = self.positionwise_grn(x)

        # Final skip connection
        x = self.decoder_gate(x)
        x = x + temporal_features
        x = self.decoder_ln(x)

        out = self.quantile_proj(x)

        return out
