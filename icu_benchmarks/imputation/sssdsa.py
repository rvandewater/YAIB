# Source: https://github.com/AI4HealthUOL/SSSD/blob/main/src/imputers/SSSDSAImputer.py
import numpy as np
import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from icu_benchmarks.imputation.layers.s4layer import S4, LinearActivation
from icu_benchmarks.models.wrappers import ImputationWrapper


@gin.configurable("SSSDSA")
class SSSDSA(ImputationWrapper):
    """ "SaShiMi model backbone. Adapted from https://github.com/AI4HealthUOL/SSSD/blob/main/src/imputers/SSSDSAImputer.py"""

    def __init__(
        self,
        d_model,
        n_layers,
        pool,
        expand,
        ff,
        glu,
        unet,
        dropout,
        in_channels,
        out_channels,
        diffusion_step_embed_dim_in,
        diffusion_step_embed_dim_mid,
        diffusion_step_embed_dim_out,
        label_embed_dim,
        label_embed_classes,
        bidirectional,
        s4_lmax,
        s4_d_state,
        s4_dropout,
        s4_bidirectional,
        diffusion_time_steps,
        beta_0,
        beta_T,
        input_size,
        *args,
        **kwargs,
    ):
        """
        SaShiMi model backbone.

        Args:
            d_model: dimension of the model. We generally use 64 for all our experiments.
            n_layers: number of (Residual (S4) --> Residual (FF)) blocks at each pooling level.
                We use 8 layers for our experiments, although we found that increasing layers even further generally
                improves performance at the expense of training / inference speed.
            pool: pooling factor at each level. Pooling shrinks the sequence length at lower levels.
                We experimented with a pooling factor of 4 with 1 to 4 tiers of pooling and found 2 tiers to be best.
                It's possible that a different combination of pooling factors and number of tiers may perform better.
            expand: expansion factor when pooling. Features are expanded (i.e. the model becomes wider) at lower levels of the
                architecture.We generally found 2 to perform best (among 2, 4).
            ff: expansion factor for the FF inverted bottleneck. We generally found 2 to perform best (among 2, 4).
            bidirectional: use bidirectional S4 layers. Bidirectional layers are suitable for use with non-causal models
                such as diffusion models like DiffWave.
            glu: use gated linear unit in the S4 layers. Adds parameters and generally improves performance.
            unet: use a unet-like architecture, adding (Residual (S4) --> Residual (FF)) layers before downpooling.
                All else fixed, this slows down inference (and slightly slows training), but generally improves performance.
                We use this variant when dropping in SaShiMi into diffusion models, and this should generally be preferred
                for non-autoregressive models.
            dropout: dropout rate. Default to 0.0, since we haven't found settings where SaShiMi overfits.
        """
        super().__init__(
            d_model=d_model,
            n_layers=n_layers,
            pool=pool,
            expand=expand,
            ff=ff,
            glu=glu,
            unet=unet,
            dropout=dropout,
            in_channels=in_channels,
            out_channels=out_channels,
            diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
            diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
            diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
            label_embed_dim=label_embed_dim,
            label_embed_classes=label_embed_classes,
            bidirectional=bidirectional,
            s4_lmax=s4_lmax,
            s4_d_state=s4_d_state,
            s4_dropout=s4_dropout,
            s4_bidirectional=s4_bidirectional,
            diffusion_time_steps=diffusion_time_steps,
            beta_0=beta_0,
            beta_T=beta_T,
            input_size=input_size,
            *args,
            **kwargs,
        )
        self.diffusion_parameters = calc_diffusion_hyperparams(diffusion_time_steps, beta_0, beta_T)
        self.d_model = H = d_model
        self.unet = unet

        def s4_block(dim, stride):
            layer = S4(
                d_model=dim,
                l_max=s4_lmax,
                d_state=s4_d_state,
                bidirectional=s4_bidirectional,
                postact="glu" if glu else None,
                dropout=dropout,
                transposed=True,
                # hurwitz=True, # use the Hurwitz parameterization for stability
                # tie_state=True, # tie SSM parameters across d_state in the S4 layer
                trainable={
                    "dt": True,
                    "A": True,
                    "P": True,
                    "B": True,
                },  # train all internal S4 parameters
            )

            return ResidualBlock(
                d_model=dim,
                layer=layer,
                dropout=dropout,
                diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                in_channels=in_channels,
                label_embed_dim=label_embed_dim,
                stride=stride,
            )

        def ff_block(dim, stride):
            layer = FFBlock(
                d_model=dim,
                expand=ff,
                dropout=dropout,
            )
            return ResidualBlock(
                d_model=dim,
                layer=layer,
                dropout=dropout,
                diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                in_channels=in_channels,
                label_embed_dim=label_embed_dim,
                stride=stride,
            )

        # Down blocks
        d_layers, H = self.init_down_blocks(pool, unet, n_layers, ff, H, expand, s4_block, ff_block)

        # Center block
        c_layers = self.init_center_blocks(pool, n_layers, ff, H, s4_block, ff_block)

        # Up blocks
        u_layers, H = self.init_up_blocks(pool, n_layers, ff, H, expand, bidirectional, s4_block, ff_block)

        self.d_layers = nn.ModuleList(d_layers)
        self.c_layers = nn.ModuleList(c_layers)
        self.u_layers = nn.ModuleList(u_layers)
        self.norm = nn.LayerNorm(H)

        self.init_conv = nn.Sequential(nn.Conv1d(in_channels, d_model, kernel_size=1), nn.ReLU())
        self.final_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1), nn.ReLU(), nn.Conv1d(d_model, out_channels, kernel_size=1)
        )
        self.fc_t1 = nn.Linear(diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid)
        self.fc_t2 = nn.Linear(diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out)
        self.cond_embedding = nn.Embedding(label_embed_classes, label_embed_dim)
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        assert H == d_model

    def init_down_blocks(self, pool, unet, n_layers, ff, H, expand, s4_block, ff_block):
        d_layers = []
        for i, p in enumerate(pool):
            if unet:
                # Add blocks in the down layers
                for _ in range(n_layers):
                    if i == 0:
                        d_layers.append(s4_block(H, 1))
                        if ff > 0:
                            d_layers.append(ff_block(H, 1))
                    elif i == 1:
                        d_layers.append(s4_block(H, p))
                        if ff > 0:
                            d_layers.append(ff_block(H, p))
            # Add sequence downsampling and feature expanding
            d_layers.append(DownPool(H, expand, p))
            H *= expand
        return d_layers, H

    def init_up_blocks(self, pool, n_layers, ff, H, expand, bidirectional, s4_block, ff_block):
        u_layers = []
        for i, p in enumerate(pool[::-1]):
            block = []
            H //= expand
            block.append(UpPool(H * expand, expand, p, causal=not bidirectional))

            for _ in range(n_layers):
                if i == 0:
                    block.append(s4_block(H, pool[0]))
                    if ff > 0:
                        block.append(ff_block(H, pool[0]))

                elif i == 1:
                    block.append(s4_block(H, 1))
                    if ff > 0:
                        block.append(ff_block(H, 1))

            u_layers.append(nn.ModuleList(block))
        return u_layers, H

    def init_center_blocks(self, pool, n_layers, ff, H, s4_block, ff_block):
        c_layers = []
        for _ in range(n_layers):
            c_layers.append(s4_block(H, pool[1] * 2))
            if ff > 0:
                c_layers.append(ff_block(H, pool[1] * 2))
        return c_layers

    def on_fit_start(self) -> None:
        self.diffusion_parameters = {
            k: v.to(self.device) for k, v in self.diffusion_parameters.items() if isinstance(v, torch.Tensor)
        }
        return super().on_fit_start()

    def get_device(self):
        return next(self.parameters()).device

    def forward(self, input_data):
        # (transformed_X, cond, mask, diffusion_steps.view(B,1),))
        # audio_cond: same shape as audio, audio_mask: same shape as audio but binary to be imputed where zero
        noise, conditional, mask, diffusion_steps = input_data

        conditional = conditional * mask
        conditional = torch.cat([conditional, mask.float()], dim=1)

        diffusion_step_embed = calc_diffusion_step_embedding(
            diffusion_steps, self.diffusion_step_embed_dim_in, self.get_device()
        )
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

        x = noise
        x = self.init_conv(x)

        # Down blocks
        outputs = []
        outputs.append(x)
        for layer in self.d_layers:
            if isinstance(layer, ResidualBlock):
                x = layer((x, conditional, diffusion_step_embed))
            else:
                x = layer(x)
            outputs.append(x)

        # Center block
        for layer in self.c_layers:
            if isinstance(layer, ResidualBlock):
                x = layer((x, conditional, diffusion_step_embed))
            else:
                x = layer(x)
        x = x + outputs.pop()  # add a skip connection to the last output of the down block

        # Up blocks
        for block in self.u_layers:
            if self.unet:
                for layer in block:
                    if isinstance(layer, ResidualBlock):
                        x = layer((x, conditional, diffusion_step_embed))
                    else:
                        x = layer(x)
                    x = x + outputs.pop()  # skip connection
            else:
                for layer in block:
                    if isinstance(layer, ResidualBlock):
                        x = layer((x, conditional, diffusion_step_embed))
                    else:
                        x = layer(x)
                    if isinstance(layer, UpPool):
                        # Before modeling layer in the block
                        x = x + outputs.pop()
                        outputs.append(x)
                x = x + outputs.pop()  # add a skip connection from the input of the modeling part of this up block

        # feature projection
        x = x.transpose(1, 2)  # (batch, length, expand)
        x = self.norm(x).transpose(1, 2)  # (batch, expand, length)

        x = self.final_conv(x)  # 128 to 12
        return x

    def step_fn(self, batch, step_prefix=""):
        amputated_data, amputation_mask, target, target_missingness = batch

        amputated_data = torch.nan_to_num(amputated_data).permute(0, 2, 1)
        amputation_mask = amputation_mask.permute(0, 2, 1).bool()

        padding_size = next_power(amputated_data.shape[2]) - amputated_data.shape[2]
        amputated_data = torch.cat(
            [
                amputated_data,
                torch.zeros((amputated_data.shape[0], amputated_data.shape[1], padding_size), device=self.device),
            ],
            dim=2,
        )
        amputation_mask = torch.cat(
            [
                amputation_mask,
                torch.zeros(
                    (amputation_mask.shape[0], amputation_mask.shape[1], padding_size), device=self.device, dtype=bool
                ),
            ],
            dim=2,
        )

        observed_mask = 1 - amputation_mask.float()
        amputation_mask = amputation_mask.bool()

        if step_prefix in ["train", "val"]:
            T, Alpha_bar = self.hparams.diffusion_time_steps, self.diffusion_parameters["Alpha_bar"]

            B, C, L = amputated_data.shape  # B is batchsize, C=1, L is audio length
            diffusion_steps = torch.randint(T, size=(B, 1, 1)).to(self.device)  # randomly sample diffusion steps from 1~T

            z = std_normal(amputated_data.shape, self.device)
            z = amputated_data * observed_mask.float() + z * (1 - observed_mask).float()
            transformed_X = (
                torch.sqrt(Alpha_bar[diffusion_steps]) * amputated_data + torch.sqrt(1 - Alpha_bar[diffusion_steps]) * z
            )  # compute x_t from q(x_t|x_0)
            epsilon_theta = self(
                (
                    transformed_X,
                    amputated_data,
                    observed_mask,
                    diffusion_steps.view(B, 1),
                )
            )  # predict \epsilon according to \epsilon_\theta

            loss = self.loss(epsilon_theta[amputation_mask], z[amputation_mask])
        else:
            target = target.permute(0, 2, 1)
            target_missingness = target_missingness.permute(0, 2, 1)
            target = torch.cat(
                [target, torch.zeros((target.shape[0], target.shape[1], padding_size), device=self.device)], dim=2
            )
            target_missingness = torch.cat(
                [
                    target_missingness,
                    torch.zeros((target_missingness.shape[0], target_missingness.shape[1], padding_size), device=self.device),
                ],
                dim=2,
            )
            imputed_data = self.sampling(amputated_data, observed_mask)
            amputated_data[amputation_mask] = imputed_data[amputation_mask]
            amputated_data[target_missingness > 0] = target[target_missingness > 0]
            loss = self.loss(amputated_data, target)
            for metric in self.metrics[step_prefix].values():
                metric.update((torch.flatten(amputated_data, start_dim=1).clone(), torch.flatten(target, start_dim=1).clone()))

        self.log(f"{step_prefix}/loss", loss.item(), prog_bar=True)
        return loss

    def default_state(self, *args, **kwargs):
        layers = list(self.d_layers) + list(self.c_layers) + [layer for block in self.u_layers for layer in block]
        return [layer.default_state(*args, **kwargs) for layer in layers]

    def step(self, x, state, **kwargs):
        """
        input: (batch, d_input)
        output: (batch, d_output)
        """
        # States will be popped in reverse order for convenience
        state = state[::-1]

        # Down blocks
        outputs = []  # Store all layers for SaShiMi
        next_state = []
        for layer in self.d_layers:
            outputs.append(x)
            x, _next_state = layer.step(x, state=state.pop(), **kwargs)
            next_state.append(_next_state)
            if x is None:
                break

        # Center block
        if x is None:
            # Skip computations since we've downsized
            skipped = len(self.d_layers) - len(outputs)
            for _ in range(skipped + len(self.c_layers)):
                next_state.append(state.pop())
            if self.unet:
                for i in range(skipped):
                    next_state.append(state.pop())
                u_layers = list(self.u_layers)[skipped // 3 :]
            else:
                for i in range(skipped):
                    for _ in range(len(self.u_layers[i])):
                        next_state.append(state.pop())
                u_layers = list(self.u_layers)[skipped:]
        else:
            outputs.append(x)
            for layer in self.c_layers:
                x, _next_state = layer.step(x, state=state.pop(), **kwargs)
                next_state.append(_next_state)
            x = x + outputs.pop()
            u_layers = self.u_layers

        x, next_state = self.up_blocks_loop(x, u_layers, next_state, state, outputs, **kwargs)

        # feature projection
        x = self.norm(x)
        return x, next_state

    def up_blocks_loop(self, x, u_layers, next_state, state, outputs, **kwargs):
        for block in u_layers:
            if self.unet:
                for layer in block:
                    x, _next_state = layer.step(x, state=state.pop(), **kwargs)
                    next_state.append(_next_state)
                    x = x + outputs.pop()
            else:
                for layer in block:
                    x, _next_state = layer.step(x, state=state.pop(), **kwargs)
                    next_state.append(_next_state)
                    if isinstance(layer, UpPool):
                        # Before modeling layer in the block
                        x = x + outputs.pop()
                        outputs.append(x)
                x = x + outputs.pop()
        return x, next_state

    def setup_rnn(self, mode="dense"):
        """
        Convert the SaShiMi model to a RNN for autoregressive generation.

        Args:
            mode: S4 recurrence mode. Using `diagonal` can speed up generation by 10-20%.
                `linear` should be faster theoretically but is slow in practice since it
                dispatches more operations (could benefit from fused operations).
                Note that `diagonal` could potentially be unstable if the diagonalization is numerically unstable
                (although we haven't encountered this case in practice), while `dense` should always be stable.
        """
        assert mode in ["dense", "diagonal", "linear"]
        for module in self.modules():
            if hasattr(module, "setup_step"):
                module.setup_step(mode)

    def sampling(self, cond, mask):
        """
        Perform the complete sampling step according to p(x_0|x_T) = prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

        Parameters:
        net (torch network):            the wavenet model
        size (tuple):                   size of tensor to be generated,
                                        usually is (number of audios to generate, channels=1, length of audio)
        diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                        note, the tensors need to be cuda tensors

        Returns:
        the generated audio(s) in torch.tensor, shape=size
        """

        Alpha, Alpha_bar, Sigma = (
            self.diffusion_parameters["Alpha"],
            self.diffusion_parameters["Alpha_bar"],
            self.diffusion_parameters["Sigma"],
        )

        T = self.hparams.diffusion_time_steps
        assert len(Alpha) == T
        assert len(Alpha_bar) == T
        assert len(Sigma) == T

        B, _, _ = cond.shape
        x = std_normal(cond.shape, self.device)

        for t in range(T - 1, -1, -1):
            x = x * (1 - mask).float() + cond * mask.float()
            diffusion_steps = (t * torch.ones((B, 1))).to(self.device)  # use the corresponding reverse step
            epsilon_theta = self(
                (
                    x,
                    cond,
                    mask,
                    diffusion_steps,
                )
            )  # predict \epsilon according to \epsilon_\theta
            # update x_{t-1} to \mu_\theta(x_t)
            x = (x - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])
            if t > 0:
                x = x + Sigma[t] * std_normal(cond.shape, self.device)  # add the variance term to x_{t-1}

        return x


def swish(x):
    return x * torch.sigmoid(x)


def std_normal(size, device):
    """
    Generate the standard Gaussian variable of a certain size
    """

    return torch.normal(0, 1, size=size).to(device)


def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in, device):
    """
    Embed a diffusion step $t$ into a higher dimensional space
    E.g. the embedding vector in the 128-dimensional space is
    [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]
    Parameters:
    diffusion_steps (torch.long tensor, shape=(batchsize, 1)):
                                diffusion steps for batch data
    diffusion_step_embed_dim_in (int, default=128):
                                dimensionality of the embedding space for discrete diffusion steps

    Returns:
    the embedding vectors (torch.tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
    """

    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim) * -_embed).to(device)
    _embed = diffusion_steps * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed), torch.cos(_embed)), 1)

    return diffusion_step_embed


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding, stride=stride)

        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        return out


class DownPool(nn.Module):
    def __init__(self, d_input, expand, pool):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_input * expand
        self.pool = pool

        self.linear = LinearActivation(
            d_input * pool,
            self.d_output,
            transposed=True,
            weight_norm=True,
        )

    def forward(self, x):
        x = rearrange(x, "... h (l s) -> ... (h s) l", s=self.pool)
        x = self.linear(x)
        return x

    def step(self, x, state, **kwargs):
        """
        x: (..., H)
        """

        if x is None:
            return None, state
        state.append(x)
        if len(state) == self.pool:
            x = rearrange(torch.stack(state, dim=-1), "... h s -> ... (h s)")
            x = x.unsqueeze(-1)
            x = self.linear(x)
            x = x.squeeze(-1)
            return x, []
        else:
            return None, state

    def default_state(self, *args, **kwargs):
        return []


class UpPool(nn.Module):
    def __init__(self, d_input, expand, pool, causal=True):
        super().__init__()
        self.d_output = d_input // expand
        self.pool = pool
        self.causal = causal

        self.linear = LinearActivation(
            d_input,
            self.d_output * pool,
            transposed=True,
            weight_norm=True,
        )

    def forward(self, x):
        x = self.linear(x)

        if self.causal:
            x = F.pad(x[..., :-1], (1, 0))  # Shift to ensure causality
        x = rearrange(x, "... (h s) l -> ... h (l s)", s=self.pool)

        return x

    def step(self, x, state, **kwargs):
        """
        x: (..., H)
        """
        assert len(state) > 0
        y, state = state[0], state[1:]
        if len(state) == 0:
            assert x is not None
            x = x.unsqueeze(-1)
            x = self.linear(x)
            x = x.squeeze(-1)
            x = rearrange(x, "... (h s) -> ... h s", s=self.pool)
            state = list(torch.unbind(x, dim=-1))
        else:
            assert x is None
        return y, state

    def default_state(self, *batch_shape, device=None):
        state = torch.zeros(batch_shape + (self.d_output, self.pool), device=device)  # (batch, h, s)
        state = list(torch.unbind(state, dim=-1))  # List of (..., H)
        return state


class FFBlock(nn.Module):
    def __init__(self, d_model, expand=2, dropout=0.0):
        """
        Feed-forward block.

        Args:
            d_model: dimension of input
            expand: expansion factor for inverted bottleneck
            dropout: dropout rate
        """
        super().__init__()

        input_linear = LinearActivation(
            d_model,
            d_model * expand,
            transposed=True,
            activation="gelu",
            activate=True,
        )
        dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        output_linear = LinearActivation(
            d_model * expand,
            d_model,
            transposed=True,
            activation=None,
            activate=False,
        )

        self.ff = nn.Sequential(
            input_linear,
            dropout,
            output_linear,
        )

    def forward(self, x):
        return self.ff(x), None

    def default_state(self, *args, **kwargs):
        return None

    def step(self, x, state, **kwargs):
        # expects: (B, D, L)
        return self.ff(x.unsqueeze(-1)).squeeze(-1), state


class ResidualBlock(nn.Module):
    def __init__(self, d_model, layer, dropout, diffusion_step_embed_dim_out, in_channels, label_embed_dim, stride):
        """
        Residual S4 block.

        Args:
            d_model: dimension of the model
            bidirectional: use bidirectional S4 layer
            glu: use gated linear unit in the S4 layer
            dropout: dropout rate
        """
        super().__init__()

        self.layer = layer
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        self.fc_t = nn.Linear(diffusion_step_embed_dim_out, d_model)
        self.cond_conv = Conv(2 * in_channels, d_model, kernel_size=stride, stride=stride)
        self.fc_label = nn.Linear(label_embed_dim, d_model) if label_embed_dim is not None else None

    def forward(self, input_data):
        """
        Input x is shape (B, d_input, L)
        """
        x, cond, diffusion_step_embed = input_data

        # add in diffusion step embedding
        part_t = self.fc_t(diffusion_step_embed).unsqueeze(2)
        z = x + part_t

        # Prenorm
        z = self.norm(z.transpose(-1, -2)).transpose(-1, -2)

        z, _ = self.layer(z)

        cond = self.cond_conv(cond)
        # cond = self.fc_label(cond)

        z = z + cond

        # Dropout on the output of the layer
        z = self.dropout(z)

        # Residual connection
        x = z + x

        return x

    def default_state(self, *args, **kwargs):
        return self.layer.default_state(*args, **kwargs)

    def step(self, x, state, **kwargs):
        z = x

        # Prenorm
        z = self.norm(z)

        # Apply layer
        z, state = self.layer.step(z, state, **kwargs)

        # Residual connection
        x = z + x

        return x, state


def largets_component(number):
    """
    returns the prime number that divides number the most times.
    """
    for i in range(2, int(number**0.5) + 1):
        if number % i == 0:
            return i
    return number


def next_power(number):
    """
    returns the next power of 2.
    """
    return 1 << (number - 1).bit_length()


def calc_diffusion_hyperparams(diffusion_time_steps, beta_0, beta_T):
    """
    Compute diffusion process hyperparameters

    Params:
        T (int): number of diffusion steps.
        beta_0 beta_T (float) :beta schedule start/end value, where any beta_t in the middle is linearly interpolated

    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """

    Beta = torch.linspace(beta_0, beta_T, diffusion_time_steps)  # Linear schedule
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, diffusion_time_steps):
        Alpha_bar[t] *= Alpha_bar[t - 1]  # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (1 - Alpha_bar[t])  # \tilde{\beta}_t = \beta_t * (1-\bar{\alpha}_{t-1})
        # / (1-\bar{\alpha}_t)
    Sigma = torch.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t

    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = diffusion_time_steps, Beta, Alpha, Alpha_bar, Sigma
    diffusion_hyperparams = _dh
    return diffusion_hyperparams
