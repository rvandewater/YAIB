import math

import gin
import numpy as np
import torch
import torch.nn as nn

from icu_benchmarks.models.wrappers import ImputationWrapper


@gin.configurable("DiffWave")
class DiffWaveImputer(ImputationWrapper):
    """Imputation model based on DiffWave (https://arxiv.org/abs/2009.09761). Adapted from
    https://github.com/AI4HealthUOL/SSSD/blob/main/src/imputers/DiffWaveImputer.py"""

    def __init__(
        self,
        in_channels,
        res_channels,
        skip_channels,
        out_channels,
        num_res_layers,
        dilation_cycle,
        diffusion_step_embed_dim_in,
        diffusion_step_embed_dim_mid,
        diffusion_step_embed_dim_out,
        diffusion_time_steps,
        beta_0,
        beta_T,
        *args,
        **kwargs,
    ):
        super(DiffWaveImputer, self).__init__(
            in_channels=in_channels,
            res_channels=res_channels,
            skip_channels=skip_channels,
            out_channels=out_channels,
            num_res_layers=num_res_layers,
            dilation_cycle=dilation_cycle,
            diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
            diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
            diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
            diffusion_time_steps=diffusion_time_steps,
            beta_0=beta_0,
            beta_T=beta_T,
            *args,
            **kwargs,
        )

        self.init_conv = nn.Sequential(Conv(in_channels, res_channels, kernel_size=1), nn.ReLU())

        self.residual_layer = Residual_group(
            res_channels=res_channels,
            skip_channels=skip_channels,
            num_res_layers=num_res_layers,
            dilation_cycle=dilation_cycle,
            diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
            diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
            diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
            in_channels=in_channels,
        )

        self.final_conv = nn.Sequential(
            Conv(skip_channels, skip_channels, kernel_size=1),
            nn.ReLU(),
            ZeroConv1d(skip_channels, out_channels),
        )

        self.diffusion_parameters = calc_diffusion_hyperparams(diffusion_time_steps, beta_0, beta_T)

    def on_fit_start(self) -> None:
        self.diffusion_parameters = {
            k: v.to(self.device) for k, v in self.diffusion_parameters.items() if isinstance(v, torch.Tensor)
        }
        return super().on_fit_start()

    def forward(self, input_data):
        noise, conditional, mask, diffusion_steps = input_data

        conditional = conditional * mask
        conditional = torch.cat([conditional, mask.float()], dim=1)

        x = noise
        x = self.init_conv(x)
        x = self.residual_layer((x, conditional, diffusion_steps))
        y = self.final_conv(x)

        return y

    def step_fn(self, batch, step_prefix=""):
        amputated_data, amputation_mask, target, target_missingness = batch

        amputated_data = torch.nan_to_num(amputated_data).permute(0, 2, 1)
        amputation_mask = amputation_mask.permute(0, 2, 1).bool()
        observed_mask = 1 - amputation_mask.float()

        if step_prefix in ["train", "val"]:
            T, Alpha_bar = (
                self.hparams.diffusion_time_steps,
                self.diffusion_parameters["Alpha_bar"],
            )

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

            loss = self.loss(epsilon_theta[amputation_mask.bool()], z[amputation_mask.bool()])
        else:
            target = target.permute(0, 2, 1)
            target_missingness = target_missingness.permute(0, 2, 1)
            imputed_data = self.sampling(amputated_data, observed_mask)
            amputated_data[amputation_mask.bool()] = imputed_data[amputation_mask.bool()]
            amputated_data[target_missingness > 0] = target[target_missingness > 0]
            loss = self.loss(amputated_data, target)
            for metric in self.metrics[step_prefix].values():
                metric.update(
                    (
                        torch.flatten(amputated_data, start_dim=1).clone(),
                        torch.flatten(target, start_dim=1).clone(),
                    )
                )

        self.log(f"{step_prefix}/loss", loss.item(), prog_bar=True)
        return loss

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


def calc_diffusion_hyperparams(diffusion_time_steps, beta_0, beta_T):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value,
                                where any beta_t in the middle is linearly interpolated

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
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = (
        diffusion_time_steps,
        Beta,
        Alpha,
        Alpha_bar,
        Sigma,
    )
    diffusion_hyperparams = _dh
    return diffusion_hyperparams


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=self.padding,
        )
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        return out


class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)
        return out


class Residual_block(nn.Module):
    def __init__(
        self,
        res_channels,
        skip_channels,
        dilation,
        diffusion_step_embed_dim_out,
        in_channels,
    ):
        super(Residual_block, self).__init__()

        self.res_channels = res_channels
        # the layer-specific fc for diffusion step embedding
        self.fc_t = nn.Linear(diffusion_step_embed_dim_out, self.res_channels)

        # dilated conv layer
        self.dilated_conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3, dilation=dilation)

        # add mel spectrogram upsampler and conditioner conv1x1 layer  (In adapted to S4 output)
        self.cond_conv = Conv(2 * in_channels, 2 * self.res_channels, kernel_size=1)  # 80 is mel bands

        # residual conv1x1 layer, connect to next residual layer
        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)

        # skip conv1x1 layer, add to all skip outputs through skip connections
        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.skip_conv = nn.utils.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.skip_conv.weight)

    def forward(self, input_data):
        x, cond, diffusion_step_embed = input_data
        h = x
        B, C, L = x.shape
        assert C == self.res_channels

        part_t = self.fc_t(diffusion_step_embed)
        part_t = part_t.view([B, self.res_channels, 1])
        h = h + part_t

        h = self.dilated_conv_layer(h)
        # add (local) conditioner
        assert cond is not None

        cond = self.cond_conv(cond)
        h += cond

        out = torch.tanh(h[:, : self.res_channels, :]) * torch.sigmoid(h[:, self.res_channels:, :])

        res = self.res_conv(out)
        assert x.shape == res.shape
        skip = self.skip_conv(out)

        return (x + res) * math.sqrt(0.5), skip


class Residual_group(nn.Module):
    def __init__(
        self,
        res_channels,
        skip_channels,
        num_res_layers,
        dilation_cycle,
        diffusion_step_embed_dim_in,
        diffusion_step_embed_dim_mid,
        diffusion_step_embed_dim_out,
        in_channels,
    ):
        super(Residual_group, self).__init__()
        self.num_res_layers = num_res_layers
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        self.fc_t1 = nn.Linear(diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid)
        self.fc_t2 = nn.Linear(diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out)

        self.residual_blocks = nn.ModuleList()
        for n in range(self.num_res_layers):
            self.residual_blocks.append(
                Residual_block(
                    res_channels,
                    skip_channels,
                    dilation=2 ** (n % dilation_cycle),
                    diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                    in_channels=in_channels,
                )
            )

    def get_device(self):
        return next(self.parameters()).device

    def forward(self, input_data):
        noise, conditional, diffusion_steps = input_data

        diffusion_step_embed = calc_diffusion_step_embedding(
            diffusion_steps, self.diffusion_step_embed_dim_in, self.get_device()
        )
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

        h = noise
        skip = 0
        for n in range(self.num_res_layers):
            h, skip_n = self.residual_blocks[n]((noise, conditional, diffusion_step_embed))
            skip += skip_n

        return skip * math.sqrt(1.0 / self.num_res_layers)  # normalize for training stability


def std_normal(size, device):
    """
    Generate the standard Gaussian variable of a certain size
    """

    return torch.normal(0, 1, size=size).to(device)
