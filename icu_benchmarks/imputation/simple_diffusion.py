from icu_benchmarks.models.wrappers import ImputationWrapper
import gin
import math
import torch
from torch import nn
import torch.nn.functional as F


@gin.configurable("Simple_Diffusion")
class SimpleDiffusionModel(ImputationWrapper):
    """Imputation model based on a Simple Diffusion Model.
    Adapted from https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL."""

    requires_backprop = True

    input_size = []

    def __init__(self, *args, input_size, **kwargs):
        super().__init__(*args, input_size=input_size, **kwargs)

        down_channels = (25, 20, 18, 15)
        up_channels = (15, 18, 20, 25)
        time_emb_dim = 6

        self.input_size = input_size

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim), nn.Linear(time_emb_dim, time_emb_dim), nn.ReLU()
        )

        # Initial projection
        self.conv0 = nn.Conv1d(input_size[1], down_channels[0], 2)

        # Downsample
        self.downs = nn.ModuleList(
            [Block(down_channels[i], down_channels[i + 1], time_emb_dim) for i in range(len(down_channels) - 1)]
        )

        # Upsample
        self.ups = nn.ModuleList(
            [Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True) for i in range(len(up_channels) - 1)]
        )

        # Final Output
        self.output = nn.ConvTranspose1d(up_channels[-1], input_size[1], 2)

    def forward(self, amputated, timestep):
        amputated = torch.nan_to_num(amputated, nan=0.0)
        # model_input = torch.cat((amputated, amputation_mask), dim=1)

        # output = self.model(model_input)
        # output = output.reshape(amputated.shape)

        # Embedd time
        t = self.time_mlp(timestep)

        # Initial Convolution
        x = self.conv0(amputated)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)

        # Output Layer
        output = self.output(x)

        return output

    def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
        return torch.linspace(start, end, timesteps)

    def get_index_from_list(self, vals, t, x_shape):
        """
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_diffusion_sample(self, x_0, t):
        """
        Takes an image and a timestep as input and
        returns the noisy version of it
        """
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        # mean + variance
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

    # Define beta schedule
    T = 300
    betas = linear_beta_schedule(timesteps=T)

    # Pre-calculate different terms for closed form
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def get_loss(self, x_0, t):
        x_noisy, noise = self.forward_diffusion_sample(x_0, t)
        noise_pred = self(x_noisy, t)
        return F.l1_loss(noise, noise_pred)

    def on_fit_start(self) -> None:
        self.betas = self.betas.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(self.device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(self.device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(self.device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(self.device)
        self.posterior_variance = self.posterior_variance.to(self.device)
        super().on_fit_start()

    def training_step(self, batch):
        amputated, amputation_mask, target, target_missingness = batch
        amputated = torch.nan_to_num(amputated, nan=0.0)

        t = torch.randint(0, self.T, (self.input_size[0],), device=self.device).long()
        loss = self.get_loss(target, t)

        self.log("train/loss", loss.item(), prog_bar=True)

        for metric in self.metrics["train"].values():
            metric.update((torch.flatten(target, start_dim=1), torch.flatten(target, start_dim=1)))

        return loss

    def validation_step(self, batch, batch_index):
        amputated, amputation_mask, target, target_missingness = batch
        amputated = torch.nan_to_num(amputated, nan=0.0)
        # imputated = self(amputated, amputation_mask)

        t = torch.randint(0, self.T, (1,), device=self.device).long()

        betas_t = self.get_index_from_list(self.betas, t, amputated.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, amputated.shape)
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, amputated.shape)

        model_mean = sqrt_recip_alphas_t * (amputated - betas_t * self(amputated, t) / sqrt_one_minus_alphas_cumprod_t)

        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, amputated.shape)

        if t == 0:
            imputated = model_mean
        else:
            noise = torch.randn_like(amputated)
            imputated = model_mean + torch.sqrt(posterior_variance_t) * noise

        # imputated = amputated.masked_scatter_(amputation_mask.bool(), imputated)

        amputated[amputation_mask > 0] = imputated[amputation_mask > 0]
        amputated[target_missingness > 0] = target[target_missingness > 0]

        loss = self.loss(amputated, target)
        self.log("val/loss", loss.item(), prog_bar=True)

        for metric in self.metrics["val"].values():
            metric.update((torch.flatten(amputated, start_dim=1), torch.flatten(target, start_dim=1)))

    def test_step(self, batch, batch_index):
        amputated, amputation_mask, target, target_missingness = batch
        amputated = torch.nan_to_num(amputated, nan=0.0)
        # imputated = self(amputated, amputation_mask)

        t = torch.randint(0, self.T, (1,), device=self.device).long()

        betas_t = self.get_index_from_list(self.betas, t, amputated.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, amputated.shape)
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, amputated.shape)

        model_mean = sqrt_recip_alphas_t * (amputated - betas_t * self(amputated, t) / sqrt_one_minus_alphas_cumprod_t)

        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, amputated.shape)

        if t == 0:
            imputated = model_mean
        else:
            noise = torch.randn_like(amputated)
            imputated = model_mean + torch.sqrt(posterior_variance_t) * noise

        # imputated = amputated.masked_scatter_(amputation_mask.bool(), imputated)

        amputated[amputation_mask > 0] = imputated[amputation_mask > 0]
        amputated[target_missingness > 0] = target[target_missingness > 0]

        loss = self.loss(amputated, target)
        self.log("test/loss", loss.item(), prog_bar=True)

        for metric in self.metrics["test"].values():
            metric.update((torch.flatten(amputated, start_dim=1), torch.flatten(target, start_dim=1)))


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        time_dim = 5 if in_ch == 25 else 4 if in_ch == 20 else 3 if in_ch == 18 else 2
        if up:
            # take 2 times the number of input channels because residuals were added in the upsampling process
            in_ch *= 2
            self.conv1 = nn.ConvTranspose1d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose1d(out_ch, out_ch, 2)
        else:
            self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv1d(out_ch, out_ch, 2)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm1d(out_ch)
        self.bnorm2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()

        # Transformer Encoder for Feature Self-Attention
        self.feature_layer = nn.TransformerEncoderLayer(d_model=in_ch, nhead=1, dim_feedforward=64, activation="gelu")
        self.feature_transformer = nn.TransformerEncoder(self.feature_layer, num_layers=1)

        # Transformer Encoder for Time Self-Attention
        self.time_layer = nn.TransformerEncoderLayer(d_model=time_dim, nhead=1, dim_feedforward=64, activation="gelu")
        self.time_transformer = nn.TransformerEncoder(self.time_layer, num_layers=1)

    def forward(self, x, t):
        # Apply Feature Self-Attention
        h = self.feature_transformer(x.permute(0, 2, 1)).permute(0, 2, 1)
        # Apply Time Self-Attention
        h = self.time_transformer(h)
        # First Convolution
        h = self.bnorm1(self.relu(self.conv1(h)))
        # Time Embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last dimension
        time_emb = time_emb[(...,) + (None,)]
        # Add time
        h += time_emb
        # Second Convolution
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1 + 0.05)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
