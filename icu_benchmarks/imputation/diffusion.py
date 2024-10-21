# Source:
# https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=qWw50ui9IZ5q
# https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb#scrollTo=290edb0b
# Tutorial:
# https://m.youtube.com/watch?v=a4Yfz2FxXiY
# Source Paper:
# https://arxiv.org/abs/2006.11239
# Paper for Cosine Schedule:
# https://arxiv.org/abs/2102.09672

from icu_benchmarks.models.wrappers import ImputationWrapper
import gin
import math
import torch
from torch import nn
import torch.nn.functional as F


@gin.configurable("Diffusion")
class SimpleDiffusionModel(ImputationWrapper):
    """Simple Diffusion Model for Imputation. See https://arxiv.org/abs/2006.11239 for more details."""

    requires_backprop = True

    input_size = []

    def __init__(self, input_size, n_onedirectional_conv, T, min_noise, max_noise, noise_scheduler, *args, **kwargs):
        super().__init__(
            n_onedirectional_conv=n_onedirectional_conv,
            T=T,
            min_noise=min_noise,
            max_noise=max_noise,
            noise_scheduler=noise_scheduler,
            *args,
            **kwargs
        )

        self.n_onedirectional_conv = n_onedirectional_conv
        self.T = T
        self.min_noise = min_noise
        self.max_noise = max_noise
        self.noise_scheduler = noise_scheduler

        # == Noise Schedulers == #
        # Linear
        if self.noise_scheduler == "linear":
            self.betas = torch.linspace(self.min_noise, self.max_noise, self.T)
        # Quadratic
        elif self.noise_scheduler == "quadratic":
            self.betas = torch.linspace(self.min_noise**0.5, self.max_noise**0.5, self.T) ** 2
        # Cosine
        elif self.noise_scheduler == "cosine":
            x = torch.linspace(0, self.T, self.T + 1)
            alphas_cumprod = torch.cos(((x / self.T) + 0.008) / (1 + 0.008) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        # Sigmoid
        elif self.noise_scheduler == "sigmoid":
            betas = torch.linspace(-6, 6, self.T)
            self.betas = torch.sigmoid(betas) * (self.max_noise - self.min_noise) + self.min_noise
        # Error
        else:
            raise NotImplementedError(
                "Noise Scheduler must be linear, quadratic, cosine or sigmoid.\n Your Entry: [%s] is not implemented"
                % self.noise_scheduler
            )

        # Helper Values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

        # Store Input Size
        self.input_size = input_size

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(input_size[2]), nn.Linear(input_size[2], input_size[2]), nn.ReLU()
        )

        # Blocks
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        for i in range(self.n_onedirectional_conv):
            self.downs.append(Block(input_size, i))
            self.ups.append(Block(input_size, (self.n_onedirectional_conv - i), up=True))

    def forward(self, amputated, timestep):
        amputated = torch.nan_to_num(amputated, nan=0.0)
        amputated = amputated[:, None, :, :]
        x = amputated

        # Embedd time
        t = self.time_mlp(timestep)

        # Residual Connections
        residuals = []

        for down in self.downs:
            x = down(x, t)
            residuals.append(x)
        for up in self.ups:
            residual = residuals.pop()
            x = torch.cat((x, residual), dim=1)
            x = up(x, t)
        return x.squeeze()

    def training_step(self, batch):
        amputated, amputation_mask, target, target_missingness = batch
        amputated = torch.nan_to_num(amputated, nan=0.0)

        self.input_size = amputated.shape

        # # Context / Target Split (Credits @Allie)
        # # Take an inverse of the amputed mask - to get the observation mask
        # observed_mask = (~(amputation_mask > 0)).float()

        # # Generate a random tensor with the same dimensions as mask and multiply mask by it
        # # This removes all missing values from the following calculations
        # rand_for_mask = torch.rand_like(observed_mask) * observed_mask

        # # Create a context mask - the selection of the elements is so that only 50% of all observed values are selected
        # context_mask = (rand_for_mask > 0.5).float()

        # # Create a target mask - the selection of the elements is so that all values not selected by the context mask
        # # but are still observed are selected
        # target_mask = (~(rand_for_mask > 0.5)).float() * observed_mask

        # context = amputated * context_mask
        # target = amputated * target_mask

        # x_0 = context

        x_0 = amputated

        # Take a random timestep
        t = torch.randint(0, self.T, (self.input_size[0],)).long()

        # Introduce Noise into the samples according
        x_t, noise = self.forward_diffusion_sample(x_0, t)

        # Let the model predict the noise in the noised sample
        noise_pred = self(x_t, t)

        # Calculate Loss: Difference between actual noise and noise prediction
        loss = F.l1_loss(noise, noise_pred)

        self.log("train/loss", loss.item(), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        amputated, amputation_mask, target, target_missingness = batch
        amputated = torch.nan_to_num(amputated, nan=0.0)

        self.input_size = amputated.shape

        # TODO: - Context / Target Split
        x_0 = amputated

        # Take a random timestep
        t = torch.randint(0, self.T, (self.input_size[0],)).long()

        # Introduce Noise into the samples according
        x_t, noise = self.forward_diffusion_sample(x_0, t)

        # Let the model predict the noise in the noised sample
        noise_pred = self(x_t, t)

        # Calculate Loss: Difference between actual noise and noise prediction
        loss = F.l1_loss(noise, noise_pred)

        self.log("val/loss", loss.item(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        amputated, amputation_mask, target, target_missingness = batch
        amputated = torch.nan_to_num(amputated, nan=0.0)

        self.input_size = amputated.shape

        x_0 = amputated

        # Take the last timestep
        t = torch.full((self.input_size[0],), self.T - 1)

        # Let the Model predict the noise
        noise_pred = self(x_0, t)

        # Calculate the forward sample
        x_t, _ = self.forward_diffusion_sample(x_0, t)

        # Calculate the backward sample for timestep 0 replacing the original x_0 and having imputed data
        x_0 = self.backward_diffusion_sample(noise_pred, x_t, t)

        # Use the prediction only where the original data is missing
        x_0 = amputated.masked_scatter_(amputation_mask.bool(), x_0)

        # Calculate Loss: Difference between imputed and target
        loss = self.loss(x_0, target)

        self.log("test/loss", loss.item(), prog_bar=True)

        x_0[target_missingness > 0] = target[target_missingness > 0]
        # Update Metrics
        for metric in self.metrics["test"].values():
            metric.update((torch.flatten(x_0, start_dim=1), torch.flatten(target, start_dim=1)))

    # Helper function to return a value for a specific timestep t from a list reformatted for the current input size
    def get_index_from_list(self, values, t, x_shape):
        batch_size = t.shape[0]
        out = values.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    # Function that takes an original sample x_0 and introduces random noise for some timestep t
    def forward_diffusion_sample(self, x_0, t):
        # Random Noise
        noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        # Mean
        mean = sqrt_alphas_cumprod_t * x_0

        # Variance
        variance = sqrt_one_minus_alphas_cumprod_t * noise

        # Forward Sample
        forward_sample = mean + variance

        return forward_sample, noise

    #
    def backward_diffusion_sample(self, noise_pred, x_t, t, t_index=0):
        """Function that takes a noised image at some timestep t and the noise prediction and tries to compute the original
        sample. The t needs to be one specific timestamp -> always the same value. It does not have to be like
        this in the forward diffusion sample function"""

        betas_t = self.get_index_from_list(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, x_t.shape)
        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x_t.shape)

        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise


class Block(nn.Module):
    def __init__(self, input_size, i, up=False):
        super().__init__()

        n_timestamps = input_size[1] - 3 * i

        self.time_mlp = nn.Linear(input_size[2], n_timestamps)
        if up:
            # take 2 times the number of input channels because residuals were added in the upsampling process
            self.conv1 = nn.ConvTranspose2d(2, 1, 3, padding=1)
            self.transform = nn.ConvTranspose2d(1, 1, (4, 2))
        else:
            self.conv1 = nn.Conv2d(1, 1, 3, padding=1)
            self.transform = nn.Conv2d(1, 1, (4, 2))
        self.conv2 = nn.Conv2d(1, 1, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(1)
        self.bnorm2 = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        # First Convolution
        h = self.bnorm1(self.relu(self.conv1(x)))
        # TODO: - Add Attention Layer before Time Embedding

        # Time Embedding
        time_emb = self.relu(self.time_mlp(t[:, None, :]))
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
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1 + 0.05)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
