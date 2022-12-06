# Source: 
# https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=qWw50ui9IZ5q
# Tutorial:
# https://m.youtube.com/watch?v=a4Yfz2FxXiY

from icu_benchmarks.models.wrappers import ImputationWrapper
import gin
import math
import torch
from torch import nn

@gin.configurable("Simple_Diffusion")
class Simple_Diffusion_Model(ImputationWrapper):

    needs_training = True
    needs_fit = False

    def __init__(self, *args, input_size, **kwargs):
        super().__init__(*args, **kwargs)

        down_channels = (25, 20, 18, 15)
        up_channels = (15, 18, 20, 25)
        time_emb_dim = 6

        # Time embedding
        # self.time_mlp = nn.Sequential(
        #     SinusoidalPositionEmbeddings(time_emb_dim),
        #     nn.Linear(time_emb_dim, time_emb_dim),
        #     nn.ReLU()
        # )

        # Initial projection
        self.conv0 = nn.Conv1d(input_size[1], down_channels[0], 2)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], time_emb_dim) for i in range(len(down_channels) - 1)])

        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True) for i in range(len(up_channels) - 1)])

        # Final Output
        self.output = nn.ConvTranspose1d(up_channels[-1], input_size[1], 2)


    def forward(self, amputated, amputation_mask): # how to hand over timestep
        amputated = torch.nan_to_num(amputated, nan=0.0)
        # model_input = torch.cat((amputated, amputation_mask), dim=1)

        # output = self.model(model_input)
        # output = output.reshape(amputated.shape)

        # Embedd time
        # t = self.time_mlp(timestep)

        # Initial Convolution
        x = self.conv0(amputated)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x) #, t
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x) #, t

        # Output Layer
        output = self.output(x)

        return output


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            # take 2 times the number of input channels because residuals were added in the upsampling process
            self.conv1 = nn.ConvTranspose1d(2 * in_ch, out_ch, 2) #, padding=1
            # self.transform = nn.ConvTranspose1d(out_ch, out_ch, 3, 1)
        else:
            self.conv1 = nn.Conv1d(in_ch, out_ch, 2) #, padding=1
            # self.transform = nn.Conv1d(out_ch, out_ch, 3, 1)
        self.bnorm = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x): #, t
        # time_emb = self.relu(self.time_mlp(t))
        h = self.bnorm(self.relu(self.conv1(x)))
        # output = h + x
        return h # output


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        print(f"embeddings: {embeddings.shape}")
        return embeddings
