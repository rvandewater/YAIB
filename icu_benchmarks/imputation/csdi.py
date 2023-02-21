# Source: https://github.com/ermongroup/CSDI

import gin
from icu_benchmarks.models.wrappers import ImputationWrapper
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionStepEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diff_CSDI(nn.Module):
    def __init__(self, channels, num_diffusion_steps, diffusion_step_embedding_dim, side_dim, nheads, num_residual_blocks, inputdim=2):
        super().__init__()
        self.channels = channels

        self.diffusion_step_embedding = DiffusionStepEmbedding(
            num_steps=num_diffusion_steps,
            embedding_dim=diffusion_step_embedding_dim,
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=side_dim,
                    channels=self.channels,
                    diffusion_embedding_dim=diffusion_step_embedding_dim,
                    nheads=nheads,
                )
                for _ in range(num_residual_blocks)
            ]
        )

    def forward(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_step_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip


@gin.configurable("CSDI")
class CSDI(ImputationWrapper):
    def __init__(
            self,
            input_size,
            time_step_embedding_size,
            feature_embedding_size,
            unconditional,
            target_strategy,
            num_diffusion_steps,
            diffusion_step_embedding_dim,
            n_attention_heads,
            num_residual_layers,
            noise_schedule,
            beta_start,
            beta_end,
            n_samples,
            conv_channels,
            *args,
            **kwargs,
        ):

        super().__init__(
            input_size=input_size,
            time_step_embedding_size=time_step_embedding_size,
            feature_embedding_size=feature_embedding_size,
            unconditional=unconditional,
            target_strategy=target_strategy,
            num_diffusion_steps=num_diffusion_steps,
            diffusion_step_embedding_dim=diffusion_step_embedding_dim,
            n_attention_heads=n_attention_heads,
            num_residual_layers=num_residual_layers,
            noise_schedule=noise_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            n_samples=n_samples,
            conv_channels=conv_channels,
            *args, **kwargs,
        )
        self.target_dim = input_size[2]
        self.n_samples = n_samples

        self.emb_time_dim = time_step_embedding_size
        self.emb_feature_dim = feature_embedding_size
        self.is_unconditional = unconditional
        self.target_strategy = target_strategy

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )


        input_dim = 1 if self.is_unconditional else 2
        self.diffmodel = diff_CSDI(conv_channels, num_diffusion_steps, diffusion_step_embedding_dim, self.emb_total_dim, n_attention_heads, num_residual_layers, input_dim)

        # parameters for diffusion models
        self.num_steps = num_diffusion_steps
        if noise_schedule == "quad":
            self.beta = np.linspace(
                beta_start ** 0.5, beta_end ** 0.5, self.num_steps
            ) ** 2
        elif noise_schedule == "linear":
            self.beta = np.linspace(
                beta_start, beta_end, self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().unsqueeze(1).unsqueeze(1)
    
    def on_fit_start(self) -> None:
        self.alpha_torch = self.alpha_torch.to(self.device)
        self.alpha_hat = torch.from_numpy(self.alpha_hat).to(self.device)
        self.beta = torch.from_numpy(self.beta).to(self.device)
        self.alpha = torch.from_numpy(self.alpha).to(self.device)
        return super().on_fit_start()

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        sample_ratios = torch.rand((len(observed_mask), ), device=self.device)
        for i in range(len(observed_mask)):
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratios[i].item())
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        random_tensor = torch.rand((len(cond_mask), ), device=self.device)
        for i in range(len(cond_mask)):
            mask_choice = random_tensor[i]
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:  # draw another sample for histmask (i-1 corresponds to another sample)
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1] 
        return cond_mask

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps
        

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def impute(self, amputated_data, cond_mask, side_info, n_samples):
        B, K, L = amputated_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional:
                noisy_obs = amputated_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(amputated_data)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional:
                    diff_input = noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    cond_obs = amputated_data.unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def get_conditional_mask(self, observed_mask):
        if self.target_strategy == "random":
            return self.get_randmask(observed_mask)
        return self.get_hist_mask(observed_mask)

    def forward(self, amputated_data, amputation_mask):

        amputated_data = amputated_data.permute(0, 2, 1)
        amputation_mask = amputation_mask.permute(0, 2, 1)
        observed_mask = torch.ones_like(amputation_mask) - amputation_mask
        B, K, L = amputated_data.shape
        
        observed_time_points = torch.arange(0, L, 1, device=self.device).expand(B, L)
        
        cond_mask = self.get_conditional_mask(observed_mask)

        side_info = self.get_side_info(observed_time_points, cond_mask)

        # return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)
        t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(amputated_data)
        noisy_data = (current_alpha ** 0.5) * amputated_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, amputated_data, cond_mask)

        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)

        target_mask = observed_mask - cond_mask
        return noise * target_mask,  predicted * target_mask
    
    def step_fn(self, batch, step_prefix):
        amputated_data, amputation_mask, target = batch
        amputated_data = amputated_data.nan_to_num()
        
        if step_prefix == "test":
            prediction = self.evaluate(amputated_data, amputation_mask, self.n_samples)
            amputated_data[amputation_mask > 0] = prediction[amputation_mask > 0]
            loss = self.loss(target, amputated_data)
            for metric in self.metrics[step_prefix].values():
                metric.update(
                    (torch.flatten(amputated_data.detach(), start_dim=1).clone(), torch.flatten(target.detach(), start_dim=1).clone())
                )
        else:
            noise, prediction = self(amputated_data, amputation_mask)
            loss = self.loss(noise, prediction)
        
        self.log(f"{step_prefix}/loss", loss.item(), prog_bar=True)
        return loss

    def predict_step(self, data, amputation_mask):
        return self.evaluate(data, amputation_mask, self.n_samples)

    def evaluate(self, amputated_data, amputation_mask, n_samples):
        amputated_data = amputated_data.permute(0, 2, 1)
        amputation_mask = amputation_mask.permute(0, 2, 1)
        B, K, L = amputated_data.shape
        
        observed_time_points = torch.arange(0, L, 1, device=self.device).expand(B, L)
        

        cond_mask = torch.ones_like(amputation_mask) - amputation_mask

        side_info = self.get_side_info(observed_time_points, cond_mask)

        samples = self.impute(amputated_data, cond_mask, side_info, n_samples)
        
        previous_deterministic_setting = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)
        samples = samples.median(dim=1)[0].permute(0, 2, 1)
        torch.use_deterministic_algorithms(previous_deterministic_setting)

        return samples

