import gin

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from icu_benchmarks.models.wrappers import ImputationWrapper


@gin.configurable("NP")
class NPImputation(ImputationWrapper):
    """Imputation using Neural Processes. Implementation adapted from https://github.com/EmilienDupont/neural-processes/.
    Provides imputation wrapper for NeuralProcess class."""

    requires_backprop = True

    def __init__(
        self,
        input_size,
        encoder_layers,
        encoder_h_dim,
        decoder_layers,
        decoder_h_dim,
        r_dim,
        z_dim,
        train_sample_times,
        val_sample_times,
        test_sample_times,
        predict_sample_times,
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            input_size=input_size,
            encoder_layers=encoder_layers,
            encoder_h_dim=encoder_h_dim,
            decoder_layers=decoder_layers,
            decoder_h_dim=decoder_h_dim,
            r_dim=r_dim,
            z_dim=z_dim,
            train_sample_times=train_sample_times,
            val_sample_times=val_sample_times,
            test_sample_times=test_sample_times,
            predict_sample_times=predict_sample_times,
            *args,
            **kwargs
        )

        self.x_dim = input_size[2]
        self.y_dim = input_size[2]
        self.z_dim = z_dim

        self.train_sample_times = train_sample_times
        self.val_sample_times = val_sample_times
        self.test_sample_times = test_sample_times
        self.predict_sample_times = predict_sample_times

        self.model = NeuralProcess(
            self.x_dim,
            self.y_dim,
            encoder_layers,
            encoder_h_dim,
            decoder_layers,
            decoder_h_dim,
            r_dim,
            z_dim,
        )

    def forward(self, x_context, y_context, x_target, y_target=None):
        return self.model(x_context, y_context, x_target, y_target)

    # Override the training step - needed for the custom loss calculation
    def training_step(self, batch, _):
        self.model.train(True)

        # Unpack batch into three values
        amputed, mask, _, _ = batch
        batch_size, num_timesteps, num_obs_var = amputed.shape

        amputed = torch.nan_to_num(amputed, nan=0.0).to(self.device)

        # Create and rearrange x to be the same shape as variables (x is timesteps)
        x = torch.arange(0, num_timesteps, device=self.device)
        x = x.repeat(batch_size)
        x = x.repeat(num_obs_var)
        x = x.reshape(batch_size, num_timesteps, num_obs_var)
        # Resulting size is [batch size, number of timesteps, number of observed variables]

        # Do a context/target split with mask - see CSDI implemnetation - line 56
        # https://github.com/ermongroup/CSDI/blob/main/main_model.py
        x_context, y_context, x_target, y_target = self._context_target_split(x, amputed, mask)

        # Get the predicted probability distribution
        p_y_pred, _, _ = self(x_context, y_context, x_target, y_target)

        # Sample K times to ensure that we select the best sample for gradient descent
        best_loss = self.loss(p_y_pred.rsample(), y_target)

        for _ in range(0, self.train_sample_times):
            loss = self.loss(p_y_pred.rsample(), y_target)
            if best_loss < loss:
                best_loss = loss

        self.log("train/loss", best_loss.item(), prog_bar=True)
        return best_loss

    # Override the validation step - needed for the custom loss calculation
    @torch.no_grad()
    def validation_step(self, batch, _):
        self.model.eval()
        # Unpack batch into three values
        amputed, mask, complete, complete_missingness_mask = batch
        batch_size, num_timesteps, num_obs_var = amputed.shape

        amputed = torch.nan_to_num(amputed, nan=0.0).to(self.device)

        # Create and rearrange x to be the same shape as variables (x is timesteps)
        x = torch.arange(0, num_timesteps, device=self.device)
        x = x.repeat(batch_size)
        x = x.repeat(num_obs_var)
        x = x.reshape(batch_size, num_timesteps, num_obs_var)
        # Resulting size is [batch size, number of timesteps, number of observed variables]

        # Do a context/target split with mask - see CSDI implemnetation - line 56
        # https://github.com/ermongroup/CSDI/blob/main/main_model.py
        x_context, y_context, x_target, y_target = self._context_target_split(x, amputed, mask)

        # Get the predicted probability distribution
        p_y_pred, _, _ = self(x_context, y_context, x_target, y_target)

        # Sample K times to ensure that we select the best sample for gradient descent
        best_loss = self.loss(p_y_pred.rsample(), y_target)

        for _ in range(0, self.val_sample_times):
            loss = self.loss(p_y_pred.rsample(), y_target)
            if best_loss < loss:
                best_loss = loss

        self.log("val/loss", best_loss.item(), prog_bar=True)

        # Do metric calculations - take x_target to be the full size now
        x_target = x

        # Get the predicted probability distribution
        p_y_pred = self(x_context, y_context, x_target)

        # Sample the distribution K times to put the values from it into the amputed dataset
        generated_list = []
        for _ in range(0, self.val_sample_times):
            generated = p_y_pred.sample()
            generated_list.append(generated)

        # Calculate mean of all K samples - dim = 0 is required to do a element-wise mean
        #   calculation on multidimensional tensor stack
        generated = torch.mean(torch.stack(generated_list), dim=0).to(self.device)
        # Use the indexing functionality of tensor to impute values into the indicies
        # specified by the mask
        amputed[mask > 0] = generated[mask > 0]
        amputed[complete_missingness_mask > 0] = complete[complete_missingness_mask > 0]

        # Update the metrics
        for metric in self.metrics["val"].values():
            metric.update(
                (
                    torch.flatten(amputed, start_dim=1),
                    torch.flatten(complete, start_dim=1),
                )
            )

    @torch.no_grad()
    def test_step(self, batch, _):
        self.model.eval()
        # Unpack batch into three values
        amputed, mask, complete, complete_missingness_mask = batch
        batch_size, num_timesteps, num_obs_var = amputed.shape

        # Create and rearrange x to be the same shape as variables (x is timesteps)
        x = torch.arange(0, num_timesteps, device=self.device)
        x = x.repeat(batch_size)
        x = x.repeat(num_obs_var)
        x = x.reshape(batch_size, num_timesteps, num_obs_var)
        # Resulting size is [batch size, number of timesteps, number of observed variables]

        # For now, do the most basic thing - put 0s instead of nans
        amputed = torch.nan_to_num(amputed, nan=0.0).to(self.device)

        x_context, y_context, _, _ = self._context_target_split(x, amputed, mask)

        x_target = x

        # Get the predicted probability distribution
        p_y_pred = self(x_context, y_context, x_target)

        # Sample the distribution K times to put the values from it into the amputed dataset
        generated_list = []
        for _ in range(0, self.test_sample_times):
            generated = p_y_pred.sample()
            generated_list.append(generated)

        # Calculate mean of all K samples - dim = 0 is required to do a element-wise mean
        #   calculation on multidimensional tensor stack
        generated = torch.mean(torch.stack(generated_list), dim=0).to(self.device)
        # Use the indexing functionality of tensor to impute values into the indicies
        # specified by the mask
        amputed[mask > 0] = generated[mask > 0]

        # In val/test loops, use the MSE loss - KL divergence can't be calculated
        # without target distribution
        loss = self.loss(amputed, complete)

        self.log("test/loss", loss.item(), prog_bar=True)

        amputed[complete_missingness_mask > 0] = complete[complete_missingness_mask > 0]
        # Update the metrics
        for metric in self.metrics["test"].values():
            metric.update(
                (
                    torch.flatten(amputed, start_dim=1),
                    torch.flatten(complete, start_dim=1),
                )
            )

    def predict(self, data):
        self.model.eval()

        data = data.to(self.device)
        batch_size, num_timesteps, num_obs_var = data.shape

        # Take an inverse of missingness mask for a mask of observed values
        observation_mask = ~(torch.isnan(data))

        # Create and rearrange x to be the same shape as variables (x is timesteps)
        x = torch.arange(0, num_timesteps, device=self.device)
        x = x.repeat(batch_size)
        x = x.repeat(num_obs_var)
        x = x.reshape(batch_size, num_timesteps, num_obs_var)

        x_context = x * observation_mask
        y_context = torch.nan_to_num(data, nan=0.0).to(self.device)

        x_target = x.to(self.device)

        p_y_pred = self(x_context, y_context, x_target)

        # Sample the distribution K times to put the values from it into the amputed dataset
        generated_list = []
        for _ in range(0, self.predict_sample_times):
            generated = p_y_pred.sample()
            generated_list.append(generated)

        # Calculate mean of all K samples - dim = 0 is required to do a element-wise mean calculation on
        #   multidimensional tensor stack
        generated = torch.mean(torch.stack(generated_list), dim=0).to(self.device)
        data[observation_mask == 0] = generated[observation_mask == 0]

        return data

    def _context_target_split(self, x, y, amputed_mask):
        # Take an inverse of the amputed mask - to get the observation mask
        observed_mask = (~(amputed_mask > 0)).float()

        # Generate a random tensor with the same dimensions as mask and multiply mask by it
        # This removes all missing values from the following calculations
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        # Create a context mask - the selection of the elements is so that only
        # 50% of all observed values are selected
        context_mask = (rand_for_mask > 0.5).float()
        # Create a target mask - the selection of the elements is so that all values
        # not selected by the context mask but are still observed are selected
        target_mask = (~(rand_for_mask > 0.5)).float() * observed_mask

        # Multiply x and y by masks to get the context/target split
        x_context = x * context_mask
        y_context = y * context_mask

        x_target = x * target_mask
        y_target = y * target_mask

        return x_context, y_context, x_target, y_target


# Actual class that implements neural processes
class NeuralProcess(nn.Module):
    """Class that implements neural processes."""

    def __init__(
        self,
        x_dim,
        y_dim,
        encoder_layers,
        encoder_h_dim,
        decoder_layers,
        decoder_h_dim,
        r_dim,
        z_dim,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.z_dim = z_dim

        # Initialize encoders/decoder
        self.encoder = MLPEncoder(x_dim, y_dim, encoder_h_dim, encoder_layers, r_dim)

        self.latent_encoder = MuEncoder(r_dim=r_dim, z_dim=z_dim)

        self.decoder = Decoder(decoder_h_dim, decoder_layers, x_dim, y_dim, z_dim)

    def forward(self, x_context, y_context, x_target, y_target=None):
        if y_target is not None:
            # Encode target and context (context needs to be encoded to
            # calculate kl term)
            mu_target, sigma_target = self._encode(x_target, y_target)
            mu_context, sigma_context = self._encode(x_context, y_context)
            # Sample from encoded distribution using reparameterization trick
            q_target = Normal(mu_target, sigma_target)
            q_context = Normal(mu_context, sigma_context)
            z_sample = q_target.rsample()
            # Get parameters of output distribution
            y_pred_mu, y_pred_sigma = self.decoder(x_target, z_sample)
            p_y_pred = Normal(y_pred_mu, y_pred_sigma)

            return p_y_pred, q_target, q_context
        else:
            # At testing time, encode only context
            mu_context, sigma_context = self._encode(x_context, y_context)
            # Sample from distribution based on context
            q_context = Normal(mu_context, sigma_context)
            z_sample = q_context.rsample()
            # Predict target points based on context
            y_pred_mu, y_pred_sigma = self.decoder(x_target, z_sample)
            p_y_pred = Normal(y_pred_mu, y_pred_sigma)

            return p_y_pred

    def _aggregate(self, r_i):
        return torch.mean(r_i, dim=1)

    def _encode(self, x, y):
        # Encode each point into a representation r_i
        r_i = self.encoder(x, y)
        # Aggregate representations r_i into a single representation r
        r = self._aggregate(r_i)
        # Return parameters of distribution
        return self.latent_encoder(r)


# This class describes the deterministic encoder
#   The encoding is (x_i, y_i) to representation r_i
class MLPEncoder(nn.Module):
    def __init__(self, x_dim, y_dim, h_dim, h_layers, r_dim):
        super().__init__()

        # Define the first input layer
        layers = [nn.Linear(x_dim + y_dim, h_dim), nn.ReLU(inplace=True)]
        # Define the multilayer structure
        for _ in range(h_layers):
            layers.append(nn.Linear(h_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
        # Add the final layer (without ReLU)
        layers.append(nn.Linear(h_dim, r_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        input_pairs = torch.cat((x, y), dim=2)
        return self.model(input_pairs)


# This class describes the latent encoder
#   The encoding is r_i to mu and sigma of the distribution from which to sample latent variable z
class MuEncoder(nn.Module):
    def __init__(self, r_dim, z_dim):
        super().__init__()

        self.model_hidden = nn.Linear(r_dim, r_dim)
        self.model_mu = nn.Linear(r_dim, z_dim)
        self.model_sigma = nn.Linear(r_dim, z_dim)

    def forward(self, r):
        hidden = torch.relu(self.model_hidden(r))
        mu = self.model_mu(hidden)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * torch.sigmoid(self.model_sigma(hidden))
        return mu, sigma


# This class describes the decoder
#   The encoding is from x_target and z to y_target (i.e. making a prediction of y)
class Decoder(nn.Module):
    def __init__(self, h_dim, h_layers, x_dim, y_dim, z_dim):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim

        layers = [nn.Linear(x_dim + z_dim, h_dim), nn.ReLU(inplace=True)]

        for _ in range(h_layers):
            layers.append(nn.Linear(h_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
        self.model_hidden = nn.Sequential(*layers)

        self.model_mu = nn.Linear(h_dim, y_dim)
        self.model_sigma = nn.Linear(h_dim, y_dim)

    def forward(self, x, z):
        batch_size, num_points, _ = x.size()
        # Repeat z, so it can be concatenated with every x. This changes shape
        # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
        z = z.unsqueeze(1).repeat(1, num_points, 1)
        # Flatten x and z to fit with linear layer
        x_flat = x.view(batch_size * num_points, self.x_dim)
        z_flat = z.view(batch_size * num_points, self.z_dim)
        # Input is concatenation of z with every row of x
        input_pairs = torch.cat((x_flat, z_flat), dim=1)
        hidden = self.model_hidden(input_pairs)
        mu = self.model_mu(hidden)
        pre_sigma = self.model_sigma(hidden)
        # Reshape output into expected shape
        mu = mu.view(batch_size, num_points, self.y_dim)
        pre_sigma = pre_sigma.view(batch_size, num_points, self.y_dim)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * F.softplus(pre_sigma)
        return mu, sigma
