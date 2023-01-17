# The following code is largerly (stolen) copied from: 
# https://github.com/EmilienDupont/neural-processes/

import math

import torch
import gin
import logging

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from pytorch_lightning.utilities import finite_checks

from numpy.random import permutation

from icu_benchmarks.models.wrappers import ImputationWrapper

@gin.configurable("NP")
class NPImputation(ImputationWrapper):
    """A wrapper for the NeuralProcess class that allows for imputation.
    """
    needs_training = True
    needs_fit = False
    def __init__(
        self, 
        input_size, 
        encoder_layers,
        encoder_h_dim,
        decoder_layers, 
        decoder_h_dim,
        r_dim,
        z_dim,
        *args, 
        **kwargs
        ) -> None:
        super().__init__(*args, input_size, encoder_layers, encoder_h_dim, decoder_layers,  decoder_h_dim, r_dim, z_dim, **kwargs)

        self.x_dim = input_size[2]
        self.y_dim = input_size[2]
        self.z_dim = z_dim

        self.model = NeuralProcess(
                    self.x_dim, 
                    self.y_dim, 
                    encoder_layers, 
                    encoder_h_dim, 
                    decoder_layers, 
                    decoder_h_dim, 
                    r_dim, 
                    z_dim
                )
        
    def forward(self, x_context, y_context, x_target, y_target = None):
        return self.model(x_context, y_context, x_target, y_target)

    # Override the training step - needed for the custom loss calculation
    def training_step(self, batch, _):
        self.model.train(True)

        # Unpack batch into three values
        _, _, complete = batch
        batch_size, num_timesteps, num_obs_var = complete.shape

        # Create and rearrange x to be the same shape as variables (x is timesteps)
        x = torch.arange(0, num_timesteps).to(self.device)
        x = x.repeat(batch_size)
        x = x.repeat(num_obs_var)
        x = x.reshape(batch_size, num_timesteps, num_obs_var)
        # Resulting size is [batch size, number of timesteps, number of observed variables]

        # For now, do the most basic thing - train on complete data
        x_context, y_context, x_target, y_target =\
            self._context_target_split(x, complete)

        # Get the predicted probability distribution
        p_y_pred, q_context, q_target =\
            self(x_context, y_context, x_target, y_target)

        loss = self._loss(p_y_pred, y_target, q_target, q_context)

        self.log("train/loss", loss.item(), prog_bar = True)
        return loss

    # Do a sanity check when training ends for any nans
    def on_train_end(self) -> None:
        try:
            finite_checks.detect_nan_parameters(self.model)
        except ValueError:
            logging.error('Found nan values in the model gradients:')
            finite_checks.print_nan_gradients(self.model)
            quit()
        return super().on_train_end()

    # Override the validation step - needed for the custom loss calculation
    @torch.no_grad()
    def validation_step(self, batch, _):
        self.model.eval()
        # Unpack batch into three values
        amputed, mask, complete = batch
        batch_size, num_timesteps, num_obs_var = complete.shape
        
        # Create and rearrange x to be the same shape as variables (x is timesteps)
        x = torch.arange(0, num_timesteps).to(self.device)
        x = x.repeat(batch_size)
        x = x.repeat(num_obs_var)
        x = x.reshape(batch_size, num_timesteps, num_obs_var)
        # Resulting size is [batch size, number of timesteps, number of observed variables]

        # For now, do the most basic thing - put 0s instead of nans
        amputed = torch.nan_to_num(amputed, nan = 0.0).to(self.device)

        x_context, y_context, _, _ =\
            self._context_target_split(x, amputed)

        x_target = x.to(self.device)

        # Get the predicted probability distribution
        p_y_pred =\
            self(x_context, y_context, x_target)

        # Sample the distribution to put the values from it into the amputed dataset
        generated = p_y_pred.sample()
        # Use the indexing functionality of tensor to impute values into the indicies specified by the mask
        amputed[mask > 0] = generated[mask > 0]

        # In val/test loops, use the MSE loss - KL divergence can't be calculated without target distribution
        loss = self.loss(amputed, complete)

        self.log("val/loss", loss.item(), prog_bar = True)

        # Update the metrics
        for metric in self.metrics["val"].values():
            metric.update((torch.flatten(amputed, start_dim=1), torch.flatten(complete, start_dim=1)))

    @torch.no_grad()
    def test_step(self, batch, _):
        self.model.eval()
        # Unpack batch into three values
        amputed, mask, complete = batch
        batch_size, num_timesteps, num_obs_var = amputed.shape
        
        # Create and rearrange x to be the same shape as variables (x is timesteps)
        x = torch.arange(0, num_timesteps).to(self.device)
        x = x.repeat(batch_size)
        x = x.repeat(num_obs_var)
        x = x.reshape(batch_size, num_timesteps, num_obs_var)
        # Resulting size is [batch size, number of timesteps, number of observed variables]
        
        # For now, do the most basic thing - put 0s instead of nans
        amputed = torch.nan_to_num(amputed, nan = 0.0).to(self.device)

        x_context, y_context, _, _ =\
            self._context_target_split(x, amputed)

        x_target = x.to(self.device)

        # Get the predicted probability distribution
        p_y_pred =\
            self(x_context, y_context, x_target)

        # Sample the distribution to put the values from it into the amputed dataset
        generated = p_y_pred.sample()
        # Use the indexing functionality of tensor to impute values into the indicies specified by the mask
        amputed[mask > 0] = generated[mask > 0]

        # In val/test loops, use the MSE loss - KL divergence can't be calculated without target distribution
        loss = self.loss(amputed, complete)

        self.log("test/loss", loss.item(), prog_bar = True)

        # Update the metrics
        for metric in self.metrics["test"].values():
            metric.update((torch.flatten(amputed, start_dim=1), torch.flatten(complete, start_dim=1)))

    def predict_step(self, batch, _):
        raise NotImplementedError()

    # Loss function - with KL divergence
    def _loss(self, p_y_pred, y_target, q_context, q_target):
        log_likelihood = p_y_pred.log_prob(y_target).mean(dim = 0).sum()
        # KL Divergence
        kl = kl_divergence(q_target, q_context).mean(dim = 0).sum()
        return -log_likelihood + kl
 
    def _context_target_split(self, x, y):
        # Calculate how many points we can have for context/target split
        num_points = x.shape[1]
        num_context = math.floor(num_points / 2)
        # Randomly rearrange the indexes (so that the split is not on the same locations every time)
        locations = permutation(num_points)
        # Sample locations of context and target points
        x_context = x[:, locations[:num_context], :]
        y_context = y[:, locations[:num_context], :]
        x_target = x[:, locations, :]
        y_target = y[:, locations, :]

        return x_context, y_context, x_target, y_target

# Actual class that implements neural processes
class NeuralProcess(nn.Module):
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
        self.encoder = MLPEncoder(
            x_dim,
            y_dim,
            encoder_h_dim,
            encoder_layers,
            r_dim
        )

        self.latent_encoder = MuEncoder(
            r_dim = r_dim,
            z_dim = z_dim
        )

        self.decoder = Decoder(
            decoder_h_dim,
            decoder_layers,
            x_dim,
            y_dim,
            z_dim
        )

    def forward(self, x_context, y_context, x_target, y_target = None):
        if self.training:
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
        return torch.mean(r_i, dim = 1)

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
    def __init__(
        self,
        x_dim,
        y_dim,
        h_dim,
        h_layers,
        r_dim
    ):
        super().__init__()
        
        # Define the first input layer
        layers = [
            nn.Linear(x_dim + y_dim, h_dim),
            nn.ReLU(inplace = True)
        ]       
        # Define the multilayer structure
        for _ in range(h_layers):
            layers.append(nn.Linear(h_dim, h_dim))
            layers.append(nn.ReLU(inplace = True))
        # Add the final layer (without ReLU)
        layers.append(nn.Linear(h_dim, r_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        input_pairs = torch.cat((x, y), dim = 2)
        return self.model(input_pairs)

# This class describes the latent encoder 
#   The encoding is r_i to mu and sigma of the distribution from which to sample latent variable z
class MuEncoder(nn.Module):
    def __init__(
        self,
        r_dim,
        z_dim
    ):
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
    def __init__(
        self,
        h_dim,
        h_layers,
        x_dim,
        y_dim,
        z_dim
        ):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim

        layers = [
            nn.Linear(x_dim + z_dim, h_dim),
            nn.ReLU(inplace = True)
        ]

        for _ in range(h_layers):
            layers.append(nn.Linear(h_dim, h_dim))
            layers.append(nn.ReLU(inplace = True))
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
