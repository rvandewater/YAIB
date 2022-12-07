# The following code is largerly (stolen) copied from: 
# https://github.com/EmilienDupont/neural-processes/

import torch
import gin
import logging

import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from numpy.random import choice

from icu_benchmarks.models.wrappers import ImputationWrapper

@gin.configurable("NP")
class NPImputation(ImputationWrapper):
    """A wrapper for the NeuralProcess class that allows for imputation.
    ImputationWrapper superclass is a subclass of LightningModule, so the
    override for the training_step function should also work.
    """
    needs_training = True
    needs_fit = False
    def __init__(
        self, 
        *args, 
        input_size, 
        encoder_layers,
        encoder_h_dim,
        decoder_layers, 
        decoder_h_dim,
        r_dim,
        z_dim,
        **kwargs
        ) -> None:
        super().__init__(*args, **kwargs)
        
        # TODO: make dynamic
        self.x_dim = 6
        self.y_dim = 6

        self.z_dim = z_dim

        # TODO: test whether this works? it seems to not need all points during training - which is weird?
        #   see https://github.com/EmilienDupont/neural-processes/blob/master/example-1d.ipynb
        self.num_context = 10
        self.num_extra_target = 10

        # Initialize the actual model
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

    def forward(self, x_context, y_context, x_target, y_target):
        return self.model(x_context, y_context, x_target, y_target)

    # Override the training step - needed for the custom loss calculation
    def training_step(self, batch, _):
        """
        TODO: make this work better - right now, the NaN values are 0s and 
        the model might not perform well. Need to figure out a way to only feed the model
        the data that is complete (i.e. so that it can construct a correct regression)
        """
        # NOTE: checking if the batch is passed as list 
        if isinstance(batch, list):
            logging.warning('The batch is passed as list of size {}'.format(len(batch)))
            batch = batch[0]
        # NOTE: check the batch is of correct shape    
        # The expected input is of shape (batch size, number of timesteps, number of observed variables)
        assert batch.shape == torch.Size([64, 25, 6])
        batch_size, num_timesteps, num_obs_var = batch.shape

        # Define x as timesteps and make it of shape [batch size, number of timesteps, number of observed variables]
        # TODO: figure out how to make input to be just timesteps
        x = torch.arange(0, num_timesteps)
        x = x.repeat(num_obs_var)
        x = x.repeat(batch_size)
        x = x.reshape(batch_size * num_timesteps, num_obs_var)
        # NOTE: right now, the y is the shape of [batch size, number of timesteps, number of observed variables]
        y = torch.nan_to_num(batch, nan=0.0)

        # Make a split for context and target (needed by the neural processes architecture)
        x_context, y_context, x_target, y_target =\
            self._context_target_split(x, y)

        # Get the predicted probability distribution
        p_y_pred, q_context, q_target =\
            self(x_context, y_context, x_target, y_target) 

        # Calculate loss
        loss = self._loss(p_y_pred, y_target, q_context, q_target)
        return loss

    def predict_step(self, batch, _):
        # NOTE: checking if the batch is passed as list 
        if isinstance(batch, list):
            logging.warning('The batch is passed as list of size {}'.format(len(batch)))
            batch = batch[0]
        # The expected input is of shape (batch size, number of timesteps, number of observed variables)
        batch_size, num_timesteps, num_obs_var = batch.shape

        # Define x as timesteps and make it of shape [batch size, number of timesteps, number of observed variables]
        # TODO: figure out how to make input to be just timesteps
        x = torch.arange(0, num_timesteps)
        x = x.repeat(num_obs_var)
        x = x.repeat(batch_size)
        x = x.reshape(batch_size * num_timesteps, num_obs_var)
        # NOTE: right now, the y is the shape of [batch size, number of timesteps, number of observed variables]
        y = torch.nan_to_num(batch, nan=0.0)

        # NOTE: set x_target as x
        x_target = x

        x_context, y_context, _, _ = self._context_target_split(x, y)

        p_y_pred = self.model(x_context, y_context, x_target)

        return p_y_pred.sample(x_target.shape)

    def validation_step(self, batch, _):
         # NOTE: checking if the batch is passed as list 
        if isinstance(batch, list):
            logging.warning('The batch is passed as list of size {}'.format(len(batch)))
            batch = batch[0]
        # The expected input is of shape (batch size, number of timesteps, number of observed variables)
        batch_size, num_timesteps, num_obs_var = batch.shape

        # Define x as timesteps and make it of shape [batch size, number of timesteps, number of observed variables]
        # TODO: figure out how to make input to be just timesteps
        x = torch.arange(0, num_timesteps)
        x = x.repeat(num_obs_var)
        x = x.repeat(batch_size)
        x = x.reshape(batch_size * num_timesteps, num_obs_var)
        # NOTE: right now, the y is the shape of [batch size, number of timesteps, number of observed variables]
        y = torch.nan_to_num(batch, nan=0.0)

        # Make a split for context and target (needed by the neural processes architecture)
        x_context, y_context, x_target, y_target =\
            self._context_target_split(x, y)

        # Get the predicted probability distribution
        p_y_pred, q_context, q_target =\
            self(x_context, y_context, x_target, y_target) 

        # Calculate loss
        loss = self._loss(p_y_pred, y_target, q_context, q_target)
        return loss

    # Loss function - with KL divergence
    def _loss(self, p_y_pred, y_target, q_context, q_target):
        log_likelihood = p_y_pred.log_prob(y_target).mean(dim = 0).sum()
        # KL Divergence
        kl = kl_divergence(q_target, q_context).mean(dim = 0).sum()
        return -log_likelihood + kl

    def _context_target_split(self, x, y):
        num_points = x.shape[1]
        # Sample locations of context and target points
        locations = choice(num_points,
                            size=self.num_context + self.num_extra_target,
                            replace=False)
        x_context = x[:, locations[:self.num_context], :]
        y_context = y[:, locations[:self.num_context], :]
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
        """
        Aggregates representations for every (x_i, y_i) pair into a single
        representation.
        Parameters
        ----------
        r_i : torch.Tensor
            Shape (batch_size, num_points, r_dim)
        """
        return torch.mean(r_i)

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
        input_pairs = torch.cat((x, y))
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
        
        # Note - models were changed to sequential (the original code was using torch.F namespace)
        layers_hidden = [
            nn.Linear(r_dim, r_dim),
            nn.ReLU(inplace = True)
            ]
        self.model_hidden = nn.Sequential(*layers_hidden)

        self.model_mu = nn.Linear(r_dim, z_dim)
        layers_sigma = [
            nn.Linear(r_dim, z_dim),
            nn.Sigmoid()
        ]
        self.model_sigma = nn.Sequential(*layers_sigma)
        
    def forward(self, r):
        hidden = self.model_hidden(r)

        mu = self.model_mu(hidden)
        sigma = 0.1 + 0.9 * self.model_sigma(hidden)

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

        layers_sigma = [
            nn.Linear(h_dim, y_dim),
            nn.Softplus()
        ]
        self.model_sigma = nn.Sequential(*layers_sigma)

    def forward(self, x, z):
        batch_size, num_points, _ = x.size()
        z = z.unsqueeze(1).repeat(1, num_points, 1)

        x_flat = x.view(batch_size * num_points, self.x_dim)
        z_flat = z.view(batch_size * num_points, self.z_dim)

        input_pairs = torch.cat((x_flat, z_flat), dim=1)
        hidden = self.model_hidden(input_pairs)

        mu = self.model_mu(hidden)
        pre_sigma = self.model_sigma(hidden)

        mu = mu.view(batch_size, num_points, self.y_dim)
        pre_sigma = pre_sigma.view(batch_size, num_points, self.y_dim)

        sigma = 0.1 + 0.9 * pre_sigma

        return mu, sigma
