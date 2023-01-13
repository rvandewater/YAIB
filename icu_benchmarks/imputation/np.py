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
        
        # NOTE: set the manual optimization (as we need an optimizer for each model)
        self.automatic_optimization = False

        self.x_dim = 1
        self.y_dim = 1
        self.z_dim = z_dim

        model_list = []
        # NOTE: For each variable, initialize a separate Neural Process
        for i in range(input_size[2]):
            model_list.append(
                NeuralProcess(
                    self.x_dim, 
                    self.y_dim, 
                    encoder_layers, 
                    encoder_h_dim, 
                    decoder_layers, 
                    decoder_h_dim, 
                    r_dim, 
                    z_dim
                    )
                )     
        # Cast the list to ModuleList to properly register all NPs
        self.model = torch.nn.ModuleList(model_list)

    # NOTE: index is required to specify which variable is being processed
    def forward(self, index, x_context, y_context, x_target, y_target = None):
        return self.model[index](x_context, y_context, x_target, y_target)

    # Override the training step - needed for the custom loss calculation
    def training_step(self, batch, _):
        # Set models to training mode
        for model in self.model:
            model.training = True

        # Unpack batch into three values
        amputated, _, _ = batch
        batch_size, num_timesteps, num_obs_var = amputated.shape
               
        # Split batch across the last dimension (i.e., have 6 tensors of shape [batch size, number of timesteps, 1])       
        amputated = torch.split(amputated, split_size_or_sections = 1, dim = 2)
        
        # Create and rearrange x to be the same shape as variables (x is timesteps)
        batch_x = torch.arange(0, num_timesteps).to(self.device)
        batch_x = batch_x.repeat(batch_size)
        batch_x = batch_x.reshape(batch_size, num_timesteps, 1)
        # Resulting size is [batch size, number of timesteps, 1]

        optimizers = self.optimizers()

        loss = 0.
        epoch_loss = 0.
        # Take one variable at a time (from a list of tensors amputated)
        for index in range(num_obs_var):
            # Set optimizer zero_grad for the model
            optimizers[index].zero_grad()
            
            curr_var = amputated[index]
            for pt_id in range(batch_size):
                # Get the x and y for the batch
                x = batch_x[pt_id]
                y = curr_var[index]

                x_context, y_context, x_target, y_target =\
                    self._context_target_split(x, y)

                # Get the predicted probability distribution
                p_y_pred, q_context, q_target =\
                    self(index, x_context, y_context, x_target, y_target)

                loss = self._loss(p_y_pred, y_target, q_target, q_context)
                epoch_loss += loss.item()
                loss.backward()

            optimizers[index].step() 

        self.log("train/loss", epoch_loss, prog_bar = True)

    # Do a sanity check when training ends for any nans
    def on_train_end(self) -> None:
        try:
            for model in self.model:
                finite_checks.detect_nan_parameters(model)
        except ValueError:
            for model in self.model:
                finite_checks.print_nan_gradients(model)
            quit()
        return super().on_train_end()

    # Override the validation step - needed for the custom loss calculation
    @torch.no_grad()
    def validation_step(self, batch, _):
        # Unset training mode for each model
        for model in self.model:
            model.eval()

        # Unpack batch into three values
        amputated, amputation_mask, target = batch
        batch_size, num_timesteps, num_obs_var = amputated.shape
               
        # Split batch across the last dimension (i.e., have 6 tensors of shape [batch size, number of timesteps, 1])       
        amputated = torch.split(amputated, split_size_or_sections = 1, dim = 2)
        amputation_mask = torch.split(amputation_mask, split_size_or_sections = 1, dim = 2)
        
        # Create and rearrange x to be the same shape as variables (x is timesteps)
        batch_x = torch.arange(0, num_timesteps).to(self.device)
        batch_x = batch_x.repeat(batch_size)
        batch_x = batch_x.reshape(batch_size, num_timesteps, 1)
        # Resulting size is [batch size, number of timesteps, 1]

        # Create an empty tensor to keep values that we are imputing
        imputed_values = torch.Tensor([]).to(self.device)

        loss = 0.
        # Take one variable at a time (from a list of tensors amputated)
        for index in range(num_obs_var):
            curr_var = amputated[index]
            # Store mask for the current variable
            curr_mask = amputation_mask[index]
            for pt_id in range(batch_size):
                # Get the x and y for the batch
                x = batch_x[pt_id]
                y = curr_var[index]

                # Get current mask for the patient's variable and make a x_target out of it - i.e. what values we need to impute
                x_target = torch.nonzero(curr_mask[pt_id].squeeze()) 

                # Squeeze the target into a 1-d tensor
                x_target = x_target.squeeze()

                # If there are no values to impute, skip this variable
                if x_target.shape == torch.Size([0]):
                    pass
                # If there is only one value to impute, transfor the tensor into a 1D tensor
                elif x_target.ndim == 0:
                    x_target_copy = x_target.reshape(1, 1, 1)
                # Otherwise, reshape the x_target tensor into [1, number of values to impute, 1] for less code changes
                else:
                    x_target_copy = x_target.reshape(1, x_target.size(dim = 0), 1)
                
                # Make a split for context (target is ignored as we are making predictions)
                x_context, y_context, _, _ =\
                    self._context_target_split(x, y)

                # Get the predicted probability distribution
                p_y_pred = self(index, x_context, y_context, x_target_copy) 

                # Take probability distribution and create a sample
                y_hat = p_y_pred.sample()
                # Flatten the tensor to 1-D to concatenate correctly
                y_hat = y_hat.flatten()
                
                # Put the y_hat values into imputed values tensor to put it into places where values are missing in the amputated data 
                imputed_values = torch.cat([imputed_values, y_hat])
        
        # Concatenate back all tensors
        amputated = torch.cat(amputated, dim = 2)
        amputation_mask = torch.cat(amputation_mask, dim = 2)
        
        # Use masked_scatter_ to put imputed values into positions where items are missing
        # Before that, convert the mask into boolean so that masked_scatter_ would work
        amputation_mask = amputation_mask > 0
        imputed = amputated.masked_scatter_(mask = amputation_mask, source = imputed_values)

        # Calculate MSE loss over predictions
        loss = self.loss(imputed, target)

        # Report the loss for current batch
        self.log("val/loss", loss.item(), prog_bar = True)
        
        # Update the metrics
        for metric in self.metrics["val"].values():
            metric.update((torch.flatten(imputed, start_dim=1), torch.flatten(target, start_dim=1)))

    @torch.no_grad()
    def test_step(self, batch, _):
        # Unset training mode for each model
        for model in self.model:
            model.eval()

        # Unpack batch into three values
        amputated, amputation_mask, target = batch
        batch_size, num_timesteps, num_obs_var = amputated.shape
               
        # Split batch across the last dimension (i.e., have 6 tensors of shape [batch size, number of timesteps, 1])       
        amputated = torch.split(amputated, split_size_or_sections = 1, dim = 2)
        amputation_mask = torch.split(amputation_mask, split_size_or_sections = 1, dim = 2)
        
        # Create and rearrange x to be the same shape as variables (x is timesteps)
        batch_x = torch.arange(0, num_timesteps).to(self.device)
        batch_x = batch_x.repeat(batch_size)
        batch_x = batch_x.reshape(batch_size, num_timesteps, 1)
        # Resulting size is [batch size, number of timesteps, 1]

        # Create an empty tensor to keep values that we are imputing
        imputed_values = torch.Tensor([]).to(self.device)

        loss = 0.
        # Take one variable at a time (from a list of tensors amputated)
        for index in range(num_obs_var):
            curr_var = amputated[index]
            # Store mask for the current variable
            curr_mask = amputation_mask[index]
            for pt_id in range(batch_size):
                # Get the x and y for the batch
                x = batch_x[pt_id]
                y = curr_var[index]

                # Get current mask for the patient's variable and make a x_target out of it - i.e. what values we need to impute
                x_target = torch.nonzero(curr_mask[pt_id].squeeze()) 

                # Squeeze the target into a 1-d tensor
                x_target = x_target.squeeze()

                # If there are no values to impute, skip this variable
                if x_target.shape == torch.Size([0]):
                    pass
                # If there is only one value to impute, transfor the tensor into a 1D tensor
                elif x_target.ndim == 0:
                    x_target_copy = x_target.reshape(1, 1, 1)
                # Otherwise, reshape the x_target tensor into [1, number of values to impute, 1] for less code changes
                else:
                    x_target_copy = x_target.reshape(1, x_target.size(dim = 0), 1)
                
                # Make a split for context (target is ignored as we are making predictions)
                x_context, y_context, _, _ =\
                    self._context_target_split(x, y)

                # Get the predicted probability distribution
                p_y_pred = self(index, x_context, y_context, x_target_copy) 

                # Take probability distribution and create a sample
                y_hat = p_y_pred.sample()
                # Flatten the tensor to 1-D to concatenate correctly
                y_hat = y_hat.flatten()
                
                # Put the y_hat values into imputed values tensor to put it into places where values are missing in the amputated data 
                imputed_values = torch.cat([imputed_values, y_hat])
        
        # Concatenate back all tensors
        amputated = torch.cat(amputated, dim = 2)
        amputation_mask = torch.cat(amputation_mask, dim = 2)
        
        # Use masked_scatter_ to put imputed values into positions where items are missing
        # Before that, convert the mask into boolean so that masked_scatter_ would work
        amputation_mask = amputation_mask > 0
        imputed = amputated.masked_scatter_(mask = amputation_mask, source = imputed_values)

        # Calculate MSE loss over predictions
        loss = self.loss(imputed, target)

        # Report the loss for current batch
        self.log("test/loss", loss.item(), prog_bar = True)
        
        # Update the metrics
        for metric in self.metrics["test"].values():
            metric.update((torch.flatten(imputed, start_dim=1), torch.flatten(target, start_dim=1)))

    def predict_step(self, batch, _):
        raise NotImplementedError()

    # Override the configuration (separate optimizer is required for each model) 
    def configure_optimizers(self):
        optimizer = []
        for i in range(len(self.model)):
            optimizer.append(self.optimizer(self.model[i].parameters()))
        return optimizer

    # Loss function - with KL divergence
    def _loss(self, p_y_pred, y_target, q_context, q_target):
        log_likelihood = p_y_pred.log_prob(y_target).mean(dim = 0).sum()
        # KL Divergence
        kl = kl_divergence(q_target, q_context).mean(dim = 0).sum()
        return -log_likelihood + kl

    # TODO: think of a better way to do this split
    # Right now: 
    #   * we take each variable separately (for 6 variables there are 6 neural processes),
    #   * instead of taking a batch of patients and processing it as a batch for a given variable, we forloop over the entire batch,
    #   * for each patient's variable, there are 25 observations but some of them are NaN, so we remove them from x and y,
    #   * with the remaining x and y we do the context/target split 
    def _context_target_split(self, x, y):
        # We expect the size to be [number of timesteps, 1]
        x = torch.flatten(x)
        y = torch.flatten(y)

        # As both x and y are one-dimensional, we can look at positions where there is missingness 
        idx = x.isfinite() & y.isfinite()
        # Then take only those points from both x and y
        x = x[idx]
        y = y[idx]

        # Return them into shapes of [1, number of timesteps, 1] for less code changes
        x = x.unflatten(0, (1, x.shape[0], 1))
        y = y.unflatten(0, (1, y.shape[0], 1))

        # Calculate how many points we can have for context/target split
        num_points = x.shape[1]
        # If the number of points is not enough for a context/target split (at least 2 points), then pass None
        #   (this would skip the batch)
        if num_points < 2:
            return None, None, None, None
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

        # Set training to true
        self.training = True

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
