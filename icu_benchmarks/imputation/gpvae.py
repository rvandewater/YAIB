import gin

from icu_benchmarks.models.wrappers import ImputationWrapper
from pypots.imputation import GPVAE
import torch
from pypots.optim import Adam


@gin.configurable("GPVAE")
class GPVAEImputation(ImputationWrapper):
    """Gaussian Process Variational Autoencoder (GPVAE) imputation using PyPots pacakge."""
    requires_backprop = True
    def __init__(self, input_size, epochs, optimizer, encoder_sizes, decoder_sizes, kernel, latent_size, batch_size, *args, **kwargs):
        super().__init__(
            *args, input_size=input_size, epochs=epochs, batch_size=batch_size, **kwargs
        )

        self.imputer = GPVAE(
            n_steps=input_size[1],
            n_features=input_size[2],
            encoder_sizes=encoder_sizes,
            decoder_sizes=decoder_sizes,
            latent_size=latent_size,
            kernel=kernel,
            batch_size=batch_size,
            epochs=epochs,
            optimizer=Adam(lr=0.02)
        )

    def fit(self, train_dataset, val_dataset):
        self.imputer.fit(
            torch.Tensor(
                train_dataset.amputated_values.values.reshape(-1, train_dataset.maxlen, train_dataset.features_df.shape[1])
            )
        )

    def forward(self, amputated_values, amputation_mask):
        debatched_values = amputated_values.to(self.imputer.device).squeeze()
        self.imputer.model = self.imputer.model.to(self.imputer.device)
        output = torch.Tensor(self.imputer.impute(debatched_values)).to(self.device)

        output = output.reshape(amputated_values.shape)
        return output
