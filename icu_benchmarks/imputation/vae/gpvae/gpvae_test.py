from pypots.data.generating import gene_physionet2012
from pypots.utils.random import set_random_seed
from pypots.optim import Adam
from pypots.imputation import GPVAE
from pypots.utils.metrics import cal_mae

set_random_seed()

# Load the PhysioNet-2012 dataset
physionet2012_dataset = gene_physionet2012(artificially_missing_rate=0.1)


# Assemble the datasets for training, validating, and testing.

dataset_for_training = {
    "X": physionet2012_dataset['train_X'],
}

dataset_for_validating = {
    "X": physionet2012_dataset['val_X'],
    "X_intact": physionet2012_dataset['val_X_intact'],
    "indicating_mask": physionet2012_dataset['val_X_indicating_mask'],
}

dataset_for_testing = {
    "X": physionet2012_dataset['test_X'],
}



# initialize the model
gp_vae = GPVAE(
    n_steps=physionet2012_dataset['n_steps'],
    n_features=physionet2012_dataset['n_features'],
    latent_size=37,
    encoder_sizes=(128,128),
    decoder_sizes=(256,256),
    kernel="cauchy",
    beta=0.2,
    M=1,
    K=1,
    sigma=1.005,
    length_scale=7.0,
    kernel_scales=1,
    window_size=24,
    batch_size=32,
    # here we set epochs=10 for a quick demo, you can set it to 100 or more for better performance
    epochs=10,
    # here we set patience=3 to early stop the training if the evaluting loss doesn't decrease for 3 epoches.
    # You can leave it to defualt as None to disable early stopping.
    patience=3,
    # give the optimizer. Different from torch.optim.Optimizer, you don't have to specify model's parameters when
    # initializing pypots.optim.Optimizer. You can also leave it to default. It will initilize an Adam optimizer with lr=0.001.
    optimizer=Adam(lr=1e-3),
    # this num_workers argument is for torch.utils.data.Dataloader. It's the number of subprocesses to use for data loading.
    # Leaving it to default as 0 means data loading will be in the main process, i.e. there won't be subprocesses.
    # You can increase it to >1 if you think your dataloading is a bottleneck to your model training speed
    num_workers=0,
    # just leave it to default, PyPOTS will automatically assign the best device for you.
    # Set it to 'cpu' if you don't have CUDA devices. You can also set it to 'cuda:0' or 'cuda:1' if you have multiple CUDA devices.
    device='cuda:0',
    # set the path for saving tensorboard and trained model files
    saving_path="tutorial_results/imputation/gp_vae",
    # only save the best model after training finished.
    # You can also set it as "better" to save models performing better ever during training.
    model_saving_strategy="best",
)

# train the model on the training set, and validate it on the validating set to select the best model for testing in the next step
gp_vae.fit(train_set=dataset_for_training, val_set=dataset_for_validating)

# the testing stage, impute the originally-missing values and artificially-missing values in the test set
gp_vae_imputation = gp_vae.impute(dataset_for_testing)

# calculate mean absolute error on the ground truth (artificially-missing values)
testing_mae = cal_mae(gp_vae_imputation,
                      physionet2012_dataset['test_X_intact'], physionet2012_dataset['test_X_indicating_mask'])
print("Testing mean absolute error: %.4f" % testing_mae)