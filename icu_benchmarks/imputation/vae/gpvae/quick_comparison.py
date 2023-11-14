import logging

from pypots.imputation import SAITS, BRITS, LOCF, USGAN, MRNN
from pypots.data.generating import gene_physionet2012
from pypots.utils.random import set_random_seed
from pypots.optim import Adam
from pypots.imputation import GPVAE
from pypots.utils.metrics import cal_mae, cal_rmse
import argparse

parser = argparse.ArgumentParser("Model")
parser.add_argument("Model", help="Model to test", type=str, default="gpvae")
args = parser.parse_args()
print(f"Testing {args.Model}")
args = parser.parse_args()

set_random_seed(1111)

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

epochs = 1000
patience = 10

if (args.Model == "saits"):
    # initialize the model
    model = SAITS(
        n_steps=physionet2012_dataset['n_steps'],
        n_features=physionet2012_dataset['n_features'],
        n_layers=2,
        d_model=256,
        d_inner=128,
        n_heads=4,
        d_k=64,
        d_v=64,
        dropout=0.1,
        attn_dropout=0.1,
        diagonal_attention_mask=True,  # otherwise the original self-attention mechanism will be applied
        ORT_weight=1,  # you can adjust the weight values of arguments ORT_weight
        # and MIT_weight to make the SAITS model focus more on one task. Usually you can just leave them to the default values, i.e. 1.
        MIT_weight=1,
        batch_size=32,
        # here we set epochs=10 for a quick demo, you can set it to 100 or more for better performance
        epochs=epochs,
        # here we set patience=3 to early stop the training if the evaluting loss doesn't decrease for 3 epoches.
        # You can leave it to defualt as None to disable early stopping.
        patience=patience,
        # give the optimizer. Different from torch.optim.Optimizer, you don't have to specify model's parameters when
        # initializing pypots.optim.Optimizer. You can also leave it to default. It will initilize an Adam optimizer with lr=0.001.
        optimizer=Adam(lr=1e-3),
        # this num_workers argument is for torch.utils.data.Dataloader. It's the number of subprocesses to use for data loading.
        # Leaving it to default as 0 means data loading will be in the main process, i.e. there won't be subprocesses.
        # You can increase it to >1 if you think your dataloading is a bottleneck to your model training speed
        num_workers=0,
        # Set it to None to use the default device (will use CPU if you don't have CUDA devices).
        # You can also set it to 'cpu' or 'cuda' explicitly, or ['cuda:0', 'cuda:1'] if you have multiple CUDA devices.
        device=None,
        # set the path for saving tensorboard and trained model files
        saving_path="tutorial_results/imputation/saits",
        # only save the best model after training finished.
        # You can also set it as "better" to save models performing better ever during training.
        model_saving_strategy="best",
    )
elif (args.Model == "gpvae"):
    # initialize the model
    model = GPVAE(
        n_steps=physionet2012_dataset['n_steps'],
        n_features=physionet2012_dataset['n_features'],
        latent_size=35,  # 37,
        encoder_sizes=(128, 128),
        decoder_sizes=(256, 256),
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
        epochs=epochs,
        # here we set patience=3 to early stop the training if the evaluting loss doesn't decrease for 3 epoches.
        # You can leave it to defualt as None to disable early stopping.
        patience=patience,
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
elif (args.Model == "brits"):
    # initialize the model
    model = BRITS(
        n_steps=physionet2012_dataset['n_steps'],
        n_features=physionet2012_dataset['n_features'],
        rnn_hidden_size=128,
        batch_size=32,
        # here we set epochs=10 for a quick demo, you can set it to 100 or more for better performance
        epochs=epochs,
        # here we set patience=3 to early stop the training if the evaluting loss doesn't decrease for 3 epoches.
        # You can leave it to defualt as None to disable early stopping.
        patience=patience,
        # give the optimizer. Different from torch.optim.Optimizer, you don't have to specify model's parameters when
        # initializing pypots.optim.Optimizer. You can also leave it to default. It will initilize an Adam optimizer with lr=0.001.
        optimizer=Adam(lr=1e-3),
        # this num_workers argument is for torch.utils.data.Dataloader. It's the number of subprocesses to use for data loading.
        # Leaving it to default as 0 means data loading will be in the main process, i.e. there won't be subprocesses.
        # You can increase it to >1 if you think your dataloading is a bottleneck to your model training speed
        num_workers=0,
        # Set it to None to use the default device (will use CPU if you don't have CUDA devices).
        # You can also set it to 'cpu' or 'cuda' explicitly, or ['cuda:0', 'cuda:1'] if you have multiple CUDA devices.
        device=None,
        # set the path for saving tensorboard and trained model files
        saving_path="tutorial_results/imputation/brits",
        # only save the best model after training finished.
        # You can also set it as "better" to save models performing better ever during training.
        model_saving_strategy="best",
    )
elif (args.Model == "locf"):
    model = LOCF(
        nan=0
        # set the value used to impute data missing at the beginning of the sequence, those cannot use LOCF mechanism to impute
    )
elif (args.Model == "usgan"):
    # initialize the model
    model = USGAN(
        n_steps=physionet2012_dataset['n_steps'],
        n_features=physionet2012_dataset['n_features'],
        rnn_hidden_size=256,
        lambda_mse=1,
        dropout_rate=0.1,
        G_steps=1,
        D_steps=1,
        batch_size=32,
        # here we set epochs=10 for a quick demo, you can set it to 100 or more for better performance
        epochs=epochs,
        # here we set patience=3 to early stop the training if the evaluting loss doesn't decrease for 3 epoches.
        # You can leave it to defualt as None to disable early stopping.
        patience=patience,
        # give the optimizer. Different from torch.optim.Optimizer, you don't have to specify model's parameters when
        # initializing pypots.optim.Optimizer. You can also leave it to default. It will initilize an Adam optimizer with lr=0.001.
        G_optimizer=Adam(lr=1e-3),
        D_optimizer=Adam(lr=1e-3),
        # this num_workers argument is for torch.utils.data.Dataloader. It's the number of subprocesses to use for data loading.
        # Leaving it to default as 0 means data loading will be in the main process, i.e. there won't be subprocesses.
        # You can increase it to >1 if you think your dataloading is a bottleneck to your model training speed
        num_workers=0,
        # just leave it to default, PyPOTS will automatically assign the best device for you.
        # Set it to 'cpu' if you don't have CUDA devices. You can also set it to 'cuda:0' or 'cuda:1' if you have multiple CUDA devices.
        device='cuda:0',
        # set the path for saving tensorboard and trained model files
        saving_path="tutorial_results/imputation/us_gan",
        # only save the best model after training finished.
        # You can also set it as "better" to save models performing better ever during training.
        model_saving_strategy="best",
    )
elif (args.Model == "rnn"):
    model = mrnn = MRNN(
        n_steps=physionet2012_dataset['n_steps'],
        n_features=physionet2012_dataset['n_features'],
        rnn_hidden_size=128,
        batch_size=32,
        # here we set epochs=10 for a quick demo, you can set it to 100 or more for better performance
        epochs=epochs,
        # here we set patience=3 to early stop the training if the evaluting loss doesn't decrease for 3 epoches.
        # You can leave it to defualt as None to disable early stopping.
        patience=patience,
        # give the optimizer. Different from torch.optim.Optimizer, you don't have to specify model's parameters when
        # initializing pypots.optim.Optimizer. You can also leave it to default. It will initilize an Adam optimizer with lr=0.001.
        optimizer=Adam(lr=1e-3),
        # this num_workers argument is for torch.utils.data.Dataloader. It's the number of subprocesses to use for data loading.
        # Leaving it to default as 0 means data loading will be in the main process, i.e. there won't be subprocesses.
        # You can increase it to >1 if you think your dataloading is a bottleneck to your model training speed
        num_workers=0,
        # Set it to None to use the default device (will use CPU if you don't have CUDA devices).
        # You can also set it to 'cpu' or 'cuda' explicitly, or ['cuda:0', 'cuda:1'] if you have multiple CUDA devices.
        device=None,
        # set the path for saving tensorboard and trained model files
        saving_path="tutorial_results/imputation/mrnn",
        # only save the best model after training finished.
        # You can also set it as "better" to save models performing better ever during training.
        model_saving_strategy="best", )
else:
    print("Model not found")
    exit()

# train the model on the training set, and validate it on the validating set to select the best model for testing in the next step
model.fit(train_set=dataset_for_training, val_set=dataset_for_validating)

# the testing stage, impute the originally-missing values and artificially-missing values in the test set
imputation = model.impute(dataset_for_testing)

# calculate mean absolute error on the ground truth (artificially-missing values)
testing_mae = cal_mae(imputation, physionet2012_dataset['test_X_intact'], physionet2012_dataset['test_X_indicating_mask'])
testing_rmse = cal_rmse(imputation, physionet2012_dataset['test_X_intact'], physionet2012_dataset['test_X_indicating_mask'])
print("Testing mean absolute error: %.4f" % testing_mae)
print("Testing rmse: %.4f" % testing_rmse)
