import os
import random
import gin
import torch
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from skopt import gp_minimize

from icu_benchmarks.data.loader import RICUDataset
from icu_benchmarks.models.wrappers import MLWrapper
from icu_benchmarks.models.utils import save_config_file


def get_predictions_for_single_model(model: MLWrapper, dataset: RICUDataset, model_dir: Path, log_dir: Path):
    """Get predictions for a single model.

    Args:
        model: Model to get predictions for.
        dataset: Dataset to get predictions for.
        model_dir: Path to directory where model weights are stored.
        log_dir: Path to directory where model output should be saved.

    Returns:
        Tuple of predictions and labels.
    """
    model.set_log_dir(log_dir)
    if (model_dir / "model.torch").is_file():
        model.load_weights(model_dir / "model.torch")
    elif (model_dir / "model.txt").is_file():
        model.load_weights(model_dir / "model.txt")
    elif (model_dir / "model.joblib").is_file():
        model.load_weights(model_dir / "model.joblib")
    else:
        raise Exception("No weights to load at path : {}".format(model_dir / "model.*"))
    return model.predict(dataset)


@gin.configurable("domain_adaptation")
def evaluate_model_combination(
    data: dict[str, pd.DataFrame],
    log_dir: Path,
    source_dir: Path = None,
    seed: int = 1234,
    reproducible: bool = True,
    model: object = MLWrapper,
    weight: str = None,
    test_on: str = "Test",

):
    """Common wrapper to train all benchmarked models.

    Args:
        data: Dict containing data to be trained on.
        log_dir: Path to directory where model output should be saved.
        source_dir: If set to load weights, path to directory containing trained weights.
        seed: Common seed used for any random operation.
        reproducible: If set to true, set torch to run reproducibly.
    """

    # Setting the seed before gin parsing
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if reproducible:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dataset = RICUDataset(data, split="train")
    test_dataset = RICUDataset(data, split=test_on)
    weight = dataset.get_balance()

    predictions = []
    for source_dataset in source_dir.iterdir():
        model_dir = source_dir / source_dataset
        predictions.append(get_predictions_for_single_model(model, dataset, model_dir, log_dir))

    test_pred = np.average(predictions, axis=0, weights=dataset_weights)

    # save config file again to capture missing gin parameters
    return log_loss(test_label, test_pred)


@gin.configurable("tune_hyperparameters")
def choose_and_bind_hyperparameters(
    data_dir: Path,
    log_dir: Path,
    seed: int,
    n_initial_points: int = 3,
    n_calls: int = 20,
    folds_to_tune_on: int = gin.REQUIRED,
    debug: bool = False,
):
    """Choose hyperparameters to tune and bind them to gin.

    Args:
        data_dir: Path to the data directory.
        log_dir: Path to the log directory.
        seed: Random seed.
        n_initial_points: Number of initial points to explore.
        n_calls: Number of iterations to optimize the hyperparameters.
        folds_to_tune_on: Number of folds to tune on.
        debug: Whether to load less data and enable more logging.

    Raises:
        ValueError: If checkpoint is not None and the checkpoint does not exist.
    """

    def convex_model_combination(hyperparams):
        return preprocess_and_train_for_folds(
            data_dir,
            Path(temp_dir),
            seed,
            num_folds_to_train=folds_to_tune_on,
            use_cache=True,
            test_on="val",
            debug=debug,
        )

    res = gp_minimize(
        bind_params_and_train,
        hyperparams_bounds,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        random_state=seed,
        noise=1e-10,  # the models are deterministic, but noise is needed for the gp to work
    )

    print(res)
