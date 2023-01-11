import os
import random
import gin
import torch
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from skopt import gp_minimize
from sklearn.metrics import log_loss

from icu_benchmarks.data.loader import RICUDataset
from icu_benchmarks.data.preprocess import preprocess_data
from icu_benchmarks.hyperparameter_tuning import choose_and_bind_hyperparameters
from icu_benchmarks.models.train import train_common
from icu_benchmarks.models.wrappers import MLWrapper
from icu_benchmarks.run_utils import log_full_line


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
    model = MLWrapper()
    model.set_log_dir(log_dir)
    if (model_dir / "model.torch").is_file():
        model.load_weights(model_dir / "model.torch")
    elif (model_dir / "model.txt").is_file():
        model.load_weights(model_dir / "model.txt")
    elif (model_dir / "model.joblib").is_file():
        model.load_weights(model_dir / "model.joblib")
    else:
        raise Exception("No weights to load at path : {}".format(model_dir / "model.*"))
    return model.predict(dataset, None, None)


def get_predictions_for_all_models(
    data: dict[str, pd.DataFrame],
    log_dir: Path,
    source_dir: Path = None,
    seed: int = 1234,
    reproducible: bool = True,
    model: object = MLWrapper,
    weight: str = None,
    test_on: str = "test",
    target_model: object = None,
):
    """Common wrapper to train all benchmarked models.

    Args:
        data: Dict containing data to be trained on.
        log_dir: Path to directory where model output should be saved.
        source_dir: If set to load weights, path to directory containing trained weights.
        seed: Common seed used for any random operation.
        reproducible: If set to true, set torch to run reproducibly.
    """
    model = MLWrapper()

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

    test_dataset = RICUDataset(data, split=test_on)
    val_dataset = RICUDataset(data, split="val")
    # weight = test_dataset.get_balance()
    _, val_labels = val_dataset.get_data_and_labels()
    _, test_labels = test_dataset.get_data_and_labels()

    val_predictions = {}
    test_predictions = {}
    for model_dir in source_dir.iterdir():
        if model_dir.is_dir():
            val_predictions[model_dir.name] = get_predictions_for_single_model(model, val_dataset, model_dir, log_dir)
            test_predictions[model_dir.name] = get_predictions_for_single_model(model, test_dataset, model_dir, log_dir)
    val_predictions["target"] = target_model.output_transform(target_model.predict(val_dataset, None, None))
    test_predictions["target"] = target_model.output_transform(target_model.predict(test_dataset, None, None))

    return val_predictions, val_labels, test_predictions, test_labels


def get_model_metrics(test_predictions: np.ndarray, test_labels: np.ndarray):
    """Evaluate a combination of models.

    Args:
        test_predictions: Predictions for test set.
        test_labels: Labels for test set.
    """
    model = MLWrapper()
    model.set_metrics(test_labels)
    test_metric_results = {}
    for name, metric in model.metrics.items():
        value = metric(model.label_transform(test_labels), test_predictions)
        test_metric_results[name] = value
        # Only log float values
        if isinstance(value, np.float):
            logging.info("test {}: {}".format(name, value))
    return test_metric_results


def domain_adaptation(
    data_dir: Path,
    run_dir: Path,
    seed: int,
    n_initial_points: int = 10,
    n_calls: int = 50,
    debug: bool = False,
):
    """Choose hyperparameters to tune and bind them to gin.

    Args:
        data_dir: Path to the data directory.
        run_dir: Path to the log directory.
        seed: Random seed.
        n_initial_points: Number of initial points to explore.
        n_calls: Number of iterations to optimize the hyperparameters.
        folds_to_tune_on: Number of folds to tune on.
        debug: Whether to load less data and enable more logging.

    Raises:
        ValueError: If checkpoint is not None and the checkpoint does not exist.
    """

    # train target baselines
    
    
    agg_loss = 0
    cv_repetitions = 5
    cv_repetitions_to_train = 5
    cv_folds = 5
    cv_folds_to_train = 5
    datasets = ["hirid", "eicu", "aumc", "miiv"]
    weight_bounds = ((0.0001, 1.0) for _ in range(len(datasets)))
    task_dir = Path("../data/mortality24/")

    # evaluate models on same test split
    for dataset in datasets:
        log_full_line(f"STARTING {dataset}", char="#", num_newlines=2)
        choose_and_bind_hyperparameters(True, task_dir / dataset, run_dir, seed, debug=debug)
        for repetition in range(cv_repetitions_to_train):
            for fold_index in range(cv_folds_to_train):
                data = preprocess_data(
                    data_dir,
                    seed=seed,
                    debug=debug,
                    use_cache=True,
                    cv_repetitions=cv_repetitions,
                    repetition_index=repetition,
                    cv_folds=cv_folds,
                    fold_index=fold_index,
                )

                run_dir_seed = run_dir / f"seed_{seed}" / f"fold_{fold_index}"
                run_dir_seed.mkdir(parents=True, exist_ok=True)

                # evaluate target baselines
                curr_loss, target_model = train_common(
                    data,
                    log_dir=run_dir_seed,
                    seed=seed,
                    reproducible=True,
                    test_on="test",
                    return_model=True,
                )
                agg_loss += curr_loss

                val_predictions, val_labels, test_predictions, test_labels = get_predictions_for_all_models(
                    data,
                    run_dir,
                    source_dir=Path("../models/best_models/Mortality24/LGBMClassifier"),
                    seed=seed,
                    target_model=target_model,
                )

                # evaluate source baselines and oracle
                for source in datasets:
                    if source == dataset:
                        continue
                    logging.info("Evaluating model: {}".format(source))
                    get_model_metrics(test_predictions[source], test_labels)

                # evaluate convex combination of models
                val_predictions_wo_oracle = [pred for source, pred in val_predictions.items() if source != dataset]
                test_predictions_wo_oracle = [pred for source, pred in test_predictions.items() if source != dataset]
                def convex_model_combination(model_weights):
                    val_pred = np.average(val_predictions_wo_oracle, axis=0, weights=model_weights)
                    return log_loss(val_labels, val_pred)

                logging.disable(logging.INFO)
                res = gp_minimize(
                    convex_model_combination,
                    weight_bounds,
                    n_calls=n_calls,
                    n_initial_points=n_initial_points,
                    random_state=seed,
                    noise=1e-10,  # the models are deterministic, but noise is needed for the gp to work
                )
                logging.disable(logging.NOTSET)
                best_model_weights = res.x
                logging.info(best_model_weights)
                test_pred = np.average(test_predictions_wo_oracle, axis=0, weights=best_model_weights)
                get_model_metrics(test_pred, test_labels)


                log_full_line(f"FINISHED FOLD {fold_index}", level=logging.INFO)
            log_full_line(f"FINISHED CV REPETITION {repetition}", level=logging.INFO, char="=", num_newlines=3)
        log_full_line(f"EVALUATED {dataset}", char="#", num_newlines=10)

    return agg_loss / (cv_repetitions_to_train * cv_folds_to_train)
 