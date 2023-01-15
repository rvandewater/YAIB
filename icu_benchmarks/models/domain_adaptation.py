import inspect
import json
import os
import random
import gin
import torch
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.stats as stats
from sklearn.metrics import log_loss, roc_auc_score

from icu_benchmarks.data.loader import RICUDataset
from icu_benchmarks.data.preprocess import preprocess_data
from icu_benchmarks.hyperparameter_tuning import choose_and_bind_hyperparameters
from icu_benchmarks.models.metric_constants import MLMetrics
from icu_benchmarks.models.train import train_common
from icu_benchmarks.models.wrappers import DLWrapper, MLWrapper
from icu_benchmarks.models.utils import JsonResultLoggingEncoder
from icu_benchmarks.run_utils import log_full_line


def load_model(model_dir: Path, log_dir: Path):
    """Load model from gin config."""
    gin.parse_config_file(model_dir / "train_config.gin")
    model_type = gin.query_parameter("train_common.model")
    if str(model_type) == "@DLWrapper()":
        model = DLWrapper()
    elif str(model_type) == "@MLWrapper()":
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
    return model


def get_predictions_for_single_model(dataset: RICUDataset, model_dir: Path, log_dir: Path):
    """Get predictions for a single model.

    Args:
        target_model: Model to get predictions for.
        dataset: Dataset to get predictions for.
        model_dir: Path to directory where model weights are stored.
        log_dir: Path to directory where model output should be saved.

    Returns:
        Tuple of predictions and labels.
    """
    model = load_model(model_dir, log_dir)
    return model.predict(dataset, None, None)


def calculate_metrics(predictions: np.ndarray, labels: np.ndarray):
    metric_results = {}
    for name, metric in MLMetrics.BINARY_CLASSIFICATION.items():
        value = metric(labels, predictions)
        metric_results[name] = value
        # Only log float values
        # if isinstance(value, np.float):
        #     logging.info("Test {}: {}".format(name, value))
    return metric_results


def get_predictions_for_all_models(
    target_model: object,
    data: dict[str, pd.DataFrame],
    log_dir: Path,
    source_dir: Path = None,
    seed: int = 1234,
    reproducible: bool = True,
    test_on: str = "test",
    source_datasets: list = None,
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

    test_dataset = RICUDataset(data, split=test_on)
    _, test_labels = test_dataset.get_data_and_labels()

    test_predictions = {}
    test_predictions["target"] = target_model.predict(test_dataset, None, None)
    for source in source_datasets:
        model_dir = source_dir / source
        test_predictions[model_dir.name] = get_predictions_for_single_model(test_dataset, model_dir, log_dir)

    for name, prediction in test_predictions.items():
        if isinstance(target_model, MLWrapper) and prediction.ndim == 2:
            test_predictions[name] = prediction[:, 1]

    return test_predictions, test_labels


def domain_adaptation(
    dataset: str,
    data_dir: Path,
    run_dir: Path,
    seed: int,
    task: str = None,
    model: str = None,
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
    cv_repetitions = 5
    cv_repetitions_to_train = 5
    cv_folds = 5
    cv_folds_to_train = 5
    target_sizes = [500, 1000, 2000]
    datasets = ["aumc", "eicu", "hirid", "miiv"]
    target_weights = [0.1, 0.2, 0.5, 1, 2, 5]
    # weights = [1] * (len(datasets) - 1)
    auc_functions = [
        lambda x: (x-0.5) ** 1,
        lambda x: (x-0.5) ** 2,
        lambda x: (x-0.5) ** 3,
        lambda x: (x-0.5) ** 4,
        lambda x: (x-0.5) ** 5,
        lambda x: ((2 ** (10*(x-0.5))) - 1),
        lambda x: ((3 ** (10*(x-0.5))) - 1),
    ]
    loss_functions = [
        lambda x: (1-x) ** 1,
        lambda x: (1-x) ** 2,
        lambda x: (1-x) ** 3,
        lambda x: (1-x) ** 4,
        lambda x: (1-x) ** 5,
        lambda x: ((2 ** (10*(1-x))) - 1),
        lambda x: ((3 ** (10*(1-x))) - 1),
    ]
    task_dir = data_dir / task
    model_path = Path("../yaib_models/best_models/")
    old_run_dir = Path("../DA_logs/")
    gin_config_before_tuning = gin.config_str()

    # evaluate models on same test split
    data_dir = task_dir / dataset
    source_datasets = [d for d in datasets if d != dataset]
    log_full_line(f"STARTING {dataset}", char="#", num_newlines=2)
    for target_size in target_sizes:
        gin.clear_config()
        gin.parse_config(gin_config_before_tuning)
        log_full_line(f"STARTING TARGET SIZE {target_size}", char="*", num_newlines=1)
        gin.bind_parameter("preprocess.fold_size", target_size)
        log_dir = run_dir / task / model / dataset / f"target_{target_size}"
        log_dir.mkdir(parents=True, exist_ok=True)
        # choose_and_bind_hyperparameters(False, data_dir, log_dir, seed, debug=debug)
        # gin_config_with_target_hyperparameters = gin.config_str()
        results = {}
        for repetition in range(cv_repetitions_to_train):
            for fold_index in range(cv_folds_to_train):
                # gin.parse_config(gin_config_with_target_hyperparameters)
                results[f"{repetition}_{fold_index}"] = {}
                fold_results = results[f"{repetition}_{fold_index}"]

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

                log_dir_fold = log_dir / f"cv_rep_{repetition}" / f"fold_{fold_index}"
                log_dir_fold.mkdir(parents=True, exist_ok=True)

                # train target model
                # target_model = train_common(data, log_dir=log_dir_fold, seed=seed, return_model=True)
                target_model = load_model(old_run_dir / task / model / dataset / f"target_{target_size}" / f"cv_rep_{repetition}" / f"fold_{fold_index}", log_dir_fold)
                
                # generate predictions and write to file if not already done
                if not (log_dir_fold / "val_predictions.json").exists():
                    val_predictions, val_labels = get_predictions_for_all_models(
                        target_model,
                        data,
                        log_dir_fold,
                        source_dir=model_path / task / model,
                        seed=seed,
                        source_datasets=source_datasets,
                        test_on="val",
                    )
                    with open(log_dir_fold / "val_predictions.json", "w") as f:
                        json.dump(val_predictions, f, cls=JsonResultLoggingEncoder)
                else:
                    with open(log_dir_fold / "val_predictions.json", "r") as f:
                        val_predictions = json.load(f)
                    _, val_labels = RICUDataset(data, split="val").get_data_and_labels()
                val_losses = {}
                val_aucs = {}
                val_losses["target"] = log_loss(val_labels, val_predictions["target"])
                val_aucs["target"] = roc_auc_score(val_labels, val_predictions["target"])
                for baseline, predictions in val_predictions.items():
                    val_losses[baseline] = log_loss(val_labels, predictions)
                    val_aucs[baseline] = roc_auc_score(val_labels, predictions)
                logging.info("Validation losses: %s", val_losses)

                # generate predictions and write to file if not already done
                if not (log_dir_fold / "test_predictions.json").exists():
                    test_predictions, test_labels = get_predictions_for_all_models(
                        target_model,
                        data,
                        log_dir_fold,
                        source_dir=model_path / task / model,
                        seed=seed,
                        source_datasets=source_datasets,
                    )
                    with open(log_dir_fold / "test_predictions.json", "w") as f:
                        json.dump(test_predictions, f, cls=JsonResultLoggingEncoder)
                else:
                    with open(log_dir_fold / "test_predictions.json", "r") as f:
                        test_predictions = json.load(f)
                    _, test_labels = RICUDataset(data, split="test").get_data_and_labels()


                for baseline, predictions in test_predictions.items():
                    # logging.info("Evaluating model: {}".format(baseline))
                    fold_results[baseline] = calculate_metrics(predictions, test_labels)
                # evaluate baselines

                # evaluate convex combination of models
                test_predictions_list = list(test_predictions.values())
                test_predictions_list_without_target = test_predictions_list[1:]

                # logging.info("Evaluating convex combination of models without target.")
                test_pred_without_target = np.average(test_predictions_list_without_target, axis=0, weights=[1,1,1])
                fold_results[f"convex_combination_without_target"] = calculate_metrics(test_pred_without_target, test_labels)

                # logging.info("Evaluating convex combination of models.")
                # for w in weights:
                #     # w =  weights + [t * sum(weights)]
                #     # logging.info(f"Evaluating target weight: {t}")
                #     logging.info(f"Evaluating weights: {w}")
                #     test_pred = np.average(test_predictions_list, axis=0, weights=w)
                #     fold_results[f"convex_combination_{w}"] = calculate_metrics(test_pred, test_labels)

                # find top three auc functions
                rated_auc_functions = []
                for f in auc_functions:
                    f_str = inspect.getsource(f).replace(" ", "")[:-2]
                    # logging.info(f"Evaluating convex combination of models with AUC function {f_str}.")
                    weights = [f(x) for x in val_aucs.values()]
                    # logging.info(f"weights: {weights}")
                    test_pred = np.average(test_predictions_list, axis=0, weights=weights)
                    fold_results[f"AUC_{f_str}"] = calculate_metrics(test_pred, test_labels)
                    rated_auc_functions.append((f_str, fold_results[f"AUC_{f_str}"]["AUC"]))
                rated_auc_functions.sort(key=lambda x: x[1], reverse=True)
                

                # find top three loss functions
                rated_loss_functions = []
                for f in loss_functions:
                    # strip whitespace
                    f_str = inspect.getsource(f).replace(" ", "")[:-2]
                    # logging.info(f"Evaluating convex combination of models with loss function {f_str}.")
                    weights = [f(x) for x in val_losses.values()]
                    # logging.info(f"losses: {val_losses.values()}")
                    # logging.info(f"weights: {weights}")
                    test_pred = np.average(test_predictions_list, axis=0, weights=weights)
                    fold_results[f"loss_{f_str}"] = calculate_metrics(test_pred, test_labels)
                    rated_loss_functions.append((f_str, fold_results[f"loss_{f_str}"]["AUC"]))
                rated_loss_functions.sort(key=lambda x: x[1], reverse=True)

                # logging.info(f"Top three AUC functions: {rated_auc_functions[:3]}")
                # logging.info(f"Top three loss functions: {rated_loss_functions[:3]}")

                log_full_line(f"FINISHED FOLD {fold_index}", level=logging.INFO)            
            # average results over folds
            agg_aucs = {}
            for fold_results in results.values():
                for source, metrics in fold_results.items():
                    agg_aucs.setdefault(source, []).append(metrics["AUC"])

            avg_aucs = {}
            for source, aucs in agg_aucs.items():
                avg_aucs[source] = np.mean(aucs)

            # print baselines first, then top three AUC, then top three loss
            for source, auc in avg_aucs.items():
                if source in ["target", "convex_combination_without_target"] + datasets:
                    logging.info(f"{source}: {auc}")
            avg_aucs_list = sorted(avg_aucs.items(), key=lambda x: x[1], reverse=True)
            i = 0
            for source, auc in avg_aucs_list:
                if "AUC" in source:
                    i += 1
                    logging.info(f"{source}: {auc}")
                    if i == 3:
                        break
            i = 0
            for source, auc in avg_aucs_list:
                if "loss" in source:
                    i += 1
                    logging.info(f"{source}: {auc}")
                    if i == 3:
                        break
            log_full_line(f"FINISHED CV REPETITION {repetition}", level=logging.INFO, char="=", num_newlines=3)

        source_metrics = {}
        for result in results.values():
            for source, source_stats in result.items():
                for metric, score in source_stats.items():
                    if isinstance(score, (float, int)):
                        source_metrics.setdefault(source, {}).setdefault(metric, []).append(score)

        # Compute statistical metric over aggregated results
        averaged_metrics = {}
        for source, source_stats in source_metrics.items():
            for metric, scores in source_stats.items():
                averaged_metrics.setdefault(source, {}).setdefault(metric, []).append({
                    "avg": np.mean(scores),
                    "std": np.std(scores),
                    "CI_0.95": stats.t.interval(0.95, len(scores) - 1, loc=np.mean(scores), scale=stats.sem(scores)),
                })

        with open(log_dir / "aggregated_source_metrics.json", "w") as f:
            json.dump(results, f, cls=JsonResultLoggingEncoder)

        with open(log_dir / "averaged_source_metrics.json", "w") as f:
            json.dump(averaged_metrics, f, cls=JsonResultLoggingEncoder)

        logging.info(f"Averaged results: {averaged_metrics}")
        log_full_line(f"EVALUATED TARGET SIZE {target_size}", char="*", num_newlines=5)

    log_full_line(f"EVALUATED {dataset}", char="#", num_newlines=5)
