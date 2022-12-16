import json
from argparse import ArgumentParser, BooleanOptionalAction
from datetime import datetime
import gin
from pathlib import Path
import scipy.stats as stats
from statistics import mean, stdev

from icu_benchmarks.data.preprocess import preprocess_data
from icu_benchmarks.models.train import train_common
from icu_benchmarks.models.utils import JsonMetricsEncoder


def build_parser() -> ArgumentParser:
    """Builds an ArgumentParser for the command line.

    Returns:
        The configured ArgumentParser.
    """
    parser = ArgumentParser(description="Benchmark lib for processing and evaluation of deep learning models on ICU data")

    parent_parser = ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(title="Commands", dest="command", required=True)

    # ARGUMENTS FOR ALL COMMANDS
    general_args = parent_parser.add_argument_group("General arguments")
    general_args.add_argument("-d", "--data-dir", required=True, type=Path, help="Path to the parquet data directory.")
    general_args.add_argument("-n", "--name", required=True, help="Name of the (target) dataset.")
    general_args.add_argument("-t", "--task", default="BinaryClassification", help="Name of the task gin.")
    general_args.add_argument("-tn", "--task-name", help="Name of the task.")
    general_args.add_argument("-m", "--model", default="LGBMClassifier", help="Name of the model gin.")
    general_args.add_argument("-e", "--experiment", help="Name of the experiment gin.")
    general_args.add_argument("-l", "--log-dir", required=True, type=Path, help="Log directory with model weights.")
    general_args.add_argument(
        "-s", "--seeds", default=[1111], nargs="+", type=int, help="Random seed for processing, tuning and training."
    )
    general_args.add_argument("-db", "--debug", default=False, action=BooleanOptionalAction, help="Set to load less data.")
    general_args.add_argument("-c", "--cache", action=BooleanOptionalAction, help="Set to cache and use preprocessed data.")
    general_args.add_argument("-p", "--plot", action=BooleanOptionalAction, help="Generate common plots.")

    # MODEL TRAINING ARGUMENTS
    prep_and_train = subparsers.add_parser("train", help="Preprocess data and train model.", parents=[parent_parser])
    prep_and_train.add_argument("--reproducible", default=True, action=BooleanOptionalAction, help="Make torch reproducible.")
    prep_and_train.add_argument("--cpu", default=False, action=BooleanOptionalAction, help="Set to train on CPU.")
    prep_and_train.add_argument("-hp", "--hyperparams", nargs="+", help="Hyperparameters for model.")
    prep_and_train.add_argument("--tune", default=False, action=BooleanOptionalAction, help="Find best hyperparameters.")
    prep_and_train.add_argument("--checkpoint", type=Path, help="Use previous checkpoint.")

    # EVALUATION PARSER
    evaluate = subparsers.add_parser("evaluate", help="Evaluate trained model on data.", parents=[parent_parser])
    evaluate.add_argument("-sn", "--source-name", required=True, type=Path, help="Name of the source dataset.")
    evaluate.add_argument("--source-dir", required=True, type=Path, help="Directory containing gin and model weights.")

    return parser


def create_run_dir(log_dir: Path, randomly_searched_params: str = None) -> Path:
    """Creates a log directory with the current time as name.

    Also creates a file in the log directory, if any parameters were randomly searched.
    The filename contains the fixed hyperparameters to check against in future runs.

    Args:
        log_dir: Parent directory to create run directory in.
        randomly_searched_params: String representing the randomly searched params.

    Returns:
        Path to the created run log directory.
    """
    log_dir_run = log_dir / str(datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
    log_dir_run.mkdir(parents=True)
    if randomly_searched_params:
        (log_dir_run / randomly_searched_params).touch()
    return log_dir_run


@gin.configurable
def preprocess_and_train_for_folds(
    data_dir: Path,
    log_dir: Path,
    seed: int,
    load_weights: bool = False,
    source_dir: Path = None,
    num_folds: int = gin.REQUIRED,
    num_folds_to_train: int = None,
    reproducible: bool = True,
    debug: bool = False,
    use_cache: bool = False,
    test_on: str = "test",
) -> float:
    """Preprocesses data and trains a model for each fold.

    Args:
        data_dir: Path to the data directory.
        log_dir: Path to the log directory.
        seed: Random seed.
        load_weights: Whether to load weights from source_dir.
        source_dir: Path to the source directory.
        num_folds: Number of folds for preprocessing.
        num_folds_to_train: Number of folds to train on. If None, all folds are trained on.
        reproducible: Whether to make torch reproducible.
        debug: Whether to load less data and enable more logging.
        use_cache: Whether to cache and use cached data.
        test_on: Dataset to test on.

    Returns:
        The average loss of all folds.
    """
    if not num_folds_to_train:
        num_folds_to_train = num_folds
    agg_loss = 0
    for fold_index in range(num_folds_to_train):
        data = preprocess_data(
            data_dir, seed=seed, debug=debug, use_cache=use_cache, num_folds=num_folds, fold_index=fold_index
        )

        run_dir_seed = log_dir / f"seed_{seed}" / f"fold_{fold_index}"
        run_dir_seed.mkdir(parents=True, exist_ok=True)

        agg_loss += train_common(
            data,
            log_dir=run_dir_seed,
            load_weights=load_weights,
            source_dir=source_dir,
            seed=seed,
            reproducible=reproducible,
            test_on=test_on,
        )

    return agg_loss / num_folds


def aggregate_results(log_dir: Path):
    """Aggregates results from all folds and writes to JSON file.

    Args:
        log_dir: Path to the log directory.
    """
    aggregated = {}
    for seed in log_dir.iterdir():
        if seed.is_dir():
            aggregated[seed.name] = {}
            for fold in seed.iterdir():
                with open(fold / "test_metrics.json", "r") as f:
                    result = json.load(f)
                    aggregated[seed.name][fold.name] = result

    # Aggregate results per metric
    list_scores = {}
    for seed, folds in aggregated.items():
        for fold, result in folds.items():
            for metric, score in result.items():
                if isinstance(score, (float, int)):
                    list_scores[metric] = list_scores.setdefault(metric, [])
                    list_scores[metric].append(score)

    # Compute statistical metric over aggregated results
    averaged_scores = {metric: (mean(list)) for metric, list in list_scores.items()}
    std_scores = {metric: (stdev(list)) for metric, list in list_scores.items()}
    confidence_interval = {
        metric: (stats.t.interval(0.95, len(list) - 1, loc=mean(list), scale=stats.sem(list)))
        for metric, list in list_scores.items()
    }

    accumulated_metrics = {"avg": averaged_scores, "std": std_scores, "CI_0.95": confidence_interval}

    with open(log_dir / "aggregated_test_metrics.json", "w") as f:
        json.dump(aggregated, f, cls=JsonMetricsEncoder)

    with open(log_dir / "accumulated_test_metrics.json", "w") as f:
        json.dump(accumulated_metrics, f, cls=JsonMetricsEncoder)
