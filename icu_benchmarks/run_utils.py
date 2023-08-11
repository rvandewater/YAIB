import warnings
from math import sqrt

import torch
import json
from argparse import ArgumentParser, BooleanOptionalAction
from datetime import datetime, timedelta
import logging
from pathlib import Path
import scipy.stats as stats
import shutil
from statistics import mean, pstdev
from icu_benchmarks.models.utils import JsonResultLoggingEncoder
from icu_benchmarks.wandb_utils import wandb_log


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
    general_args.add_argument("-t", "--task", default="BinaryClassification", required=True, help="Name of the task gin.")
    general_args.add_argument("-n", "--name", required=False, help="Name of the (target) dataset.")
    general_args.add_argument("-tn", "--task-name", required=False, help="Name of the task, used for naming experiments.")
    general_args.add_argument("-m", "--model", default="LGBMClassifier", required=False, help="Name of the model gin.")
    general_args.add_argument("-e", "--experiment", required=False, help="Name of the experiment gin.")
    general_args.add_argument(
        "-l", "--log-dir", required=False, default=Path("../yaib_logs/"), type=Path, help="Log directory with model weights."
    )
    general_args.add_argument(
        "-s", "--seed", required=False, default=1234, type=int, help="Random seed for processing, tuning and training."
    )
    general_args.add_argument(
        "-v",
        "--verbose",
        default=False,
        required=False,
        action=BooleanOptionalAction,
        help="Whether to use verbose logging. Disable for clean logs.",
    )
    general_args.add_argument("--cpu", default=False, required=False, action=BooleanOptionalAction, help="Set to use CPU.")
    general_args.add_argument(
        "-db", "--debug", required=False, default=False, action=BooleanOptionalAction, help="Set to load less data."
    )
    general_args.add_argument(
        "-lc",
        "--load_cache",
        required=False,
        default=False,
        action=BooleanOptionalAction,
        help="Set to load generated data cache.",
    )
    general_args.add_argument(
        "-gc",
        "--generate_cache",
        required=False,
        default=False,
        action=BooleanOptionalAction,
        help="Set to generate data cache.",
    )
    general_args.add_argument("-p", "--preprocessor", required=False, type=Path, help="Load custom preprocessor from file.")
    general_args.add_argument("-pl", "--plot", required=False, action=BooleanOptionalAction, help="Generate common plots.")
    general_args.add_argument(
        "-wd", "--wandb-sweep", required=False, action="store_true", help="Activates wandb hyper parameter sweep."
    )
    general_args.add_argument(
        "-imp", "--pretrained-imputation", required=False, type=str, help="Path to pretrained imputation model."
    )

    # MODEL TRAINING ARGUMENTS
    prep_and_train = subparsers.add_parser("train", help="Preprocess features and train model.", parents=[parent_parser])
    prep_and_train.add_argument(
        "--reproducible", required=False, default=True, action=BooleanOptionalAction, help="Make torch reproducible."
    )
    prep_and_train.add_argument("-hp", "--hyperparams", required=False, nargs="+", help="Hyperparameters for model.")
    prep_and_train.add_argument("--tune", default=False, action=BooleanOptionalAction, help="Find best hyperparameters.")
    prep_and_train.add_argument("--checkpoint", required=False, type=Path, help="Use previous checkpoint.")

    # EVALUATION PARSER
    evaluate = subparsers.add_parser("evaluate", help="Evaluate trained model on data.", parents=[parent_parser])
    evaluate.add_argument("-sn", "--source-name", required=True, type=Path, help="Name of the source dataset.")
    evaluate.add_argument("--source-dir", required=True, type=Path, help="Directory containing gin and model weights.")

    # DOMAIN ADAPTATION ARGUMENTS
    prep_and_train = subparsers.add_parser("da", help="Run DA experiment.", parents=[parent_parser])

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


def aggregate_results(log_dir: Path, execution_time: timedelta = None):
    """Aggregates results from all folds and writes to JSON file.

    Args:
        log_dir: Path to the log directory.
        execution_time: Overall execution time.
    """
    aggregated = {}
    for repetition in log_dir.iterdir():
        if repetition.is_dir():
            aggregated[repetition.name] = {}
            for fold_iter in repetition.iterdir():
                aggregated[repetition.name][fold_iter.name] = {}
                if (fold_iter / "test_metrics.json").is_file():
                    with open(fold_iter / "test_metrics.json", "r") as f:
                        result = json.load(f)
                        aggregated[repetition.name][fold_iter.name].update(result)
                elif (fold_iter / "val_metrics.csv").is_file():
                    with open(fold_iter / "val_metrics.csv", "r") as f:
                        result = json.load(f)
                        aggregated[repetition.name][fold_iter.name].update(result)
                # Add durations to metrics
                if (fold_iter / "durations.json").is_file():
                    with open(fold_iter / "durations.json", "r") as f:
                        result = json.load(f)
                        aggregated[repetition.name][fold_iter.name].update(result)

    # Aggregate results per metric
    list_scores = {}
    for repetition, folds in aggregated.items():
        for fold, result in folds.items():
            for metric, score in result.items():
                if isinstance(score, (float, int)):
                    list_scores[metric] = list_scores.setdefault(metric, [])
                    list_scores[metric].append(score)

    # Compute statistical metric over aggregated results
    averaged_scores = {metric: (mean(list)) for metric, list in list_scores.items()}

    # Calculate the population standard deviation over aggregated results over folds/iterations
    # Divide by sqrt(n) to get standard deviation.
    std_scores = {metric: (pstdev(list) / sqrt(len(list))) for metric, list in list_scores.items()}

    confidence_interval = {
        metric: (stats.t.interval(0.95, len(list) - 1, loc=mean(list), scale=stats.sem(list)))
        for metric, list in list_scores.items()
    }

    accumulated_metrics = {
        "avg": averaged_scores,
        "std": std_scores,
        "CI_0.95": confidence_interval,
        "execution_time": execution_time.total_seconds() if execution_time is not None else 0.0,
    }

    with open(log_dir / "aggregated_test_metrics.json", "w") as f:
        json.dump(aggregated, f, cls=JsonResultLoggingEncoder)

    with open(log_dir / "accumulated_test_metrics.json", "w") as f:
        json.dump(accumulated_metrics, f, cls=JsonResultLoggingEncoder)

    logging.info(f"Accumulated results: {accumulated_metrics}")

    wandb_log(json.loads(json.dumps(accumulated_metrics, cls=JsonResultLoggingEncoder)))


def log_full_line(msg: str, level: int = logging.INFO, char: str = "-", num_newlines: int = 0):
    """Logs a full line of a given character with a message centered.

    Args:
        msg: Message to log.
        level: Logging level.
        char: Character to use for the line.
        num_newlines: Number of newlines to append.
    """
    terminal_size = shutil.get_terminal_size((80, 20))
    reserved_chars = len(logging.getLevelName(level)) + 28
    logging.log(
        level,
        "{0:{char}^{width}}{1}".format(msg, "\n" * num_newlines, char=char, width=terminal_size.columns - reserved_chars),
    )


def load_pretrained_imputation_model(use_pretrained_imputation):
    """Loads a pretrained imputation model.

    Args:
        use_pretrained_imputation: Path to the pretrained imputation model.
    """
    if use_pretrained_imputation is not None and not Path(use_pretrained_imputation).exists():
        logging.warning("The specified pretrained imputation model does not exist.")
        use_pretrained_imputation = None

    if use_pretrained_imputation is not None:
        logging.info("Using pretrained imputation from" + str(use_pretrained_imputation))
        pretrained_imputation_model_checkpoint = torch.load(use_pretrained_imputation, map_location=torch.device("cpu"))
        if isinstance(pretrained_imputation_model_checkpoint, dict):
            imputation_model_class = pretrained_imputation_model_checkpoint["class"]
            pretrained_imputation_model = imputation_model_class(**pretrained_imputation_model_checkpoint["hyper_parameters"])
            pretrained_imputation_model.set_trained_columns(pretrained_imputation_model_checkpoint["trained_columns"])
            pretrained_imputation_model.load_state_dict(pretrained_imputation_model_checkpoint["state_dict"])
        else:
            pretrained_imputation_model = pretrained_imputation_model_checkpoint
        pretrained_imputation_model = pretrained_imputation_model.to("cuda" if torch.cuda.is_available() else "cpu")
        try:
            logging.info(f"imputation model device: {next(pretrained_imputation_model.parameters()).device}")
            pretrained_imputation_model.device = next(pretrained_imputation_model.parameters()).device
        except Exception as e:
            logging.debug(f"Could not set device of imputation model: {e}")
    else:
        pretrained_imputation_model = None

    return pretrained_imputation_model


def setup_logging(date_format, log_format, verbose):
    """
    Set up all loggers to use the same format and date format.

    Args:
        date_format: Format for the date.
        log_format: Format for the log.
        verbose: Whether to log debug messages.
    """
    logging.basicConfig(format=log_format, datefmt=date_format)
    loggers = ["pytorch_lightning", "lightning_fabric"]
    for logger in loggers:
        logging.getLogger(logger).handlers[0].setFormatter(logging.Formatter(log_format, datefmt=date_format))

    if not verbose:
        logging.getLogger().setLevel(logging.INFO)
        for logger in loggers:
            logging.getLogger(logger).setLevel(logging.INFO)
        warnings.filterwarnings("ignore")
    else:
        logging.getLogger().setLevel(logging.DEBUG)
        for logger in loggers:
            logging.getLogger(logger).setLevel(logging.DEBUG)
        warnings.filterwarnings("default")
