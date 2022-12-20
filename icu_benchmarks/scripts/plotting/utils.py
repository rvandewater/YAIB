import json
import logging

from icu_benchmarks.scripts.plotting.plotting import Plotter


def plot_fold(metrics, save_dir):
    """
    Plots the results of a single fold.
    Args:
        metrics: Metrics dictionary.
        save_dir: Directory to save the plots to.
    """
    plotter = Plotter({"fold": metrics}, save_dir)
    plotter.calibration_curve()
    plotter.precision_recall_curve()
    plotter.receiver_operator_curve()


def plot_agg_results(log_dir, metrics_path):
    """Aggregates results from all folds for one seed and generates plots.

    Args:
        log_dir: Path to the log directory.
        metrics_path: Metrics JSON file
    """
    with open(log_dir / f"{metrics_path}.json") as metrics_file:
        metrics = json.load(metrics_file)
        for seed in metrics:
            plotter = Plotter(metrics[seed], log_dir, seed)
            # Check if there are multiple folds
            if len(metrics[seed]) > 1:
                base = metrics[seed]["fold_0"]
            else:
                base = metrics[seed]
            if "ROC" and "AUC" in base:
                plotter.receiver_operator_curve()
            if "PRC" and "PR" in base:
                plotter.precision_recall_curve()
            if "Calibration" in base:
                plotter.calibration_curve()
    logging.info(f"Generated plots in {log_dir}")
