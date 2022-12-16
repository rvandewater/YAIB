import json
import logging

from icu_benchmarks.scripts.plotting.plotting import plot_calibration_curve, plot_prc, plot_roc


def plot_fold(metrics, save_dir):
    """
    Plots the results of a single fold.
    Args:
        metrics: Metrics dictionary.
        save_dir: Directory to save the plots to.
    """
    plot_calibration_curve({"fold": metrics}, save_dir)
    plot_prc({"fold": metrics}, save_dir)
    plot_roc({"fold": metrics}, save_dir)


def plot_agg_results(log_dir, metrics_path):
    """Aggregates results from all folds for one seed and generates plots.

    Args:
        log_dir: Path to the log directory.
        metrics_path: Metrics JSON file
    """
    with open(log_dir / f"{metrics_path}.json") as metrics_file:
        metrics = json.load(metrics_file)
        for seed in metrics:
            if "ROC" and "AUC" in metrics[seed]:
                plot_roc(metrics[seed], log_dir, seed)
            if "PRC" and "PR" in metrics[seed]:
                plot_prc(metrics[seed], log_dir, seed)
            if "Calibration" in metrics[seed]:
                plot_calibration_curve(metrics[seed], log_dir, seed)
    logging.info(f"Generated plots in {log_dir}")
