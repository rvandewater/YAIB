import json
import matplotlib.pyplot as plt
import logging

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


def plot_roc(results, save_dir, specifier=""):
    for fold in results:
        fold_result = results[fold]
        auc = fold_result["AUC"]
        plt.plot(fold_result["ROC"][0], fold_result["ROC"][1], label=f"ROC curve {fold} {auc:0.3f}")
    plt.plot([0, 1], [0, 1], "k--")  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate or (1 - Specificity)")
    plt.ylabel("True Positive Rate or (Sensitivity)")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(save_dir / f"roc_curve_{specifier}.png")
    plt.clf()


def plot_prc(results, save_dir, specifier=""):
    for fold in results:
        fold_result = results[fold]
        prc = fold_result["PR"]
        plt.plot(fold_result["PRC"][0], fold_result["PRC"][1], label=f"PRC curve {fold} {prc:0.3f}")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.5, 1.0])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve")
    plt.legend(loc="lower right")
    plt.savefig(save_dir / f"prc_curve_{specifier}.png")
    plt.clf()


def plot_calibration_curve(results, save_dir, specifier=""):
    for fold in results:
        fold_result = results[fold]
        plt.plot(fold_result["Calibration"][0], fold_result["Calibration"][1], label=f"Calibration curve {fold}")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.legend(loc="lower right")
    plt.savefig(save_dir / f"call_curve {specifier}.png")
    plt.clf()
