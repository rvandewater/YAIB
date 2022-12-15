import json
import matplotlib.pyplot as plt


def plot_results(log_dir, metrics_path):
    """Plots results from all folds and generates images.

    Args:
        log_dir: Path to the log directory.
    """
    for seed in log_dir.iterdir():
        if seed.is_dir():
            for fold in seed.iterdir():
                with open(fold / f"{metrics_path}.json") as metrics_file:
                    metrics = json.load(metrics_file)
                plot_fold(metrics, fold)


def plot_fold(metrics, save_dir):
    plot_calibration_curve({"fold": metrics}, save_dir)
    plot_prc({"fold": metrics}, save_dir)
    plot_roc({"fold": metrics}, save_dir)


def plot_roc(results, save_dir):
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
    plt.savefig(save_dir / "roc_curve.png")
    plt.clf()


def plot_prc(results, save_dir):
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
    plt.savefig(save_dir / "prc_curve.png")
    plt.clf()


def plot_calibration_curve(results, save_dir):
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
    plt.savefig(save_dir / "call_curve.png")
    plt.clf()
