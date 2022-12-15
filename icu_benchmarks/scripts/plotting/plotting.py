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
    plot_calibration_curve(metrics["Calibration"][0], metrics["Calibration"][1], save_dir)
    plot_prc(metrics["PRC"][0], metrics["PRC"][1], save_dir)
    plot_roc(metrics["ROC"][0], metrics["ROC"][1], metrics["AUC"], save_dir)


def plot_roc(fpr, tpr, auc, save_dir):
    plt.plot(fpr, tpr, label="ROC curve (area = %0.3f)" % auc)
    plt.plot([0, 1], [0, 1], "k--")  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate or (1 - Specificity)")
    plt.ylabel("True Positive Rate or (Sensitivity)")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(save_dir / "roc_curve.png")
    plt.clf()


def plot_prc(rec, prec, auprc, save_dir):
    plt.plot(rec, prec, label="PRC curve (area = %0.3f)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.5, 1.0])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve")
    plt.legend(loc="lower right")
    plt.savefig(save_dir / "prc_curve.png")
    plt.clf()


def plot_calibration_curve(mpp, fop, save_dir):
    plt.plot(mpp, fop, label="Calibration Curve")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.legend(loc="lower right")
    plt.savefig(save_dir / "call_curve.png")
    plt.clf()
