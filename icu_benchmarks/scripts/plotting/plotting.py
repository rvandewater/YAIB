import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, results, save_dir, specifier=""):
        self.results = results
        self.save_dir = save_dir
        self.specifier = specifier

    def receiver_operator_curve(self):
        for fold in self.results:
            fold_result = self.results[fold]
            auc = fold_result["AUC"]
            plt.plot(fold_result["ROC"][0], fold_result["ROC"][1], label=f"ROC curve {fold} {auc:0.3f}")
        plt.plot([0, 1], [0, 1], "k--")  # random predictions curve
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel("False Positive Rate or (1 - Specificity)")
        plt.ylabel("True Positive Rate or (Sensitivity)")
        plt.title(f"Receiver Operating Characteristic for {self.specifier}")
        plt.legend(loc="lower right")
        plt.savefig(self.save_dir / f"roc_curve_{self.specifier}.png")
        plt.clf()

    def precision_recall_curve(self):
        for fold in self.results:
            fold_result = self.results[fold]
            prc = fold_result["PR"]
            plt.plot(fold_result["PRC"][0], fold_result["PRC"][1], label=f"PRC curve {fold} {prc:0.3f}")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.5, 1.0])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision Recall Curve for {self.specifier}")
        plt.legend(loc="lower right")
        plt.savefig(self.save_dir / f"prc_curve_{self.specifier}.png")
        plt.clf()

    def calibration_curve(self):
        for fold in self.results:
            fold_result = self.results[fold]
            plt.plot(fold_result["Calibration"][0], fold_result["Calibration"][1], label=f"Calibration curve {fold}")
        plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")  # random predictions curve
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title(f"Calibration Curve for {self.specifier}")
        plt.legend(loc="lower right")
        plt.savefig(self.save_dir / f"call_curve {self.specifier}.png")
        plt.clf()
