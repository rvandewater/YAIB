import matplotlib.pyplot as plt
import numpy as np


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

    def plot_XAI_Metrics(accumulated_metrics, log_dir_plots):
        groups = {}
        for key in accumulated_metrics["avg"]:
            if key in ["loss", "MAE"]:
                continue
            suffix = key.split("_")[-1]
            if suffix not in groups:
                groups[suffix] = []
            groups[suffix].append(key)

        # Define a dictionary for legend labels
        legend_labels = {
            "IG": "Integrated Gradient",
            "G": "Gradient",
            "R": "Random",
            "FA": "Feature Ablation",
            "Att": "Attention",
            "VSN": "Variable Selection Network",
            "L": "Lime",
        }
        colors = [
            "navy",
            "skyblue",
            "crimson",
            "salmon",
            "teal",
            "orange",
            "darkgreen",
            "lightgreen",
        ]

        # Plotting
        num_groups = len(groups)
        fig, axs = plt.subplots(num_groups, 1, figsize=(10, num_groups * 5))

        # Custom handles for the legend
        # handles = [plt.Rectangle((0, 0), 1, 1, color="none",
        # label=f"{key}: {value}") for key, value in legend_labels.items()]

        for i, (suffix, keys) in enumerate(groups.items()):
            ax = axs[i] if num_groups > 1 else axs
            # Extract values and errors
            avg_values = [accumulated_metrics["avg"][key] for key in keys]
            ci_lower = [accumulated_metrics["CI_0.95"][key][0] for key in keys]
            ci_upper = [accumulated_metrics["CI_0.95"][key][1] for key in keys]
            ci_error = [np.abs([a - b, c - a]) for a, b, c in zip(avg_values, ci_lower, ci_upper)]

            # Sort by absolute values of avg_values
            sorted_indices = np.argsort([np.abs(val) for val in avg_values])[::-1]  # Indices to sort in descending order
            sorted_keys = np.array(keys)[sorted_indices]
            sorted_avg_values = np.array(avg_values)[sorted_indices]
            sorted_ci_error = np.array(ci_error)[sorted_indices]

            # Plot bars
            bars = ax.bar(
                sorted_keys,
                np.abs(sorted_avg_values),
                yerr=np.array(sorted_ci_error).T,
                capsize=5,
                color=colors,
            )

            # Set titles and labels
            title_suffix = sorted_keys[0].split("_")[1]
            ax.set_title(f'Metric: "{title_suffix}"')
            ax.set_ylabel("Values")
            ax.axhline(0, color="grey", linewidth=0.8)
            ax.grid(axis="y")

            # Set x-ticks
            ax.set_xticks(sorted_keys)
            ax.set_xticklabels([key.split("_")[0] for key in sorted_keys], rotation=45, ha="right")
            # Create a custom legend for each subplot
            custom_labels = [legend_labels[key.split("_")[0]] for key in sorted_keys]
            ax.legend(bars, custom_labels, loc="upper right")

        plt.tight_layout()
        plt.savefig(log_dir_plots / "metrics_plot.png", bbox_inches="tight")

        def plot_attributions(self, features_attrs, timestep_attrs, method_name, log_dir):
            """
            Plots the attribution values for features and timesteps.

            Args:
                - features_attrs: Array of feature attribution values.
                - timestep_attrs: Array of timestep attribution values.
                - method_name: Name of the attribution method.
                - log_dir: Directory to save the plots.
            Returns:
                Nothing
            """

            # Plot for feature attributions
            x_values = np.arange(1, len(features_attrs) + 1)
            plt.figure(figsize=(8, 6))
            plt.plot(
                x_values,
                features_attrs,
                marker="o",
                color="skyblue",
                linestyle="-",
                linewidth=2,
                markersize=8,
            )
            plt.xlabel("Feature")
            plt.ylabel("{} Attribution".format(method_name))
            plt.title("{} Attribution Values".format(method_name))
            plt.xticks(
                x_values,
                [
                    "height",
                    "weight",
                    "age",
                    "sex",
                    "time_idx",
                    "alb",
                    "alp",
                    "alt",
                    "ast",
                    "be",
                    "bicar",
                    "bili",
                    "bili_dir",
                    "bnd",
                    "bun",
                    "ca",
                    "cai",
                    "ck",
                    "ckmb",
                    "cl",
                    "crea",
                    "crp",
                    "dbp",
                    "fgn",
                    "fio2",
                    "glu",
                    "hgb",
                    "hr",
                    "inr_pt",
                    "k",
                    "lact",
                    "lymph",
                    "map",
                    "mch",
                    "mchc",
                    "mcv",
                    "methb",
                    "mg",
                    "na",
                    "neut",
                    "o2sat",
                    "pco2",
                    "ph",
                    "phos",
                    "plt",
                    "po2",
                    "ptt",
                    "resp",
                    "sbp",
                    "temp",
                    "tnt",
                    "urine",
                    "wbc",
                ],
                rotation=90,
            )
            plt.tight_layout()
            plt.savefig(
                log_dir / "{}_attribution_features_plot.png".format(method_name),
                bbox_inches="tight",
            )

            # Plot for timestep attributions
            x_values = np.arange(1, len(timestep_attrs) + 1)
            plt.figure(figsize=(8, 6))
            plt.plot(
                x_values,
                timestep_attrs,
                marker="o",
                color="skyblue",
                linestyle="-",
                linewidth=2,
                markersize=8,
            )
            plt.xlabel("Time Step")
            plt.ylabel("{} Attribution".format(method_name))
            plt.title("{} Attribution Values".format(method_name))
            plt.xticks(x_values)
            plt.tight_layout()
            plt.savefig(log_dir / "{}_attribution_plot.png".format(method_name), bbox_inches="tight")
