import json
from pathlib import Path
import csv

models_dir = Path("../DA_logs")
for metric in ["AUC", "PR"]:
    for endpoint in models_dir.iterdir():
        if endpoint.is_dir():
            with open(models_dir / f"{endpoint.name}_{metric}_results.csv", "w") as csv_file:
                writer = csv.writer(csv_file)
                info = ["model", "target", "target_size"]
                source_names = [
                    "target",
                    "aumc",
                    "eicu",
                    "hirid",
                    "miiv",
                    "convex_combination_without_target",
                    "target_weight_0.5",
                    "target_weight_1",
                    "target_weight_2",
                    "loss_weighted",
                    "target_with_predictions",
                    "cc_with_preds",
                ]
                stats_basis = ["avg", "std", "CI_0.95"]
                stats_basis = ["avg"]
                stats = ["avg", "std", "CI_0.95_min", "CI_0.95_max"]
                stats = ["avg"]
                # combine fieldnames and stats
                full_fields = [f"{source}_{stat}" for source in source_names for stat in stats]
                writer = csv.DictWriter(csv_file, fieldnames=info + full_fields)

                writer.writeheader()
                for model in endpoint.iterdir():
                    for target in ["aumc", "eicu", "hirid", "miiv"]:
                        target_sizes = ["target_500", "target_1000", "target_2000"]
                        for target_size in target_sizes:
                            with open(model / target / target_size / "averaged_source_metrics.json", "r") as f:
                                results = json.load(f)

                                row_data = {"model": model.name, "target": target, "target_size": target_size}
                                for stat in stats_basis:
                                    for source, source_metrics in results.items():
                                        if stat == "CI_0.95":
                                            row_data[f"{source}_{stat}_min"] = source_metrics[metric][0][stat][0] * 100
                                            row_data[f"{source}_{stat}_max"] = source_metrics[metric][0][stat][1] * 100
                                        else:
                                            row_data[f"{source}_{stat}"] = source_metrics[metric][0][stat] * 100
                                writer.writerow(row_data)
