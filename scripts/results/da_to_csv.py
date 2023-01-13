import json
from pathlib import Path
import csv

models_dir = Path("../DA_logs")
for metric in ["AUC", "PR"]:
    for endpoint in models_dir.iterdir():
        if endpoint.is_dir():
            with open(models_dir / f'{endpoint.name}_{metric}_results.csv', 'w') as csv_file:
                writer = csv.writer(csv_file)
                info = ["model", "target", "target_size"]
                source_names = ['target', 'aumc', 'eicu', 'hirid', 'miiv', 'convex_combination_without_target', 'convex_combination_0.1', 'convex_combination_0.2', 'convex_combination_0.5', 'convex_combination_1', 'convex_combination_2', 'convex_combination_5']
                stats = ['avg', 'std', 'CI_0.95']
                # combine fieldnames and stats
                full_fields = [f'{source}_{stat}' for source in source_names for stat in stats]
                writer = csv.DictWriter(csv_file, fieldnames=info+full_fields)

                writer.writeheader()
                for model in endpoint.iterdir():
                    if model.name == "LSTM":
                        continue
                    for target in model.iterdir():
                        for target_size in target.iterdir():
                            with open(target_size / 'averaged_source_metrics.json', 'r') as f:
                                results = json.load(f)
                                # source_metrics = [source_metrics[metric] for source_name, source_metrics in results.items()]
                                # source_metrics = [[metr[0]["avg"], metr[0]["std"], metr[0]["CI_0.95"]] for metr in source_metrics]
                                # source_metrics_flat = [item for sublist in source_metrics for item in sublist]
                                # writer.writerow([model.name, target.name, target_size.name] + source_metrics_flat)

                                info = {
                                    'model': model.name,
                                    'target': target.name,
                                    'target_size': target_size.name
                                }
                                metrics_row = {f'{source}_{stat}': source_metrics[metric][0][stat] for source, source_metrics in results.items() for stat in stats}
                                writer.writerow(info + metrics_row)
