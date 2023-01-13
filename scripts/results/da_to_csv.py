import json
from pathlib import Path
import csv

models_dir = Path("../DA_logs")
for metric in ["AUC", "PR"]:
    for endpoint in models_dir.iterdir():
        with open(models_dir / f'{endpoint.name}_{metric}_results.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for model in endpoint.iterdir():
                for target in model.iterdir():
                    for target_size in target.iterdir():
                        with open(target_size / 'averaged_source_metrics.json', 'r') as f:
                            results = json.load(f)
                            writer.writerow([model.name, target, target_size] + [source[metric] for source in results])
