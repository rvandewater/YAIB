from pathlib import Path
import pickle
import sys

log_dir = Path(sys.argv[1])
results = {}

for run in log_dir.iterdir():
    results[run.name] = 0
    for seed in run.iterdir():
        if seed.is_dir():
            with open(seed / "test_metrics.pkl", "rb") as f:
                results[run.name] += pickle.load(f)["AUC"]

sorted_results = sorted(results.items(), key=lambda item: item[1])
for run, auc in reversed(sorted_results[-5:]):
    print(f"{run}: {auc}")
    run_dir = log_dir / run
    for child in run_dir.iterdir():
        if child.is_file():
            for hyperparam in child.name.split("-"):
                print(hyperparam)
