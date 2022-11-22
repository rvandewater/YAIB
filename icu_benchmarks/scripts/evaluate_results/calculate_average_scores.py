from pathlib import Path
import pickle
import sys

log_dir = Path(sys.argv[1])

def sorted_dir(dir: Path) -> list[Path]:
    return sorted(list(dir.iterdir()))

for dataset_dir in sorted_dir(log_dir):
    for experiment_dir in sorted_dir(dataset_dir):
        for source_dir in sorted_dir(experiment_dir):
            PR = 0
            AUC = 0
            num_seeds = 0
            run_dirs = sorted(list(source_dir.iterdir()))
            for seed in run_dirs[-1].iterdir():
                num_seeds += 1
                with open(seed / "test_metrics.pkl", "rb") as f:
                    result = pickle.load(f)
                    PR += result['PR']
                    AUC += result['AUC']

            PR = PR / num_seeds
            AUC = AUC / num_seeds

            print(f"{experiment_dir.name} on {dataset_dir.name} {source_dir.name}")
            print(f"PR: {PR:.4f}")
            print(f"AUC: {AUC:.4f}")
