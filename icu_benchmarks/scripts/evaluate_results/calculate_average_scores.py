from pathlib import Path
import pickle
import sys

log_dir = Path(sys.argv[1])
PR = 0
AUC = 0
num_seeds = 0

for seed in log_dir.iterdir():
    num_seeds += 1
    with open(seed / "test_metrics.pkl", "rb") as f:
        result = pickle.load(f)
        PR += result['PR']
        AUC += result['AUC']

PR = PR / num_seeds
AUC = AUC / num_seeds

print(f"PR: {PR:.4f}")
print(f"AUC: {AUC:.4f}")
