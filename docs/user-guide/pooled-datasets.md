# Pooled datasets

Pooling combines the cohorts of several datasets into one training set, for example to train on three datasets and test
on the fourth. `icu_benchmarks/data/pooling.py` generates pooled cohorts from cohorts that share the same variables.

```python
from pathlib import Path

from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.pooling import PooledData, PooledDataset

file_names = {"DYNAMIC": "dyn.parquet", "STATIC": "sta.parquet", "OUTCOME": "outc.parquet"}
vars = {
    "GROUP": "stay_id",
    "SEQUENCE": "time",
    "LABEL": "label",
    "DYNAMIC": [...],  # as in the task config
    "STATIC": ["age", "sex", "height", "weight"],
}

# data_dir contains one folder per dataset (aumc, eicu, hirid, miiv) with the cohort files
data_dir = Path("cohorts/mortality24")
pooled = PooledData(data_dir, vars, datasets=None, file_names=file_names, runmode=RunMode.classification)
for datasets in [PooledDataset.aumc_eicu_miiv, PooledDataset.hirid_eicu_miiv, PooledDataset.aumc_hirid_eicu_miiv]:
    pooled.generate(datasets, samples=10000, seed=42)
```

`generate` draws `samples` stays from **each** dataset (stratified by label for classification), gives the stay IDs of
each dataset a distinct prefix so they cannot clash, and saves the pooled cohort as a new dataset folder in `data_dir`,
named after the datasets and the sample count, e.g. `aumc_eicu_miiv_10000`. Train on it like any other cohort with `-d`.

| `PooledDataset`        | Datasets                |
| ---------------------- | ----------------------- |
| `hirid_eicu_miiv`      | HiRID, eICU, MIMIC-IV   |
| `aumc_hirid_eicu`      | AUMCdb, HiRID, eICU     |
| `aumc_eicu_miiv`       | AUMCdb, eICU, MIMIC-IV  |
| `aumc_hirid_miiv`      | AUMCdb, HiRID, MIMIC-IV |
| `aumc_hirid_eicu_miiv` | All four                |

With `save_test=True` (the default), the stays of each dataset that were not drawn are saved in
`<dataset>_test_<n>`, so the pooled model can be tested on held-out data without leakage.

!!! note

    The held-out folder names include neither the seed nor the pool, so a second pooling run with a different seed
    overwrites the first run's held-out sets.

The [experiment configs](https://github.com/rvandewater/YAIB/tree/development/experiments) `experiment_pooled.yml`
and `experiment_eval_pooled.yml` show how the pooled cohorts were trained and evaluated with Weights & Biases sweeps.
