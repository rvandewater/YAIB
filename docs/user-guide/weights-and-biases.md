# Weights & Biases

YAIB integrates with [Weights & Biases](https://wandb.ai/) (W&B) to track runs and to run grids of experiments as
sweeps. You need a W&B account; log in once with `uv run wandb login`.

## Sweeps

A sweep runs `icu-benchmarks` once for every combination of parameters. The sweep files in `experiments/` reproduce
the paper's benchmarks; `demo_benchmark_classification.yml` runs the demo cohorts:

```yaml title="experiments/demo_benchmark_classification.yml (shortened)"
command:
  - ${env}
  - ${program}
  - train
  - -d
  - ../data/
  - -t
  - BinaryClassification
  - --log-dir
  - ../yaib_logs
  - --tune
  - --wandb-sweep
  - -gc
  - -lc
method: grid
name: yaib_demo_classification_benchmark
parameters:
  data_dir:
    values:
      - demo_data/mortality24/eicu_demo
      - demo_data/mortality24/mimic_demo
      # ...
  model:
    values:
      - LogisticRegression
      - LGBMClassifier
      - GRU
      # ...
```

Create the sweep and start an agent from the repository root:

```bash
uv run wandb sweep --verbose experiments/demo_benchmark_classification.yml
uv run wandb agent <sweep_id>
```

!!! tip "SLURM"

    To run each configuration as its own job on a cluster, start agents with `uv run wandb agent --count 1 <sweep_id>`.
    `experiments/slurm_base_template.sh` is a template for the job script.

## How sweep parameters are applied

`--wandb-sweep` (`-wd`) makes YAIB read the parameters of the current sweep run. Each parameter is applied in two ways:

- If its name is a command-line option (`data_dir`, `model`, `seed`, `task_name`, ...), it sets that option. This is how
    the sweep above varies the dataset and the model.
- It is also passed as a gin binding, as with `-hp`. So any gin binding can be swept:

```yaml
parameters:
  ImputationPandasDataset.mask_method:
    values: [MCAR, MAR, MNAR]
  train_common.batch_size:
    values: [32, 64]
```

Runs are named `<mode>_<model>_<dataset>`, with the source dataset and fine-tuning size added for transfer runs.
With `--tune`, the tuning trials are logged to W&B too.

More examples are in `scripts/sweep_configs/`.
