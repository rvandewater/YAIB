# Hyperparameter Optimization using Weights and Biases Sweeps

[This sweep file](wandb_sweep.yaml) shows an example on how to run a hyperparameter sweep with W&B. The general structure of the YAML should look like this:
``` yaml
program: icu-benchmarks
command:
  - ${env}
  - ${program}
  - "train"
# .... other program parameters ....
  - "--wandb-sweep"
method: grid
parameters:
  # gin config parameter name:
    # values: [a, b, etc...]
  # example:
  ImputationDataset.mask_method:
    values: ["MCAR", "MAR", "MNAR"]
```

You can then create a sweep with 
``` bash
wandb sweep path/to/sweep_file.yaml
```
which will give you a sweep id.

and start an agent to perform the optimization using the following command:
``` bash
wandb agent YOUR_SWEEP_ID
```