# Hyperparameter tuning

With `--tune`, YAIB searches the hyperparameters with [Optuna](https://optuna.org/) before the cross-validation run,
then trains every fold with the best configuration found.

## Defining the search space

A model config declares its search space with `hyperparameter` bindings in named *scopes*. For example, `GRU.gin`:

```python title="configs/prediction_models/GRU.gin"
--8<--
configs/prediction_models/GRU.gin
--8<--
```

Each scope has a `class_to_tune`, and the other bindings in the scope are arguments of that class:

```python
<scope>/hyperparameter.class_to_tune = @<SomeClass>
<scope>/hyperparameter.<argument> = <value, range or choices>
```

The scopes to search are set in the tuning settings, `configs/prediction_models/common/DLTuning.gin` for deep learning
models (scopes `model` and `optimizer`) and `MLTuning.gin` for classical models (scope `model`):

```python title="configs/prediction_models/common/DLTuning.gin"
--8<--
configs/prediction_models/common/DLTuning.gin
--8<--
```

### Values, ranges and choices

| Binding                 | Meaning                     | Optuna call                    |
| ----------------------- | --------------------------- | ------------------------------ |
| `= 1e-6`                | Fixed value, not tuned      | (bound directly)               |
| `= (1, 10)`             | Integer range, uniform      | `suggest_int(1, 10)`           |
| `= (0.0, 0.4)`          | Float range, uniform        | `suggest_float(0.0, 0.4)`      |
| `= (1e-6, 1e-4, "log")` | Range sampled log-uniformly | `suggest_float(..., log=True)` |
| `= [32, 64, 128]`       | Choice between values       | `suggest_categorical([...])`   |

`"log-uniform"` is accepted as a spelling of `"log"`, and a fourth element (the log base, as in older configs such as
`(32, 256, "log-uniform", 2)`) is ignored: the base does not change a log-uniform distribution. A range whose bounds are
both integers samples integers.

Log-uniform sampling draws `log(x)` uniformly between `log(low)` and `log(high)` (the reciprocal distribution, with
density proportional to `1/x`), so each order of magnitude is equally likely. This suits parameters such as learning
rates that span several orders of magnitude. Both bounds must be positive.

### Tuner settings

| Binding                                 | Default in the configs                          | Meaning                                                      |
| --------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------ |
| `tune_hyperparameters.scopes`           | `["model", "optimizer"]` (DL), `["model"]` (ML) | Scopes to search                                             |
| `tune_hyperparameters.n_calls`          | `30`                                            | Number of trials                                             |
| `tune_hyperparameters.n_initial_points` | `5`                                             | Random trials before the Gaussian-process sampler takes over |
| `tune_hyperparameters.folds_to_tune_on` | `5`                                             | Folds each trial is cross-validated on                       |
| `tune_hyperparameters.sampler`          | `optuna.samplers.GPSampler`                     | Optuna sampler class                                         |

Each trial trains on the folds of the first repetition and is scored on their **validation** splits, never on the test
splits. The score is the validation loss; for regression it is the mean squared error in the original target units.

## Which value wins

A hyperparameter can be set in several places. YAIB applies them in this order:

1. A direct binding to the class always wins and is not tuned: `GRUNet.hidden_dim = 64`, in any config or `-hp`.
2. Otherwise the `hyperparameter` binding is used: a single value is bound as is, a range or list is tuned.
3. Among bindings at the same level, the last one parsed wins: `-hp` over the configs.

For example, to fix the learning rate and narrow the hidden size of the GRU from the command line:

```bash
uv run icu-benchmarks train ... -m GRU --tune \
    -hp optimizer/hyperparameter.lr=1e-4 'model/hyperparameter.hidden_dim=[64,128]'
```

## Without `--tune`

Without `--tune`, YAIB still binds a value for every range or list: it draws one random configuration (seeded with
`--seed`) from the search space. To train with tuned values, use `--tune` or load an earlier study with
`--hp-checkpoint`.

## Results and checkpoints

The study is saved in the run folder as `hyperparameter_tuning_logs.db` (an Optuna SQLite storage), together with the
plots `optimization_history.png` and `param_importances.png`. The log prints a table of every trial and
highlights the best one so far.

To reuse a study, pass it with `--hp-checkpoint` (relative to `<log-dir>/<name>/<task-name>/<model>/`, or absolute):

- with `--tune`, tuning resumes and runs the remaining trials up to `n_calls`;
- without `--tune`, YAIB trains directly with the study's best configuration.

```bash
uv run icu-benchmarks train ... -m GRU --hp-checkpoint 2026-11-06T14-08-10/hyperparameter_tuning_logs.db
```

To log the trials to Weights & Biases, add `--wandb-sweep`; see [Weights & Biases](weights-and-biases.md).
