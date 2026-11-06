# Contributing

YAIB is in active development and we welcome contributions. Please discuss the change you want to make in an
[issue](https://github.com/rvandewater/YAIB/issues) first, and follow the
[code of conduct](https://github.com/rvandewater/YAIB/blob/development/CONTRIBUTING.md#code-of-conduct).

## Development setup

```bash
git clone https://github.com/rvandewater/YAIB.git
cd YAIB
git checkout development
uv python install 3.12
uv sync --dev --extra imputation
uv run pre-commit install
```

Branch from `development` and open pull requests against it.

## Tests

```bash
uv run pytest tests/
```

With coverage of the `icu_benchmarks` package:

```bash
uv run coverage run -m pytest tests/
uv run coverage report
uv run coverage html  # writes htmlcov/index.html
```

CI runs the test suite with coverage on Python 3.10 to 3.13, uploads the coverage to Codecov, and trains a model on the
demo data end to end. New code should come with tests; a bug fix should come with a test that fails without it.

## Formatting and linting

[ruff](https://docs.astral.sh/ruff/) formats and lints the code, with a line length of 127:

```bash
uv run ruff format --line-length 127
uv run ruff check --line-length 127 --statistics
```

The pre-commit hooks run ruff, codespell, Prettier (YAML) and mdformat (Markdown) on every commit. To run them on all
files:

```bash
uv run pre-commit run --all-files
```

## Pull requests

1. Update the documentation for any change to the interface, including installation, configuration and command-line
    options; see [Documentation](documentation.md).
2. Keep `pyproject.toml` and `uv.lock` in sync: after changing dependencies, run `uv lock` and commit both. CI installs
    with `uv sync --locked`.
3. Use [SemVer](https://semver.org/) for version changes.
4. A pull request can be merged once it has the required reviewer approval.

## Libraries

YAIB builds on:

- [Polars](https://pola.rs/) and [pandas](https://pandas.pydata.org/) for data handling.
- [ReciPys](https://github.com/rvandewater/ReciPys) for the preprocessing recipes.
- [PyTorch](https://pytorch.org/), [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/),
    [PyTorch Ignite](https://pytorch.org/ignite/) and [TorchMetrics](https://lightning.ai/docs/torchmetrics/) for deep
    learning.
- [scikit-learn](https://scikit-learn.org/), [LightGBM](https://lightgbm.readthedocs.io/),
    [XGBoost](https://xgboost.readthedocs.io/), [CatBoost](https://catboost.ai/) and
    [imbalanced-learn](https://imbalanced-learn.org/) for classical models.
- [Optuna](https://optuna.org/) for hyperparameter tuning.
- [gin-config](https://github.com/google/gin-config) for configuration.
- [SHAP](https://shap.readthedocs.io/) for feature explanations.
- [Weights & Biases](https://wandb.ai/) for experiment tracking.
- [HyperImpute](https://github.com/vanderschaarlab/hyperimpute) and [PyPOTS](https://github.com/WenjieDu/PyPOTS) for
    imputation baselines (optional `imputation` extra).
- [pytest](https://docs.pytest.org/) for testing.
