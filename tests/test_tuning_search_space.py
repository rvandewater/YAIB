"""The Optuna tuner must sample the search space the gin configs describe (#240)."""

from pathlib import Path

import gin
import optuna
import pytest
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution

import icu_benchmarks.run  # noqa: F401  registers the Run configurable used by the task configs
from icu_benchmarks.tuning.hyperparameters import collect_bound_hyperparameters, make_sampler, suggest_hyperparameter

CONFIGS = sorted(Path("configs/prediction_models").glob("*.gin"))
REGRESSORS = {"ElasticNet", "LGBMRegressor"}


def distribution(value):
    trial = optuna.create_study().ask()
    suggest_hyperparameter(trial, "p", value)
    return trial.distributions["p"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ((1e-3, 1e1, "log-uniform"), FloatDistribution(1e-3, 1e1, log=True)),
        ((1e-6, 1e-4, "log"), FloatDistribution(1e-6, 1e-4, log=True)),
        ((0.33, 1.0), FloatDistribution(0.33, 1.0)),
        ((8, 128, "log", 2), IntDistribution(8, 128, log=True)),
        ((32, 256, "log-uniform", 2), IntDistribution(32, 256, log=True)),
        ((1, 3), IntDistribution(1, 3)),
        ([10, 50, 100], CategoricalDistribution([10, 50, 100])),
        (("gelu", "relu"), CategoricalDistribution(["gelu", "relu"])),
    ],
)
def test_suggest_hyperparameter(value, expected):
    assert distribution(value) == expected


def test_numeric_tuple_without_prior_is_rejected():
    # (5, 10, 20) used to be sampled as an integer range [5, 10]
    with pytest.raises(ValueError, match="use a list for choices"):
        distribution((5, 10, 20))


@pytest.mark.parametrize("config", CONFIGS, ids=lambda p: p.stem)
def test_model_search_spaces_are_valid(config):
    gin.clear_config()
    try:
        # Model configs use macros from the task config, as in run.py
        task = "Regression" if config.stem in REGRESSORS else "BinaryClassification"
        gin.parse_config_files_and_bindings([str(config), f"configs/tasks/{task}.gin"], None, finalize_config=False)
        bounds, names = collect_bound_hyperparameters({}, ["model"])
        trial = optuna.create_study().ask()
        for name, value in zip(names, bounds):
            suggest_hyperparameter(trial, name, value)
    finally:
        gin.clear_config()


def test_gp_sampler_uses_n_initial_points():
    sampler = make_sampler(optuna.samplers.GPSampler, seed=0, n_initial_points=5)
    assert sampler._n_startup_trials == 5
    assert isinstance(make_sampler(optuna.samplers.TPESampler, seed=0, n_initial_points=5), optuna.samplers.TPESampler)
