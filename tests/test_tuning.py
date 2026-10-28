"""Optuna tuning must store its trials, resume from a checkpoint and bind the best hyperparameters (on the MIMIC demo)."""

from pathlib import Path

import gin
import matplotlib.pyplot as plt
import optuna
import pytest

import icu_benchmarks.run  # noqa: F401 (registers the Run configurable of the task configs)
import icu_benchmarks.tuning.hyperparameters
from icu_benchmarks.tuning.hyperparameters import choose_and_bind_hyperparameters_optuna

ROOT = Path(__file__).parent.parent
DATA = ROOT / "demo_data" / "mortality24" / "mimic_demo"
TUNED = {"LogisticRegression.C", "LogisticRegression.penalty", "LogisticRegression.l1_ratio"}


@pytest.fixture(autouse=True)
def logistic_regression_config(monkeypatch):
    # The configs include each other relative to the repository
    monkeypatch.chdir(ROOT)
    gin.clear_config()
    parse_config()
    yield
    gin.clear_config()


def parse_config():
    gin.parse_config_files_and_bindings(
        ["configs/prediction_models/LogisticRegression.gin", "configs/tasks/BinaryClassification.gin"],
        ["LogisticRegression.max_iter = 50"],
        finalize_config=False,
    )


def tune(log_dir, **kwargs):
    kwargs = {"do_tune": True, "seed": 1, "n_calls": 2, "folds_to_tune_on": 1, "plot": False, **kwargs}
    return choose_and_bind_hyperparameters_optuna(data_dir=DATA, log_dir=log_dir, **kwargs)


def load(db: Path) -> optuna.Study:
    return optuna.load_study(study_name="tuning", storage=f"sqlite:///{db}")


def bound(names) -> dict:
    return {name: gin.query_parameter(name) for name in names}


def forbid_training(monkeypatch):
    monkeypatch.setattr(
        icu_benchmarks.tuning.hyperparameters, "execute_repeated_cv", lambda *a, **k: pytest.fail("Trained a model")
    )


@pytest.fixture(scope="module")
def tuned(tmp_path_factory):
    """A finished two-trial study on the demo data, tuned once for the module."""
    log_dir = tmp_path_factory.mktemp("tuned")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.chdir(ROOT)
        gin.clear_config()
        parse_config()
        tune(log_dir, plot=True)
        best = bound(TUNED)
    gin.clear_config()
    plt.close("all")
    return log_dir, best


def test_tuning_stores_trials_and_binds_the_best(tuned):
    log_dir, best = tuned
    study = load(log_dir / "hyperparameter_tuning_logs.db")
    assert len(study.trials) == 2 and all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    # Only the ranges are tuned; the bound max_iter and the fixed solver are not
    assert set(study.best_params) == TUNED
    assert best == study.best_params
    assert study.best_value == min(t.value for t in study.trials) and study.best_value > 0
    # The trials train in temporary directories inside the run directory, which are removed afterwards
    assert not [p for p in log_dir.iterdir() if p.is_dir()]


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="The optuna.visualization plots are plotly figures, so plt.savefig saves an empty matplotlib figure",
)
@pytest.mark.parametrize("name", ["param_importances.png", "optimization_history.png"])
def test_tuning_plots_are_not_blank(name, tuned):
    image = plt.imread(tuned[0] / name)
    assert image.min() < image.max()


@pytest.mark.parametrize("do_tune", [True, False])
def test_finished_checkpoint_binds_best_without_training(do_tune, tuned, tmp_path, monkeypatch):
    log_dir, best = tuned
    db = log_dir / "hyperparameter_tuning_logs.db"
    forbid_training(monkeypatch)
    tune(tmp_path, checkpoint=db, do_tune=do_tune)
    assert bound(TUNED) == best
    assert len(load(db).trials) == 2


def test_resuming_runs_only_the_remaining_trials(tuned, tmp_path):
    db = tmp_path / "checkpoint.db"
    db.write_bytes((tuned[0] / "hyperparameter_tuning_logs.db").read_bytes())
    tune(tmp_path, checkpoint=db, n_calls=3)
    study = load(db)
    assert len(study.trials) == 3
    assert bound(TUNED) == study.best_params


def test_missing_checkpoint_starts_a_new_study(tmp_path):
    tune(tmp_path, checkpoint=tmp_path / "missing.db", n_calls=1)
    assert not (tmp_path / "missing.db").exists()
    assert len(load(tmp_path / "hyperparameter_tuning_logs.db").trials) == 1


def test_without_tuning_or_checkpoint_binds_one_untrained_point(tmp_path, monkeypatch):
    forbid_training(monkeypatch)
    tune(tmp_path, do_tune=False)
    (trial,) = load(tmp_path / "hyperparameter_tuning_logs.db").trials
    assert bound(TUNED) == trial.params


@pytest.mark.parametrize("kwargs", [{"scopes": []}, {"folds_to_tune_on": None}], ids=["no_scopes", "no_folds"])
def test_tuning_is_skipped_without_scopes_or_folds(kwargs, tmp_path, monkeypatch):
    forbid_training(monkeypatch)
    tune(tmp_path, **kwargs)
    assert not any(tmp_path.iterdir())
    with pytest.raises(ValueError):
        gin.query_parameter("LogisticRegression.C")


def test_tuning_is_skipped_when_everything_is_bound(tmp_path, monkeypatch):
    forbid_training(monkeypatch)
    for name, value in {"C": 1.0, "penalty": "l2", "l1_ratio": 0.5}.items():
        gin.bind_parameter(f"LogisticRegression.{name}", value)
    tune(tmp_path)
    assert not any(tmp_path.iterdir())
    assert gin.query_parameter("LogisticRegression.C") == 1.0
