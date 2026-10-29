"""End to end: train on the MIMIC demo through the CLI entry point, then evaluate the saved model on the eICU demo."""

import json
import sys
from pathlib import Path

import gin
import joblib
import numpy as np
import pytest

import icu_benchmarks.run
from icu_benchmarks.run import main

ROOT = Path(__file__).parent.parent
FAST = ["execute_repeated_cv.cv_repetitions_to_train=1", "execute_repeated_cv.cv_folds_to_train=3"]


@pytest.fixture(autouse=True)
def clean_gin():
    gin.clear_config()
    yield
    gin.clear_config()


def run(args, monkeypatch, cwd=ROOT):
    # main reads the configs relative to the repository. setup_logging (#260) expects the handlers that lightning only adds
    # when the root logger has none (not under pytest), and it would silence warnings for the remaining tests.
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(icu_benchmarks.run, "setup_logging", lambda *args: None)
    main(args)
    gin.clear_config()


def test_train_then_evaluate_on_demo_data(tmp_path, monkeypatch):
    common = ["-t", "BinaryClassification", "-tn", "Mortality24", "-m", "LogisticRegression", "--cpu", "-l", str(tmp_path)]
    run(["-d", "demo_data/mortality24/mimic_demo", "-n", "mimic_demo", *common, "-hp", *FAST], monkeypatch)
    (run_dir,) = (tmp_path / "mimic_demo" / "Mortality24" / "LogisticRegression").iterdir()
    folds = sorted(p.relative_to(run_dir).as_posix() for p in run_dir.glob("repetition_*/fold_*"))
    assert folds == ["repetition_0/fold_0", "repetition_0/fold_1", "repetition_0/fold_2"]
    summary = json.loads((run_dir / "accumulated_test_metrics.json").read_text())
    assert 0 <= summary["avg"]["AUC"] <= 1 and summary["execution_time"] > 0
    source = run_dir / "repetition_0" / "fold_0"
    assert {"last.joblib", "test_metrics.json", "durations.json", "train_config.gin"} <= {p.name for p in source.iterdir()}

    run(
        [
            "-d",
            "demo_data/mortality24/eicu_demo",
            "-n",
            "eicu_demo",
            *common,
            "--eval",
            "-sn",
            "mimic",
            "--source-dir",
            str(source),
        ],
        monkeypatch,
    )
    (eval_dir,) = (tmp_path / "eicu_demo" / "Mortality24" / "LogisticRegression" / "_from_mimic").iterdir()
    evaluated = json.loads((eval_dir / "repetition_0" / "fold_0" / "test_metrics.json").read_text())
    assert 0 <= evaluated["AUC"] <= 1
    # Evaluation reuses the trained model instead of fitting a new one
    assert not (eval_dir / "repetition_0" / "fold_0" / "last.joblib").exists()


def test_complete_train_of_dl_model_on_hourly_task(tmp_path, monkeypatch):
    hyperparams = ["train_common.epochs=1", "GRUNet.hidden_dim=4", "GRUNet.layer_dim=1"]
    args = ["-d", "demo_data/aki/mimic_demo", "-t", "BinaryClassification", "-tn", "AKI", "-m", "GRU", "--cpu"]
    run([*args, "--complete-train", "-l", str(tmp_path), "-hp", *hyperparams], monkeypatch)
    # Without a name, the run is named after the data directory
    (run_dir,) = (tmp_path / "mimic_demo" / "AKI" / "GRU").iterdir()
    assert [p.name for p in run_dir.glob("repetition_*/fold_*")] == ["fold_0"]
    fold = run_dir / "repetition_0" / "fold_0"
    assert (fold / "last.ckpt").exists() and not (fold / "test_metrics.json").exists()
    assert "GRUNet.hidden_dim = 4" in (fold / "train_config.gin").read_text()


def test_unknown_model_is_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Invalid task or model"):
        run(
            ["-d", "demo_data/mortality24/mimic_demo", "-t", "BinaryClassification", "-m", "NoSuchModel", "-l", str(tmp_path)],
            monkeypatch,
        )


MORTALITY = ["-t", "BinaryClassification", "-tn", "Mortality24", "-m", "LogisticRegression", "--cpu"]
ONE_FOLD = ["execute_repeated_cv.cv_repetitions_to_train=1", "execute_repeated_cv.cv_folds_to_train=1"]
EICU = ["-d", "demo_data/mortality24/eicu_demo", "-n", "eicu_demo", *MORTALITY]


def bound(fold: Path, name: str):
    """The value a run saved for a gin parameter."""
    gin.clear_config()
    gin.parse_config_file(fold / "train_config.gin")
    value = gin.query_parameter(name)
    gin.clear_config()
    return value


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    """One fold of a model trained on the MIMIC demo, to start the transfer runs from."""
    log_dir = tmp_path_factory.mktemp("source")
    with pytest.MonkeyPatch.context() as monkeypatch:
        gin.clear_config()
        run(
            ["-d", "demo_data/mortality24/mimic_demo", "-n", "mimic_demo", *MORTALITY, "-l", str(log_dir), "-hp", *ONE_FOLD],
            monkeypatch,
        )
    (fold,) = log_dir.glob("mimic_demo/Mortality24/LogisticRegression/*/repetition_0/fold_0")
    return fold


def test_fine_tune_retrains_on_target_data(source, tmp_path, monkeypatch):
    run([*EICU, "-l", str(tmp_path), "-ft", "40", "-sn", "mimic", "--source-dir", str(source)], monkeypatch)
    (run_dir,) = (tmp_path / "eicu_demo" / "Mortality24" / "LogisticRegression" / "_from_mimic" / "fine_tune_40").iterdir()
    # The fold settings and the hyperparameters come from the source run
    assert [p.relative_to(run_dir).as_posix() for p in run_dir.glob("repetition_*/fold_*")] == ["repetition_0/fold_0"]
    fold = run_dir / "repetition_0" / "fold_0"
    assert bound(fold, "LogisticRegression.C") == bound(source, "LogisticRegression.C")
    # Unlike evaluation, fine-tuning fits and saves a new model
    assert not np.allclose(joblib.load(fold / "last.joblib").coef_, joblib.load(source / "last.joblib").coef_)
    assert 0 <= json.loads((fold / "test_metrics.json").read_text())["AUC"] <= 1


def test_samples_train_new_model_with_source_config(source, tmp_path, monkeypatch):
    run([*EICU, "-l", str(tmp_path), "--samples", "40", "--source-dir", str(source)], monkeypatch)
    # The run directory is named after the fine-tune size instead of the samples (#241), so it is not checked here
    (fold,) = (tmp_path / "eicu_demo" / "Mortality24" / "LogisticRegression").glob("*/*/repetition_0/fold_0")
    assert bound(fold, "LogisticRegression.C") == bound(source, "LogisticRegression.C")
    assert (fold / "last.joblib").exists() and (fold / "test_metrics.json").exists()


def test_experiment_config_replaces_model_and_task_configs(tmp_path, monkeypatch):
    # Experiment configs live in configs/experiments, which the repository does not ship
    configs = tmp_path / "configs"
    configs.mkdir()
    for path in (ROOT / "configs").iterdir():
        (configs / path.name).symlink_to(path)
    (configs / "experiments").mkdir()
    (configs / "experiments" / "small_lr.gin").write_text(
        'include "configs/prediction_models/LogisticRegression.gin"\n'
        'include "configs/tasks/BinaryClassification.gin"\n'
        "LogisticRegression.C = 0.25\n" + "".join(f"{binding}\n" for binding in ONE_FOLD)
    )
    data = str(ROOT / "demo_data" / "mortality24" / "mimic_demo")
    run(
        ["-d", data, "-n", "mimic_demo", *MORTALITY, "-e", "small_lr", "-l", str(tmp_path / "logs")], monkeypatch, cwd=tmp_path
    )
    # The experiment names the log directory instead of the task and model
    (fold,) = (tmp_path / "logs" / "mimic_demo" / "small_lr").glob("*/repetition_*/fold_*")
    assert bound(fold, "LogisticRegression.C") == 0.25
    assert (fold / "test_metrics.json").exists()


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#274: main binds the custom preprocessor before it parses the task config, which binds the default one again",
)
def test_custom_preprocessor_is_used(tmp_path, monkeypatch):
    marker = tmp_path / "applied"
    path = tmp_path / "custom.py"
    path.write_text(
        "from pathlib import Path\n"
        "from icu_benchmarks.data.preprocessor import PolarsClassificationPreprocessor\n\n\n"
        "class CustomPreprocessor(PolarsClassificationPreprocessor):\n"
        "    def apply(self, data, vars):\n"
        f"        Path({str(marker)!r}).touch()\n"
        "        return super().apply(data, vars)\n"
    )
    monkeypatch.setitem(sys.modules, "preprocessor", None)
    args = ["-d", "demo_data/mortality24/mimic_demo", *MORTALITY, "-l", str(tmp_path / "logs"), "-p", str(path)]
    run([*args, "-hp", *ONE_FOLD], monkeypatch)
    assert marker.exists()
