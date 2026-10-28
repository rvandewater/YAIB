"""End to end: train on the MIMIC demo through the CLI entry point, then evaluate the saved model on the eICU demo."""

import json
from pathlib import Path

import gin
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


def run(args, monkeypatch):
    # main reads the configs relative to the repository. setup_logging expects the handlers that lightning only adds
    # when the root logger has none (not under pytest), and it would silence warnings for the remaining tests.
    monkeypatch.chdir(ROOT)
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
