"""Run helpers must lay out run directories, aggregate fold results and bind the configuration they are given."""

import json
import logging
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import gin
import polars as pl
import pytest

from icu_benchmarks.constants import RunMode
from icu_benchmarks.run import get_mode
import icu_benchmarks.wandb_utils
from icu_benchmarks.run_utils import (
    aggregate_results,
    build_parser,
    create_run_dir,
    get_config_files,
    import_preprocessor,
    load_pretrained_imputation_model,
    log_full_line,
    name_datasets,
)
from icu_benchmarks.tuning.gin_utils import bind_gin_params, get_gin_hyperparameters
from icu_benchmarks.wandb_utils import (
    apply_wandb_sweep,
    set_wandb_experiment_name,
    update_wandb_config,
    wandb_log,
    wandb_running,
)

CONFIGS = Path(__file__).parent.parent / "configs"


@pytest.fixture(autouse=True)
def clean_gin():
    gin.clear_config()
    yield
    gin.clear_config()


def test_create_run_dir_is_unique(tmp_path):
    first = create_run_dir(tmp_path)
    second = create_run_dir(tmp_path, randomly_searched_params="lr_0.1")
    assert first != second and first.is_dir() and second.parent == tmp_path
    assert (second / "lr_0.1").is_file()


def write_fold(fold_dir: Path, auc: float, loss: float, shap_rows: int = 0):
    fold_dir.mkdir(parents=True)
    (fold_dir / "test_metrics.json").write_text(json.dumps({"AUC": auc, "loss": loss}))
    (fold_dir / "durations.json").write_text(json.dumps({"train_duration": "0:00:01", "epochs": 2}))
    if shap_rows:
        pl.DataFrame({"x": [0.1] * shap_rows}).write_parquet(fold_dir / "test_shap_values.parquet")


def test_aggregate_results(tmp_path):
    write_fold(tmp_path / "repetition_0" / "fold_0", 0.6, 1.0, shap_rows=2)
    write_fold(tmp_path / "repetition_0" / "fold_1", 0.8, 3.0, shap_rows=3)
    write_fold(tmp_path / "repetition_1" / "fold_0", 0.7, 2.0)
    aggregate_results(tmp_path)

    per_fold = json.loads((tmp_path / "aggregated_test_metrics.json").read_text())
    assert per_fold["repetition_0"]["fold_1"] == {"AUC": 0.8, "loss": 3.0, "train_duration": "0:00:01", "epochs": 2}
    summary = json.loads((tmp_path / "accumulated_test_metrics.json").read_text())
    assert summary["avg"] == pytest.approx({"AUC": 0.7, "loss": 2.0, "epochs": 2})
    # Population standard deviation divided by sqrt(n)
    assert summary["std"]["loss"] == pytest.approx((2 / 3) ** 0.5 / 3**0.5)
    low, high = summary["CI_0.95"]["AUC"]
    assert low < 0.7 < high
    assert summary["execution_time"] == 0.0
    assert pl.read_parquet(tmp_path / "aggregated_shap_values.parquet").height == 5


def test_get_config_files():
    tasks, models = get_config_files(CONFIGS)
    assert {"BinaryClassification", "Regression"} <= set(tasks)
    assert {"LGBMClassifier", "GRU", "LogisticRegression"} <= set(models)
    assert "common" not in tasks + models
    assert get_config_files(Path("does_not_exist")) == ([], [])


def test_get_config_files_skips_common_files(tmp_path):
    for config in ("tasks/Task.gin", "tasks/common.gin", "prediction_models/Model.gin", "prediction_models/common"):
        (tmp_path / config).parent.mkdir(exist_ok=True)
        (tmp_path / config).touch()
    (tmp_path / "prediction_models" / "nested").mkdir()
    assert get_config_files(tmp_path) == (["Task"], ["Model"])


def test_unreadable_config_dir_has_no_configs(monkeypatch):
    def unreadable(*args):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "glob", unreadable)
    assert get_config_files(CONFIGS) == ([], [])


def test_name_datasets_and_run_mode_bindings():
    name_datasets("mimic", "mimic", "eicu")
    assert gin.query_parameter("train_common.dataset_names") == {"train": "mimic", "val": "mimic", "test": "eicu"}
    gin.bind_parameter("Run.mode", "Regression")
    assert get_mode() is RunMode.regression


def test_log_full_line_centers_message(caplog):
    with caplog.at_level(logging.INFO):
        log_full_line("done", char="=")
    message = caplog.messages[-1]
    assert "=done=" in message and set(message.replace("done", "")) == {"="}


def test_missing_pretrained_imputation_model_is_ignored(tmp_path):
    assert load_pretrained_imputation_model(None) is None
    assert load_pretrained_imputation_model(str(tmp_path / "missing.pt")) is None


def test_import_preprocessor_without_custom_class_binds_nothing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "preprocessor", None)
    path = tmp_path / "custom.py"
    path.write_text("class OtherPreprocessor:\n    pass\n")
    import_preprocessor(str(path))
    with pytest.raises(ValueError):
        gin.query_parameter("preprocess.preprocessor")


def test_import_custom_preprocessor(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "preprocessor", None)
    path = tmp_path / "custom.py"
    path.write_text("class CustomPreprocessor:\n    marker = 'custom'\n")
    import_preprocessor(str(path))
    assert gin.query_parameter("preprocess.preprocessor").marker == "custom"


@gin.configurable
class TunedModel:
    def __init__(self, depth=1, lr=0.1, layers=1, dropout=0.0):
        pass


def test_get_gin_hyperparameters_binds_fixed_values_and_returns_ranges():
    gin.bind_parameter("TunedModel.depth", 3)
    to_tune = get_gin_hyperparameters(TunedModel, depth=(1, 5), lr=(1e-4, 1e-2, "log"), layers=[1, 2], dropout=0.1)
    # depth is bound already and dropout is fixed, so only lr and layers are left to tune
    assert to_tune == {"TunedModel.lr": (1e-4, 1e-2, "log"), "TunedModel.layers": [1, 2]}
    assert gin.query_parameter("TunedModel.dropout") == 0.1 and gin.query_parameter("TunedModel.depth") == 3
    bind_gin_params({"TunedModel.lr": 1e-3})
    assert gin.query_parameter("TunedModel.lr") == 1e-3


def test_wandb_helpers_without_a_run():
    assert not wandb_running()
    update_wandb_config({"a": 1})
    wandb_log({"a": 1})
    args = Namespace(name=None, data_dir="data/mimic_demo", model="GRU", modalities=None, fine_tune=None, eval=True)
    args.source_name = "eicu"
    set_wandb_experiment_name(args, "run")
    assert args.name == "mimic_demo"


@pytest.fixture
def fake_wandb(monkeypatch):
    """A running wandb whose config and run are plain objects, so nothing reaches the network."""
    calls = {"init": [], "config": {}, "log": []}

    class Config(dict):
        def update(self, values, **kwargs):
            super().update(values)
            calls["config"].update(values)

    wandb = icu_benchmarks.wandb_utils.wandb
    monkeypatch.setattr(wandb, "init", lambda **kwargs: calls["init"].append(kwargs))
    monkeypatch.setattr(wandb, "config", Config({"TunedModel.lr": 0.01, "TunedModel.layers": 3, "seed": 7}))
    monkeypatch.setattr(wandb, "run", SimpleNamespace(name=None))
    monkeypatch.setattr(wandb, "log", calls["log"].append)
    return calls


def test_apply_wandb_sweep_turns_the_sweep_into_gin_bindings(fake_wandb, tmp_path):
    args = build_parser().parse_args(["-d", "data", "-t", "BinaryClassification", "-l", str(tmp_path), "--wandb-sweep"])
    icu_benchmarks.wandb_utils.wandb.config["TunedModel.dropout"] = "none"
    args = apply_wandb_sweep(args)
    assert fake_wandb["init"] == [{"allow_val_change": True, "dir": tmp_path}]
    # Sweep values override the arguments of the same name and are passed on as hyperparameters
    assert args.seed == 7
    assert args.hyperparams[:2] == ["TunedModel.lr=0.01", "TunedModel.layers=3"]
    assert "TunedModel.dropout='none'" in args.hyperparams
    gin.parse_config([h for h in args.hyperparams if h.startswith("TunedModel")])
    assert gin.query_parameter("TunedModel.lr") == 0.01 and gin.query_parameter("TunedModel.dropout") == "none"


def test_apply_wandb_sweep_keeps_cli_hyperparameters(fake_wandb):
    args = build_parser().parse_args(["-d", "data", "-t", "BinaryClassification", "-hp", "TunedModel.depth=2"])
    assert apply_wandb_sweep(args).hyperparams[:2] == ["TunedModel.depth=2", "TunedModel.lr=0.01"]


@pytest.mark.parametrize(
    "flags,suffix",
    [
        ({}, ""),
        ({"modalities": ["STATIC"]}, "_mods_['STATIC']"),
        ({"fine_tune": 10, "eval": True}, "_source_eicu_fine-tune_10_samples"),
        ({"eval": True}, "_source_eicu"),
        ({"samples": 50}, "_train_size_50_samples"),
        ({"complete_train": True}, "_complete_training"),
    ],
)
def test_wandb_run_name_describes_the_run(flags, suffix, fake_wandb):
    args = Namespace(name="mimic", data_dir="data", model="GRU", source_name="eicu", **{"modalities": None, **flags})
    for flag in ("fine_tune", "eval", "samples", "complete_train"):
        setattr(args, flag, getattr(args, flag, None))
    set_wandb_experiment_name(args, "run")
    expected = f"run_GRU_mimic{suffix}"
    assert icu_benchmarks.wandb_utils.wandb.run.name == expected and fake_wandb["config"]["run-name"] == expected


def test_wandb_helpers_with_a_run(fake_wandb):
    assert wandb_running()
    update_wandb_config({"a": 1})
    wandb_log({"loss": 0.5})
    assert fake_wandb["config"]["a"] == 1 and fake_wandb["log"] == [{"loss": 0.5}]
