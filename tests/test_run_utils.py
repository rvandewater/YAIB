"""Run helpers must lay out run directories, aggregate fold results and bind the configuration they are given."""

import json
import logging
import sys
from argparse import Namespace
from pathlib import Path

import gin
import polars as pl
import pytest

from icu_benchmarks.constants import RunMode
from icu_benchmarks.run import get_mode
from icu_benchmarks.run_utils import (
    aggregate_results,
    create_run_dir,
    get_config_files,
    import_preprocessor,
    load_pretrained_imputation_model,
    log_full_line,
    name_datasets,
)
from icu_benchmarks.tuning.gin_utils import bind_gin_params, get_gin_hyperparameters
from icu_benchmarks.wandb_utils import set_wandb_experiment_name, update_wandb_config, wandb_log, wandb_running

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
