"""Attributions and explanation metrics for deep learning models (--explain-features with the explain extra)."""

import json
import sys
from types import SimpleNamespace

import gin
import numpy as np
import polars as pl
import pytest
import torch
from torch import nn

from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.constants import DataSegment as Segment, DataSplit as Split
from icu_benchmarks.data.loader import PredictionPolarsDataset
from icu_benchmarks.models.dl_models.bat import BAT
from icu_benchmarks.models.dl_models.grud import GRUD
from icu_benchmarks.models.dl_models.mamba import Mamba
from icu_benchmarks.models.dl_models.rnn import GRUNet, LSTMNet, RNNet
from icu_benchmarks.models.dl_models.tcn import TemporalConvNet
from icu_benchmarks.models.dl_models.transformer import LocalTransformer, Transformer
from icu_benchmarks.models.explain import explain_predictions, require_captum
from icu_benchmarks.models.explain.attribution import missing_baseline, variable_groups
from icu_benchmarks.models.explain.metrics import deletion_aopc, faithfulness_correlation, sparsity
from icu_benchmarks.models.train import train_common
from icu_benchmarks.run_utils import aggregate_results, check_explain_features

pytest.importorskip("captum")

COLUMNS = ["hr", "map", "noise", "MissingIndicator_hr", "MissingIndicator_map", "MissingIndicator_noise"]
VARS = {"GROUP": "stay_id", "SEQUENCE": "time", "LABEL": "label", "DYNAMIC": ["hr", "map", "noise"], "STATIC": []}


@pytest.fixture(autouse=True)
def clean_gin():
    gin.clear_config()
    yield
    gin.clear_config()


def hourly_data(stays=12, hours=5, seed=0, hourly_labels=True):
    """Stays with values, missingness indicators and a label at every hour (or once, at the last hour)."""
    rng = np.random.default_rng(seed)
    data = {}
    for split in (Split.train, Split.val, Split.test):
        n = stays * hours
        missing = rng.random((n, 3)) < 0.3
        values = np.where(missing, 0.0, rng.normal(size=(n, 3)))
        frame = {"stay_id": np.repeat(np.arange(stays), hours), "time": np.tile(np.arange(hours), stays)}
        frame |= {name: values[:, i] for i, name in enumerate(["hr", "map", "noise"])}
        frame |= {f"MissingIndicator_{name}": missing[:, i].astype(float) for i, name in enumerate(["hr", "map", "noise"])}
        label = (np.cumsum(values[:, 0].reshape(stays, hours), axis=1) > 0).astype(int)
        label[0], label[1] = 0, 1  # Both classes in every split
        if hourly_labels:
            outcome = pl.DataFrame({"stay_id": frame["stay_id"], "time": frame["time"], "label": label.ravel()})
        else:
            outcome = pl.DataFrame({"stay_id": np.arange(stays), "label": label[:, -1]})
        data[split] = {Segment.features: pl.DataFrame(frame), Segment.outcome: outcome}
    return data


class CumulativeLinear(nn.Module):
    """A causal model whose log-odds at hour t is the sum of x[s] @ w over s <= t, so attributions have a known value."""

    def __init__(self, weights):
        super().__init__()
        self.linear = nn.Linear(len(weights), 1, bias=False)
        self.linear.weight.data = torch.tensor([weights], dtype=torch.float32)
        self.target_transform = None

    def forward(self, x):
        score = self.linear(x).cumsum(dim=1)
        return torch.cat([torch.zeros_like(score), score], dim=-1)  # logits of class 0 and 1


def test_value_and_indicator_form_one_variable():
    names, group_of = variable_groups(["age", "hr", "MissingIndicator_age", "MissingIndicator_hr", "MissingIndicator_x"])
    assert names == ["age", "hr", "MissingIndicator_x"]  # an indicator without its value column stays on its own
    assert group_of.tolist() == [0, 1, 0, 1, 2]
    # "Not measured": the value at the training mean (0) and the indicator set
    assert missing_baseline(["age", "MissingIndicator_age"]).tolist() == [0.0, 1.0]


def test_metrics_on_known_attributions():
    assert sparsity(np.array([[0.0, 0.0, 0.0, 4.0]]))[0] == pytest.approx(0.75)  # all on one of 4 variables
    assert sparsity(np.array([[1.0, -1.0, 1.0, -1.0]]))[0] == pytest.approx(0.0)
    assert np.isnan(sparsity(np.zeros((1, 3)))[0])
    weights = np.array([[3.0, -1.0, 0.5]])
    subsets = np.array([[[1, 0, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1]]], dtype=bool)
    drops = (subsets * weights[:, None]).sum(axis=-1)  # an additive model drops by the removed attributions
    assert faithfulness_correlation(weights, subsets, drops)[0] == pytest.approx(1.0)
    assert faithfulness_correlation(-weights, subsets, drops)[0] == pytest.approx(-1.0)
    assert deletion_aopc(np.array([5.0]), np.array([[2.0, 1.0]]))[0] == pytest.approx(3.5)


@pytest.mark.parametrize("hourly_labels", [True, False])
def test_attributions_of_a_linear_model_are_exact(tmp_path, hourly_labels):
    data = hourly_data(stays=6, hours=4, hourly_labels=hourly_labels)
    dataset = PredictionPolarsDataset(data, split=Split.test, vars=VARS, ram_cache=False)
    assert dataset.get_input_names() == COLUMNS
    weights = [2.0, -1.0, 0.0, 0.5, 0.0, 0.0]  # hr's indicator also matters, noise does not
    model = CumulativeLinear(weights)

    results = explain_predictions(model, dataset, tmp_path, max_stays=6, ig_steps=8, faithfulness_subsets=6)

    attributions = pl.read_parquet(tmp_path / "test_attributions.parquet")
    labeled_hours = 6 * 4 if hourly_labels else 6
    assert attributions.columns == ["stay_id", "time", "method", "hr", "map", "noise"]
    assert attributions.group_by("method").len().sort("method")["len"].to_list() == [labeled_hours] * 3
    if not hourly_labels:
        assert attributions["time"].unique().to_list() == [3]  # the label sits at the last hour

    # Exact value: the sum over the hours up to the explained one of (x - baseline) @ w, per variable
    x = torch.stack([dataset[i][0] for i in range(len(dataset))]).numpy()
    baseline = missing_baseline(COLUMNS).numpy()
    contribution = np.cumsum((x - baseline) * np.array(weights), axis=1)  # (stays, hours, columns)
    for method in ["IntegratedGradients", "FeatureAblation"]:
        rows = attributions.filter(pl.col("method") == method)
        for stay_id, hour, hr, map_, noise in rows.select("stay_id", "time", "hr", "map", "noise").iter_rows():
            expected = contribution[stay_id, hour]
            assert hr == pytest.approx(expected[0] + expected[3], abs=1e-4)
            assert map_ == pytest.approx(expected[1], abs=1e-4)
            assert noise == pytest.approx(0.0, abs=1e-6)

    assert results["explain/IntegratedGradients/convergence_delta"] < 1e-4
    # For an additive model, removing a subset drops the output by exactly its attribution
    assert results["explain/IntegratedGradients/faithfulness"] == pytest.approx(1.0, abs=1e-3)
    assert results["explain/FeatureAblation/deletion"] >= results["explain/Random/deletion"]
    saved = json.loads((tmp_path / "test_explanation_metrics.json").read_text())
    assert saved == pytest.approx(results, nan_ok=True)
    importance = pl.read_parquet(tmp_path / "test_feature_importance.parquet")
    assert importance.columns == ["method", "variable", "mean_abs_attribution", "mean_attribution"]
    assert importance.height == 3 * 3


class CumulativeRegression(CumulativeLinear):
    def forward(self, x):
        return self.linear(x).cumsum(dim=1)  # a single output: the predicted (scaled) target


def test_regression_attributions_are_in_native_units(tmp_path):
    dataset = PredictionPolarsDataset(hourly_data(stays=4, hours=3), split=Split.test, vars=VARS, ram_cache=False)
    model = CumulativeRegression([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    (tmp_path / "scaled").mkdir()
    explain_predictions(model, dataset, tmp_path / "scaled", methods=("FeatureAblation",), metrics=())
    model.target_transform = SimpleNamespace(scale=0.5)  # the model predicts targets * 0.5 + offset
    explain_predictions(model, dataset, tmp_path, methods=("FeatureAblation",), metrics=())
    scaled = pl.read_parquet(tmp_path / "scaled" / "test_attributions.parquet")["hr"]
    native = pl.read_parquet(tmp_path / "test_attributions.parquet")["hr"]
    assert native.to_numpy() == pytest.approx(scaled.to_numpy() / 0.5)


class ExponentialRegression(CumulativeRegression):
    def forward(self, x):
        return super().forward(x).exp()  # not polynomial, so Integrated Gradients with few steps leaves a convergence delta


def test_regression_metrics_are_in_native_units(tmp_path):
    dataset = PredictionPolarsDataset(hourly_data(stays=4, hours=3), split=Split.test, vars=VARS, ram_cache=False)
    model = ExponentialRegression([1.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    kwargs = dict(methods=("IntegratedGradients",), metrics=("deletion",), ig_steps=2)
    scaled = explain_predictions(model, dataset, tmp_path, **kwargs)
    model.target_transform = SimpleNamespace(scale=0.5)
    native = explain_predictions(model, dataset, tmp_path, **kwargs)
    for metric in ["convergence_delta", "deletion"]:
        assert scaled[f"explain/IntegratedGradients/{metric}"] > 1e-3
        assert native[f"explain/IntegratedGradients/{metric}"] == pytest.approx(
            scaled[f"explain/IntegratedGradients/{metric}"] / 0.5
        )


DL_MODELS = {
    "GRUNet": lambda size: GRUNet(size, hidden_dim=4, layer_dim=1, num_classes=2),
    "LSTMNet": lambda size: LSTMNet(size, hidden_dim=4, layer_dim=1, num_classes=2),
    "RNNet": lambda size: RNNet(size, hidden_dim=4, layer_dim=1, num_classes=2),
    "GRUD": lambda size: GRUD(size, hidden_dim=4, num_classes=2),
    "TemporalConvNet": lambda size: TemporalConvNet(size, num_channels=[4], num_classes=2, kernel_size=2),
    "Transformer": lambda size: Transformer(size, hidden=4, heads=1, ff_hidden_mult=1, depth=1, num_classes=2),
    "LocalTransformer": lambda size: LocalTransformer(
        size, hidden=4, heads=1, ff_hidden_mult=1, depth=1, num_classes=2, local_context=2
    ),
    "BAT": lambda size: BAT(size, num_classes=2, value_embed_size=2, depth=1, heads=1, ff_hidden_mult=1),
    "Mamba": lambda size: Mamba(size, hidden=4, depth=1, num_classes=2, d_state=4),
}


@pytest.mark.parametrize("name", DL_MODELS)
def test_every_dl_model_can_be_explained(name, tmp_path):
    # GRU-D and BAT read the missingness indicators, so the data has them
    torch.manual_seed(0)
    dataset = PredictionPolarsDataset(hourly_data(stays=4, hours=4), split=Split.test, vars=VARS, ram_cache=False)
    model = DL_MODELS[name](torch.Size([4, 4, len(COLUMNS)]))
    model.set_trained_columns(dataset.get_feature_names())
    model.set_explain_features(True)
    model.train()
    results = explain_predictions(model, dataset, tmp_path, ig_steps=16, faithfulness_subsets=5)
    assert all(value is None or np.isfinite(value) for value in results.values())
    assert results["explain/IntegratedGradients/convergence_delta"] < 0.1
    assert pl.read_parquet(tmp_path / "test_attributions.parquet").height == 3 * 4 * 4
    assert model.training  # explained in eval mode, then restored to the mode it was handed over in


def test_train_common_explains_a_gru(tmp_path):
    gin.bind_parameter("PredictionPolarsDataset.vars", VARS)
    gin.bind_parameter("DLPredictionWrapper.loss", torch.nn.functional.cross_entropy)
    gin.bind_parameter("GRUNet.hidden_dim", 4)
    gin.bind_parameter("GRUNet.layer_dim", 1)
    gin.bind_parameter("GRUNet.num_classes", 2)
    gin.bind_parameter("explain_predictions.max_stays", 5)
    gin.bind_parameter("explain_predictions.ig_steps", 4)
    kwargs = dict(model=GRUNet, mode=RunMode.classification, epochs=1, batch_size=4, cpu=True, num_workers=0)
    train_common(hourly_data(), log_dir=tmp_path, weight="balanced", explain_features=True, **kwargs)

    metrics = json.loads((tmp_path / "test_explanation_metrics.json").read_text())
    for method in ["IntegratedGradients", "FeatureAblation", "Random"]:
        for metric in ["faithfulness", "deletion", "sparsity"]:
            assert np.isfinite(metrics[f"explain/{method}/{metric}"])
    attributions = pl.read_parquet(tmp_path / "test_attributions.parquet")
    assert attributions.height == 3 * 5 * 5  # 3 methods, 5 stays, every one of their 5 labeled hours
    assert {"hr", "map", "noise"} <= set(attributions.columns)


def test_explanations_use_the_tested_best_checkpoint(tmp_path, monkeypatch):
    # Shift the weights after training: testing must reload the best checkpoint, and the explanations must use it
    import icu_benchmarks.models.train as train

    class ShiftingTrainer(train.Trainer):
        def fit(self, model, *args, **kwargs):
            super().fit(model, *args, **kwargs)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(1.0)

    explained = {}
    monkeypatch.setattr(train, "Trainer", ShiftingTrainer)
    monkeypatch.setattr(train, "explain_predictions", lambda model, *args: explained.update(model.state_dict()))
    gin.bind_parameter("PredictionPolarsDataset.vars", VARS)
    gin.bind_parameter("DLPredictionWrapper.loss", torch.nn.functional.cross_entropy)
    for name, value in {"hidden_dim": 4, "layer_dim": 1, "num_classes": 2}.items():
        gin.bind_parameter(f"GRUNet.{name}", value)
    kwargs = dict(model=GRUNet, mode=RunMode.classification, epochs=2, batch_size=4, cpu=True, num_workers=0)
    train_common(hourly_data(), log_dir=tmp_path, weight="balanced", explain_features=True, **kwargs)
    best = torch.load(tmp_path / "model.ckpt", weights_only=False)["state_dict"]
    assert explained.keys() == best.keys()
    assert all(torch.equal(explained[name].cpu(), best[name].cpu()) for name in best)


def test_explanations_are_opt_in(tmp_path):
    gin.bind_parameter("PredictionPolarsDataset.vars", VARS)
    gin.bind_parameter("DLPredictionWrapper.loss", torch.nn.functional.cross_entropy)
    for name, value in {"hidden_dim": 4, "layer_dim": 1, "num_classes": 2}.items():
        gin.bind_parameter(f"GRUNet.{name}", value)
    kwargs = dict(model=GRUNet, mode=RunMode.classification, epochs=1, batch_size=4, cpu=True, num_workers=0)
    train_common(hourly_data(), log_dir=tmp_path, weight="balanced", **kwargs)
    assert not (tmp_path / "test_attributions.parquet").exists()


def test_multiclass_models_are_rejected():
    model = GRUNet((4, 5, 6), hidden_dim=4, layer_dim=1, num_classes=3)
    with pytest.raises(NotImplementedError, match="binary classification and regression"):
        model.set_explain_features(True)


def test_multiclass_models_are_rejected_before_tuning():
    gin.parse_config(
        "import icu_benchmarks.models.dl_models.rnn\n"
        "import icu_benchmarks.tuning.gin_utils\n"
        "train_common.model = @GRUNet\n"
        "NUM_CLASSES = 3\n"
        "model/hyperparameter.num_classes = %NUM_CLASSES"
    )
    with pytest.raises(NotImplementedError, match="not 3 classes"):
        check_explain_features(True)


def test_pandas_datasets_are_rejected_before_training(tmp_path):
    gin.parse_config("import icu_benchmarks.models.dl_models.rnn\ntrain_common.model = @GRUNet\ntrain_common.polars = False")
    with pytest.raises(NotImplementedError, match="Polars"):
        check_explain_features(True)
    kwargs = dict(model=GRUNet, mode=RunMode.classification, epochs=1, batch_size=4, cpu=True, num_workers=0)
    with pytest.raises(NotImplementedError, match="Polars"):
        train_common(hourly_data(), log_dir=tmp_path, explain_features=True, polars=False, **kwargs)
    assert not (tmp_path / "model.ckpt").exists()


def test_dl_models_pass_the_check_before_tuning():
    gin.parse_config("import icu_benchmarks.models.dl_models.rnn\ntrain_common.model = @GRUNet")
    check_explain_features(True)


def test_missing_captum_fails_early_with_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "captum", None)  # import captum raises ImportError
    with pytest.raises(ImportError, match="uv sync --extra explain"):
        require_captum()
    gin.parse_config("import icu_benchmarks.models.dl_models.rnn\ntrain_common.model = @GRUNet")
    with pytest.raises(ImportError, match="captum"):
        check_explain_features(True)


def test_explanation_metrics_are_averaged_over_folds(tmp_path):
    for fold, (auc, faithfulness) in enumerate([(0.7, 0.5), (0.9, 0.7)]):
        fold_dir = tmp_path / "repetition_0" / f"fold_{fold}"
        fold_dir.mkdir(parents=True)
        (fold_dir / "test_metrics.json").write_text(json.dumps({"AUC": auc}))
        (fold_dir / "test_explanation_metrics.json").write_text(
            json.dumps({"explain/IntegratedGradients/faithfulness": faithfulness})
        )
        pl.DataFrame(
            {"method": ["Random"], "variable": ["hr"], "mean_abs_attribution": [1.0], "mean_attribution": [0.0]}
        ).write_parquet(fold_dir / "test_feature_importance.parquet")
    aggregate_results(tmp_path)
    accumulated = json.loads((tmp_path / "accumulated_test_metrics.json").read_text())
    assert accumulated["avg"]["explain/IntegratedGradients/faithfulness"] == pytest.approx(0.6)
    importance = pl.read_parquet(tmp_path / "aggregated_feature_importance.parquet")
    assert importance.height == 2 and set(importance["fold"]) == {"fold_0", "fold_1"}
