"""Each imputation model must take a training step to a finite loss and impute without overwriting observed values."""

import importlib
import inspect
import re
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import gin
import pandas as pd
import pytest
import torch

import icu_benchmarks.models.train  # noqa: F401  (registers train_common for the config tests)
from icu_benchmarks.imputation import baselines
from icu_benchmarks.imputation.csdi import CSDI
from icu_benchmarks.imputation.diffusion import SimpleDiffusionModel as Diffusion
from icu_benchmarks.imputation.diffwave import DiffWaveImputer
from icu_benchmarks.imputation.mlp import MLPImputation
from icu_benchmarks.imputation.np import NPImputation
from icu_benchmarks.imputation.rnn import BRNNImputation, RNNImputation
from icu_benchmarks.imputation.simple_diffusion import SimpleDiffusionModel as SimpleDiffusion

BATCH, STEPS, FEATURES = 2, 8, 6  # Simple_Diffusion hard-codes six features
INPUT_SIZE = (BATCH, STEPS, FEATURES)
CONFIGS = sorted(Path(__file__).parents[1].glob("configs/imputation_models/*.gin"))
INSTALLED = {package: find_spec(package) is not None for package in ["hyperimpute", "pypots", "einops", "opt_einsum"]}
# The SSSD models' S4 layers need einops and opt_einsum from the imputation extra
S4_INSTALLED = INSTALLED["einops"] and INSTALLED["opt_einsum"]
if S4_INSTALLED:
    from icu_benchmarks.imputation.sssds4 import SSSDS4
    from icu_benchmarks.imputation.sssdsa import SSSDSA

DIFFUSION_STEPS = dict(diffusion_time_steps=3, beta_0=1e-4, beta_T=2e-2)
STEP_EMBEDDING = dict(diffusion_step_embed_dim_in=8, diffusion_step_embed_dim_mid=8, diffusion_step_embed_dim_out=8)
CSDI_PARAMS = dict(
    input_size=INPUT_SIZE,
    time_step_embedding_size=4,
    feature_embedding_size=4,
    unconditional=False,
    target_strategy="hist",
    num_diffusion_steps=3,
    diffusion_step_embedding_dim=8,
    n_attention_heads=1,
    num_residual_layers=1,
    noise_schedule="quad",
    beta_start=1e-4,
    beta_end=0.5,
    n_samples=2,
    conv_channels=4,
)
S4 = dict(s4_lmax=STEPS, s4_d_state=4, s4_dropout=0.0, s4_bidirectional=True)
SKLEARN = {
    "knn": lambda: baselines.KNNImputation(n_neighbors=2),
    "mice": lambda: baselines.MICEImputation(max_iter=2, verbose=0),
    "mean": baselines.MeanImputation,
    "median": baselines.MedianImputation,
    "zero": baselines.ZeroImputation,
    "most_frequent": baselines.MostFrequentImputation,
}
MODELS = {
    **SKLEARN,
    "mlp": lambda: MLPImputation(input_size=INPUT_SIZE, num_hidden_layers=1, hidden_layer_size=5),
    "rnn": lambda: RNNImputation(input_size=INPUT_SIZE, hidden_size=5),
    "brnn": lambda: BRNNImputation(input_size=INPUT_SIZE, hidden_size=5),
    "csdi": lambda: CSDI(**CSDI_PARAMS),
    "csdi_unconditional": lambda: CSDI(**CSDI_PARAMS | dict(unconditional=True, noise_schedule="linear")),
    "csdi_mix": lambda: CSDI(**CSDI_PARAMS | dict(target_strategy="mix")),  # Mixes random and historical targets
    "diffusion": lambda: Diffusion(
        input_size=INPUT_SIZE, n_onedirectional_conv=2, T=3, min_noise=1e-4, max_noise=0.02, noise_scheduler="linear"
    ),
    "simple_diffusion": lambda: SimpleDiffusion(input_size=INPUT_SIZE),
    "diffwave": lambda: DiffWaveImputer(
        in_channels=FEATURES,
        res_channels=4,
        skip_channels=4,
        out_channels=FEATURES,
        num_res_layers=2,
        dilation_cycle=2,
        **STEP_EMBEDDING,
        **DIFFUSION_STEPS,
    ),
    "sssds4": lambda: SSSDS4(
        input_size=INPUT_SIZE,
        res_channels=4,
        skip_channels=4,
        num_res_layers=1,
        s4_layernorm=True,
        **S4,
        **STEP_EMBEDDING,
        **DIFFUSION_STEPS,
    ),
    "sssdsa": lambda: SSSDSA(
        input_size=INPUT_SIZE,
        d_model=4,
        n_layers=1,
        pool=[2, 2],
        expand=2,
        ff=2,
        glu=True,
        unet=True,
        dropout=0.0,
        in_channels=FEATURES,
        out_channels=FEATURES,
        label_embed_dim=4,
        label_embed_classes=2,
        bidirectional=True,
        **S4,
        **STEP_EMBEDDING,
        **DIFFUSION_STEPS,
    ),
    "np": lambda: NPImputation(
        input_size=INPUT_SIZE,
        encoder_layers=1,
        encoder_h_dim=4,
        decoder_layers=1,
        decoder_h_dim=4,
        r_dim=3,
        z_dim=3,
        train_sample_times=2,
        val_sample_times=2,
        test_sample_times=2,
        predict_sample_times=2,
    ),
}
if not S4_INSTALLED:
    MODELS = {name: factory for name, factory in MODELS.items() if not name.startswith("sssd")}


def batch(seed=0):
    """A batch shaped like ImputationPandasDataset items: amputated values, amputation mask, target, target missingness."""
    generator = torch.Generator().manual_seed(seed)
    target = torch.randn(INPUT_SIZE, generator=generator)
    target_missing = torch.zeros(INPUT_SIZE)
    target_missing[0, 0, 0] = target_missing[1, 2, 3] = 1
    target[target_missing.bool()] = 0
    # Values missing in the target are always part of the amputation mask
    amputation_mask = ((torch.rand(INPUT_SIZE, generator=generator) < 0.3) | target_missing.bool()).float()
    amputated = target.masked_fill(amputation_mask.bool(), float("nan"))
    return amputated, amputation_mask, target, target_missing


def ready(name):
    """Build the model and run what Lightning (or, for the sklearn baselines, train_common) does before the first step."""
    torch.manual_seed(0)
    model = MODELS[name]()
    model.on_fit_start()
    if name in SKLEARN:
        model.fit(SimpleNamespace(amputated_values=pd.DataFrame(batch()[0].reshape(-1, FEATURES).numpy())), None)
    return model


def run_step(model, prefix):
    """Call a Lightning step hook the way Lightning does and return what the model logged as its loss."""
    logged = {}
    model.log = lambda key, value, **kwargs: logged.setdefault(key, value)
    hook = getattr(model, {"train": "training_step", "val": "validation_step", "test": "test_step"}[prefix])
    args = (batch(), 0) if len(inspect.signature(hook).parameters) > 1 else (batch(),)
    return hook(*args), logged[f"{prefix}/loss"]


@pytest.mark.parametrize("name", MODELS)
def test_training_step_gives_finite_loss(name):
    model = ready(name)
    loss, logged = run_step(model, "train")
    assert loss.ndim == 0 and torch.isfinite(loss) and loss.item() == pytest.approx(logged)
    if model.requires_backprop:
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


@pytest.mark.parametrize("prefix", ["val", "test"])
@pytest.mark.parametrize("name", MODELS)
def test_evaluation_step_gives_finite_loss(name, prefix):
    model = ready(name)
    with torch.inference_mode():  # As in Lightning's validation and test loops
        _, logged = run_step(model, prefix)
    assert torch.isfinite(torch.as_tensor(logged))


@pytest.mark.parametrize("name", [name for name in MODELS if name not in {"mlp", "rnn", "brnn"}])  # Covered in test_imputation
def test_predict_fills_only_missing_values(name):
    model = ready(name)
    data = batch()[0]
    observed = ~torch.isnan(data)
    with torch.no_grad():
        imputed = model.predict(data.clone())
    assert imputed.shape == data.shape and torch.isfinite(imputed).all()
    assert torch.equal(imputed[observed], data[observed])  # Observed values are returned unchanged


@pytest.mark.parametrize("name", ["mlp", "rnn", "diffusion", "simple_diffusion", "diffwave"])
def test_predict_builds_no_autograd_graph(name):
    # Called directly, without the caller's torch.no_grad(); a diffusion graph spans all T reverse steps
    imputed = ready(name).predict(batch()[0])
    assert not imputed.requires_grad


def test_diffusion_test_step_imputes_values_in_place(monkeypatch):
    model = ready("diffusion")
    target = batch()[2]
    # With a reverse step that recovers the target exactly, imputing at the missing positions reproduces the target
    monkeypatch.setattr(model, "backward_diffusion_sample", lambda *args, **kwargs: target.clone())
    _, logged = run_step(model, "test")
    assert logged == pytest.approx(0)


@pytest.mark.parametrize("name", ["diffusion", "simple_diffusion"])
def test_diffusion_schedule_moves_with_the_model(name):
    model = MODELS[name]()
    assert {"betas", "sqrt_alphas_cumprod", "posterior_variance"} <= dict(model.named_buffers()).keys()
    # Derived from the configuration and not saved, so checkpoints from before the schedule became buffers still load
    assert "betas" not in model.state_dict()
    model.to(torch.float64)
    assert model.betas.dtype == torch.float64


ACCELERATOR = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None


@pytest.mark.skipif(ACCELERATOR is None, reason="needs a CUDA or MPS device")
@pytest.mark.parametrize("name", ["diffusion", "simple_diffusion", "diffwave", "sssdsa"])
def test_restored_diffusion_model_predicts_on_accelerator(name):
    # A restored model is moved to the device without on_fit_start, so everything predict uses must follow .to()
    if name == "sssdsa" and not S4_INSTALLED:
        pytest.skip("SSSDSA needs einops and opt_einsum")
    torch.manual_seed(0)
    model = MODELS[name]().to(ACCELERATOR)
    with torch.no_grad():
        imputed = model.predict(batch()[0].to(ACCELERATOR))
    assert imputed.device.type == ACCELERATOR and torch.isfinite(imputed).all()


@pytest.mark.parametrize("schedule", ["linear", "quadratic", "cosine", "sigmoid"])
def test_diffusion_noise_schedules_are_valid(schedule):
    model = Diffusion(
        input_size=INPUT_SIZE, n_onedirectional_conv=1, T=5, min_noise=1e-4, max_noise=0.02, noise_scheduler=schedule
    )
    assert model.betas.shape == (5,) and ((model.betas > 0) & (model.betas < 1)).all()


def test_diffusion_rejects_unknown_schedule():
    with pytest.raises(NotImplementedError):
        Diffusion(input_size=INPUT_SIZE, n_onedirectional_conv=1, T=5, min_noise=1e-4, max_noise=0.02, noise_scheduler="x")


PYPOTS = dict(input_size=INPUT_SIZE, epochs=1)
PYPOTS_LAYERS = PYPOTS | dict(n_layers=1, d_model=8, d_inner=8, n_head=2, d_k=4, d_v=4, dropout=0.0)
WRAPPED = {  # Name: (package, model, parameters)
    **{
        name: ("hyperimpute", getattr(baselines, f"{name}Imputation"), {})
        for name in ["GAIN", "ICE", "MissForest", "SoftImpute", "Sinkhorn", "Miracle", "Miwae", "Hyper"]
    },
    "BRITS": ("pypots", baselines.BRITSImputation, PYPOTS | dict(rnn_hidden_size=4, batch_size=2)),
    "SAITS": ("pypots", baselines.SAITSImputation, PYPOTS_LAYERS),
    "Attention": ("pypots", baselines.AttentionImputation, PYPOTS_LAYERS),
}


@pytest.mark.parametrize("package, model, params", list(WRAPPED.values()), ids=list(WRAPPED))
def test_wrapped_package_models_construct(package, model, params):
    if INSTALLED[package]:
        assert model(**params).imputer is not None
    else:  # The package is installed with the imputation extra
        with pytest.raises(ModuleNotFoundError, match=package):
            model(**params)


def imputation_split(seed=0, stays=4 * BATCH):
    """A split as ImputationPandasDataset holds it: values with real missingness and YAIB's amputation of them."""
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(stays * STEPS, FEATURES, generator=generator)
    never_measured = torch.rand(values.shape, generator=generator) < 0.2
    amputated = torch.rand(values.shape, generator=generator) < 0.3
    measured = pd.DataFrame(values.masked_fill(never_measured, float("nan")).numpy())
    split = SimpleNamespace(
        maxlen=STEPS,
        features_df=measured.fillna(0),  # as the dataset stores it
        target_missingness_mask=measured.isna(),
        amputated_values=measured.mask(amputated.numpy()),
    )
    # The values to impute that have a ground truth: amputated, but measured
    split.scored = (amputated & ~never_measured).reshape(stays, STEPS, FEATURES)
    return split


# MissForest goes through the same wrapper as ICE but takes about 25 s
@pytest.mark.parametrize("name", ["ICE", "SoftImpute", "BRITS", "SAITS", "Attention"])
def test_wrapped_package_models_fit_and_impute(name):
    package, model, params = WRAPPED[name]
    if not INSTALLED[package]:
        pytest.skip(f"{package} is installed with the imputation extra")
    torch.manual_seed(0)
    train = imputation_split(seed=0)
    stays = torch.tensor(train.amputated_values.values).reshape(-1, STEPS, FEATURES)
    imputer = model(**params)
    imputer.fit(train, imputation_split(seed=1))
    batch = stays[:BATCH]
    observed = ~batch.isnan()
    imputed = imputer(batch, observed)
    assert imputed.shape == batch.shape and not imputed.isnan().any()
    assert torch.allclose(imputed[observed], batch[observed], atol=1e-5)


@pytest.mark.parametrize("name", ["SAITS", "Attention"])
def test_pypots_masked_models_learn_to_impute_the_yaib_amputation(name, monkeypatch):
    # PyPOTS's own fit hides a fresh random 20% of the observed values instead (#162)
    package, model, params = WRAPPED[name]
    if not INSTALLED[package]:
        pytest.skip(f"{package} is installed with the imputation extra")
    imputer = model(**params)
    loaders = {}
    train_model = imputer.imputer._train_model

    def record(train_loader, val_loader=None):
        loaders.update(train=train_loader, val=val_loader)
        train_model(train_loader, val_loader)

    monkeypatch.setattr(imputer.imputer, "_train_model", record)
    train, val = imputation_split(seed=0), imputation_split(seed=1)
    imputer.fit(train, val)

    for split, loader in [(train, loaders["train"]), (val, loaders["val"])]:
        dataset = loader.dataset
        assert torch.equal(dataset.indicating_mask.bool(), split.scored)  # the loss and validation targets
        observed = ~torch.tensor(split.amputated_values.values).reshape(split.scored.shape).isnan()
        assert torch.equal(dataset.missing_mask.bool(), observed)  # the input: YAIB's amputation and no more
        first, again = dataset[0], dataset[0]
        assert all(torch.equal(a, b) for a, b in zip(first, again))  # no fresh random mask per epoch
    assert imputer.imputer.best_model_dict is not None


def test_pypots_brits_keeps_the_epoch_that_best_imputes_the_yaib_amputation(monkeypatch):
    package, model, params = WRAPPED["BRITS"]
    if not INSTALLED[package]:
        pytest.skip(f"{package} is installed with the imputation extra")
    imputer = model(**params)
    calls = {}
    fit = imputer.imputer.fit

    def record(train_set, val_set=None, **kwargs):
        calls.update(train=train_set, val=val_set)
        fit(train_set, val_set=val_set, **kwargs)

    monkeypatch.setattr(imputer.imputer, "fit", record)
    train, val = imputation_split(seed=0), imputation_split(seed=1)
    imputer.fit(train, val)

    assert set(calls["train"]) == {"X"}  # BRITS learns from the observed values only
    amputated_but_measured = torch.tensor(calls["val"]["X"]).isnan() & ~torch.tensor(calls["val"]["X_ori"]).isnan()
    assert torch.equal(amputated_but_measured, val.scored)  # PyPOTS validates on these


def imports(config):
    return re.findall(r"^import ([\w.]+)", config.read_text(), re.MULTILINE)


S4_MODULES = {"icu_benchmarks.imputation.sssds4", "icu_benchmarks.imputation.sssdsa"}


def skip_without_s4(config):
    if not S4_INSTALLED and S4_MODULES & set(imports(config)):
        pytest.skip("The SSSD models need einops and opt_einsum from the imputation extra")


@pytest.mark.parametrize("module", sorted({module for config in CONFIGS for module in imports(config)}))
def test_config_imports_exist(module):
    assert find_spec(module) is not None


@pytest.mark.parametrize("config", CONFIGS, ids=lambda config: config.stem)
def test_config_model_is_registered(config):
    skip_without_s4(config)
    for module in imports(config):
        if module.startswith("icu_benchmarks.imputation."):
            importlib.import_module(module)
    model = re.search(r"^train_common\.model\s*=\s*@(\w+)", config.read_text(), re.MULTILINE).group(1)
    assert gin.config._REGISTRY.get_match(model) is not None


@pytest.mark.parametrize("config", CONFIGS, ids=lambda config: config.stem)
def test_config_bindings_parse(config):
    skip_without_s4(config)
    gin.clear_config()
    try:
        gin.parse_config(config.read_text())
    finally:
        gin.clear_config()
