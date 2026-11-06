# Installation

YAIB supports Python 3.10 to 3.13 on Linux, macOS and Windows. It is installed from source with
[uv](https://docs.astral.sh/uv/), which reads `pyproject.toml` and the locked versions in `uv.lock`.

## Install from source

Clone the repository. The newest features and fixes are on the `development` branch:

```bash
git clone https://github.com/rvandewater/YAIB.git
cd YAIB
git checkout development
```

Install uv if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install Python and the project with its dependencies into `.venv`:

```bash
uv python install 3.12
uv sync
```

Check that the command-line interface works:

```bash
uv run icu-benchmarks train -h
```

Every command in this documentation runs through `uv run`, which uses the project's environment. You can also activate
it with `source .venv/bin/activate` and drop the `uv run` prefix.

## Optional dependencies

| Command                      | Adds                                                                                                                                                                                                   |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `uv sync --extra imputation` | The hyperimpute and PyPOTS imputation baselines (MissForest, GAIN, MIWAE, MIRACLE, Sinkhorn, SoftImpute, BRITS, SAITS, Attention) and the SSSD models. About 100 extra packages, including TensorFlow. |
| `uv sync --extra explain`    | [Captum](https://captum.ai/), for the feature attributions of deep learning models with `--explain-features`.                                                                                          |
| `uv sync --dev`              | Tests, coverage, linting and pre-commit, for [development](../development/contributing.md).                                                                                                            |
| `uv sync --only-group docs`  | The tools to build this documentation, see [Documentation](../development/documentation.md).                                                                                                           |

## Platform notes

- **GPUs:** on Linux and Windows, uv installs the CUDA build of PyTorch from the PyTorch index. YAIB lists the GPUs it
    finds at the start of each run. Pass `--cpu` to train on the CPU anyway.
- **macOS:** uv installs the default PyPI build of PyTorch. On Macs with Metal Performance Shaders, set
    `PYTORCH_ENABLE_MPS_FALLBACK=1` if a model uses an operation that MPS does not support.
- **Windows:** replace the line continuation `\` in the examples with `^` (Command Prompt) or `` ` `` (PowerShell).
- **Mamba:** the `Mamba` model runs in plain PyTorch. On NVIDIA GPUs it uses the fused kernels of
    [`mamba-ssm`](https://github.com/state-spaces/mamba) if you install it separately.

## PyPI

YAIB is also published on PyPI as [`yaib`](https://pypi.org/project/yaib/). The CLI reads its configs from the
`configs/` folder of the current directory, so for running experiments a clone of the repository is the simplest setup.
