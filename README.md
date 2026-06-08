![YAIB logo](https://github.com/rvandewater/YAIB/blob/development/docs/figures/yaib_logo.png?raw=true)

# 🧪 Yet Another ICU Benchmark

[![CI](https://github.com/rvandewater/YAIB/actions/workflows/ci.yml/badge.svg?branch=development)](https://github.com/rvandewater/YAIB/actions/workflows/ci.yml)
![Platform](https://img.shields.io/badge/platform-linux--64%20%7C%20win--64%20%7C%20osx--64-lightgrey)
[![arXiv](https://img.shields.io/badge/arXiv-2306.05109-b31b1b.svg)](http://arxiv.org/abs/2306.05109)
[![PyPI version shields.io](https://img.shields.io/pypi/v/yaib.svg)](https://pypi.python.org/pypi/yaib/)
[![python](https://img.shields.io/badge/-Python_3.13-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3130/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Yet another ICU benchmark (YAIB) provides a framework for doing clinical machine learning experiments on Intensive Care Unit
(ICU) EHR data.

We support the following datasets out of the box:

| **Dataset**                 | [MIMIC-III](https://physionet.org/content/mimiciii/) / [IV](https://physionet.org/content/mimiciv/) | [eICU-CRD](https://physionet.org/content/eicu-crd/) | [HiRID](https://physionet.org/content/hirid/1.1.1/) | [AUMCdb](https://doi.org/10.17026/dans-22u-f8vd) |
| --------------------------- | --------------------------------------------------------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------ |
| **Admissions**              | 40k / 73k                                                                                           | 200k                                                | 33k                                                 | 23k                                              |
| **Version**                 | v1.4 / v2.2                                                                                         | v2.0                                                | v1.1.1                                              | v1.0.2                                           |
| **Frequency** (time-series) | 1 hour                                                                                              | 5 minutes                                           | 2 / 5 minutes                                       | up to 1 minute                                   |
| **Originally published**    | 2015 / 2020                                                                                         | 2017                                                | 2020                                                | 2019                                             |
| **Origin**                  | USA                                                                                                 | USA                                                 | Switzerland                                         | Netherlands                                      |

New datasets can also be added. We are currently working on a package to make this process as smooth as possible.
The benchmark is designed for operating on preprocessed parquet files.

We provide five common tasks for clinical prediction by default:

| No  | Task                      | Frequency                 | Type                  |
| --- | ------------------------- | ------------------------- | --------------------- |
| 1   | ICU Mortality             | Once per Stay (after 24H) | Binary Classification |
| 2   | Acute Kidney Injury (AKI) | Hourly (within 6H)        | Binary Classification |
| 3   | Sepsis                    | Hourly (within 6H)        | Binary Classification |
| 4   | Kidney Function (KF)      | Once per stay             | Regression            |
| 5   | Length of Stay (LoS)      | Hourly (within 7D)        | Regression            |

New tasks can be easily added.
To get started right away, we include the eICU and MIMIC-III demo datasets in our repository.

The following repositories may be relevant as well:

- [YAIB-cohorts](https://github.com/rvandewater/YAIB-cohorts): Cohort generation for YAIB.
- [YAIB-models](https://github.com/rvandewater/YAIB-models): Pretrained models for YAIB.
- [ReciPys](https://github.com/rvandewater/ReciPys): Preprocessing package for YAIB pipelines.
- For all YAIB-related repositories, please see: https://github.com/stars/rvandewater/lists/yaib.

# 📄 Paper

To reproduce the benchmarks in our paper, see the [ML reproducibility document](PAPER.md).
If you use this code in your research, please cite the following publication:

```
@inproceedings{vandewaterYetAnotherICUBenchmark2024,
  title = {Yet Another ICU Benchmark: A Flexible Multi-Center Framework for Clinical ML},
  shorttitle = {Yet Another ICU Benchmark},
  booktitle = {The Twelfth International Conference on Learning Representations},
  author = {van de Water, Robin and Schmidt, Hendrik Nils Aurel and Elbers, Paul and Thoral, Patrick and Arnrich, Bert and Rockenschaub, Patrick},
  year = {2024},
  month = oct,
  urldate = {2024-02-19},
  langid = {english},
}
```

This paper can also be found on arXiv: [2306.05109](https://arxiv.org/abs/2306.05109).

# 💿 Installation

YAIB is installed from source using `uv` and the repository's `pyproject.toml`.

## Installation from source

First, clone this repository:

```bash
git clone https://github.com/rvandewater/YAIB.git
cd YAIB
```

Please note the branch. The newest features and fixes are available on the development branch:

```bash
git checkout development
```

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Pin Python, create the virtual environment, and install all project dependencies:

```bash
uv python install 3.12
uv sync
```

To install development dependencies as well:

```bash
uv sync --dev
```

### Platform notes

- On macOS, YAIB should use the default PyPI `torch` wheel.
- On Linux and Windows, the project is configured to resolve the CUDA-enabled `torch` wheel from the PyTorch index.
- On Macs with Metal Performance Shaders, run YAIB with `PYTORCH_ENABLE_MPS_FALLBACK=1` if needed.

After installation, check that the CLI is available:

```bash
uv run icu-benchmarks train -h
```

YAIB will automatically list available processors at initialization in its log files.

# 👩‍💻 Usage

Please refer to [our wiki](https://github.com/rvandewater/YAIB/wiki) for detailed information on how to use YAIB.

## Quickstart 🚀 (demo data)

The authors of MIMIC-III and eICU have made a small demo dataset available to demonstrate their use. They can be found on PhysioNet: [MIMIC-III Clinical Database Demo](https://physionet.org/content/mimiciii-demo/1.4/) and [eICU Collaborative Research Database Demo](https://physionet.org/content/eicu-crd-demo/2.0.1/). These datasets are published under the [Open Data Commons Open Database License v1.0](https://opendatacommons.org/licenses/odbl/1-0/) and can be used without credentialing procedure. We have created demo cohorts processed **solely from these datasets** for each of our currently supported task endpoints. To the best of our knowledge, this complies with the license and the respective dataset author's instructions. Usage of the task cohorts and the dataset is only permitted with the above license.

We **strongly recommend** completing a human subject research training to ensure you properly handle human subject research data.

In the folder `demo_data` we provide processed publicly available demo datasets from eICU and MIMIC with the necessary labels for `Mortality at 24h`, `Sepsis`, `Acute Kidney Injury`, `Kidney Function`, and `Length of Stay`.

If you do not yet have access to the ICU datasets, you can run the following command to train models for the included demo cohorts:

```bash
uv run icu-benchmarks train \
    -d demo_data/mortality24/mimic_demo \
    -n mimic_demo \
    -t BinaryClassification \
    -tn Mortality24 \
    -m XGBClassifier \
    --seed 2222 \
    -l ../yaib_logs/ \
    --tune
```

If you want to reproduce the LightGBM example with cache generation:

```bash
uv run icu-benchmarks train \
    -d demo_data/mortality24/mimic_demo \
    -n mimic_demo \
    -t BinaryClassification \
    -tn Mortality24 \
    -m LGBMClassifier \
    -hp LGBMClassifier.min_child_samples=10 \
    --generate_cache \
    --load_cache \
    --seed 2222 \
    -l ../yaib_logs/ \
    --tune
```

> For a list of available flags, run `uv run icu-benchmarks train -h`.

> On Macs with Metal Performance Shaders, run with `PYTORCH_ENABLE_MPS_FALLBACK=1` if needed.

> For Windows based systems, the next line character (`\`) needs to be replaced by `^` (Command Prompt) or `` ` `` (PowerShell).

Alternatively, the easiest method to train all the models in the paper is to run these commands from the repository root:

```bash
uv run wandb sweep --verbose experiments/benchmark_classification.yml
uv run wandb sweep --verbose experiments/benchmark_regression.yml
```

This will create two hyperparameter sweeps for Weights & Biases for the classification and regression tasks. You can then run the following command to train the models:

```bash
uv run wandb agent <sweep_id>
```

> Tip: You can choose to run each of the configurations on a SLURM cluster instance with `uv run wandb agent --count 1 <sweep_id>`.

> Note: You will need to have a Weights & Biases account and be logged in to run the above commands.

## Getting the datasets

HiRID, eICU, and MIMIC-IV can be accessed through [PhysioNet](https://physionet.org/). A guide to this process can be found [here](https://eicu-crd.mit.edu/gettingstarted/access/).
AUMCdb can be accessed through a separate [procedure](https://github.com/AmsterdamUMC/AmsterdamUMCdb). We do not have involvement in the access procedure and cannot answer requests for data access.

## Cohort creation

Since the datasets were created independently of each other, they do not share the same data structure or data identifiers. In order to make them interoperable, use the preprocessing utilities provided by the [ricu package](https://github.com/eth-mds/ricu).
Ricu pre-defines a large number of clinical concepts and how to load them from a given dataset, providing a common interface to the data that is used in this benchmark. Please refer to our [cohort definition](https://github.com/rvandewater/YAIB-cohorts) code for generating the cohorts using our Python interface for ricu.
After this, you can run the benchmark once you have gained access to the datasets.

# 👟 Running YAIB

## Preprocessing and Training

The following command will run training and evaluation on the MIMIC demo dataset for binary mortality prediction at 24h with `LGBMClassifier`:

```bash
uv run icu-benchmarks train \
    -d demo_data/mortality24/mimic_demo \
    -n mimic_demo \
    -t BinaryClassification \
    -tn Mortality24 \
    -m LGBMClassifier \
    -hp LGBMClassifier.min_child_samples=10 \
    --generate_cache \
    --load_cache \
    --seed 2222 \
    -l ../yaib_logs/ \
    --tune
```

> For a list of available flags, run `uv run icu-benchmarks train -h`.

> Run with `PYTORCH_ENABLE_MPS_FALLBACK=1` on Macs with Metal Performance Shaders.

> For Windows based systems, the next line character (`\`) needs to be replaced by `^` (Command Prompt) or `` ` `` (PowerShell), respectively.

Alternatively, the easiest method to train all the models in the paper is to run these commands from the directory root:

```bash
uv run wandb sweep --verbose experiments/benchmark_classification.yml
uv run wandb sweep --verbose experiments/benchmark_regression.yml
```

This will create two hyperparameter sweeps for WandB for the classification and regression tasks.
This configuration will train all the models in the paper. You can then run the following command to train the models:

```bash
uv run wandb agent <sweep_id>
```

> Tip: You can choose to run each of the configurations on a SLURM cluster instance by `uv run wandb agent --count 1 <sweep_id>`.

> Note: You will need to have a WandB account and be logged in to run the above commands.

## Evaluate or Finetune

It is possible to evaluate a model trained on another dataset without additional training. In this case, the source dataset is the demo data from MIMIC and the target is the eICU demo:

```bash
uv run icu-benchmarks \
    --eval \
    -d demo_data/mortality24/eicu_demo \
    -n eicu_demo \
    -t BinaryClassification \
    -tn Mortality24 \
    -m LGBMClassifier \
    --generate_cache \
    --load_cache \
    -s 2222 \
    -l ../yaib_logs \
    -sn mimic \
    --source-dir ../yaib_logs/mimic_demo/Mortality24/LGBMClassifier/2022-12-12T15-24-46/repetition_0/fold_0
```

> A similar syntax is used for finetuning, where a model is loaded and then retrained. To run finetuning, replace `--eval` with `-ft`.

## Models

We provide several existing machine learning models that are commonly used for multivariate time-series data.
`pytorch` is used for the deep learning models, `lightgbm` for the boosted tree approaches, and `sklearn` for other classical machine learning models.
The benchmark provides (among others) the following built-in models:

- [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic+regression): Standard regression approach.
- [Elastic Net](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html): Linear regression with combined L1 and L2 priors as regularizer.
- [LightGBM](https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf): Efficient gradient boosting trees.
- [Long Short-term Memory (LSTM)](https://ieeexplore.ieee.org/document/818041): The most commonly used type of recurrent neural networks for long sequences.
- [Gated Recurrent Unit (GRU)](https://arxiv.org/abs/1406.1078): An extension to LSTM which showed improvements ([paper](https://arxiv.org/abs/1412.3555)).
- [Temporal Convolutional Networks (TCN)](https://arxiv.org/pdf/1803.01271): 1D convolution approach to sequence data. By using dilated convolution to extend the receptive field of the network it has shown great performance on long-term dependencies.
- [Transformers](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf): The most common attention-based approach.

# 🛠️ Development

To adapt YAIB to your own use case, you can use the [development information](https://github.com/rvandewater/YAIB/wiki/Contribution-and-development) page as a reference.
We appreciate contributions to the project. Please read the [contribution guidelines](CONTRIBUTING.md) before submitting a pull request.

# Acknowledgements

This project has been developed partially under the funding of “Gemeinsamer Bundesausschuss (G-BA) Innovationsausschuss” in the framework of “CASSANDRA - Clinical ASSist AND aleRt Algorithms” (project number 01VSF20015).
We would like to acknowledge the work of Alisher Turubayev, Anna Shopova, Fabian Lange, Mahmut Kamalak, Paul Mattes, and Victoria Ayvasky for adding PyTorch Lightning, Weights and Biases compatibility, and several optional imputation methods to a later version of the benchmark repository.

We do not own any of the datasets used in this benchmark. This project uses heavily adapted components of the [HiRID benchmark](https://github.com/ratschlab/HIRID-ICU-Benchmark/). We thank the authors for providing this codebase and encourage further development to benefit the scientific community. The demo datasets have been released under an [Open Data Commons Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1-0/).

# License

This source code is released under the MIT license, included [here](LICENSE). We do not own any of the datasets used or included in this repository.
