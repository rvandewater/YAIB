# Generating cohorts

The [YAIB-cohorts](https://github.com/rvandewater/YAIB-cohorts) repository extracts the task cohorts from the ICU
datasets in the format YAIB reads. It uses the [ricu](https://github.com/eth-mds/ricu) R package, which defines clinical
concepts and how to load them from each dataset, so that every dataset is accessed through the same interface.

Clone it next to YAIB:

```bash
git clone https://github.com/rvandewater/YAIB-cohorts.git
```

The cohorts can be extracted from R or from Python. The README of YAIB-cohorts is the authoritative reference; the steps
below give an overview.

=== "R"

    Run these commands from `YAIB-cohorts/R`.

    **1. Set up the environment.** The extractions for the paper ran with R 4.2.2. An `renv` lock file pins all
    package versions:

    ```bash
    Rscript setup_env.R
    ```

    **2. Load the datasets into ricu.** Download the datasets you have access to and import them as described in
    `?ricu::import_src`. ricu also ships two demo datasets, `mimic.demo` and `eicu.demo`, for quick experiments; `renv`
    should install them (otherwise see [mimic-demo](https://github.com/eth-mds/mimic-demo) and
    [eicu-demo](https://github.com/eth-mds/eicu-demo)). Then set the data path in `.Rprofile`:

    ```r
    Sys.setenv(RICU_DATA_PATH = "/path/to/your/ricu/data/folder")
    ```

    **3. Generate the cohorts.** All tasks share a data-cleaning step, `base_cohort.R`, that must run first:

    ```bash
    Rscript base_cohort.R --src mimic_demo
    Rscript mortality.R --src mimic_demo
    ```

    Replace `mortality.R` with the task and `mimic_demo` with the dataset: one of `mimic_demo`, `eicu_demo`, `aumc`,
    `hirid`, `eicu`, `mimic`, `miiv`. To extract a task from all datasets at once, run `bash gen_cohort.sh mortality`.

=== "Python"

    Run these commands from `YAIB-cohorts/Python`.

    **1. Set up the environment:**

    ```bash
    python setup_env.py
    ```

    **2. Load the datasets into ricu.** As for R: import the datasets with `?ricu::import_src` and set
    `RICU_DATA_PATH` in `.Rprofile`.

    **3. Generate the cohorts:**

    ```bash
    python mortality.py --src mimic_demo
    ```

    Replace `mortality.py` with the task and `mimic_demo` with the dataset, as above.

## Tasks

| Script            | Task                                              | Type           |
| ----------------- | ------------------------------------------------- | -------------- |
| `mortality`       | ICU mortality after 24 hours                      | Classification |
| `aki`             | Acute kidney injury within the next 6 hours       | Classification |
| `sepsis`          | Sepsis within the next 6 hours                    | Classification |
| `kidney_function` | Kidney function on the second day of the ICU stay | Regression     |
| `los`             | Remaining length of stay                          | Regression     |

The output directory is set in `config.yaml` at the root of YAIB-cohorts. Each cohort is written as `dyn.parquet`,
`sta.parquet` and `outc.parquet`, which you pass to YAIB with `-d`. The exact label definitions, such as the time window
around AKI onset that counts as positive, are in the task scripts of YAIB-cohorts (for example `R/aki.R`).
