# Datasets and access

YAIB works on preprocessed cohorts in a common [parquet format](../extending/datasets.md). To run it on the full ICU
datasets, you need three steps:

1. Get access to the datasets (this page).
2. [Generate the cohorts](cohorts.md) with YAIB-cohorts.
3. [Run YAIB](../user-guide/running-experiments.md) on the generated cohorts.

To try YAIB first, use the demo cohorts in the repository; see the [Quickstart](quickstart.md).

## Supported datasets

| Dataset                 | [MIMIC-III](https://physionet.org/content/mimiciii/) / [IV](https://physionet.org/content/mimiciv/) | [eICU-CRD](https://physionet.org/content/eicu-crd/) | [HiRID](https://physionet.org/content/hirid/1.1.1/) | [AUMCdb](https://doi.org/10.17026/dans-22u-f8vd) |
| ----------------------- | --------------------------------------------------------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------ |
| Admissions              | 40k / 73k                                                                                           | 200k                                                | 33k                                                 | 23k                                              |
| Version                 | v1.4 / v2.2                                                                                         | v2.0                                                | v1.1.1                                              | v1.0.2                                           |
| Frequency (time series) | 1 hour                                                                                              | 5 minutes                                           | 2 / 5 minutes                                       | up to 1 minute                                   |
| Originally published    | 2015 / 2020                                                                                         | 2017                                                | 2020                                                | 2019                                             |
| Origin                  | USA                                                                                                 | USA                                                 | Switzerland                                         | Netherlands                                      |

## Getting access

All four datasets require a credentialing procedure.

- **MIMIC-III, MIMIC-IV, eICU-CRD and HiRID** are available on [PhysioNet](https://physionet.org/). The
    [eICU getting-started guide](https://eicu-crd.mit.edu/gettingstarted/access/) describes the process.
- **AUMCdb** has its own [access procedure](https://github.com/AmsterdamUMC/AmsterdamUMCdb).

We are not involved in these access procedures and cannot answer requests for data access.

## Other datasets

Any dataset can be used once it is in YAIB's format; see [Adding datasets](../extending/datasets.md). Since the
datasets were collected independently, they do not share a data structure or identifiers. YAIB-cohorts uses the
[ricu](https://github.com/eth-mds/ricu) R package to harmonize them.
