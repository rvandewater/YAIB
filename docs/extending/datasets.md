# Adding datasets

YAIB reads any cohort in its data format, so a new dataset needs no code changes once it is in that format. The
[YAIB-cohorts](https://github.com/rvandewater/YAIB-cohorts) code produces exactly this format for the supported
datasets.

## The data format

A cohort is a folder with three [Apache Parquet](https://parquet.apache.org/) files:

| File           | Contents                                                                     | Rows                                                         |
| -------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `dyn.parquet`  | Dynamic variables that change during the stay (vital signs, lab values, ...) | One per stay and time step                                   |
| `sta.parquet`  | Static variables (age, sex, height, weight, ...)                             | One per stay                                                 |
| `outc.parquet` | The label to predict                                                         | One per stay, or one per stay and time step for hourly tasks |

The task config tells YAIB which file is which and what role each column has:

```python title="configs/tasks/common/PredictionTaskVariables.gin (shortened)"
preprocess.file_names = {
    "DYNAMIC": "dyn.parquet",
    "OUTCOME": "outc.parquet",
    "STATIC": "sta.parquet",
}

vars = {
    "GROUP": "stay_id",
    "SEQUENCE": "time",
    "LABEL": "label",
    "DYNAMIC": ["alb", "alp", "alt", ..., "wbc"],
    "STATIC": ["age", "sex", "height", "weight"],
}
```

| Role       | Meaning                                                                                    |
| ---------- | ------------------------------------------------------------------------------------------ |
| `GROUP`    | The ID column that groups rows into stays; it is in all three files.                       |
| `SEQUENCE` | The time column of the dynamic data (and of hourly labels).                                |
| `LABEL`    | The label column in `outc.parquet`. It can be a list of labels; choose one with `--label`. |
| `DYNAMIC`  | The dynamic feature columns.                                                               |
| `STATIC`   | The static feature columns.                                                                |

The roles come from [ReciPys](https://github.com/rvandewater/ReciPys), the preprocessing package developed alongside
YAIB.

If your files or columns are named differently, copy `PredictionTaskVariables.gin` and the task config, change the
names, and select your task config with `-t`; see [Adding tasks](tasks.md).

## Harmonizing a new data source

To use the existing clinical concepts and task definitions with a new database, make it available to
[ricu](https://github.com/eth-mds/ricu), which YAIB-cohorts uses to harmonize the datasets. The example below follows
the integration of SICdb.

### Define ID types and table structure

ricu needs to know which tables and columns exist in the data source. This is specified in a JSON configuration file,
partially shown below:

```json
{
  "name": "sic",
  "id_cfg": {
    "patient": {
      "id": "patientid",
      "position": 1,
      "start": "firstadmission",
      "end": "offsetofdeath",
      "table": "cases"
    },
    "icustay": {
      "id": "caseid",
      "position": 2,
      "start": "offsetafterfirstadmission",
      "end": "timeofstay",
      "table": "cases"
    }
  },
  "tables": {
    "cases": {
      "files": "cases.csv.gz",
      "defaults": {
        "index_var": "offsetafterfirstadmission",
        "time_vars": [
          "offsetafterfirstadmission",
          "offsetofdeath"
        ]
      },
      "cols": {
        "caseid": {
          "name": "CaseID",
          "spec": "col_integer"
        },
        "patientid": {
          "name": "PatientID",
          "spec": "col_integer"
        },
        "admissionyear": {
          "name": "AdmissionYear",
          "spec": "col_integer"
        }
      }
    }
  }
}
```

Tables are defined under `tables`. Each table lists its source file (usually `.csv` or `.csv.gz`) and its columns with
their data types. Default roles can be set for some columns, usually the time index, the other time columns, and the
column with the value of interest.

ricu also needs the main ID types of the dataset. Information in ICU datasets is linked to a unit of observation: the
patient, the hospital admission (`hospadm`) or the ICU stay (`icustay`). Knowing which ID each piece of information
belongs to lets ricu relate all information in time. For example, MIMIC-IV records `labevents` per hospital admission
and `chartevents` per ICU stay; defining how the two ID systems relate maps both to a common time scale, such as the
time since ICU admission. Knowing that an ICU stay is described in `icustays` in MIMIC and in `cases` in SICdb defines
the same reference point in both databases.

SICdb has two ID types, the patient and the ICU stay, and no separate hospital admission. A patient is observed from
their first admission until their death (if it occurred); an ICU stay lasts from admission to the ICU until discharge.

### Calculate origin times for each ID type

ricu also needs to know when each ID starts and ends. Much of this is automated, but small adjustments are often
needed. SICdb, for example, gives discharge times in seconds since admission rather than as absolute times. Such
adjustments are made by subtyping the respective function (here `id_win_helper`) and overriding its default behavior.
ricu's helper functions also set up relative times in seconds.

With these steps, SICdb is available in ricu. Further helpers can add optional functionality, such as finding the
measurement units, which SICdb stores in a separate reference table. This only interfaces the database: each clinical
concept still has to be mapped to the database's items, for example through their measurement IDs; see
[Clinical concepts](clinical-concepts.md). ricu's helpers make this mapping much simpler. More on ricu and its design is
in Bennett et al. (2023).
