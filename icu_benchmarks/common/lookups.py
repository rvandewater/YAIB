import logging
from pathlib import Path

import numpy as np
import pandas as pd

from icu_benchmarks.common.constants import PID

STEPS_PER_HOURS = 60


def read_general_table(path: Path):
    logging.info(f"Reading general table from {path}")
    return pd.read_parquet(path, engine="pyarrow").set_index(PID)


def read_var_ref_table(var_ref_path):
    return pd.read_csv(var_ref_path, sep='\t')


def read_reference_table(varref_path):
    """
    Read variableid-metavariableid mapping table for the merge step
    """

    varref = pd.read_csv(varref_path, sep="\t", encoding='cp1252')

    pharmaref = varref[varref["type"] == "pharma"].rename(columns={"variableid": "pharmaid"})
    enum_ref = {'very short': int(STEPS_PER_HOURS / 12), 'short': 1 * STEPS_PER_HOURS, '4h': 4 * STEPS_PER_HOURS,
                '6h': 6 * STEPS_PER_HOURS, '12h': 12 * STEPS_PER_HOURS, '24h': 24 * STEPS_PER_HOURS,
                '3d': 72 * STEPS_PER_HOURS}
    pharmaref.loc[:, "pharmaactingperiod_min"] = pharmaref.pharmaactingperiod.apply(
        lambda x: enum_ref[x] if type(x) == str else np.nan)
    check_func = lambda x: float(x) if type(x)==float or "/" not in x else float(x.split("/")[0])/float(x.split("/")[1])
    pharmaref.loc[:, "unitconversionfactor"] = pharmaref.unitconversionfactor.apply(check_func)
    varref = varref[varref["type"] != "pharma"].copy()
    varref.drop(varref.index[varref.variableid.isnull()], inplace=True)
    varref.loc[:, "variableid"] = varref.variableid.astype(int)
    varref.set_index("variableid", inplace=True)
    return varref, pharmaref
