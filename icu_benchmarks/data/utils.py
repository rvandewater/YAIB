import logging
import polars as pl
import polars.selectors as cs
import numpy as np

from icu_benchmarks.data.constants import VarType as Var, DataSegment as Segment


def infinite_removal(val):
    for col in val.select(cs.numeric()).columns:
        if val[col].is_infinite().any():
            logging.info(f"Column '{col}' contains infinite values. Datatype: {val[col].dtype}")

    max_float64 = np.finfo(np.float64).max / 100
    # Replace infinite values with the maximum value for float64
    val = val.with_columns(
        [
            pl.when(pl.col(col).is_infinite()).then(max_float64).otherwise(pl.col(col)).alias(col)
            for col in val.columns
            if val[col].dtype == pl.Float64
        ]
    )
    return val


def check_sanitize_data(data, vars):
    """Check for duplicates in the loaded data and remove them."""
    group = vars[Var.group] if Var.group in vars.keys() else None
    sequence = vars[Var.sequence] if Var.sequence in vars.keys() else None
    keep = "last"
    logging.info(data.keys())
    if Segment.static in data.keys():
        old_len = len(data[Segment.static])
        data[Segment.static] = data[Segment.static].unique(subset=group, keep=keep, maintain_order=True)
        logging.warning(f"Removed {old_len - len(data[Segment.static])} duplicates from static data.")
    if Segment.dynamic in data.keys():
        old_len = len(data[Segment.dynamic])
        data[Segment.dynamic] = data[Segment.dynamic].unique(subset=[group, sequence], keep=keep, maintain_order=True)
        data[Segment.dynamic] = infinite_removal(data[Segment.dynamic])
        data[Segment.dynamic] = data[Segment.dynamic].select(~cs.starts_with("wearable_ppgfeature_HRV_SampEn"))
        vars[Segment.dynamic] = [col for col in vars[Segment.dynamic] if col in data[Segment.dynamic].columns]
        logging.warning(f"Removed {old_len - len(data[Segment.dynamic])} duplicates from dynamic data.")
    if Segment.outcome in data.keys():
        old_len = len(data[Segment.outcome])
        if sequence in data[Segment.outcome].columns:
            # We have a dynamic outcome with group and sequence
            data[Segment.outcome] = data[Segment.outcome].unique(subset=[group, sequence], keep=keep, maintain_order=True)
        else:
            data[Segment.outcome] = data[Segment.outcome].unique(subset=[group], keep=keep, maintain_order=True)
        logging.warning(f"Removed {old_len - len(data[Segment.outcome])} duplicates from outcome data.")
    return data, vars


def modality_selection(
    data: dict[pl.DataFrame], modality_mapping: dict[str], selected_modalities: list[str], vars
) -> dict[pl.DataFrame]:
    logging.info(f"Selected modalities: {selected_modalities}")
    selected_columns = [modality_mapping[cols] for cols in selected_modalities if cols in modality_mapping.keys()]
    if not any(col in modality_mapping.keys() for col in selected_modalities):
        raise ValueError("None of the selected modalities found in modality mapping.")
    if selected_columns == []:
        logging.info("No columns selected. Using all columns.")
        return data, vars
    selected_columns = sum(selected_columns, [])
    selected_columns.extend([vars[Var.group], vars[Var.label], vars[Var.sequence]])
    old_columns = []
    # Update vars dict
    for key, value in vars.items():
        if key not in [Var.group, Var.label, Var.sequence]:
            old_columns.extend(value)
            vars[key] = [col for col in value if col in selected_columns]
    # -3 because of standard columns
    logging.info(f"Selected columns: {len(selected_columns) - 3}, original columns: {len(old_columns)}, "
                 f"not using: {len(set(old_columns) - set(selected_columns))} columns")
    logging.debug(f"Not using columns: {set(old_columns) - set(selected_columns)}")
    # Update data dict
    for key in data.keys():
        sel_col = [col for col in data[key].columns if col in selected_columns]
        data[key] = data[key].select(sel_col)
        logging.debug(f"Selected columns in {key}: {len(data[key].columns)}")
    return data, vars
