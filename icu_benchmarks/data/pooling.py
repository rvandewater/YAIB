from pathlib import Path
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from .constants import DataSegment as Segment, VarType as Var
import pyarrow.parquet as pq


class PooledDataset:
    eicu_hirid_miiv = ["eicu", "hirid", "miiv"]
    eicu_hirid_aumc = ["eicu", "hirid", "aumc"]
    eicu_aumc_miiv = ["eicu", "aumc", "miiv"]
    aumc_hirid_miiv = ["aumc", "hirid", "miiv"]
    eicu_hirid_aumc_miiv = ["eicu", "hirid", "aumc", "miiv"]


def generate_pooled_data(data_dir, vars, datasets, file_names, samples=10000, seed=42, shuffle=False, stratify=None):
    data = {}
    for folder in data_dir.iterdir():
        if folder.is_dir():
            if folder.name in datasets:
                data[folder.name] = {
                    f: pq.read_table(folder / file_names[f]).to_pandas(self_destruct=True) for f in file_names.keys()
                }
    data = pool_datasets(datasets=data, samples=samples, vars=vars, shuffle=True, stratify=None)
    save_pooled_data(data_dir, data, datasets)


def save_pooled_data(data_dir, data, datasets):
    save_folder = "_".join(datasets)
    save_dir = data_dir / save_folder
    if not save_dir.exists():
        save_dir.mkdir()
    for key, value in data.items():
        value.to_parquet(save_dir / Path(key + ".parquet"))
    logging.info(f"Saved pooled data at {save_dir}")


def pool_datasets(datasets={}, samples=10000, vars=[], seed=42, shuffle=True, stratify=None, **kwargs):
    """
    Pool datasets into a single dataset.
    Args:
        datasets: list of datasets to pool
    Returns:
        pooled dataset
    """
    if len(datasets) == 0:
        raise ValueError("No datasets supplied.")
    pooled_data = {Segment.static: [], Segment.dynamic: [], Segment.outcome: []}
    id = vars[Var.group]
    int_id = 0
    for key, value in datasets.items():
        int_id += 1
        repeated_digit = str(int_id) * 4
        # outcome = value[Segment.outcome].groupby(vars["LABEL"], group_keys=False).sample(samples_per_class,random_state=seed)
        outcome = value[Segment.outcome]
        outcome = train_test_split(
            outcome, stratify=outcome[vars[Var.label]], shuffle=shuffle, random_state=seed, train_size=samples
        )[0]
        stays = pd.Series(outcome[id].unique())
        static = value[Segment.static]
        dynamic = value[Segment.dynamic]
        static = static.loc[static[id].isin(stays)]
        dynamic = dynamic.loc[dynamic[id].isin(stays)]
        # Preventing id clashing
        outcome[id] = outcome[id].map(lambda x: int(str(x) + repeated_digit))
        static[id] = static[id].map(lambda x: int(str(x) + repeated_digit))
        dynamic[id] = dynamic[id].map(lambda x: int(str(x) + repeated_digit))
        # Adding to pooled data
        pooled_data[Segment.static].append(static)
        pooled_data[Segment.dynamic].append(dynamic)
        pooled_data[Segment.outcome].append(outcome)
    # Add each datatype together
    for key, value in pooled_data.items():
        pooled_data[key] = pd.concat(value, ignore_index=True)
    return pooled_data
