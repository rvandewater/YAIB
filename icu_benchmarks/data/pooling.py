from pathlib import Path
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from .constants import DataSegment as Segment, VarType as Var
from icu_benchmarks.contants import RunMode
import pyarrow.parquet as pq


class PooledDataset:
    hirid_eicu_miiv = ["hirid", "eicu", "miiv"]
    aumc_hirid_eicu = ["aumc", "hirid","eicu"]
    aumc_eicu_miiv = ["aumc", "eicu", "miiv"]
    aumc_hirid_miiv = ["aumc", "hirid", "miiv"]
    aumc_hirid_eicu_miiv = ["aumc","hirid", "eicu", "miiv"]


def generate_pooled_data(data_dir, vars, datasets, file_names, samples=10000, seed=42, shuffle=False, stratify=None, runmode=RunMode.classification):
    data = {}
    for folder in data_dir.iterdir():
        if folder.is_dir():
            if folder.name in datasets:
                data[folder.name] = {
                    f: pq.read_table(folder / file_names[f]).to_pandas(self_destruct=True) for f in file_names.keys()
                }
    data = pool_datasets(datasets=data, samples=samples, vars=vars, shuffle=True, stratify=None, runmode=runmode)
    save_pooled_data(data_dir, data, datasets,file_names, samples=samples)


def save_pooled_data(data_dir, data, datasets, file_names, samples=10000):
    save_folder = "_".join(datasets)
    save_folder += f"_{samples}"
    save_dir = data_dir / save_folder
    if not save_dir.exists():
        save_dir.mkdir()
    # filenames = ["sta", "dyn", "outc"]
    for key, value in data.items():
        value.to_parquet(save_dir / Path(file_names[key]))
    logging.info(f"Saved pooled data at {save_dir}")


def pool_datasets(datasets={}, samples=10000, vars=[], seed=42, shuffle=True, runmode = RunMode.classification, stratify=None, **kwargs):
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
        static = value[Segment.static]
        dynamic = value[Segment.dynamic]
        # Get unique stay IDs from outcome segment
        stays = pd.Series(outcome[id].unique())

        # if(len(stays) is len(outcome[id]):
        if runmode is RunMode.classification:
            # If we have more outcomes than stays, check max label value per stay id
            labels = outcome.groupby(id).max()[vars[Var.label]].reset_index(drop=True)
            # if pd.Series(outcome[id].unique()) is outcome[id]):
            selected_stays = train_test_split(stays, stratify=labels, shuffle=shuffle, random_state=seed, train_size=samples)
        else:
            selected_stays = train_test_split(stays, shuffle=shuffle, random_state=seed, train_size=samples)
        # Select only stays that are in the selected_stays
        save_test = True
        # Save test sets to test on without leakage
        if save_test:
            data_dir = Path(r'C:\Users\Robin\Documents\Git\YAIB\data\YAIB_Datasets\data\mortality24')
            select = selected_stays[1]
            # if(runmode is RunMode.classification):
            #     select=train_test_split(stays, stratify=labels, shuffle=shuffle, random_state=seed, train_size=samples)[0]
            # else:
            #     select = train_test_split(select, shuffle=shuffle, random_state=seed, train_size=samples)[0]
            outcome = outcome.loc[outcome[id].isin(select)]
            static = static.loc[static[id].isin(select)]
            dynamic = dynamic.loc[dynamic[id].isin(select)]
            # Preventing id clashing
            outcome[id] = outcome[id].map(lambda x: int(str(x) + repeated_digit))
            static[id] = static[id].map(lambda x: int(str(x) + repeated_digit))
            dynamic[id] = dynamic[id].map(lambda x: int(str(x) + repeated_digit))
            save_folder = key
            save_folder += f"_{len(select)}"
            save_dir = data_dir / save_folder
            if not save_dir.exists():
                save_dir.mkdir()
            # filenames = ["sta", "dyn", "outc"]
            outcome.to_parquet(save_dir / Path("outc.parquet"))
            static.to_parquet(save_dir / Path("sta.parquet"))
            dynamic.to_parquet(save_dir / Path("dyn.parquet"))
            logging.info(f"Saved train data at {save_dir}")
        selected_stays = selected_stays[0]
        outcome = outcome.loc[outcome[id].isin(selected_stays)]
        static = static.loc[static[id].isin(selected_stays)]
        dynamic = dynamic.loc[dynamic[id].isin(selected_stays)]
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
