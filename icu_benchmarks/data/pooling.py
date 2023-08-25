import copy
import pickle
import torch
import logging
import gin
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from .constants import DataSplit as Split, DataSegment as Segment, VarType as Var


def pool_datasets(datasets={}, samples=10000, vars = [], seed=42,shuffle=True, stratify=None, **kwargs):
    """
    Pool datasets into a single dataset.
    Args:
        datasets: list of datasets to pool
    Returns:
        pooled dataset
    """
    if len(datasets) == 0:
        raise ValueError("No datasets supplied.")
    pooled_data = {Segment.static:[], Segment.dynamic:[], Segment.outcome:[]}
    id = vars[Var.group]
    int_id = 0
    for key, value in datasets.items():
            if samples:
                int_id += 1
                num_classes = 2
                repeated_digit = str(int_id) * 4
                samples_per_class = int(samples / num_classes)
                # outcome = value[Segment.outcome].groupby(vars["LABEL"], group_keys=False).sample(samples_per_class,random_state=seed)
                outcome = value[Segment.outcome]
                outcome = train_test_split(outcome, stratify=outcome[vars["LABEL"]], shuffle=shuffle, random_state=seed, train_size=samples)[0]
                stays = pd.Series(outcome[id].unique())
                static = value[Segment.static]
                dynamic = value[Segment.dynamic]
                static = static.loc[static[id].isin(stays)]
                dynamic = dynamic.loc[dynamic[id].isin(stays)]
                outcome[id] = outcome[id].map(lambda x: int(str(x) + repeated_digit))
                static[id] = static[id].map(lambda x: int(str(x) + repeated_digit))
                dynamic[id] = dynamic[id].map(lambda x: int(str(x) + repeated_digit))
                pooled_data[Segment.static].append(static)
                pooled_data[Segment.dynamic].append(dynamic)
                pooled_data[Segment.outcome].append(outcome)
    for key, value in pooled_data.items():
        pooled_data[key] = pd.concat(value, ignore_index=True)
    return pooled_data
