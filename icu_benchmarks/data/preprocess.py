import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

from icu_benchmarks.common import constants

VARS = constants.VARS
FILE_NAMES = constants.FILE_NAMES


def load_data(data_dir: Path) -> dict[pd.DataFrame]:
    """Load data from disk

    Args:
        data_dir (Path): path to folder with data stored as parquet files

    Returns:
        dict[pd.DataFrame]: dictionary containing data divided int OUTCOME, STATIC, and DYNAMIC. 
    """
    data = {}
    for f in ['STATIC', 'DYNAMIC', 'OUTCOME']:
        data[f] = pq.read_table(data_dir / constants.FILE_NAMES[f]).to_pandas()
    return data


def make_single_split(data: dict[pd.DataFrame], train_pct: float = 0.7, val_pct: float = 0.1, seed: int = 42) -> dict[dict[pd.DataFrame]]:
    """Randomly split the data into training, validation, and test set

    Args:
        data (dict[pd.DataFrame]): dictionary containing data divided int OUTCOME, STATIC, and DYNAMIC. 
        train_pct (float, optional): Proportion of stays assigned to training fold. Defaults to 0.7.
        val_pct (float, optional): Proportion of stays assigned to validation fold. Defaults to 0.1.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        dict[dict[pd.DataFrame]]: input data divided into 'train', 'val', and 'test'
    """
    id = VARS['STAY_ID']
    stays = data['OUTCOME'][[id]]
    stays = stays.sample(frac=1, random_state=seed)

    num_stays = len(stays)
    delims = (num_stays * np.array([0, train_pct, train_pct + val_pct, 1])).astype(int)
    
    splits = {'train': {}, 'val': {}, 'test': {}} 
    for i, fold in enumerate(splits.keys()):
        # Loop through train / val / test
        stays_in_fold = stays.iloc[delims[i]:delims[i+1], :]
        for type in data.keys():
            # Loop through DYNAMIC / STATIC / OUTCOME
            splits[fold][type] = data[type].merge(stays_in_fold, on=id, how="right")

    return splits
    
    