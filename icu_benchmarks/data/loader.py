import gin
import logging
import numpy as np
import os.path as pth
import pandas as pd
from pyarrow import parquet
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset

from icu_benchmarks.common import constants

VARS = constants.VARS
FILE_NAMES = constants.FILE_NAMES


@gin.configurable("RICUDataset")
class RICUDataset(Dataset):
    def __init__(self, source_path, split="train", scale_label=False):
        """
        Args:
            source_path (string): Path to the source folder with
            dataset_name (constants.ICUDataset): Name of the dataset to load
            split (string): Either 'train','val' or 'test'.
            scale_label (bool): Whether to train a min_max scaler on labels (For regression stability).
        """
        self.loader = RICULoader(source_path, split=split)
        self.split = split
        self.scale_label = scale_label
        if self.scale_label:
            self.scaler = MinMaxScaler()
            self.scaler.fit(self.loader.labels_splits_df.to_numpy().astype(float))
        else:
            self.scaler = None

    def __len__(self):
        return self.loader.num_stays

    def __getitem__(self, idx):
        data, label, pad_mask = self.loader.sample(idx)

        if self.scale_label:
            label = self.scaler.transform(label.reshape(-1, 1))[:, 0]
        return torch.from_numpy(data), torch.from_numpy(label), torch.from_numpy(pad_mask)

    def get_balance(self):
        """Return the weight balance for the split of interest.

        Returns: (list) Weights for each label.

        """
        counts = self.loader.outc_df.groupby("label").count()["stay_id"]
        return list((1 / counts) * np.sum(counts) / counts.shape[0])

    # TODO check whether this works for seq2seq task
    def set_scaler(self, scaler):
        """Sets the scaler for labels in case of regression.

        Args:
            scaler: sklearn scaler instance

        """
        self.scaler = scaler

    def get_data_and_labels(self):
        """Function to return all the data and labels aligned at once.
        We use this function for the ML methods which don't require an iterator.

        Returns: (np.array, np.array) a tuple containing  data points and label for the split.

        """
        logging.info("Gathering the samples for split " + self.split)
        labels = self.loader.labels_df["label"].to_numpy().astype(float)
        rep = self.loader.dyn_data_df
        if len(labels) == self.loader.num_stays:
            rep = rep.groupby(level="stay_id").last()
        rep = rep.to_numpy()

        if self.scaler is not None:
            labels = self.scaler.transform(labels.reshape(-1, 1))[:, 0]
        return rep, labels


@gin.configurable("RICULoader")
class RICULoader(object):
    def __init__(self, data_path, split="train", use_features=True, use_static=False):
        """
        Args:
            data_path (string): Path to the folder containing the preprocessed files with static and dynamic data,
            labels and splits.
            split (string): Name of split to load.
        """
        # We set sampling config
        self.split = split

        # Load parquet into dataframes, selecting the split from the data
        self.static_df = parquet.read_table(pth.join(data_path, FILE_NAMES["STATIC_IMPUTED"])).to_pandas().loc[self.split]
        self.outc_df = parquet.read_table(pth.join(data_path, FILE_NAMES["OUTCOME"])).to_pandas()
        self.labels_df = parquet.read_table(pth.join(data_path, FILE_NAMES["LABELS_SPLITS"])).to_pandas().loc[self.split]
        self.dyn_df = parquet.read_table(pth.join(data_path, FILE_NAMES["DYNAMIC_IMPUTED"])).to_pandas().loc[self.split]
        if use_features:
            self.features_df = (
                parquet.read_table(pth.join(data_path, FILE_NAMES["FEATURES_IMPUTED"])).to_pandas().loc[self.split]
            )
            self.dyn_df = pd.concat([self.dyn_df, self.features_df], axis=1)
        if use_static:
            self.dyn_df = pd.concat([self.dyn_df, self.static_df.set_index(VARS["STAY_ID"])], axis=1).drop(
                labels="sex", axis=1
            )
        self.dyn_data_df = self.dyn_df.drop(labels="time", axis=1)

        # calculate basic info for the data
        self.num_stays = self.static_df.shape[0]
        self.num_measurements = self.dyn_df.shape[0]
        self.maxlen = self.dyn_df.groupby([VARS["STAY_ID"]]).size().max()

    def get_window(self, stay_id, pad_value=0.0):
        """Windowing function

        Args:
            stay_id (int): Id of the stay we want to sample.
            pad_value (float): Value to pad with if stop - start < self.maxlen.

        Returns:
            window (np.array) : Array with data.
            pad_mask (np.array): 1D array with 0 if no labels are provided for the timestep.
            labels (np.array): 1D array with corresponding labels for each timestep.
        """
        # slice to make sure to always return a DF
        window = self.dyn_data_df.loc[stay_id:stay_id].to_numpy()
        labels = self.labels_df.loc[stay_id][["label"]].to_numpy().astype(float)

        if len(labels) == 1:
            # only one label per stay, align with window
            labels = np.concatenate([np.empty(window.shape[0] - 1) * np.nan, labels], axis=0)

        length_diff = self.maxlen - window.shape[0]

        # Padding the array to fulfill size requirement
        pad_mask = np.ones(window.shape[0])

        if length_diff > 0:
            # window shorter than longest window in dataset, pad to same length
            window = np.concatenate([window, np.ones((length_diff, window.shape[1])) * pad_value], axis=0)
            labels = np.concatenate([labels, np.ones((length_diff,)) * pad_value], axis=0)
            pad_mask = np.concatenate([pad_mask, np.zeros((length_diff,))], axis=0)

        not_labeled = np.argwhere(np.isnan(labels))
        if len(not_labeled) > 0:
            labels[not_labeled] = -1
            pad_mask[not_labeled] = 0

        pad_mask = pad_mask.astype(bool)
        labels = labels.astype(np.float32)
        window = window.astype(np.float32)
        return window, labels, pad_mask

    def sample(self, idx=None):
        """Function to sample from the data split of choice.
        Args:
            idx (int): A specific idx to sample. If None is provided, sample randomly.
        Returns:
            A sample from the desired distribution as tuple of numpy arrays (sample, label, mask).
        """
        if idx is None:
            idx = np.random.randint(self.num_stays)

        stay_id = self.static_df.iloc[idx][VARS["STAY_ID"]]

        return self.get_window(stay_id)
