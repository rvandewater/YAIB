from pandas import DataFrame
import gin
import logging
import numpy as np
import torch
from torch import Tensor
from typing import Dict, Tuple
from torch.utils.data import Dataset

from icu_benchmarks.imputation.amputations import ampute_data


@gin.configurable("ClassificationDataset")
class RICUDataset(Dataset):
    """Subclass of torch Dataset that represents the data to learn on.

    Args:
        data: Dict of the different splits of the data.
        split: Either 'train','val' or 'test'.
        vars: Contains the names of columns in the data.
    """

    def __init__(self, data: dict, split: str = "train", vars: Dict[str, str] = gin.REQUIRED):
        self.split = split
        self.vars = vars
        self.static_df = data[split]["STATIC"]
        self.outc_df = data[split]["OUTCOME"].set_index(self.vars["GROUP"])
        self.dyn_df = data[split]["DYNAMIC"].set_index(self.vars["GROUP"]).drop(labels=self.vars["SEQUENCE"], axis=1)

        # calculate basic info for the data
        self.num_stays = self.static_df.shape[0]
        self.num_measurements = self.dyn_df.shape[0]
        self.maxlen = self.dyn_df.groupby([self.vars["GROUP"]]).size().max()

    def __len__(self) -> int:
        """Returns number of stays in the data.

        Returns:
            number of stays in the data
        """
        return self.num_stays

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Function to sample from the data split of choice.

        Used for deep learning implementations.

        Args:
            idx: A specific row index to sample.

        Returns:
            A sample from the data, consisting of data, labels and padding mask.
        """
        pad_value = 0.0
        stay_id = self.static_df.iloc[idx][self.vars["GROUP"]]

        # slice to make sure to always return a DF
        window = self.dyn_df.loc[stay_id:stay_id].to_numpy()
        labels = self.outc_df.loc[stay_id:stay_id]["label"].to_numpy(dtype=float)

        if len(labels) == 1:
            # only one label per stay, align with window
            labels = np.concatenate([np.empty(window.shape[0] - 1) * np.nan, labels], axis=0)

        length_diff = self.maxlen - window.shape[0]

        pad_mask = np.ones(window.shape[0])

        # Padding the array to fulfill size requirement
        if length_diff > 0:
            # window shorter than longest window in dataset, pad to same length
            window = np.concatenate([window, np.ones((length_diff, window.shape[1])) * pad_value], axis=0)
            labels = np.concatenate([labels, np.ones(length_diff) * pad_value], axis=0)
            pad_mask = np.concatenate([pad_mask, np.zeros(length_diff)], axis=0)

        not_labeled = np.argwhere(np.isnan(labels))
        if len(not_labeled) > 0:
            labels[not_labeled] = -1
            pad_mask[not_labeled] = 0

        pad_mask = pad_mask.astype(bool)
        labels = labels.astype(np.float32)
        data = window.astype(np.float32)

        return torch.from_numpy(data), torch.from_numpy(labels), torch.from_numpy(pad_mask)

    def get_balance(self) -> list:
        """Return the weight balance for the split of interest.

        Returns:
            Weights for each label.
        """
        counts = self.outc_df.value_counts()
        return list((1 / counts) * np.sum(counts) / counts.shape[0])

    def get_data_and_labels(self) -> Tuple[np.array, np.array]:
        """Function to return all the data and labels aligned at once.

        We use this function for the ML methods which don't require an iterator.

        Returns:
            A Tuple containing data points and label for the split.
        """
        logging.info("Gathering the samples for split " + self.split)
        labels = self.outc_df["label"].to_numpy().astype(float)
        rep = self.dyn_df
        if len(labels) == self.num_stays:
            # order of groups could be random, we make sure not to change it
            rep = rep.groupby(level=self.vars["GROUP"], sort=False).last()
        rep = rep.to_numpy()

        return rep, labels


@gin.configurable("ImputationDataset")
class ImputationDataset(Dataset):
    """Subclass of torch Dataset that represents the data to learn on.

    Args:
        data: Dict of the different splits of the data.
        split: Either 'train','val' or 'test'.
        vars: Contains the names of columns in the data.
    """

    def __init__(
        self,
        data: Dict[str, DataFrame],
        split: str = "train",
        vars: Dict[str, str] = gin.REQUIRED,
        mask_proportion=0.3,
        mask_method="MCAR",
        mask_observation_proportion=0.3,
    ):
        self.split = split
        self.vars = vars
        self.static_df = data[split]["STATIC"]
        self.dyn_df = data[split]["DYNAMIC"].set_index(self.vars["GROUP"]).drop(labels=self.vars["SEQUENCE"], axis=1)
        self.dyn_df = self.dyn_df.loc[:, self.vars["DYNAMIC"]]

        # calculate basic info for the data
        self.num_stays = self.static_df.shape[0]
        self.num_measurements = self.dyn_df.shape[0]
        self.dyn_measurements = self.dyn_df.shape[1]
        self.maxlen = self.dyn_df.groupby([self.vars["GROUP"]]).size().max()

        self.amputated_values, self.amputation_mask = ampute_data(
            self.dyn_df, mask_method, mask_proportion, mask_observation_proportion
        )
        self.amputation_mask = DataFrame(self.amputation_mask, columns=self.vars["DYNAMIC"])
        self.amputation_mask[self.vars["GROUP"]] = self.dyn_df.index
        self.amputation_mask.set_index(self.vars["GROUP"], inplace=True)

    def __len__(self) -> int:
        """Returns number of stays in the data.

        Returns:
            number of stays in the data
        """
        return self.num_stays

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Function to sample from the data split of choice.

        Used for deep learning implementations.

        Args:
            idx: A specific row index to sample.

        Returns:
            A sample from the data, consisting of data, labels and padding mask.
        """
        stay_id = self.static_df.iloc[idx][self.vars["GROUP"]]

        # slice to make sure to always return a DF
        window = self.dyn_df.loc[stay_id:stay_id, self.vars["DYNAMIC"]]
        amputated_window = self.amputated_values.loc[stay_id:stay_id, self.vars["DYNAMIC"]]
        amputation_mask = self.amputation_mask.loc[stay_id:stay_id, self.vars["DYNAMIC"]]

        return (
            torch.from_numpy(amputated_window.values).to(torch.float32),
            torch.from_numpy(amputation_mask.values).to(torch.float32),
            torch.from_numpy(window.values).to(torch.float32),
        )

    def get_data_and_labels(self) -> Tuple[np.array, np.array]:
        """Function to return all the data and labels aligned at once.

        We use this function for the ML methods which don't require an iterator.

        Returns:
            A Tuple containing data points and label for the split.
        """
        logging.info("Gathering the samples for split " + self.split)
        rep: DataFrame = self.dyn_df.to_numpy()

        return self.amputated_values, rep
