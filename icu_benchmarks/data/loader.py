from typing import List
from pandas import DataFrame
import gin
import numpy as np
from torch import Tensor, cat, from_numpy, float32
from torch.utils.data import Dataset
import logging
from typing import Dict, Tuple

from icu_benchmarks.imputation.amputations import ampute_data
from .constants import DataSegment as Segment
from .constants import DataSplit as Split


class CommonDataset(Dataset):
    """Common dataset: subclass of Torch Dataset that represents the data to learn on.

    Args: data: Dict of the different splits of the data. split: Either 'train','val' or 'test'. vars: Contains the names of
    columns in the data. grouping_segment: str, optional: The segment of the data contains the grouping column with only
    unique values. Defaults to Segment.outcome. Is used to calculate the number of stays in the data.
    """

    def __init__(
        self,
        data: dict,
        split: str = Split.train,
        vars: Dict[str, str] = gin.REQUIRED,
        grouping_segment: str = Segment.outcome,
        mps: bool = False,
        name: str = "",
    ):
        self.split = split
        self.vars = vars
        self.grouping_df = data[split][grouping_segment].set_index(self.vars["GROUP"])
        self.features_df = (
            data[split][Segment.features].set_index(self.vars["GROUP"]).drop(labels=self.vars["SEQUENCE"], axis=1)
        )

        # calculate basic info for the data
        self.num_stays = self.grouping_df.index.unique().shape[0]
        self.maxlen = self.features_df.groupby([self.vars["GROUP"]]).size().max()
        self.mps = mps
        self.name = name

    def ram_cache(self, cache: bool = True):
        self._cached_dataset = None
        if cache:
            logging.info(f"Caching {self.split} dataset in ram.")
            self._cached_dataset = [self[i] for i in range(len(self))]

    def __len__(self) -> int:
        """Returns number of stays in the data.

        Returns:
            number of stays in the data
        """
        return self.num_stays

    def get_feature_names(self):
        return self.features_df.columns

    def to_tensor(self):
        values = []
        for entry in self:
            for i, value in enumerate(entry):
                if len(values) <= i:
                    values.append([])
                values[i].append(value.unsqueeze(0))
        return [cat(value, dim=0) for value in values]


@gin.configurable("PredictionDataset")
class PredictionDataset(CommonDataset):
    """Subclass of common dataset for prediction tasks.

    Args:
        ram_cache (bool, optional): Whether the complete dataset should be stored in ram. Defaults to True.
    """

    def __init__(self, *args, ram_cache: bool = True, **kwargs):
        super().__init__(*args, grouping_segment=Segment.outcome, **kwargs)
        self.outcome_df = self.grouping_df
        self.ram_cache(ram_cache)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Function to sample from the data split of choice. Used for deep learning implementations.

        Args:
            idx: A specific row index to sample.

        Returns:
            A sample from the data, consisting of data, labels and padding mask.
        """
        if self._cached_dataset is not None:
            return self._cached_dataset[idx]

        pad_value = 0.0
        stay_id = self.outcome_df.index.unique()[idx]  # [self.vars["GROUP"]]

        # slice to make sure to always return a DF
        window = self.features_df.loc[stay_id:stay_id].to_numpy()
        labels = self.outcome_df.loc[stay_id:stay_id][self.vars["LABEL"]].to_numpy(dtype=float)

        if len(labels) == 1:
            # only one label per stay, align with window
            labels = np.concatenate([np.empty(window.shape[0] - 1) * np.nan, labels], axis=0)

        length_diff = self.maxlen - window.shape[0]
        pad_mask = np.ones(window.shape[0])

        # Padding the array to fulfill size requirement
        if length_diff > 0:
            # window shorter than the longest window in dataset, pad to same length
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

        return from_numpy(data), from_numpy(labels), from_numpy(pad_mask)

    def get_balance(self) -> list:
        """Return the weight balance for the split of interest.

        Returns:
            Weights for each label.
        """
        counts = self.outcome_df[self.vars["LABEL"]].value_counts()
        return list((1 / counts) * np.sum(counts) / counts.shape[0])

    def get_data_and_labels(self) -> Tuple[np.array, np.array]:
        """Function to return all the data and labels aligned at once.

        We use this function for the ML methods which don't require an iterator.

        Returns:
            A Tuple containing data points and label for the split.
        """
        labels = self.outcome_df[self.vars["LABEL"]].to_numpy().astype(float)
        rep = self.features_df
        if len(labels) == self.num_stays:
            # order of groups could be random, we make sure not to change it
            rep = rep.groupby(level=self.vars["GROUP"], sort=False).last()
        rep = rep.to_numpy().astype(float)

        return rep, labels

    def to_tensor(self):
        data, labels = self.get_data_and_labels()
        if self.mps:
            return from_numpy(data).to(float32), from_numpy(labels).to(float32)
        else:
            return from_numpy(data), from_numpy(labels)


@gin.configurable("ImputationDataset")
class ImputationDataset(CommonDataset):
    """Subclass of Common Dataset that contains data for imputation models."""

    def __init__(
        self,
        data: Dict[str, DataFrame],
        split: str = Split.train,
        vars: Dict[str, str] = gin.REQUIRED,
        mask_proportion=0.3,
        mask_method="MCAR",
        mask_observation_proportion=0.3,
        ram_cache: bool = True,
    ):
        """
        Args:
            data (Dict[str, DataFrame]): data to use
            split (str, optional): split to apply. Defaults to Split.train.
            vars (Dict[str, str], optional): contains names of columns in the data. Defaults to gin.REQUIRED.
            mask_proportion (float, optional): proportion to artificially mask for amputation. Defaults to 0.3.
            mask_method (str, optional): masking mechanism. Defaults to "MCAR".
            mask_observation_proportion (float, optional): poportion of the observed data to be masked. Defaults to 0.3.
            ram_cache (bool, optional): if the dataset should be completely stored in ram and not generated on the fly during
                training. Defaults to True.
        """
        super().__init__(data, split, vars, grouping_segment=Segment.static)
        self.amputated_values, self.amputation_mask = ampute_data(
            self.features_df, mask_method, mask_proportion, mask_observation_proportion
        )
        self.amputation_mask = (self.amputation_mask + self.features_df.isna().values).bool()
        self.amputation_mask = DataFrame(self.amputation_mask, columns=self.vars[Segment.dynamic])
        self.amputation_mask[self.vars["GROUP"]] = self.features_df.index
        self.amputation_mask.set_index(self.vars["GROUP"], inplace=True)

        self.target_missingness_mask = self.features_df.isna()
        self.features_df.fillna(0, inplace=True)
        self.ram_cache(ram_cache)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Function to sample from the data split of choice.

        Used for deep learning implementations.

        Args:
            idx: A specific row index to sample.

        Returns:
            A sample from the data, consisting of data, labels and padding mask.
        """
        if self._cached_dataset is not None:
            return self._cached_dataset[idx]
        stay_id = self.grouping_df.iloc[idx].name

        # slice to make sure to always return a DF
        window = self.features_df.loc[stay_id:stay_id, self.vars[Segment.dynamic]]
        window_missingness_mask = self.target_missingness_mask.loc[stay_id:stay_id, self.vars[Segment.dynamic]]
        amputated_window = self.amputated_values.loc[stay_id:stay_id, self.vars[Segment.dynamic]]
        amputation_mask = self.amputation_mask.loc[stay_id:stay_id, self.vars[Segment.dynamic]]

        return (
            from_numpy(amputated_window.values).to(float32),
            from_numpy(amputation_mask.values).to(float32),
            from_numpy(window.values).to(float32),
            from_numpy(window_missingness_mask.values).to(float32),
        )


@gin.configurable("ImputationPredictionDataset")
class ImputationPredictionDataset(Dataset):
    """Subclass of torch dataset that represents data with missingness for imputation.

    Args:
        data (DataFrame): dict of the different splits of the data
        grouping_column (str, optional): column that is used for grouping. Defaults to "stay_id".
        select_columns (List[str], optional): the columns to serve as input for the imputation model. Defaults to None.
        ram_cache (bool, optional): wether the dataset should be stored in ram. Defaults to True.
    """

    def __init__(
        self,
        data: DataFrame,
        grouping_column: str = "stay_id",
        select_columns: List[str] = None,
        ram_cache: bool = True,
    ):
        self.dyn_df = data

        if select_columns is not None:
            self.dyn_df = self.dyn_df[list(select_columns) + grouping_column]

        if grouping_column is not None:
            self.dyn_df = self.dyn_df.set_index(grouping_column)
        else:
            self.dyn_df = data

        # calculate basic info for the data
        self.group_indices = self.dyn_df.index.unique()
        self.maxlen = self.dyn_df.groupby(grouping_column).size().max()

        self._cached_dataset = None
        if ram_cache:
            logging.info("Caching dataset in ram.")
            self._cached_dataset = [self[i] for i in range(len(self))]

    def __len__(self) -> int:
        """Returns number of stays in the data.

        Returns:
            number of stays in the data
        """
        return self.group_indices.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Function to sample from the data split of choice.

        Used for deep learning implementations.

        Args:
            idx: A specific row index to sample.

        Returns:
            A sample from the data, consisting of data, labels and padding mask.
        """
        if self._cached_dataset is not None:
            return self._cached_dataset[idx]
        stay_id = self.group_indices[idx]

        # slice to make sure to always return a DF
        window = self.dyn_df.loc[stay_id:stay_id, :]

        return from_numpy(window.values).to(float32)
