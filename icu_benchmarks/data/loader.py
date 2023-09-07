from typing import List
from pandas import DataFrame
import pandas as pd
import gin
import numpy as np
from torch import Tensor, cat, from_numpy, float32, empty, stack
from torch.utils.data import Dataset
import logging
from typing import Dict, Tuple
from icu_benchmarks.imputation.amputations import ampute_data
from .constants import DataSegment as Segment
from .constants import DataSplit as Split
from .constants import FeatType as Features
from collections import OrderedDict
from pytorch_forecasting import TimeSeriesDataSet


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
    ):
        self.split = split
        self.vars = vars
        self.grouping_df = data[split][grouping_segment].set_index(self.vars["GROUP"])
        self.features_df = (
            # drops time coulmn and sets index to stay_id
            data[split][Segment.features]
            .set_index(self.vars["GROUP"])
            .drop(labels=self.vars["SEQUENCE"], axis=1)
        )

        # calculate basic info for the data
        self.num_stays = self.grouping_df.index.unique().shape[0]
        self.maxlen = self.features_df.groupby([self.vars["GROUP"]]).size().max()

    def ram_cache(self, cache: bool = True):
        self._cached_dataset = None
        if cache:
            logging.info("Caching dataset in ram.")
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
        stay_id = self.outcome_df.index.unique()[idx]

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
        return from_numpy(data).to(float32), from_numpy(labels).to(float32)


@gin.configurable("PredictionDatasetTFT")
class PredictionDatasetTFT(PredictionDataset):
    """Subclass of prediction dataset for TFT as we need to define if variables are cont,static,known or observed.
    We also need to feed the model the variables in a specific order

    Args:
        ram_cache (bool, optional): Whether the complete dataset should be stored in ram. Defaults to True.
    """

    def __init__(self, *args, ram_cache: bool = True, **kwargs):
        super().__init__(*args, ram_cache=True, **kwargs)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Function to sample from the data split of choice. Used for TFT.
        The data needs to be given to the model in the following order 
        [static categorical,static contious,known catergorical,known continous, observed categorical, observed continous,target ,id]

        Args:
            idx: A specific row index to sample.

        Returns:
            A sample from the data, consisting of data, labels and padding mask.
        """
        if self._cached_dataset is not None:
            return self._cached_dataset[idx]

        pad_value = 0.0
        stay_id = self.outcome_df.index.unique()[idx]

        # We need to be sure that tensors are returned in the correct order to be processed correclty by tft
        tensors = [[] for _ in range(8)]
        for var in self.features_df.columns:
            if var == "sex":
                tensors[0].append(self.features_df.loc[stay_id:stay_id][var].to_numpy())
            elif var == "age" or var == "height" or var == "weight":
                tensors[1].append(self.features_df.loc[stay_id:stay_id][var].to_numpy())
            elif "MissingIndicator" in var:
                tensors[4].append(self.features_df.loc[stay_id:stay_id][var].to_numpy())
            else:
                tensors[5].append(self.features_df.loc[stay_id:stay_id][var].to_numpy())

        tensors[6].extend(self.outcome_df.loc[stay_id:stay_id][self.vars["LABEL"]].to_numpy(dtype=float))
        tensors[7].append(np.asarray([stay_id]))
        window_shape0 = np.shape(tensors[0])[1]

        if len(tensors[6]) == 1:
            # only one label per stay, align with window
            tensors[6] = np.concatenate([np.empty(window_shape0 - 1) * np.nan, tensors[6]], axis=0)

        length_diff = self.maxlen - window_shape0
        pad_mask = np.ones(window_shape0)
        # Padding the array to fulfill size requirement

        if length_diff > 0:
            # window shorter than the longest window in dataset, pad to same length
            tensors[0] = np.concatenate(
                [tensors[0], np.ones((np.shape(tensors[0])[0], self.maxlen - np.shape(tensors[0])[1])) * pad_value], axis=1
            )
            tensors[1] = np.concatenate(
                [tensors[1], np.ones((np.shape(tensors[1])[0], self.maxlen - np.shape(tensors[1])[1])) * pad_value], axis=1
            )
            tensors[4] = np.concatenate(
                [tensors[4], np.ones((np.shape(tensors[4])[0], self.maxlen - np.shape(tensors[4])[1])) * pad_value], axis=1
            )
            tensors[5] = np.concatenate(
                [tensors[5], np.ones((np.shape(tensors[5])[0], self.maxlen - np.shape(tensors[5])[1])) * pad_value], axis=1
            )

            tensors[6] = np.concatenate([tensors[6], np.ones(
                self.maxlen - np.shape(tensors[6])[0]) * pad_value], axis=0)
            pad_mask = np.concatenate([pad_mask, np.zeros(length_diff)], axis=0)
        tensors[7] = np.concatenate(
            [tensors[7], np.ones((np.shape(tensors[7])[0], self.maxlen - np.shape(tensors[7])[1])) * stay_id], axis=1
        )  # should be done regardless of length_diff
        not_labeled = np.argwhere(np.isnan(tensors[6]))
        if len(not_labeled) > 0:
            tensors[6][not_labeled] = -1
            pad_mask[not_labeled] = 0
        tensors[6] = [tensors[6]]
        pad_mask = pad_mask.astype(bool)

        tensors = (from_numpy(np.array(tensor)).to(float32) for tensor in tensors)
        tensors = [stack((x,), dim=-1) if x.numel() > 0 else empty(0) for x in tensors]
        return OrderedDict(zip(Features.FEAT_NAMES, tensors)), from_numpy(pad_mask)


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


@gin.configurable("PredictionDatasetTFTpytorch")
class PredictionDatasetTFTpytorch(TimeSeriesDataSet):
    """Subclass of timeseries dataset works with pyotrch forecasting library .

    Args:
        data (DataFrame): dict of the different splits of the data
        split: Either 'train','val' or 'test'
        max_prediction_length: maximum number of time steps to predict,
        max_encoder_length: maximum length of input sequence to give the model,
        ram_cache (bool, optional): wether the dataset should be stored in ram. Defaults to True.
    """

    def __init__(
        self,
        data: dict,
        split: str,
        max_prediction_length: int,
        max_encoder_length: int,
        *args,
        ram_cache: bool = False,
        **kwargs
    ):
        data[split]["FEATURES"]["time_idx"] = ((data[split]["FEATURES"]["time"] / pd.Timedelta(seconds=3600))).astype(
            int
        )  # create an incremental column indicating the time step(required by constructor)
        data = data.get(split)  # get split
        labels = data["OUTCOME"]
        features = data["FEATURES"]
        self.data = pd.merge(labels, features, on=["stay_id", "time"])  # combine labels and features
        self.split = split
        self.args = args
        self.ram_cache = ram_cache
        self.kwargs = kwargs
        self.column_names = features.columns
        super().__init__(
            data=self.data,
            time_idx="time_idx",
            target="label",
            group_ids=["stay_id"],
            min_encoder_length=max_encoder_length,
            max_encoder_length=max_encoder_length,
            min_prediction_length=max_prediction_length,
            max_prediction_length=max_prediction_length,
            static_categoricals=[],
            static_reals=["height", "weight", "age", "sex"],
            time_varying_known_categoricals=[],
            time_varying_known_reals=[],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=[
                "alb",
                "alp",
                "alt",
                "ast",
                "be",
                "bicar",
                "bili",
                "bili_dir",
                "bnd",
                "bun",
                "ca",
                "cai",
                "ck",
                "ckmb",
                "cl",
                "crea",
                "crp",
                "dbp",
                "fgn",
                "fio2",
                "glu",
                "hgb",
                "hr",
                "inr_pt",
                "k",
                "lact",
                "lymph",
                "map",
                "mch",
                "mchc",
                "mcv",
                "methb",
                "mg",
                "na",
                "neut",
                "o2sat",
                "pco2",
                "ph",
                "phos",
                "plt",
                "po2",
                "ptt",
                "resp",
                "sbp",
                "temp",
                "tnt",
                "urine",
                "wbc",
                "MissingIndicator_1",
                "MissingIndicator_2",
                "MissingIndicator_3",
                "MissingIndicator_4",
                "MissingIndicator_5",
                "MissingIndicator_6",
                "MissingIndicator_7",
                "MissingIndicator_8",
                "MissingIndicator_9",
                "MissingIndicator_10",
                "MissingIndicator_11",
                "MissingIndicator_12",
                "MissingIndicator_13",
                "MissingIndicator_14",
                "MissingIndicator_15",
                "MissingIndicator_16",
                "MissingIndicator_17",
                "MissingIndicator_18",
                "MissingIndicator_19",
                "MissingIndicator_20",
                "MissingIndicator_21",
                "MissingIndicator_22",
                "MissingIndicator_23",
                "MissingIndicator_24",
                "MissingIndicator_25",
                "MissingIndicator_26",
                "MissingIndicator_27",
                "MissingIndicator_28",
                "MissingIndicator_29",
                "MissingIndicator_30",
                "MissingIndicator_31",
                "MissingIndicator_32",
                "MissingIndicator_33",
                "MissingIndicator_34",
                "MissingIndicator_35",
                "MissingIndicator_36",
                "MissingIndicator_37",
                "MissingIndicator_38",
                "MissingIndicator_39",
                "MissingIndicator_40",
                "MissingIndicator_41",
                "MissingIndicator_42",
                "MissingIndicator_43",
                "MissingIndicator_44",
                "MissingIndicator_45",
                "MissingIndicator_46",
                "MissingIndicator_47",
                "MissingIndicator_48",
            ],
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

    def get_balance(self) -> list:
        """Return the weight balance for the split of interest.

        Returns:
            Weights for each label.
        """

        counts = self.data["target"][0].unique(return_counts=True)

        return list((1 / counts[1]) * counts[1].sum() / counts[0].shape[0])

    def get_feature_names(self):
        return self.column_names
