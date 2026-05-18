import copy
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union
import logging

import gin
import pandas as pd
import polars as pl
import recipys
import torch
from numpy import nan as np_nan
from recipys.recipe import Recipe
from recipys.selector import all_numeric_predictors, all_of, all_outcomes, has_type
from recipys.step import (
    Accumulator,
    Selector,
    StepHistorical,
    StepImputeFastForwardFill,
    StepImputeFastZeroFill,
    StepImputeFill,
    StepImputeModel,
    StepScale,
    StepSklearn,
)
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, MinMaxScaler

from icu_benchmarks.data.loader import ImputationPredictionDataset

from .constants import DataSegment as DataSegment
from .constants import DataSplit as DataSplit


class Preprocessor(ABC):
    def __init__(
        self,
        generate_features: bool = False,
        scaling: bool = True,
        use_static_features: bool = True,
        save_cache: Optional[Union[str, Path]] = None,
        load_cache: Optional[Union[str, Path]] = None,
        vars_to_exclude: Optional[list[str]] = None,
    ):
        self.generate_features = generate_features
        self.scaling = scaling
        self.use_static_features = use_static_features
        self.imputation_model = None
        self.model_impute = None
        self.save_cache = save_cache
        self.load_cache = load_cache
        self.vars_to_exclude = vars_to_exclude

    @abstractmethod
    def apply(self, data, vars):
        return data

    @abstractmethod
    def to_cache_string(self):
        return f"{self.__class__.__name__}"

    def set_imputation_model(self, imputation_model):
        self.imputation_model = imputation_model
        if self.imputation_model is not None:
            from icu_benchmarks.wandb_utils import update_wandb_config

            update_wandb_config({"imputation_model": self.imputation_model.__class__.__name__})

    def vars_selection(self, input_variables, segment: str) -> Union[str, list[str]]:
        if input_variables.get(segment, None) is None or len(input_variables[segment]) == 0:
            logging.warning("No dynamic variables provided. Skipping dynamic preprocessing.")
            return []  # Return empty list if no variables are provided
        vars_to_apply: Union[str, list[str]]
        if self.vars_to_exclude is not None:
            # Exclude vars_to_exclude from missing indicator/ feature generation
            vars_to_apply = list(set(input_variables[segment]) - set(self.vars_to_exclude))
            logging.info(f"Excluding features from in-built preprocessing: {len(self.vars_to_exclude)} : "
            f"{self.vars_to_exclude if len(self.vars_to_exclude) < 10 else f'{self.vars_to_exclude[:10]}...'}...")
            if len(vars_to_apply) == 0:
                logging.warning(
                    f"No variables left after excluding vars_to_exclude: {self.vars_to_exclude} for segment {segment}. "
                    "Skipping preprocessing for this segment."
                )
        else:
            vars_to_apply = input_variables[segment]
        return vars_to_apply

@gin.configurable("base_classification_preprocessor")
class PolarsClassificationPreprocessor(Preprocessor):
    def __init__(
        self,
        generate_features: bool = False,
        scaling: bool = True,
        use_static_features: bool = True,
        save_cache: Optional[Union[str, Path]] = None,
        load_cache: Optional[Union[str, Path]] = None,
        vars_to_exclude: Optional[list[str]] = None,
    ):
        """
        Args:
            generate_features: Generate features for dynamic data.
            scaling: Scaling of dynamic and static data.
            use_static_features: Use static features.
            save_cache: Save recipe cache from this path.
            load_cache: Load recipe cache from this path.
            vars_to_exclude: Variables to exclude from missing indicator/ feature generation.
        Returns:
            Preprocessed data.
        """
        super().__init__(
            generate_features=generate_features,
            scaling=scaling,
            use_static_features=use_static_features,
            save_cache=save_cache,
            load_cache=load_cache,
            vars_to_exclude=vars_to_exclude,
        )
        self.imputation_model = None

    def apply(
        self,
        data: dict[str, dict[str, pl.DataFrame]],
        vars: dict[str, Union[str, list[str]]],
    ) -> dict[str, dict[str, pl.DataFrame]]:
        """
        Args:
            data: Train, validation and test data dictionary. Further divided in static, dynamic, and outcome.
            vars: Variables for static, dynamic, outcome.
        Returns:
            Preprocessed data.
        """
        # Check if dynamic features are present
        if (
            self.use_static_features
            and all(DataSegment.static in value for value in data.values())
            and len(vars[DataSegment.static]) > 0
        ):
            logging.info("Preprocessing static features.")
            data = self._process_static(data, vars)
        else:
            self.use_static_features = False

        if all(DataSegment.dynamic in value for value in data.values()):
            logging.info("Preprocessing dynamic features.")
            logging.info(data.keys())
            data = self._process_dynamic(data, vars)
            if self.use_static_features:
                # Join static and dynamic data.
                data[DataSplit.train][DataSegment.dynamic] = data[DataSplit.train][DataSegment.dynamic].join(
                    data[DataSplit.train][DataSegment.static], on=vars["GROUP"]
                )
                data[DataSplit.val][DataSegment.dynamic] = data[DataSplit.val][DataSegment.dynamic].join(
                    data[DataSplit.val][DataSegment.static], on=vars["GROUP"]
                )
                data[DataSplit.test][DataSegment.dynamic] = data[DataSplit.test][DataSegment.dynamic].join(
                    data[DataSplit.test][DataSegment.static], on=vars["GROUP"]
                )

                # Remove static features from splits
                data[DataSplit.train][DataSegment.features] = data[DataSplit.train].pop(DataSegment.static)
                data[DataSplit.val][DataSegment.features] = data[DataSplit.val].pop(DataSegment.static)
                data[DataSplit.test][DataSegment.features] = data[DataSplit.test].pop(DataSegment.static)

            # Create feature splits
            data[DataSplit.train][DataSegment.features] = data[DataSplit.train].pop(DataSegment.dynamic)
            data[DataSplit.val][DataSegment.features] = data[DataSplit.val].pop(DataSegment.dynamic)
            data[DataSplit.test][DataSegment.features] = data[DataSplit.test].pop(DataSegment.dynamic)
        elif self.use_static_features:
            data[DataSplit.train][DataSegment.features] = data[DataSplit.train].pop(DataSegment.static)
            data[DataSplit.val][DataSegment.features] = data[DataSplit.val].pop(DataSegment.static)
            data[DataSplit.test][DataSegment.features] = data[DataSplit.test].pop(DataSegment.static)
        else:
            raise Exception(f"No recognized data segments data to preprocess. Available: {data.keys()}")
        logging.debug("Data head")
        logging.debug(data[DataSplit.train][DataSegment.features].head())
        logging.debug(data[DataSplit.train][DataSegment.outcome])

        # if not isinstance(vars["SEQUENCE"], str):
        #     raise TypeError(f'Expected key "SEQUENCE" to be of type str, got {type(vars["SEQUENCE"])} instead')
        if "SEQUENCE" in vars:
            for split in [DataSplit.train, DataSplit.val, DataSplit.test]:
                # Check if we have sequence in the outcome and the data and outcome length match
                if vars["SEQUENCE"] in data[split][DataSegment.outcome] and len(data[split][DataSegment.features]) != len(
                    data[split][DataSegment.outcome]
                ):
                    raise Exception(
                        f"Data and outcome length mismatch in {split} split: "
                        f"features: {len(data[split][DataSegment.features])}, outcome: {len(data[split][DataSegment.outcome])}"
                    )
        data[DataSplit.train][DataSegment.features] = data[DataSplit.train][DataSegment.features].unique()
        data[DataSplit.val][DataSegment.features] = data[DataSplit.val][DataSegment.features].unique()
        data[DataSplit.test][DataSegment.features] = data[DataSplit.test][DataSegment.features].unique()

        logging.debug(f"Generate features in preprocessing: {self.generate_features}")
        return data

    def _process_static(self, data: dict[str, dict[str, pl.DataFrame]], vars: dict[str, Union[str, list[str]]]):
        vars_to_apply = self.vars_selection(input_variables=vars, segment=DataSegment.dynamic)
        if len(vars_to_apply) == 0:
            return data
        sta_rec = Recipe(data[DataSplit.train][DataSegment.static], [], vars[DataSegment.static])
        sta_rec.add_step(StepSklearn(MissingIndicator(features="all"), sel=all_of(vars[DataSegment.static]), in_place=False))
        if self.scaling:
            sta_rec.add_step(StepScale())
        sta_rec.add_step(StepImputeFill(sel=all_numeric_predictors(), strategy="zero"))
        types = ["String", "Object", "Categorical"]
        sel = has_type(types)
        if len(sel(sta_rec.data)) > 0:
            sta_rec.add_step(StepSklearn(SimpleImputer(missing_values=np_nan, strategy="most_frequent"), sel=has_type(types)))
            sta_rec.add_step(StepSklearn(LabelEncoder(), sel=has_type(types), columnwise=True))

        data = apply_recipe_to_splits(sta_rec, data, DataSegment.static, self.save_cache, self.load_cache)

        return data

    def _model_impute(self, data: pd.DataFrame, group: Optional[str] = None):
        if not self.imputation_model:
            raise ValueError("No Imputation Model provided! Aborting...")
        dataset = ImputationPredictionDataset(data, group, self.imputation_model.trained_columns)
        input_data = torch.cat([data_point.unsqueeze(0) for data_point in dataset], dim=0)
        self.imputation_model.eval()
        with torch.no_grad():
            logging.info(f"Imputing with {self.imputation_model.__class__.__name__}.")
            imputation = self.imputation_model.predict(input_data)
            logging.info("Imputation done.")
        assert imputation.isnan().sum() == 0
        data = data.copy()
        data.loc[:, self.imputation_model.trained_columns] = imputation.flatten(end_dim=1).to("cpu")
        if group is not None:
            data.drop(columns=group, inplace=True)
        return data

    def _process_dynamic(self, data: dict[str, dict[str, pl.DataFrame]], vars: dict[str, Union[str, list[str]]]):
        vars_to_apply = self.vars_selection(input_variables=vars, segment=DataSegment.dynamic)
        if len(vars_to_apply) == 0:
            return data
        dyn_rec = Recipe(
            data[DataSplit.train][DataSegment.dynamic], [], vars[DataSegment.dynamic], vars["GROUP"], vars["SEQUENCE"]
        )
        if self.scaling:
            dyn_rec.add_step(StepScale(sel=all_numeric_predictors(backend=recipys.constants.Backend.POLARS)))
        if self.imputation_model is not None:
            dyn_rec.add_step(StepImputeModel(model=self.model_impute, sel=all_of(vars[DataSegment.dynamic])))
        dyn_rec.add_step(StepSklearn(MissingIndicator(features="all"), sel=all_of(vars_to_apply), in_place=False))
        dyn_rec.add_step(StepImputeFill(strategy="forward"))
        dyn_rec.add_step(StepImputeFill(strategy="zero"))
        if self.generate_features:
            dyn_rec = self._dynamic_feature_generation(dyn_rec, all_of(vars_to_apply))
        data = apply_recipe_to_splits(dyn_rec, data, DataSegment.dynamic, self.save_cache, self.load_cache)
        # logging.info(f"Data columns: {len(data[Split.train][Segment.dynamic].columns)} -> old columns: {len(old_columns)}, added columns: {set(data[Split.train][Segment.dynamic].columns) - set(old_columns)}")
        return data



    def _dynamic_feature_generation(self, data, dynamic_vars):
        logging.debug("Adding dynamic feature generation.")
        data.add_step(StepHistorical(sel=dynamic_vars, fun=Accumulator.MIN, suffix="min_hist"))
        data.add_step(StepHistorical(sel=dynamic_vars, fun=Accumulator.MAX, suffix="max_hist"))
        data.add_step(StepHistorical(sel=dynamic_vars, fun=Accumulator.COUNT, suffix="count_hist"))
        data.add_step(StepHistorical(sel=dynamic_vars, fun=Accumulator.MEAN, suffix="mean_hist"))
        return data

    def to_cache_string(self):
        return (
            super().to_cache_string()
            + f"_classification_{self.generate_features}_{self.scaling}_{self.imputation_model.__class__.__name__}"
        )


@gin.configurable("base_regression_preprocessor")
class PolarsRegressionPreprocessor(PolarsClassificationPreprocessor):
    # Override base classification preprocessor
    def __init__(
        self,
        generate_features: bool = False,
        scaling: bool = True,
        use_static_features: bool = True,
        outcome_max=None,
        outcome_min=None,
        save_cache: Optional[Union[str, Path]] = None,
        load_cache: Optional[Union[str, Path]] = None,
        vars_to_exclude: Optional[list[str]] = None,
    ):
        """
        Args:
            generate_features: Generate features for dynamic data.
            scaling: Scaling of dynamic and static data.
            use_static_features: Use static features.
            max_range: Maximum value in outcome.
            min_range: Minimum value in outcome.
            save_cache: Save recipe cache.
            load_cache: Load recipe cache.
        Returns:
            Preprocessed data.
        """
        super().__init__(
            generate_features=generate_features,
            scaling=scaling,
            use_static_features=use_static_features,
            save_cache=save_cache,
            load_cache=load_cache,
            vars_to_exclude=vars_to_exclude,
        )
        self.outcome_max = outcome_max
        self.outcome_min = outcome_min

    def apply(
        self, data: dict[str, dict[str, pl.DataFrame]], vars: dict[str, Union[str, list[str]]]
    ) -> dict[str, dict[str, pl.DataFrame]]:
        """
        Args:
            data: Train, validation and test data dictionary. Further divided in static, dynamic, and outcome.
            vars: Variables for static, dynamic, outcome.
        Returns:
            Preprocessed data.
        """
        for split in [DataSplit.train, DataSplit.val, DataSplit.test]:
            data = self._process_outcome(data, vars, split)

        data = super().apply(data, vars)
        return data

    def _process_outcome(
        self, data: dict[str, dict[str, pl.DataFrame]], vars: dict[str, Union[str, list[str]]], split: str
    ) -> dict[str, dict[str, pl.DataFrame]]:
        logging.debug(f"Processing {split} outcome values.")
        outcome_rec = Recipe(data[split][DataSegment.outcome], vars["LABEL"], [], vars["GROUP"])
        # If the range is predefined, use predefined transformation function
        if self.outcome_max is not None and self.outcome_min is not None:
            if self.outcome_max == self.outcome_min:
                logging.warning("outcome_max equals outcome_min. Skipping outcome scaling.")
            else:
                outcome_rec.add_step(
                    StepSklearn(
                        sklearn_transformer=FunctionTransformer(
                            func=lambda x: ((x - self.outcome_min) / (self.outcome_max - self.outcome_min))
                        ),
                        sel=all_outcomes(),
                    )
                )
        else:
            # If the range is not predefined, use MinMaxScaler
            outcome_rec.add_step(StepSklearn(MinMaxScaler(), sel=all_outcomes()))
        outcome_rec.prep()
        data[split][DataSegment.outcome] = pl.DataFrame(outcome_rec.bake())
        return data


@gin.configurable("pandas_classification_preprocessor")
class PandasClassificationPreprocessor(Preprocessor):
    def __init__(
        self,
        generate_features: bool = True,
        scaling: bool = True,
        use_static_features: bool = True,
        save_cache: Optional[Union[str, Path]] = None,
        load_cache: Optional[Union[str, Path]] = None,
    ):
        """
        Args:
            generate_features: Generate features for dynamic data.
            scaling: Scaling of dynamic and static data.
            use_static_features: Use static features.
            save_cache: Save recipe cache from this path.
            load_cache: Load recipe cache from this path.
        Returns:
            Preprocessed data.
        """
        self.generate_features = generate_features
        self.scaling = scaling
        self.use_static_features = use_static_features
        self.imputation_model = None
        self.save_cache = save_cache
        self.load_cache = load_cache

    def apply(
        self, data: dict[str, dict[str, pd.DataFrame]], vars: dict[str, Union[str, list[str]]]
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Args:
            data: Train, validation and test data dictionary. Further divided in static, dynamic, and outcome.
            vars: Variables for static, dynamic, outcome.
        Returns:
            Preprocessed data.
        """
        logging.info("Preprocessing dynamic features.")

        data = self._process_dynamic(data, vars)
        if self.use_static_features:
            logging.info("Preprocessing static features.")
            data = self._process_static(data, vars)

            # Set index to grouping variable
            data[DataSplit.train][DataSegment.static] = data[DataSplit.train][DataSegment.static].set_index(vars["GROUP"])
            data[DataSplit.val][DataSegment.static] = data[DataSplit.val][DataSegment.static].set_index(vars["GROUP"])
            data[DataSplit.test][DataSegment.static] = data[DataSplit.test][DataSegment.static].set_index(vars["GROUP"])

            # Join static and dynamic data.
            data[DataSplit.train][DataSegment.dynamic] = data[DataSplit.train][DataSegment.dynamic].join(
                data[DataSplit.train][DataSegment.static], on=vars["GROUP"]
            )
            data[DataSplit.val][DataSegment.dynamic] = data[DataSplit.val][DataSegment.dynamic].join(
                data[DataSplit.val][DataSegment.static], on=vars["GROUP"]
            )
            data[DataSplit.test][DataSegment.dynamic] = data[DataSplit.test][DataSegment.dynamic].join(
                data[DataSplit.test][DataSegment.static], on=vars["GROUP"]
            )

            # Remove static features from splits
            data[DataSplit.train][DataSegment.features] = data[DataSplit.train].pop(DataSegment.static)
            data[DataSplit.val][DataSegment.features] = data[DataSplit.val].pop(DataSegment.static)
            data[DataSplit.test][DataSegment.features] = data[DataSplit.test].pop(DataSegment.static)

        # Create feature splits
        data[DataSplit.train][DataSegment.features] = data[DataSplit.train].pop(DataSegment.dynamic)
        data[DataSplit.val][DataSegment.features] = data[DataSplit.val].pop(DataSegment.dynamic)
        data[DataSplit.test][DataSegment.features] = data[DataSplit.test].pop(DataSegment.dynamic)

        logging.debug("Data head")
        logging.debug(data[DataSplit.train][DataSegment.features].head())
        logging.debug(data[DataSplit.train][DataSegment.outcome].head())
        logging.debug(f"Generate features: {self.generate_features}")
        return data

    def _process_static(
        self, data: dict[str, dict[str, pd.DataFrame]], vars: dict[str, Union[str, list[str]]]
    ) -> dict[str, dict[str, pd.DataFrame]]:
        vars_to_apply = self.vars_selection(input_variables=vars, segment=DataSegment.static)
        if len(vars_to_apply) == 0:
            return data
        sta_rec = Recipe(data[DataSplit.train][DataSegment.static], [], vars[DataSegment.static])
        if self.scaling:
            sta_rec.add_step(StepScale())

        sta_rec.add_step(StepImputeFastZeroFill(sel=all_numeric_predictors()))
        if len(data[DataSplit.train][DataSegment.static].select_dtypes(include=["object"]).columns) > 0:
            sta_rec.add_step(
                StepSklearn(SimpleImputer(missing_values=np_nan, strategy="most_frequent"), sel=has_type("object"))
            )
            sta_rec.add_step(StepSklearn(LabelEncoder(), sel=has_type("object"), columnwise=True))

        data = apply_recipe_to_splits(sta_rec, data, DataSegment.static, self.save_cache, self.load_cache)

        return data

    def _model_impute(self, data: pd.DataFrame, group: Optional[str] = None) -> pd.DataFrame:
        if not self.imputation_model:
            raise TypeError("No Imputation Model provided. Aborting...")
        dataset = ImputationPredictionDataset(data, group, self.imputation_model.trained_columns)
        input_data = torch.cat([data_point.unsqueeze(0) for data_point in dataset], dim=0)
        self.imputation_model.eval()
        with torch.no_grad():
            logging.info(f"Imputing with {self.imputation_model.__class__.__name__}.")
            imputation = self.imputation_model.predict(input_data)
            logging.info("Imputation done.")
        assert imputation.isnan().sum() == 0
        data = data.copy()
        data.loc[:, self.imputation_model.trained_columns] = imputation.flatten(end_dim=1).to("cpu")
        if group is not None:
            data.drop(columns=group, inplace=True)
        return data

    def _process_dynamic(
        self, data: dict[str, dict[str, pd.DataFrame]], vars: dict[str, Union[str, list[str]]]
    ) -> dict[str, dict[str, pd.DataFrame]]:
        vars_to_apply = self.vars_selection(input_variables=vars, segment=DataSegment.dynamic)
        if len(vars_to_apply) == 0:
            return data
        dyn_rec = Recipe(
            data[DataSplit.train][DataSegment.dynamic], [], vars[DataSegment.dynamic], vars["GROUP"], vars["SEQUENCE"]
        )
        vars_to_apply = self.vars_selection(input_variables=vars, segment=DataSegment.dynamic)
        if len(vars_to_apply) == 0:
            return data
        if self.scaling:
            dyn_rec.add_step(StepScale())
        if self.imputation_model is not None:
            dyn_rec.add_step(StepImputeModel(model=self.model_impute, sel=all_of(vars[DataSegment.dynamic])))
        dyn_rec.add_step(StepSklearn(MissingIndicator(), sel=all_of(vars[DataSegment.dynamic]), in_place=False))
        dyn_rec.add_step(StepImputeFastForwardFill())
        dyn_rec.add_step(StepImputeFastZeroFill())
        if self.generate_features:
            dyn_rec = self._dynamic_feature_generation(dyn_rec, all_of(vars[DataSegment.dynamic]))
        data = apply_recipe_to_splits(dyn_rec, data, DataSegment.dynamic, self.save_cache, self.load_cache)
        return data

    def _dynamic_feature_generation(self, data: Recipe, dynamic_vars: Selector):
        logging.debug("Adding dynamic feature generation.")
        data.add_step(StepHistorical(sel=dynamic_vars, fun=Accumulator.MIN, suffix="min_hist"))
        data.add_step(StepHistorical(sel=dynamic_vars, fun=Accumulator.MAX, suffix="max_hist"))
        data.add_step(StepHistorical(sel=dynamic_vars, fun=Accumulator.COUNT, suffix="count_hist"))
        data.add_step(StepHistorical(sel=dynamic_vars, fun=Accumulator.MEAN, suffix="mean_hist"))
        return data

    def to_cache_string(self) -> str:
        return (
            super().to_cache_string()
            + f"_classification_{self.generate_features}_{self.scaling}_{self.imputation_model.__class__.__name__}"
        )


@gin.configurable("pandas_regression_preprocessor")
class PandasRegressionPreprocessor(PandasClassificationPreprocessor):
    # Override base classification preprocessor
    def __init__(
        self,
        generate_features: bool = True,
        scaling: bool = True,
        use_static_features: bool = True,
        outcome_max=None,
        outcome_min=None,
        save_cache: Optional[Union[str, Path]] = None,
        load_cache: Optional[Union[str, Path]] = None,
    ):
        """
        Args:
            generate_features: Generate features for dynamic data.
            scaling: Scaling of dynamic and static data.
            use_static_features: Use static features.
            max_range: Maximum value in outcome.
            min_range: Minimum value in outcome.
            save_cache: Save recipe cache.
            load_cache: Load recipe cache.
        Returns:
            Preprocessed data.
        """
        super().__init__(
            generate_features=generate_features,
            scaling=scaling,
            use_static_features=use_static_features,
            save_cache=save_cache,
            load_cache=load_cache,
        )
        self.outcome_max = outcome_max
        self.outcome_min = outcome_min

    def apply(
        self, data: dict[str, dict[str, pd.DataFrame]], vars: dict[str, Union[str, list[str]]]
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Args:
            data: Train, validation and test data dictionary. Further divided in static, dynamic, and outcome.
            vars: Variables for static, dynamic, outcome.
        Returns:
            Preprocessed data.
        """
        for split in [DataSplit.train, DataSplit.val, DataSplit.test]:
            data = self._process_outcome(data, vars, split)

        data = super().apply(data, vars)
        return data

    def _process_outcome(self, data, vars, split):
        logging.debug(f"Processing {split} outcome values.")
        outcome_rec = Recipe(data[split][DataSegment.outcome], vars["LABEL"], [], vars["GROUP"])
        # If the range is predefined, use predefined transformation function
        if self.outcome_max is not None and self.outcome_min is not None:
            outcome_rec.add_step(
                StepSklearn(
                    sklearn_transformer=FunctionTransformer(
                        func=lambda x: ((x - self.outcome_min) / (self.outcome_max - self.outcome_min))
                    ),
                    sel=all_outcomes(),
                )
            )
        else:
            # If the range is not predefined, use MinMaxScaler
            outcome_rec.add_step(StepSklearn(MinMaxScaler(), sel=all_outcomes()))
        outcome_rec.prep()
        data[split][DataSegment.outcome] = outcome_rec.bake()
        return data


@gin.configurable("base_imputation_preprocessor")
class PandasImputationPreprocessor(Preprocessor):
    def __init__(
        self,
        scaling: bool = True,
        use_static_features: bool = True,
        filter_missing_values: bool = True,
    ):
        """
        Preprocesses data for imputation.

        Args:
            scaling (bool, optional): If the values in each column should be normalized. Defaults to True.
            use_static_features (bool, optional): If static features should be included in the dataset. Defaults to True.
        """
        self.scaling = scaling
        self.use_static_features = use_static_features
        self.filter_missing_values = filter_missing_values

    def apply(
        self, data: dict[str, dict[str, pd.DataFrame]], vars: dict[str, Union[str, list[str]]]
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Args:
            data: Train, validation and test data dictionary. Further divided in static, dynamic, and outcome.
            vars: Variables for static, dynamic, outcome.
        Returns:
            Preprocessed data.
        """
        logging.info("Preprocessor static features.")
        data = {step: self._process_dynamic_data(data[step], vars) for step in data}

        dyn_rec = Recipe(
            data[DataSplit.train][DataSegment.dynamic], [], vars[DataSegment.dynamic], vars["GROUP"], vars["SEQUENCE"]
        )
        if self.scaling:
            dyn_rec.add_step(StepScale())
        data = apply_recipe_to_splits(dyn_rec, data, DataSegment.dynamic, self.save_cache, self.load_cache)

        if not (isinstance(vars["GROUP"], str) and isinstance(vars["SEQUENCE"], str)):
            raise TypeError(
                f'Expected keys "GROUP" and "SEQUENCE" to be of type str, got {type(vars["GROUP"])} and {type(vars["SEQUENCE"])} instead.'
            )
        selected_vars: list[str] = [str(item) for item in vars[DataSegment.dynamic]] + [vars["GROUP"], vars["SEQUENCE"]]
        data[DataSplit.train][DataSegment.features] = data[DataSplit.train].pop(DataSegment.dynamic).loc[:, selected_vars]
        data[DataSplit.val][DataSegment.features] = data[DataSplit.val].pop(DataSegment.dynamic).loc[:, selected_vars]
        data[DataSplit.test][DataSegment.features] = data[DataSplit.test].pop(DataSegment.dynamic).loc[:, selected_vars]
        return data

    def to_cache_string(self):
        return super().to_cache_string() + f"_imputation_{self.use_static_features}_{self.scaling}"

    def _process_dynamic_data(self, data, vars):
        if self.filter_missing_values:
            rows_to_remove = data[DataSegment.dynamic][vars[DataSegment.dynamic]].isna().sum(axis=1) != 0
            ids_to_remove = data[DataSegment.dynamic].loc[rows_to_remove][vars["GROUP"]].unique()
            data = {table_name: table.loc[~table[vars["GROUP"]].isin(ids_to_remove)] for table_name, table in data.items()}
            logging.info(f"Removed {len(ids_to_remove)} stays with missing values.")
        return data


def apply_recipe_to_splits(
    recipe: Recipe,
    data: dict[str, dict[str, Union[pd.DataFrame, pl.DataFrame]]],
    type: str,
    save_cache: Optional[Union[str, Path]] = None,
    load_cache: Optional[Union[str, Path]] = None,
) -> dict[str, dict[str, Union[pd.DataFrame, pl.DataFrame]]]:
    """
    Fits and transforms the training features, then transforms the validation and test features with the recipe.
     Works with both Polars and Pandas versions of recipys.

    Args:
        load_cache: Load recipe from cache, for e.g. transfer learning.
        save_cache: Save recipe to cache, for e.g. transfer learning.
        recipe: Object containing info about the features and steps.
        data: Dict containing 'train', 'val', and 'test' and types of features per split.
        type: Whether to apply recipe to dynamic features, static features or outcomes.

    Returns:
        Transformed features divided into 'train', 'val', and 'test'.
    """

    if isinstance(load_cache, (str, Path)):
        load_cache = Path(load_cache)
        # Load existing recipe
        recipe = restore_recipe(load_cache)
        data[DataSplit.train][type] = recipe.bake(data[DataSplit.train][type])
    elif isinstance(save_cache, (str, Path)):
        save_cache = Path(save_cache)
        # Save prepped recipe
        data[DataSplit.train][type] = recipe.prep()
        cache_recipe(recipe, save_cache)
    else:
        # No saving or loading of existing cache
        data[DataSplit.train][type] = recipe.prep()

    data[DataSplit.val][type] = recipe.bake(data[DataSplit.val][type])
    data[DataSplit.test][type] = recipe.bake(data[DataSplit.test][type])
    return data


def cache_recipe(recipe: Recipe, cache_file: Path) -> None:
    """Cache recipe to make it available for e.g. transfer learning."""
    recipe_cache = copy.deepcopy(recipe)
    recipe_cache.cache()
    if not (cache_file.parent).exists():
        (cache_file.parent).mkdir(parents=True, exist_ok=True)
    cache_file.touch()
    with open(cache_file, "wb") as f:
        pickle.dump(recipe_cache, f, pickle.HIGHEST_PROTOCOL)
    logging.info(f"Cached recipe in {cache_file}.")


def restore_recipe(cache_file: Path) -> Recipe:
    """Restore recipe from cache to use for e.g. transfer learning."""
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            logging.info(f"Loading cached recipe from {cache_file}.")
            recipe = pickle.load(f)
            return recipe
    else:
        raise FileNotFoundError(f"Cache file {cache_file} not found.")
