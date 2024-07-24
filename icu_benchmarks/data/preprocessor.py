import copy
import pickle

import torch
import logging

import gin
import pandas as pd
import polars.selectors as cs
import polars as pl
from recipys.recipe import Recipe
from recipys.selector import all_numeric_predictors, all_outcomes, has_type, all_of
from recipys.step import (
    StepScale,
    # StepImputeFastForwardFill,
    # StepImputeFastZeroFill,
    StepImputeFill,
    StepSklearn,
    StepHistorical,
    Accumulator,
    StepImputeModel,
)

from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import LabelEncoder, FunctionTransformer, MinMaxScaler

from icu_benchmarks.wandb_utils import update_wandb_config
from icu_benchmarks.data.loader import ImputationPredictionDataset
from .constants import DataSplit as Split, DataSegment as Segment
import abc


class Preprocessor:
    @abc.abstractmethod
    def apply(self, data, vars, save_cache=False, load_cache=None):
        return data

    @abc.abstractmethod
    def to_cache_string(self):
        return f"{self.__class__.__name__}"

    def set_imputation_model(self, imputation_model):
        self.imputation_model = imputation_model
        if self.imputation_model is not None:
            update_wandb_config({"imputation_model": self.imputation_model.__class__.__name__})

@gin.configurable("base_classification_preprocessor")
class PolarsClassificationPreprocessor(Preprocessor):
    def __init__(
        self,
        generate_features: bool = True,
        scaling: bool = True,
        use_static_features: bool = True,
        save_cache=None,
        load_cache=None,
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

    def apply(self, data, vars) -> dict[dict[pl.DataFrame]]:
        """
        Args:
            data: Train, validation and test data dictionary. Further divided in static, dynamic, and outcome.
            vars: Variables for static, dynamic, outcome.
        Returns:
            Preprocessed data.
        """
        logging.info("Preprocessing dynamic features.")

        data = self._process_dynamic(data, vars)
        if self.use_static_features and len(vars[Segment.static]) > 0:
            logging.info("Preprocessing static features.")
            data = self._process_static(data, vars)

            # Set index to grouping variable
            data[Split.train][Segment.static] = data[Split.train][Segment.static]#.set_index(vars["GROUP"])
            data[Split.val][Segment.static] = data[Split.val][Segment.static]#.set_index(vars["GROUP"])
            data[Split.test][Segment.static] = data[Split.test][Segment.static]#.set_index(vars["GROUP"])

            # Join static and dynamic data.
            data[Split.train][Segment.dynamic] = data[Split.train][Segment.dynamic].join(
                data[Split.train][Segment.static], on=vars["GROUP"]
            )
            data[Split.val][Segment.dynamic] = data[Split.val][Segment.dynamic].join(
                data[Split.val][Segment.static], on=vars["GROUP"]
            )
            data[Split.test][Segment.dynamic] = data[Split.test][Segment.dynamic].join(
                data[Split.test][Segment.static], on=vars["GROUP"]
            )

            # Remove static features from splits
            data[Split.train][Segment.features] = data[Split.train].pop(Segment.static)
            data[Split.val][Segment.features] = data[Split.val].pop(Segment.static)
            data[Split.test][Segment.features] = data[Split.test].pop(Segment.static)

        # Create feature splits
        data[Split.train][Segment.features] = data[Split.train].pop(Segment.dynamic)
        data[Split.val][Segment.features] = data[Split.val].pop(Segment.dynamic)
        data[Split.test][Segment.features] = data[Split.test].pop(Segment.dynamic)

        logging.debug("Data head")
        logging.debug(data[Split.train][Segment.features].head())
        logging.info(f"Generate features: {self.generate_features}")
        return data

    def _process_static(self, data, vars):
        sta_rec = Recipe(data[Split.train][Segment.static], [], vars[Segment.static])
        sta_rec.add_step(StepSklearn(MissingIndicator(features="all"), sel=all_of(vars[Segment.static]), in_place=False))
        if self.scaling:
            sta_rec.add_step(StepScale())
        sta_rec.add_step(StepImputeFill(sel=all_numeric_predictors(),strategy="zero"))
        # sta_rec.add_step(StepImputeFastZeroFill(sel=all_numeric_predictors()))
        # if len(data[Split.train][Segment.static].select_dtypes(include=["object"]).columns) > 0:
        types = ["String", "Object", "Categorical"]
        sel = has_type(types)
        if(len(sel(sta_rec.data))>0):
        # if len(data[Split.train][Segment.static].select(cs.by_dtype(types)).columns) > 0:
            sta_rec.add_step(StepSklearn(SimpleImputer(missing_values=None, strategy="most_frequent"), sel=has_type(types)))
            sta_rec.add_step(StepSklearn(LabelEncoder(), sel=has_type(types), columnwise=True))

        data = apply_recipe_to_splits(sta_rec, data, Segment.static, self.save_cache, self.load_cache)

        return data

    def _model_impute(self, data, group=None):
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

    def _process_dynamic(self, data, vars):
        dyn_rec = Recipe(data[Split.train][Segment.dynamic], [], vars[Segment.dynamic], vars["GROUP"], vars["SEQUENCE"])
        if self.scaling:
            dyn_rec.add_step(StepScale())
        if self.imputation_model is not None:
            dyn_rec.add_step(StepImputeModel(model=self.model_impute, sel=all_of(vars[Segment.dynamic])))
        dyn_rec.add_step(StepSklearn(MissingIndicator(features="all"), sel=all_of(vars[Segment.dynamic]), in_place=False))
        # dyn_rec.add_step(StepImputeFastForwardFill())
        dyn_rec.add_step(StepImputeFill(strategy="forward"))
        # dyn_rec.add_step(StepImputeFastZeroFill())
        dyn_rec.add_step(StepImputeFill(strategy="zero"))
        if self.generate_features:
            dyn_rec = self._dynamic_feature_generation(dyn_rec, all_of(vars[Segment.dynamic]))
        data = apply_recipe_to_splits(dyn_rec, data, Segment.dynamic, self.save_cache, self.load_cache)
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
        generate_features: bool = True,
        scaling: bool = True,
        use_static_features: bool = True,
        outcome_max=None,
        outcome_min=None,
        save_cache=None,
        load_cache=None,
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
        super().__init__(generate_features, scaling, use_static_features, save_cache, load_cache)
        self.outcome_max = outcome_max
        self.outcome_min = outcome_min

    def apply(self, data, vars):
        """
        Args:
            data: Train, validation and test data dictionary. Further divided in static, dynamic, and outcome.
            vars: Variables for static, dynamic, outcome.
        Returns:
            Preprocessed data.
        """
        for split in [Split.train, Split.val, Split.test]:
            data = self._process_outcome(data, vars, split)

        data = super().apply(data, vars)
        return data

    def _process_outcome(self, data, vars, split):
        logging.debug(f"Processing {split} outcome values.")
        outcome_rec = Recipe(data[split][Segment.outcome], vars["LABEL"], [], vars["GROUP"])
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
        data[split][Segment.outcome] = outcome_rec.bake()
        return data

@gin.configurable("pandas_classification_preprocessor")
class PandasClassificationPreprocessor(Preprocessor):
    def __init__(
        self,
        generate_features: bool = True,
        scaling: bool = True,
        use_static_features: bool = True,
        save_cache=None,
        load_cache=None,
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

    def apply(self, data, vars) -> dict[dict[pd.DataFrame]]:
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
            data[Split.train][Segment.static] = data[Split.train][Segment.static].set_index(vars["GROUP"])
            data[Split.val][Segment.static] = data[Split.val][Segment.static].set_index(vars["GROUP"])
            data[Split.test][Segment.static] = data[Split.test][Segment.static].set_index(vars["GROUP"])

            # Join static and dynamic data.
            data[Split.train][Segment.dynamic] = data[Split.train][Segment.dynamic].join(
                data[Split.train][Segment.static], on=vars["GROUP"]
            )
            data[Split.val][Segment.dynamic] = data[Split.val][Segment.dynamic].join(
                data[Split.val][Segment.static], on=vars["GROUP"]
            )
            data[Split.test][Segment.dynamic] = data[Split.test][Segment.dynamic].join(
                data[Split.test][Segment.static], on=vars["GROUP"]
            )

            # Remove static features from splits
            data[Split.train][Segment.features] = data[Split.train].pop(Segment.static)
            data[Split.val][Segment.features] = data[Split.val].pop(Segment.static)
            data[Split.test][Segment.features] = data[Split.test].pop(Segment.static)

        # Create feature splits
        data[Split.train][Segment.features] = data[Split.train].pop(Segment.dynamic)
        data[Split.val][Segment.features] = data[Split.val].pop(Segment.dynamic)
        data[Split.test][Segment.features] = data[Split.test].pop(Segment.dynamic)

        logging.debug("Data head")
        logging.debug(data[Split.train][Segment.features].head())
        logging.info(f"Generate features: {self.generate_features}")
        return data

    def _process_static(self, data, vars):
        sta_rec = Recipe(data[Split.train][Segment.static], [], vars[Segment.static])
        if self.scaling:
            sta_rec.add_step(StepScale())

        sta_rec.add_step(StepImputeFastZeroFill(sel=all_numeric_predictors()))
        if len(data[Split.train][Segment.static].select_dtypes(include=["object"]).columns) > 0:
            sta_rec.add_step(StepSklearn(SimpleImputer(missing_values=None, strategy="most_frequent"), sel=has_type("object")))
            sta_rec.add_step(StepSklearn(LabelEncoder(), sel=has_type("object"), columnwise=True))

        data = apply_recipe_to_splits(sta_rec, data, Segment.static, self.save_cache, self.load_cache)

        return data

    def _model_impute(self, data, group=None):
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

    def _process_dynamic(self, data, vars):
        dyn_rec = Recipe(data[Split.train][Segment.dynamic], [], vars[Segment.dynamic], vars["GROUP"], vars["SEQUENCE"])
        if self.scaling:
            dyn_rec.add_step(StepScale())
        if self.imputation_model is not None:
            dyn_rec.add_step(StepImputeModel(model=self.model_impute, sel=all_of(vars[Segment.dynamic])))
        dyn_rec.add_step(StepSklearn(MissingIndicator(), sel=all_of(vars[Segment.dynamic]), in_place=False))
        dyn_rec.add_step(StepImputeFastForwardFill())
        dyn_rec.add_step(StepImputeFastZeroFill())
        if self.generate_features:
            dyn_rec = self._dynamic_feature_generation(dyn_rec, all_of(vars[Segment.dynamic]))
        data = apply_recipe_to_splits(dyn_rec, data, Segment.dynamic, self.save_cache, self.load_cache)
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
        save_cache=None,
        load_cache=None,
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
        super().__init__(generate_features, scaling, use_static_features, save_cache, load_cache)
        self.outcome_max = outcome_max
        self.outcome_min = outcome_min

    def apply(self, data, vars):
        """
        Args:
            data: Train, validation and test data dictionary. Further divided in static, dynamic, and outcome.
            vars: Variables for static, dynamic, outcome.
        Returns:
            Preprocessed data.
        """
        for split in [Split.train, Split.val, Split.test]:
            data = self._process_outcome(data, vars, split)

        data = super().apply(data, vars)
        return data

    def _process_outcome(self, data, vars, split):
        logging.debug(f"Processing {split} outcome values.")
        outcome_rec = Recipe(data[split][Segment.outcome], vars["LABEL"], [], vars["GROUP"])
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
        data[split][Segment.outcome] = outcome_rec.bake()
        return data


@gin.configurable("base_imputation_preprocessor")
class PandasImputationPreprocessor(Preprocessor):
    def __init__(
        self,
        scaling: bool = True,
        use_static_features: bool = True,
        filter_missing_values: bool = True,
    ):
        """Preprocesses data for imputation.

        Args:
            scaling (bool, optional): If the values in each column should be normalized. Defaults to True.
            use_static_features (bool, optional): If static features should be included in the dataset. Defaults to True.
        """
        self.scaling = scaling
        self.use_static_features = use_static_features
        self.filter_missing_values = filter_missing_values

    def apply(self, data, vars):
        """
        Args:
            data: Train, validation and test data dictionary. Further divided in static, dynamic, and outcome.
            vars: Variables for static, dynamic, outcome.
        Returns:
            Preprocessed data.
        """
        logging.info("Preprocessor static features.")
        data = {step: self._process_dynamic_data(data[step], vars) for step in data}

        dyn_rec = Recipe(data[Split.train][Segment.dynamic], [], vars[Segment.dynamic], vars["GROUP"], vars["SEQUENCE"])
        if self.scaling:
            dyn_rec.add_step(StepScale())
        data = apply_recipe_to_splits(dyn_rec, data, Segment.dynamic, self.save_cache, self.load_cache)

        data[Split.train][Segment.features] = (
            data[Split.train].pop(Segment.dynamic).loc[:, vars[Segment.dynamic] + [vars["GROUP"], vars["SEQUENCE"]]]
        )
        data[Split.val][Segment.features] = (
            data[Split.val].pop(Segment.dynamic).loc[:, vars[Segment.dynamic] + [vars["GROUP"], vars["SEQUENCE"]]]
        )
        data[Split.test][Segment.features] = (
            data[Split.test].pop(Segment.dynamic).loc[:, vars[Segment.dynamic] + [vars["GROUP"], vars["SEQUENCE"]]]
        )
        return data

    def to_cache_string(self):
        return super().to_cache_string() + f"_imputation_{self.use_static_features}_{self.scaling}"

    def _process_dynamic_data(self, data, vars):
        if self.filter_missing_values:
            rows_to_remove = data[Segment.dynamic][vars[Segment.dynamic]].isna().sum(axis=1) != 0
            ids_to_remove = data[Segment.dynamic].loc[rows_to_remove][vars["GROUP"]].unique()
            data = {table_name: table.loc[~table[vars["GROUP"]].isin(ids_to_remove)] for table_name, table in data.items()}
            logging.info(f"Removed {len(ids_to_remove)} stays with missing values.")
        return data


@staticmethod
def apply_recipe_to_splits(
    recipe: Recipe, data: dict[dict[pd.DataFrame]], type: str, save_cache=None, load_cache=None
) -> dict[dict[pd.DataFrame]]:
    """Fits and transforms the training features, then transforms the validation and test features with the recipe.
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

    if isinstance(load_cache, str):
        # Load existing recipe
        recipe = restore_recipe(load_cache)
        data[Split.train][type] = recipe.bake(data[Split.train][type])
    elif isinstance(save_cache, str):
        # Save prepped recipe
        data[Split.train][type] = recipe.prep()
        cache_recipe(recipe, save_cache)
    else:
        # No saving or loading of existing cache
        data[Split.train][type] = recipe.prep()

    data[Split.val][type] = recipe.bake(data[Split.val][type])
    data[Split.test][type] = recipe.bake(data[Split.test][type])
    return data


def cache_recipe(recipe: Recipe, cache_file: str) -> None:
    """Cache recipe to make it available for e.g. transfer learning."""
    recipe_cache = copy.deepcopy(recipe)
    recipe_cache.cache()
    if not (cache_file / "..").exists():
        (cache_file / "..").mkdir()
    cache_file.touch()
    with open(cache_file, "wb") as f:
        pickle.dump(recipe_cache, f, pickle.HIGHEST_PROTOCOL)
    logging.info(f"Cached recipe in {cache_file}.")


def restore_recipe(cache_file: str) -> Recipe:
    """Restore recipe from cache to use for e.g. transfer learning."""
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            logging.info(f"Loading cached recipe from {cache_file}.")
            recipe = pickle.load(f)
            return recipe
    else:
        raise FileNotFoundError(f"Cache file {cache_file} not found.")
