import torch
import logging

import gin
import pandas as pd
from recipys.recipe import Recipe
from recipys.selector import all_numeric_predictors, all_outcomes, has_type, all_of
from recipys.step import StepScale, StepImputeFill, StepSklearn, StepHistorical, Accumulator, StepImputeModel
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from icu_benchmarks.wandb_utils import update_wandb_config
from icu_benchmarks.data.loader import ImputationPredictionDataset
from .constants import DataSplit as Split, DataSegment as Segment
import abc


class Preprocessor:
    @abc.abstractmethod
    def apply(self, data, vars):
        return data

    @abc.abstractmethod
    def to_cache_string(self):
        return f"{self.__class__.__name__}"


@gin.configurable("base_classification_preprocessor")
class DefaultClassificationPreprocessor(Preprocessor):
    def __init__(
        self, generate_features: bool = True, scaling: bool = True, use_static_features: bool = True, vars: dict = None
    ):
        """
        Args:
            generate_features: Generate features for dynamic data.
            scaling: Scaling of dynamic and static data.
            use_static_features: Use static features.
        Returns:
            Preprocessed data.
        """
        self.generate_features = generate_features
        self.scaling = scaling
        self.use_static_features = use_static_features
        self.imputation_model = None
        self.vars = vars

    def set_imputation_model(self, imputation_model):
        self.imputation_model = imputation_model
        if self.imputation_model is not None:
            update_wandb_config({"imputation_model": self.imputation_model.__class__.__name__})

    def apply(self, data, vars):
        """
        Args:
            data: Train, validation and test data dictionary. Further divided in static, dynamic, and outcome.
            vars: Variables for static, dynamic, outcome.
        Returns:
            Preprocessed data.
        """
        logging.info("Preprocessing dynamic features.")
        data = self.process_dynamic(data, vars)
        if self.use_static_features:
            logging.info("Preprocessing static features.")
            data = self.process_static(data, vars)

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
        return data

    def process_static(self, data, vars):
        sta_rec = Recipe(data[Split.train][Segment.static], [], vars[Segment.static])
        if self.scaling:
            sta_rec.add_step(StepScale())

        sta_rec.add_step(StepImputeFill(sel=all_numeric_predictors(), value=0))
        sta_rec.add_step(StepSklearn(SimpleImputer(missing_values=None, strategy="most_frequent"), sel=has_type("object")))
        sta_rec.add_step(StepSklearn(LabelEncoder(), sel=has_type("object"), columnwise=True))

        data = self.apply_recipe_to_Splits(sta_rec, data, Segment.static)

        return data

    def model_impute(self, data, group=None):
        dataset = ImputationPredictionDataset(data, group, self.imputation_model.trained_columns)
        input_data = torch.cat([data_point.unsqueeze(0) for data_point in dataset], dim=0)
        self.imputation_model.eval()
        with torch.no_grad():
            logging.info("predicting...")
            imputation = self.imputation_model.predict(input_data)
            logging.info("done predicting")
        assert imputation.isnan().sum() == 0
        data = data.copy()
        data.loc[:, self.imputation_model.trained_columns] = imputation.flatten(end_dim=1).to("cpu")
        if group is not None:
            data.drop(columns=group, inplace=True)
        return data

    def process_dynamic(self, data, vars):
        dyn_rec = Recipe(data[Split.train][Segment.dynamic], [], vars[Segment.dynamic], vars["GROUP"], vars["SEQUENCE"])
        if self.scaling:
            dyn_rec.add_step(StepScale())
        if self.imputation_model is not None:
            dyn_rec.add_step(StepImputeModel(model=self.model_impute, sel=all_of(vars[Segment.dynamic])))
        dyn_rec.add_step(StepSklearn(MissingIndicator(), sel=all_of(vars[Segment.dynamic]), in_place=False))
        dyn_rec.add_step(StepImputeFill(method="ffill"))
        dyn_rec.add_step(StepImputeFill(value=0))
        if self.generate_features:
            dyn_rec = self.dynamic_feature_generation(dyn_rec, all_of(vars[Segment.dynamic]))
        data = self.apply_recipe_to_Splits(dyn_rec, data, Segment.dynamic)
        # data[Split.train][Segment.dynamic] = data[Split.train][Segment.dynamic].loc[
        #     :, vars[Segment.dynamic] + [vars["GROUP"], vars["SEQUENCE"]]
        # ]
        # data[Split.val][Segment.dynamic] = data[Split.val][Segment.dynamic].loc[
        #     :, vars[Segment.dynamic] + [vars["GROUP"], vars["SEQUENCE"]]
        # ]
        # data[Split.test][Segment.dynamic] = data[Split.test][Segment.dynamic].loc[
        #     :, vars[Segment.dynamic] + [vars["GROUP"], vars["SEQUENCE"]]
        # ]
        return data

    def dynamic_feature_generation(self, data, dynamic_vars):
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

    @staticmethod
    def apply_recipe_to_Splits(recipe: Recipe, data: dict[dict[pd.DataFrame]], type: str) -> dict[dict[pd.DataFrame]]:
        """Fits and transforms the training features, then transforms the validation and test features with the recipe.

        Args:
            recipe: Object containing info about the features and steps.
            data: Dict containing 'train', 'val', and 'test' and types of features per Split.
            type: Whether to apply recipe to dynamic features, static features or outcomes.

        Returns:
            Transformed features divided into 'train', 'val', and 'test'.
        """
        data[Split.train][type] = recipe.prep()
        data[Split.val][type] = recipe.bake(data[Split.val][type])
        data[Split.test][type] = recipe.bake(data[Split.test][type])
        return data

@gin.configurable("base_regression_preprocessor")
class DefaultRegressionPreprocessor(DefaultClassificationPreprocessor):
    # Override base classification preprocessor
    def apply(self, data, vars):
        """
        Args:
            data: Train, validation and test data dictionary. Further divided in static, dynamic, and outcome.
            vars: Variables for static, dynamic, outcome.
        Returns:
            Preprocessed data.
        """
        logging.info("Preprocessing dynamic features.")

        data = self.process_dynamic(data, vars)

        for split in [Split.train, Split.val, Split.test]:
            data = self.process_outcome(data, vars, split)

        if self.use_static_features:
            logging.info("Preprocessing static features.")
            data = self.process_static(data, vars)

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
        return data

    def process_outcome(self, data, vars, split):
        logging.debug(f"Processing {split} outcome values.")
        outcome_rec = Recipe(data[split][Segment.outcome], vars["LABEL"], [], vars["GROUP"])
        # if(vars["SEQUENCE"] in data[split][Segment.outcome].columns):
        #     # Seq2Seq regression
        #     outcome_rec = Recipe(data[split][Segment.outcome], vars["LABEL"], [], vars["GROUP"], vars["SEQUENCE"])
        # else:
        #     # Regression
        #     outcome_rec = Recipe(data[split][Segment.outcome], vars["LABEL"], [], vars["GROUP"])
        outcome_rec.add_step(StepSklearn(sklearn_transformer=MinMaxScaler(), sel=all_outcomes()))
        outcome_rec.prep()
        data[split][Segment.outcome] = outcome_rec.bake()
        return data
@gin.configurable("base_imputation_preprocessor")
class DefaultImputationPreprocessor(Preprocessor):
    def __init__(
        self,
        scaling: bool = True,
        use_static_features: bool = True,
        filter_missing_values: bool = True,
        vars: dict = None,
    ):
        """Preprocesses data for imputation.

        Args:
            scaling (bool, optional): If the values in each column should be normalized. Defaults to True.
            use_static_features (bool, optional): If static features should be included in the dataset. Defaults to True.
            vars (dict, optional): Dict containing column names in the data. Defaults to None.
        """
        self.scaling = scaling
        self.use_static_features = use_static_features
        self.vars = vars
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
        data = {step: self.process_dynamic_data(data[step], vars) for step in data}

        dyn_rec = Recipe(data[Split.train][Segment.dynamic], [], vars[Segment.dynamic], vars["GROUP"], vars["SEQUENCE"])
        if self.scaling:
            dyn_rec.add_step(StepScale())
        data = self.apply_recipe_to_splits(dyn_rec, data, Segment.dynamic)

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

    @staticmethod
    def apply_recipe_to_splits(recipe: Recipe, data: dict[dict[pd.DataFrame]], type: str) -> dict[dict[pd.DataFrame]]:
        """Fits and transforms the training features, then transforms the validation and test features with the recipe.

        Args:
            recipe: Object containing info about the features and steps.
            data: Dict containing 'train', 'val', and 'test' and types of features per split.
            type: Whether to apply recipe to dynamic features, static features or outcomes.

        Returns:
            Transformed features divided into 'train', 'val', and 'test'.
        """
        data[Split.train][type] = recipe.prep()
        data[Split.val][type] = recipe.bake(data[Split.val][type])
        data[Split.test][type] = recipe.bake(data[Split.test][type])
        return data

    def process_dynamic_data(self, data, vars):
        if self.filter_missing_values:
            rows_to_remove = data[Segment.dynamic][vars[Segment.dynamic]].isna().sum(axis=1) != 0
            ids_to_remove = data[Segment.dynamic].loc[rows_to_remove][vars["GROUP"]].unique()
            data = {table_name: table.loc[~table[vars["GROUP"]].isin(ids_to_remove)] for table_name, table in data.items()}
            logging.info(f"Removed {len(ids_to_remove)} stays with missing values.")
        return data
