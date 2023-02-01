from tqdm import tqdm
import torch
import logging

import gin
import pandas as pd
import wandb
from recipys.recipe import Recipe
from recipys.selector import all_numeric_predictors, has_type, all_of
from recipys.step import StepScale, StepImputeFill, StepSklearn, StepHistorical, Accumulator, StepImputeModel
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import LabelEncoder

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


@gin.configurable("base_imputation_preprocessor")
class DefaultImputationPreprocessor(Preprocessor):
    def __init__(
        self,
        scaling: bool = True,
        use_static_features: bool = True,
        window_size: int = 25,
        window_stride: int = 1,
        vars: dict = None,
    ):
        """
        Args:
            scaling: Scaling of dynamic and static data.
            use_static_features: Use static features.
        Returns:
            Preprocessed data.
        """
        self.scaling = scaling
        self.use_static_features = use_static_features
        self.window_size = window_size
        self.window_stride = window_stride
        self.vars = vars

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
        return super().to_cache_string() + f"_imputation_{self.use_static_features}_{self.scaling}_{self.window_size}"

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
        maxlen = data[Segment.dynamic].groupby([vars["GROUP"]]).size().max()
        if self.window_size >= maxlen:
            rows_to_remove = data[Segment.dynamic][vars[Segment.dynamic]].isna().sum(axis=1) != 0
            ids_to_remove = data[Segment.dynamic].loc[rows_to_remove][vars["GROUP"]].unique()
            data = {table_name: table.loc[~table[vars["GROUP"]].isin(ids_to_remove)] for table_name, table in data.items()}
            logging.info(f"Removed {len(ids_to_remove)} stays with missing values.")
        else:
            # collect all stays with window size consecutive present values
            data[Segment.dynamic]["OLD_GROUP"] = data[Segment.dynamic][vars["GROUP"]]
            new_data = {name: pd.DataFrame(columns=table.columns) for name, table in data.items()}
            grouped = data[Segment.dynamic].groupby([vars["GROUP"]])
            slice_counter = 0

            for group_name, group in tqdm(grouped):
                for i in range(0, len(group) - self.window_size + 1, self.window_stride):
                    if group.iloc[i : i + self.window_size][vars[Segment.dynamic]].isna().sum().sum() == 0:
                        slice = group.iloc[i : i + self.window_size].copy()
                        slice.loc[:, vars["GROUP"]] = slice_counter
                        # use pandas.concat
                        new_data[Segment.dynamic] = pd.concat([new_data[Segment.dynamic], slice])
                        for table_name, table in data.items():
                            if table_name != Segment.dynamic:
                                new_slice_data = table.loc[table[vars["GROUP"]] == group_name].copy()
                                new_slice_data.loc[:, vars["GROUP"]] = slice_counter
                                new_data[table_name] = pd.concat([new_data[table_name], new_slice_data])
                        slice_counter += 1
            data = new_data
            logging.info(
                f"Generated {slice_counter} slices with {self.window_size} consecutive present values from {len(grouped)} records."
            )
        return data


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
        if self.imputation_model is not None and wandb.run is not None:
            wandb.run.summary["imputation_model"] = self.imputation_model.__class__.__name__

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
        dataset = ImputationPredictionDataset(data, group)
        data = torch.cat([data_point.unsqueeze(0) for data_point in dataset], dim=0)
        self.imputation_model.eval()
        with torch.no_grad():
            logging.info("predicting...")
            imputation = self.imputation_model.predict(data)
            logging.info("done predicting")
        assert imputation.isnan().sum() == 0
        return imputation.flatten(end_dim=1).to("cpu")

    def process_dynamic(self, data, vars):
        dyn_rec = Recipe(data[Split.train][Segment.dynamic], [], vars[Segment.dynamic], vars["GROUP"], vars["SEQUENCE"])
        if self.scaling:
            dyn_rec.add_step(StepScale())
        if self.imputation_model is not None:
            dyn_rec.add_step(StepImputeModel(model=self.model_impute, sel=all_of(vars[Segment.dynamic])))
        else:
            dyn_rec.add_step(StepSklearn(MissingIndicator(), sel=all_of(vars[Segment.dynamic]), in_place=False))
            dyn_rec.add_step(StepImputeFill(method="ffill"))
            dyn_rec.add_step(StepImputeFill(value=0))
        if self.generate_features:
            dyn_rec = self.dynamic_feature_generation(dyn_rec, all_of(vars[Segment.dynamic]))
        data = self.apply_recipe_to_Splits(dyn_rec, data, Segment.dynamic)
        data[Split.train][Segment.dynamic] = data[Split.train][Segment.dynamic].loc[
            :, vars[Segment.dynamic] + [vars["GROUP"], vars["SEQUENCE"]]
        ]
        data[Split.val][Segment.dynamic] = data[Split.val][Segment.dynamic].loc[
            :, vars[Segment.dynamic] + [vars["GROUP"], vars["SEQUENCE"]]
        ]
        data[Split.test][Segment.dynamic] = data[Split.test][Segment.dynamic].loc[
            :, vars[Segment.dynamic] + [vars["GROUP"], vars["SEQUENCE"]]
        ]
        return data

    def dynamic_feature_generation(self, data, dynamic_vars):
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
