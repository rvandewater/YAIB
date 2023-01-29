import torch
import logging

import gin
import pandas as pd
from recipys.recipe import Recipe
from recipys.selector import all_numeric_predictors, has_type, all_of
from recipys.step import StepScale, StepImputeFill, StepSklearn, StepHistorical, Accumulator, StepImputeModel
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import LabelEncoder

from icu_benchmarks.data.loader import ImputationPredictionDataset
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
    def __init__(self, scaling: bool = True, use_static_features: bool = True, window_size: int = 25, window_stride: int = 1):
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
    
    def apply(self, data, vars):
        """
        Args:
            data: Train, validation and test data dictionary. Further divided in static, dynamic, and outcome.
            vars: Variables for static, dynamic, outcome.
        Returns:
            Preprocessed data.
        """
        logging.info("Preprocessor static features.")
        
        maxlen = data["DYNAMIC"].groupby([vars["GROUP"]]).size().max()
        if self.window_size >= maxlen:
            rows_to_remove = data["DYNAMIC"][vars["DYNAMIC"]].isna().sum(axis=1) != 0
            ids_to_remove = data["DYNAMIC"].loc[rows_to_remove][vars["GROUP"]].unique()
            data = {table_name: table.loc[~table[vars["GROUP"]].isin(ids_to_remove)] for table_name, table in data.items()}
            logging.info(f"Removed {len(ids_to_remove)} stays with missing values.")
        else:
            # collect all stays with window size consecutive present values
            data["DYNAMIC"]["OLD_GROUP"] = data["DYNAMIC"][vars["GROUP"]]
            new_data = {name: pd.DataFrame(columns=table.columns) for name, table in data.items()}
            grouped = data["DYNAMIC"].groupby([vars["GROUP"]])
            slice_counter = 0
            
            for group_name, group in grouped:
                for i in range(0, len(group) - self.window_size + 1, self.window_stride):
                    if group.iloc[i:i+self.window_size][vars["DYNAMIC"]].isna().sum().sum() == 0:
                        slice = group.iloc[i:i+self.window_size]
                        slice[vars["GROUP"]] = slice_counter
                        new_data["DYNAMIC"] = new_data["DYNAMIC"].append(slice)
                        for table_name, table in data.items():
                            if table_name != "DYNAMIC":
                                new_slice_data = table.loc[table[vars["GROUP"]] == group_name]
                                new_slice_data[vars["GROUP"]] = slice_counter
                                new_data[table_name] = new_data[table_name].append(new_slice_data)
                        slice_counter += 1
            data = new_data
            logging.info(f"Generated {slice_counter} slices with {self.window_size} consecutive present values from {len(grouped)} records.")
        dyn_rec = Recipe(data["train"]["DYNAMIC"], [], vars["DYNAMIC"], vars["GROUP"], vars["SEQUENCE"])
        
        if self.scaling:
            dyn_rec.add_step(StepScale())
        data = self.apply_recipe_to_splits(dyn_rec, data, "DYNAMIC")
        data["train"]["FEATURES"] = data["train"].pop("DYNAMIC")
        data["val"]["FEATURES"] = data["val"].pop("DYNAMIC")
        data["test"]["FEATURES"] = data["test"].pop("DYNAMIC")
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
        data["train"][type] = recipe.prep()
        data["val"][type] = recipe.bake(data["val"][type])
        data["test"][type] = recipe.bake(data["test"][type])
        return data


@gin.configurable("base_classification_preprocessor")
class DefaultClassificationPreprocessor(Preprocessor):
    def __init__(
        self,
        generate_features: bool = True,
        scaling: bool = True,
        use_static_features: bool = True,
    ):
        """
        Args:
            generate_features: Generate features for static data.
            scaling: Scaling of dynamic and static data.
            use_static_features: Use static features.
        Returns:
            Preprocessed data.
        """
        self.generate_features = generate_features
        self.scaling = scaling
        self.use_static_features = use_static_features
        self.imputation_model = None
    
    def set_imputation_model(self, imputation_model):
        self.imputation_model = imputation_model

    def apply(self, data, vars):
        """
        Args:
            data: Train, validation and test data dictionary. Further divided in static, dynamic, and outcome.
            vars: Variables for static, dynamic, outcome.
        Returns:
            Preprocessed data.
        """
        logging.info("Preprocessor static features.")
        data = self.process_dynamic(data, vars)
        if self.use_static_features:
            data = self.process_static(data, vars)
            data["train"]["STATIC"] = data["train"]["STATIC"].set_index(vars["GROUP"])
            data["val"]["STATIC"] = data["val"]["STATIC"].set_index(vars["GROUP"])
            data["test"]["STATIC"] = data["test"]["STATIC"].set_index(vars["GROUP"])

            data["train"]["DYNAMIC"] = data["train"]["DYNAMIC"].join(data["train"]["STATIC"], on=vars["GROUP"])
            data["val"]["DYNAMIC"] = data["val"]["DYNAMIC"].join(data["val"]["STATIC"], on=vars["GROUP"])
            data["test"]["DYNAMIC"] = data["test"]["DYNAMIC"].join(data["test"]["STATIC"], on=vars["GROUP"])
        data["train"]["FEATURES"] = data["train"].pop("DYNAMIC")
        data["val"]["FEATURES"] = data["val"].pop("DYNAMIC")
        data["test"]["FEATURES"] = data["test"].pop("DYNAMIC")
        return data

    def process_static(self, data, vars):
        sta_rec = Recipe(data["train"]["STATIC"], [], vars["STATIC"])
        if self.scaling:
            sta_rec.add_step(StepScale())

        sta_rec.add_step(StepImputeFill(sel=all_numeric_predictors(), value=0))
        sta_rec.add_step(StepSklearn(SimpleImputer(missing_values=None, strategy="most_frequent"), sel=has_type("object")))
        sta_rec.add_step(StepSklearn(LabelEncoder(), sel=has_type("object"), columnwise=True))

        data = self.apply_recipe_to_splits(sta_rec, data, "STATIC")

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
        dyn_rec = Recipe(data["train"]["DYNAMIC"], [], vars["DYNAMIC"], vars["GROUP"], vars["SEQUENCE"])
        if self.scaling:
            dyn_rec.add_step(StepScale())
        if self.imputation_model is not None:
            dyn_rec.add_step(StepImputeModel(self.model_impute, sel=all_of(vars["DYNAMIC"])))
        else:
            dyn_rec.add_step(StepSklearn(MissingIndicator(), sel=all_of(vars["DYNAMIC"]), in_place=False))
            dyn_rec.add_step(StepImputeFill(method="ffill"))
            dyn_rec.add_step(StepImputeFill(value=0))
        if self.generate_features:
            dyn_rec = self.dynamic_feature_generation(dyn_rec, all_of(vars["DYNAMIC"]))
        data = self.apply_recipe_to_splits(dyn_rec, data, "DYNAMIC")
        return data

    def dynamic_feature_generation(self, data, dynamic_vars):
        data.add_step(StepHistorical(sel=dynamic_vars, fun=Accumulator.MIN, suffix="min_hist"))
        data.add_step(StepHistorical(sel=dynamic_vars, fun=Accumulator.MAX, suffix="max_hist"))
        data.add_step(StepHistorical(sel=dynamic_vars, fun=Accumulator.COUNT, suffix="count_hist"))
        data.add_step(StepHistorical(sel=dynamic_vars, fun=Accumulator.MEAN, suffix="mean_hist"))
        return data

    def to_cache_string(self):
        return super().to_cache_string() + f"_classification_{self.generate_features}_{self.scaling}_{self.imputation_model.__class__.__name__}"

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
        data["train"][type] = recipe.prep()
        data["val"][type] = recipe.bake(data["val"][type])
        data["test"][type] = recipe.bake(data["test"][type])
        return data
