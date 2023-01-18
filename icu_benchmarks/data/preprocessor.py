import logging

import gin
import pandas as pd
from recipys.recipe import Recipe
from recipys.selector import all_numeric_predictors, has_type, all_of
from recipys.step import StepScale, StepImputeFill, StepSklearn, StepHistorical, Accumulator
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import LabelEncoder

import abc


class Preprocessor:
    def __init__(self):
        pass

    @abc.abstractmethod
    def apply(self, data, vars):
        return data

    @abc.abstractmethod
    def to_cache_string(self):
        return f"{self.__class__.__name__}"


# TODO: static features or not.
# TODO: concatenate static features to dynamic features here.

@gin.configurable("base_preprocessor")
class DefaultPreprocessor(Preprocessor):
    def __init__(
        self,
        generate_features: bool = True,
        scaling: bool = True,
        use_static_features: bool = False,
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

    # TODO: pass data and vars as arguments
    def apply(self, data, vars):
        """
        Args:
            data: Train, validation and test data dictionary. Further divided in static, dynamic, and outcome.
            vars: Variables for static, dynamic, outcome.
        Returns:
            Preprocessed data.
        """
        logging.info("Preprocessor static features.")
        if self.use_static_features:
            data = self.process_static(data, vars)
        data = self.process_dynamic(data, vars)
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

    def process_dynamic(self, data, vars):
        dyn_rec = Recipe(data["train"]["DYNAMIC"], [], vars["DYNAMIC"], vars["GROUP"], vars["SEQUENCE"])
        if self.scaling:
            dyn_rec.add_step(StepScale())
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
        return super().to_cache_string() + f"_{self.generate_features}_{self.scaling}"

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
