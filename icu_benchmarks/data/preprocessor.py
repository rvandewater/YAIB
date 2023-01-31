import logging

import gin
import pandas as pd
from recipys.recipe import Recipe
from recipys.selector import all_numeric_predictors, has_type, all_of
from recipys.step import StepScale, StepImputeFill, StepSklearn, StepHistorical, Accumulator
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import LabelEncoder
from .constants import DataSplit as Split, DataSegment as Segment
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


@gin.configurable("base_preprocessor")
class DefaultPreprocessor(Preprocessor):
    def __init__(
        self,
        generate_features: bool = True,
        scaling: bool = True,
        use_static_features: bool = True,
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
            data[Split.train][Segment.static] = data[Split.train][Segment.static].set_index(vars["GROUP"])
            data[Split.val][Segment.static] = data[Split.val][Segment.static].set_index(vars["GROUP"])
            data[Split.test][Segment.static] = data[Split.test][Segment.static].set_index(vars["GROUP"])

            data[Split.train][Segment.dynamic] = data[Split.train][Segment.dynamic].join(data[Segment.train][Segment.static], on=vars["GROUP"])
            data[Split.val][Segment.dynamic] = data[Split.val][Segment.dynamic].join(data[Split.val][Segment.static], on=vars["GROUP"])
            data[Split.test][Segment.dynamic] = data[Split.test][Segment.dynamic].join(data[Split.test][Segment.static], on=vars["GROUP"])
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

    def process_dynamic(self, data, vars):
        dyn_rec = Recipe(data[Split.train][Segment.dynamic], [], vars[Segment.dynamic], vars["GROUP"], vars["SEQUENCE"])
        if self.scaling:
            dyn_rec.add_step(StepScale())
        dyn_rec.add_step(StepSklearn(MissingIndicator(), sel=all_of(vars[Segment.dynamic]), in_place=False))
        dyn_rec.add_step(StepImputeFill(method="ffill"))
        dyn_rec.add_step(StepImputeFill(value=0))
        if self.generate_features:
            dyn_rec = self.dynamic_feature_generation(dyn_rec, all_of(vars[Segment.dynamic]))
        data = self.apply_recipe_to_Splits(dyn_rec, data, Segment.dynamic)
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
