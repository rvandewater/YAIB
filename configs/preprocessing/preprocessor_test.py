import logging

import gin
import pandas as pd
from recipys.recipe import Recipe
from recipys.selector import all_numeric_predictors, has_type, all_of
from recipys.step import StepScale, StepImputeFill, StepSklearn, StepHistorical, Accumulator
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import LabelEncoder
from icu_benchmarks.data.preprocessor import Preprocessor


class CustomPreprocessor(Preprocessor):
    def __init__(
        self,
        data,
        vars: dict[str],
        generate_features: bool = True,
        scaling: bool = True,
    ):
        """
        Args:
            data: Train, validation and test data dictionary. Further divided in static, dynamic, and outcome.
            vars: Variables for static, dynamic, outcome.
            generate_features: Generate features for static data.
            scaling: Scaling of dynamic and static data.
        Returns:
            Preprocessed data.
        """
        self.data = data
        self.vars = vars
        self.generate_features = generate_features
        self.scaling = scaling

    def apply(self):
        logging.info("Preprocessor static features.")
        self.data = self.process_static()
        self.data = self.process_dynamic()
        return self.data

    def process_static(self):
        sta_rec = Recipe(self.data["train"]["STATIC"], [], self.vars["STATIC"])
        if self.scaling:
            sta_rec.add_step(StepScale())

        sta_rec.add_step(StepImputeFill(sel=all_numeric_predictors(), value=0))
        sta_rec.add_step(StepSklearn(SimpleImputer(missing_values=None, strategy="most_frequent"), sel=has_type("object")))
        sta_rec.add_step(StepSklearn(LabelEncoder(), sel=has_type("object"), columnwise=True))

        data = self.apply_recipe_to_splits(sta_rec, self.data, "STATIC")

        return data

    def process_dynamic(self):
        vars = self.vars
        dyn_rec = Recipe(self.data["train"]["DYNAMIC"], [], vars["DYNAMIC"], vars["GROUP"], vars["SEQUENCE"])
        if self.scaling:
            dyn_rec.add_step(StepScale())
        dyn_rec.add_step(StepSklearn(MissingIndicator(), sel=all_of(vars["DYNAMIC"]), in_place=False))
        dyn_rec.add_step(StepImputeFill(method="ffill"))
        dyn_rec.add_step(StepImputeFill(value=0))
        if self.generate_features:
            dyn_rec = self.dynamic_feature_generation(dyn_rec)

        data = self.apply_recipe_to_splits(dyn_rec, self.data, "DYNAMIC")
        return data

    def dynamic_feature_generation(self, data):
        data.add_step(StepHistorical(sel=all_of(self.vars["DYNAMIC"]), fun=Accumulator.MIN, suffix="min_hist"))
        data.add_step(StepHistorical(sel=all_of(self.vars["DYNAMIC"]), fun=Accumulator.MAX, suffix="max_hist"))
        data.add_step(StepHistorical(sel=all_of(self.vars["DYNAMIC"]), fun=Accumulator.COUNT, suffix="count_hist"))
        data.add_step(StepHistorical(sel=all_of(self.vars["DYNAMIC"]), fun=Accumulator.MEAN, suffix="mean_hist"))
        return data

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
