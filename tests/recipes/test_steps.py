from textwrap import fill
import pytest
import numpy as np
from sklearn.preprocessing import *
from sklearn.impute import *

from icu_benchmarks.recipes.recipe import Recipe
from icu_benchmarks.recipes.selector import all_numeric_predictors
from icu_benchmarks.recipes.step import StepSklearn

from tests.recipes.test_recipe import example_df


class TestSklearnStep:
    def test_simple_imputer(self, example_df):
        example_df.loc[[1,2,4,7], 'x1'] = np.nan
        rec = Recipe(example_df, ['y'], ['x1', 'x2'], ['id']) # FIXME: add squence when merged
        rec.add_step(StepSklearn(sel=all_numeric_predictors(), sklearn_transform=SimpleImputer(strategy='constant', fill_value=0)))
        df = rec.prep()
        assert (df.loc[[1,2,4,7], 'x1'] == 0).all()
