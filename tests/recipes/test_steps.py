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

    def test_standard_scaler(self, example_df):
        rec = Recipe(example_df, ['y'], ['x1', 'x2'], ['id']) # FIXME: add squence when merged
        rec.add_step(StepSklearn(sel=all_numeric_predictors(), sklearn_transform=StandardScaler()))
        df = rec.prep()
        assert abs(df['x1'].mean()) < 0.00001
        assert abs(df['x2'].mean()) < 0.00001

    def test_min_max_scaler(self, example_df):
        rec = Recipe(example_df, ['y'], ['x1', 'x2'], ['id']) # FIXME: add squence when merged
        rec.add_step(StepSklearn(sel=all_numeric_predictors(), sklearn_transform=MinMaxScaler()))
        df = rec.prep()
        assert ((0 <= df['x1']) & (df['x1'] <= 1)).all()
        assert ((0 <= df['x2']) & (df['x2'] <= 1)).all()

    def test_max_abs_scaler(self, example_df):
        rec = Recipe(example_df, ['y'], ['x1', 'x2'], ['id']) # FIXME: add squence when merged
        rec.add_step(StepSklearn(sel=all_numeric_predictors(), sklearn_transform=MaxAbsScaler()))
        df = rec.prep()
        assert ((-1 <= df['x1']) & (df['x1'] <= 1)).all()
        assert ((-1 <= df['x2']) & (df['x2'] <= 1)).all()

    def test_robust_scaler(self, example_df):
        rec = Recipe(example_df, ['y'], ['x1', 'x2'], ['id']) # FIXME: add squence when merged
        rec.add_step(StepSklearn(sel=all_numeric_predictors(), sklearn_transform=RobustScaler()))
        df = rec.prep()
        assert df['x1'].median() == 0
