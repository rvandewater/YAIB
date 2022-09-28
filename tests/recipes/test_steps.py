from textwrap import fill
import pytest
import numpy as np
from sklearn.preprocessing import *
from sklearn.impute import *

from icu_benchmarks.recipes.recipe import Recipe
from icu_benchmarks.recipes.selector import all_numeric_predictors, has_types
from icu_benchmarks.recipes.step import StepSklearn

from tests.recipes.test_recipe import example_df


class TestSklearnStep:
    @pytest.fixture()
    def example_recipe(self, example_df):
        return Recipe(example_df, ['y'], ['x1', 'x2', 'x3'], ['id']) # FIXME: add squence when merged

    def test_simple_imputer(self, example_df):
        example_df.loc[[1,2,4,7], 'x1'] = np.nan
        rec = Recipe(example_df, ['y'], ['x1', 'x2', 'x3'], ['id']) # FIXME: add squence when merged
        rec.add_step(StepSklearn(sel=all_numeric_predictors(), sklearn_transform=SimpleImputer(strategy='constant', fill_value=0)))
        df = rec.prep()
        assert (df.loc[[1,2,4,7], 'x1'] == 0).all()

    def test_standard_scaler(self, example_recipe):
        example_recipe.add_step(StepSklearn(sel=all_numeric_predictors(), sklearn_transform=StandardScaler()))
        df = example_recipe.prep()
        assert abs(df['x1'].mean()) < 0.00001
        assert abs(df['x2'].mean()) < 0.00001

    def test_min_max_scaler(self, example_recipe):
        example_recipe.add_step(StepSklearn(sel=all_numeric_predictors(), sklearn_transform=MinMaxScaler()))
        df = example_recipe.prep()
        assert ((0 <= df['x1']) & (df['x1'] <= 1)).all()
        assert ((0 <= df['x2']) & (df['x2'] <= 1)).all()

    def test_max_abs_scaler(self, example_recipe):
        example_recipe.add_step(StepSklearn(sel=all_numeric_predictors(), sklearn_transform=MaxAbsScaler()))
        df = example_recipe.prep()
        assert ((-1 <= df['x1']) & (df['x1'] <= 1)).all()
        assert ((-1 <= df['x2']) & (df['x2'] <= 1)).all()

    def test_robust_scaler(self, example_recipe):
        example_recipe.add_step(StepSklearn(sel=all_numeric_predictors(), sklearn_transform=RobustScaler()))
        df = example_recipe.prep()
        assert df['x1'].median() == 0
        assert df['x2'].median() == 0

    def test_binarizer(self, example_recipe):
        example_recipe.add_step(StepSklearn(sel=all_numeric_predictors(), sklearn_transform=Binarizer()))
        df = example_recipe.prep()
        assert (df['x1'].isin([0, 1])).all()
        assert (df['x2'].isin([0, 1])).all()

    def test_ordinal_encoder(self, example_recipe):
        example_recipe.add_step(StepSklearn(sel=has_types(['category']), sklearn_transform=OrdinalEncoder(), in_place=False))
        df = example_recipe.prep()
        assert ((0 <= df['OrdinalEncoder_1']) & (df['OrdinalEncoder_1'] <= 2)).all()
    
    def test_label_encoder(self, example_recipe):
        example_recipe.add_step(StepSklearn(sel=has_types(['category']), sklearn_transform=LabelEncoder(), columnwise=True))
        df = example_recipe.prep()
        assert ((0 <= df['x3']) & (df['x3'] <= 2)).all()

    def test_onehot_encoder(self, example_recipe):
        example_recipe.add_step(StepSklearn(sel=has_types(['category']), sklearn_transform=OneHotEncoder(sparse=False), in_place=False))
        df = example_recipe.prep()
        assert (df['OneHotEncoder_1'].isin([0, 1])).all()
        assert (df['OneHotEncoder_2'].isin([0, 1])).all()
        assert (df['OneHotEncoder_3'].isin([0, 1])).all()
        assert (df['OneHotEncoder_4'].isin([0, 1])).all()
        assert (df['OneHotEncoder_5'].isin([0, 1])).all()

    def test_wrong_columnwise(self, example_recipe):
        example_recipe.add_step(StepSklearn(sel=has_types(['category']), sklearn_transform=LabelEncoder(), columnwise=False))
        with pytest.raises(ValueError) as exc_info:
            example_recipe.prep()
        assert 'columnwise=True' in str(exc_info.value)

    def test_wrong_in_place(self, example_recipe):
        example_recipe.add_step(StepSklearn(sel=has_types(['category']), sklearn_transform=OneHotEncoder(sparse=False), in_place=True))
        with pytest.raises(ValueError) as exc_info:
            example_recipe.prep()
        assert 'in_place=False' in str(exc_info.value)

    def test_sparse_error(self, example_recipe):
        example_recipe.add_step(StepSklearn(sel=has_types(['category']), sklearn_transform=OneHotEncoder(sparse=True), in_place=False))
        with pytest.raises(ValueError) as exc_info:
            example_recipe.prep()
        assert 'sparse=False' in str(exc_info.value)