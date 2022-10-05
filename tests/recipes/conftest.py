import pytest
import numpy as np
import pandas as pd

from icu_benchmarks.recipes.recipe import Recipe
from icu_benchmarks.recipes.ingredients import Ingredients


@pytest.fixture()
def example_df():
    rand_state = np.random.RandomState(42)
    df = pd.DataFrame(
        {
            "id": [1] * 6 + [2] * 4,
            "time": np.concatenate((np.arange(6), np.arange(4))),
            "y": rand_state.normal(size=(10,)),
            "x1": rand_state.normal(loc=10, scale=5, size=(10,)),
            "x2": rand_state.binomial(n=1, p=0.3, size=(10,)),
            "x3": pd.Series(["a", "b", "c", "a", "c", "b", "c", "a", "b", "c"], dtype="category"),
            "x4": pd.Series(["x", "y", "y", "x", "y", "y", "x", "x", "y", "x"], dtype="category"),
        }
    )
    return df


@pytest.fixture
def example_ingredients(example_df):
    return Ingredients(example_df)


@pytest.fixture()
def example_recipe(example_df):
    return Recipe(example_df, ["y"], ["x1", "x2", "x3", "x4"], ["id"])  # FIXME: add squence when merged


@pytest.fixture()
def example_recipe_w_nan(example_df):
    example_df.loc[[1, 2, 4, 7], "x1"] = np.nan
    return Recipe(example_df, ["y"], ["x1", "x2", "x3", "x4"], ["id"])  # FIXME: add squence when merged
