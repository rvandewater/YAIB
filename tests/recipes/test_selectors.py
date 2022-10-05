import pytest

from icu_benchmarks.recipes.selector import (
    Selector,
    all_outcomes,
    all_of,
    # starts_with,
    # ends_with,
    # contains,
    has_role,
    has_type,
    groups,
    all_numeric_predictors,
    all_predictors,
    intersection,
    enlist_str,
)


def test_no_description():
    with pytest.raises(TypeError) as e_info:
        Selector()
    assert e_info.match("missing 1 required positional argument")


def test_not_ingredients(example_df):
    with pytest.raises(TypeError) as e_info:
        Selector("test step")(example_df)
    assert e_info.match("Expected Ingredients")


def test_intersection():
    assert intersection(["a", "b"], ["b", "c"]) == ["b"]


def test_enlist_str():
    assert enlist_str("string") == ["string"]


def test_enlist_str_list():
    assert enlist_str(["string1", "string2"]) == ["string1", "string2"]


def test_enlist_str_None():
    assert enlist_str(None) == None


def test_enlist_str_other():
    with pytest.raises(TypeError) as e_info:
        enlist_str({"k": "string"})
    assert e_info.match("Expected str or list of str")


def test_enlist_str_other_list():
    with pytest.raises(TypeError) as e_info:
        enlist_str(["outer", {"k": "inner"}])
    assert e_info.match("Only lists of str are allowed.")


def test_all_of(example_ingredients):
    sel = all_of(["y", "x1"])
    assert sel(example_ingredients) == ["y", "x1"]


# def test_starts_with(example_ingredients):
#     sel = starts_with('x')
#     assert sel(example_ingredients) == ['x1', 'x2']


# def test_ends_with(example_ingredients):
#     sel = ends_with('1')
#     assert sel(example_ingredients) == ['x1']


# def test_contains(example_ingredients):
#     sel = contains('i')
#     assert sel(example_ingredients) == ['id', 'time']


def test_has_role(example_ingredients):
    example_ingredients.update_role("x1", "predictor")
    example_ingredients.update_role("x2", "predictor")
    sel = has_role("predictor")
    assert sel(example_ingredients) == ["x1", "x2"]


def test_has_type(example_ingredients):
    sel = has_type("float64")
    assert sel(example_ingredients) == ["y", "x1"]


def test_groups(example_ingredients):
    example_ingredients.update_role("id", "group")
    sel = groups()
    assert sel(example_ingredients) == ["id"]


def test_all_predictors(example_ingredients):
    example_ingredients.update_role("x1", "predictor")
    example_ingredients.update_role("x2", "predictor")
    sel = all_predictors()
    assert sel(example_ingredients) == ["x1", "x2"]


def test_all_numeric_predictors(example_ingredients):
    example_ingredients.update_role("x1", "predictor")
    example_ingredients.update_role("x2", "predictor")
    sel = all_numeric_predictors()
    assert sel(example_ingredients) == ["x1", "x2"]


def test_all_outcomes(example_ingredients):
    example_ingredients.update_role("y", "outcome")
    sel = all_outcomes()
    assert sel(example_ingredients) == ["y"]
