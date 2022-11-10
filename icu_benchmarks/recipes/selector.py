import re

from .ingredients import Ingredients
from typing import Union


class Selector:
    """Class responsible for selecting the variables affected by a recipe step

    Args:
        description: Text used to represent Selector when printed in summaries
        names: Column names to select. Defaults to None.
        roles: Column roles to select, see also Ingredients. Defaults to None.
        types: Column data types to select. Defaults to None.
        pattern: Regex pattern to search column names with. Defaults to None.
    """

    def __init__(
        self,
        description: str,
        names: Union[str, list[str]] = None,
        roles: Union[str, list[str]] = None,
        types: Union[str, list[str]] = None,
        pattern: re.Pattern = None,
    ):
        self.description = description
        self.set_names(names)
        self.set_roles(roles)
        self.set_types(types)
        self.set_pattern(pattern)

    def set_names(self, names: Union[str, list[str]]):
        """Set the column names to select with this Selector

        Args:
            names: column names to select
        """
        self.names = enlist_str(names)

    def set_roles(self, roles: Union[str, list[str]]):
        """Set the column roles to select with this Selector

        Args:
            roles: column roles to select, see also Ingredients
        """
        self.roles = enlist_str(roles)

    def set_types(self, roles: Union[str, list[str]]):
        """Set the column data types to select with this Selector

        Args:
            roles: column data types to select
        """
        self.types = enlist_str(roles)

    def set_pattern(self, pattern: re.Pattern):
        """Set the pattern to search with this Selector

        Args:
            pattern: Regex pattern to search column names with.
        """
        self.pattern = pattern

    def __call__(self, ingr: Ingredients) -> list[str]:
        """Select variables from Ingredients

        Args:
            ingr: object from which to select the variables

        Raises:
            TypeError: when something other than an Ingredient object is passed

        Returns:
            Selected variables.
        """

        if not isinstance(ingr, Ingredients):
            raise TypeError(f"Expected Ingredients, got {ingr.__class__}")

        vars = ingr.columns.tolist()

        if self.roles is not None:
            sel_roles = [v for v, r in ingr.roles.items() if intersection(r, self.roles)]
            vars = intersection(vars, sel_roles)

        if self.types is not None:
            sel_types = ingr.select_dtypes(include=self.types).columns.tolist()
            vars = intersection(vars, sel_types)

        if self.names is not None:
            vars = intersection(vars, self.names)

        if self.pattern is not None:
            vars = list(filter(self.pattern.search, vars))

        return vars

    def __repr__(self):
        return self.description


def enlist_str(x: Union[str, list[str], None]) -> Union[list[str], None]:
    """Wrap a str in a list if it isn't a list yet

    Args:
        x: object to wrap.

    Raises:
        TypeError: If neither a str nor a list of strings is passed

    Returns:
        _description_
    """
    if isinstance(x, str):
        return [x]
    elif isinstance(x, list):
        if not all(isinstance(i, str) for i in x):
            raise TypeError("Only lists of str are allowed.")
        return x
    elif x is None:
        return x
    else:
        raise TypeError(f"Expected str or list of str, got {x.__class__}")


def intersection(x: list, y: list) -> list:
    """Intersection of two lists

    Note:
        maintains the order of the first list
        does not deduplicate items (i.e., does not return a set)

    Args:
        x: first list
        y: second list

    Returns:
        Elements in `x` that are also in `y`.
    """
    if isinstance(x, str):
        x = [x]
    if isinstance(y, str):
        y = [y]
    return [i for i in x if i in y]


def all_of(names: Union[str, list[str]]) -> Selector:
    """Define selector for any columns with one of the given names

    Args:
        names: names to select

    Returns:
        Object representing the selection rule.
    """
    return Selector(description=str(names), names=names)


def regex_names(regex: str) -> Selector:
    """Define selector for any columns where the name matches the regex pattern

    Args:
        pattern: string to be transformed to regex pattern to search for

    Returns:
        Object representing the selection rule.
    """
    pattern = re.compile(regex)
    return Selector(description=f"regex: {regex}", pattern=pattern)


def starts_with(prefix: str) -> Selector:
    """Define selector for any columns where the name starts with the prefix

    Args:
        prefix: prefix to search for

    Returns:
        Object representing the selection rule.
    """
    return regex_names(f"^{prefix}")


def ends_with(suffix: str) -> Selector:
    """Define selector for any columns where the name ends with the suffix

    Args:
        prsuffixefix: suffix to search for

    Returns:
        Object representing the selection rule.
    """
    return regex_names(f"{suffix}$")


def contains(substring: str) -> Selector:
    """Define selector for any columns where the name contains the substring

    Args:
        substring: substring to search for

    Returns:
        Object representing the selection rule.
    """
    return regex_names(f"{substring}")


def has_role(roles: Union[str, list[str]]) -> Selector:
    """Define selector for any columns with one of the given roles

    Args:
        roles: roles to select

    Returns:
       Object representing the selection rule.
    """
    return Selector(description=f"roles: {roles}", roles=roles)


def has_type(types: Union[str, list[str]]) -> Selector:
    """Define selector for any columns with one of the given types

    Args:
        types: data types to select

    Note:
        Data types are selected based on string representation as returned by `df[[varname]].dtype.name`.

    Returns:
        Object representing the selection rule.
    """
    return Selector(description=f"types: {types}", types=types)


def all_predictors() -> Selector:
    """Define selector for all predictor columns

    Returns:
        Object representing the selection rule.
    """
    sel = has_role(["predictor"])
    sel.description = "all predictors"
    return sel


def all_numeric_predictors() -> Selector:
    """Define selector for all numerical predictor columns

    Returns:
        Object representing the selection rule.
    """
    sel = all_predictors()
    sel.set_types(["int16", "int32", "int64", "float16", "float32", "float64"])
    sel.description = "all numeric predictors"
    return sel


def all_outcomes() -> Selector:
    """Define selector for all outcome columns

    Returns:
        Object representing the selection rule.
    """
    sel = has_role(["outcome"])
    sel.description = "all outcomes"
    return sel


def all_groups() -> Selector:
    """Define selector for all grouping variables

    Returns:
        Object representing the selection rule.
    """
    return Selector(description="all grouping variables", roles=["group"])


def select_groups(ingr: Ingredients) -> list[str]:
    """Select any grouping columns

    Defines and directly applies Selector(roles=["group"])

    Returns:
        grouping columns
    """
    return all_groups()(ingr)


def all_sequences() -> Selector:
    """Define selector for all grouping variables

    Returns:
        Object representing the selection rule.
    """
    return Selector(description="all sequence variables", roles=["sequence"])


def select_sequence(ingr: Ingredients) -> list[str]:
    """Select any sequence columns

    Defines and directly applies Selector(roles=["sequence"])

    Returns:
        Grouping columns.
    """
    return all_sequences()(ingr)
