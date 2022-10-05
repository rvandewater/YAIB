import re

from .ingredients import Ingredients


class Selector():
    """Class responsible for selecting the variables affected by a step"""
    def __init__(self, description, names=None, roles=None, types=None, pattern=None):
        self.description = description
        self.set_names(names)
        self.set_roles(roles)
        self.set_types(types)
        self.set_pattern(pattern)

    def set_names(self, names):
        self.names = names

    def set_roles(self, roles):
        self.roles = roles

    def set_types(self, types):
        self.types = types

    def set_pattern(self, pattern):
        self.pattern = pattern

    def __call__(self, data):
        if not isinstance(data, Ingredients):
            raise TypeError(f'Expected Ingredients, got {data.__class__}')

        vars = data.columns.tolist()

        if self.roles is not None:
            sel_roles = [v for v, r in data.roles.items() if intersection(r, self.roles)]
            vars = intersection(vars, sel_roles)

        if self.types is not None:
            sel_types = data.select_dtypes(include=self.types).columns.tolist()
            vars = intersection(vars, sel_types)

        if self.names is not None:
            vars = intersection(vars, self.names)

        if self.pattern is not None:
            vars = list(filter(self.pattern.search, vars))

        return vars

    def __repr__(self):
        return self.description


def intersection(x, y):
    if isinstance(x, str):
        x = [x]
    if isinstance(y, str):
        y = [y]
    return [i for i in x if i in y]


def all_of(names):
    return Selector(description=str(names), names=names)


def regex_names(regex):
    pattern = re.compile(regex)
    return Selector(description=f'regex: {regex}', pattern=pattern)


def starts_with(prefix):
    return regex_names(f'^{prefix}')


def ends_with(suffix):
    return regex_names(f'{suffix}$')


def contains(substring):
    return regex_names(f'{substring}')


def has_role(roles):
    return Selector(description=f'roles: {roles}', roles=roles)


def has_type(types):
    return Selector(description=f'types: {types}', types=types)


def groups():
    return Selector(description='grouping variables', roles=['group'])


def all_predictors():
    sel = has_role(['predictor'])
    sel.description = 'all predictors'
    return sel


def all_numeric_predictors():
    sel = all_predictors()
    sel.set_types(['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
    sel.description = 'all numeric predictors'
    return sel


def all_outcomes():
    sel = has_role(['outcome'])
    sel.description = 'all outcomes'
    return sel
