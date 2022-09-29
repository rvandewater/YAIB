from .ingredients import Ingredients

class Selector():
    """Class responsible for selecting the variables affected by a step"""
    def __init__(self, description, names=None, roles=None, types=None):
        self.description = description
        self.set_names(names)
        self.set_roles(roles)
        self.set_types(types)

    def set_names(self, names):
        self.names = names

    def set_roles(self, roles):
        self.roles = roles

    def set_types(self, types):
        self.types = types

    def __call__(self, data):
        if not isinstance(data, Ingredients):
            raise TypeError(f'Expected Ingredients, got {data.__class__}')
        
        vars = data.columns.tolist()

        if self.roles is not None:
            sel_roles = [v for v, r in data.roles.items() if len(intersection(r, self.roles)) > 0]
            vars = intersection(vars, sel_roles)

        if self.types is not None:
            # currently matches types by name. is this problematic?
            sel_types = [v for v, t in data.dtypes.items() if t.name in self.types]
            vars = intersection(vars, sel_types)

        if self.names is not None:
            vars = intersection(vars, self.names)

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


def starts_with(pattern):
    raise NotImplementedError()


def ends_with(pattern):
    raise NotImplementedError()


def contains(pattern):
    raise NotImplementedError()


def has_role(roles):
    return Selector(description=f'roles: {roles}', roles=roles)


def has_type(types):
    return Selector(description=f'types: {types}', types=types)


def groups():
    return Selector(description=f'grouping variables', roles=['group'])


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
