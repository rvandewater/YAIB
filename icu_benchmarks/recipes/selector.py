class Selector():
    """Class responsible for selecting the variables affected by a step"""
    def __init__(self, description, names=None, roles=None, types=None):
        self.description = description
        self.names = names
        self.roles = roles if roles else []
        self.types = types if types else []

    def set_types(self, types):
        self.types = types

    def __call__(self, data):
        # FIXME: think about how to combine names, roles, and types
        vars = self.names if self.names else data.columns.tolist()
        if self.roles:
            vars = [v for v in vars if intersection(data.roles.get(v, []), self.roles)]
        if self.types:
            vars = [v for v in vars if data.dtypes[v] in self.types]
        return vars

    def __repr__(self):
        return self.description


def intersection(x, y):
    return set(x).intersection(set(y))


def all_of(names):
    return Selector(descripton=str(names), names=names)


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


def all_outcomess():
    sel = has_role(['outcome'])
    sel.description = 'all outcomes'
    return sel
