from typing import final
from copy import copy
from itertools import chain
from collections import Counter
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Selector():
    """Class responsible for selecting the variables affected by a step"""
    def __init__(self, description, names=None, roles=None, types=None):
        self.description = description
        self.names = names
        self.roles = roles
        self.types = types

    def set_type(self, types):
        self.types = types

    def __call__(self, data):
        # FIXME: think about how to combine names, roles, and types
        if self.names is None:
            vars = data.columns.tolist()
        else:
            vars = self.names

        with_role = [v for v, r in data.roles.items() if len(intersection(r, self.roles)) > 0]
        vars = [v for v in vars if v in with_role]

        # FIXME: filter types

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
    sel.add_type(['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
    sel.description = 'all numeric predictors'
    return sel

def all_outcomess():
    sel = has_role(['outcome'])
    sel.description = 'all outcomes'
    return sel

class Data(pd.DataFrame):
    _metadata = ["roles"]

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=None,):
        super().__init__(data, index, columns, dtype, )
        self.roles = {}
    
    @property
    def _constructor(self):
        return Data

    @final
    def __finalize__(self, other, method=None, **kwargs):
        super().__finalize__(other, method, **kwargs)
        self.roles = copy({c: r for c, r in self.roles.items() if c in self.columns})
        return self

    def _check_column(self, column):
        if not isinstance(column, str):
            raise ValueError(f'Expected string, got {column}')
        if not column in self.columns:
            raise ValueError(f'{column} does not exist in this Data object')

    def _check_role(self, new_role):
        if not isinstance(new_role, str):
            raise ValueError(f'new_role must be string, was {new_role.__class__}')

    def add_role(self, column, new_role):
        self._check_column(column)
        self._check_role(new_role)
        if column in self.roles.keys():
            raise RuntimeError(f'{column} already has role(s): f{self.roles[column]}')
        self.roles[column] = [new_role]

    def update_role(self, column, new_role):
        self._check_column(column)
        self._check_role(new_role)
        if column not in self.roles.keys():
            self.add_role(column, new_role)
        else:
            self.roles[column] += [new_role]

class Recipe():
    def __init__(self, data, outcomes=None, predictors=None, groups=None) -> None:
        self.data = Data(data)
        self.steps = []

        if outcomes:
            self.add_role(outcomes, 'outcome')
        if predictors:
            self.add_role(predictors, 'predictor')
        if groups:
            self.add_role(groups, 'group')

    def add_role(self, vars, new_role='predictor'):
        if isinstance(vars, str):
            vars = [vars]
        for v in vars:
            self.data.add_role(v, new_role)

    def update_role(self, vars, new_role='predictor'):
        if isinstance(vars, str):
            vars = [vars]
        for v in vars:
            self.data.update_role(v, new_role)

    def add_step(self, step):
        self.steps.append(step)
        return self

    def _check_data(self, data):
        if data is None:
            data = self.data
        if not data.columns.equals(self.data.columns):
            raise ValueError('Columns of data argument differs from recipe data.')
        return data 

    def _apply_group(self, data, step):
        if step.group:
            group_vars = groups()(data)
            data = data.groupby(group_vars)
        return data

    def prep(self, data=None, fresh=False):
        data = self._check_data(data)
        data = copy(data)

        for step in self.steps:
            data = self._apply_group(data, step)
            if fresh or not step.trained:
                data = step.fit_transform(data)
            else:
                data = step.transform(data)

        return self

    def bake(self, data=None):
        data = self._check_data(data)
        data = copy(data)
        
        for step in self.steps:
            data = self._apply_group(data, step)
            if not step.trained:
                raise RuntimeError(f'Step {step} not trained. Run prep first.')
            else:
                data = step.transform(data)

        return data

    def __repr__(self):
        repr = 'Recipe\n\n'
        
        # Print all existing roles and how many variables are assigned to each
        num_roles = Counter(chain.from_iterable(self.data.roles.values()))
        num_roles = pd.DataFrame({
            'role': [r for r in num_roles.keys()],
            '#variables': [n for n in num_roles.values()]
        })
        repr += 'Inputs:\n\n' + num_roles.__repr__() + '\n\n'

        # Print all steps
        repr += 'Operations:\n\n'
        for step in self.steps:
            repr += str(step) + '\n'

        return repr

class Step():
    def __init__(self, sel=all_predictors()) -> None:
        self.sel = sel
        self.columns = []
        self._trained = False
        self._group = True
    
    @property
    def trained(self):
        return self._trained

    @property
    def group(self):
        return self._group

    def fit(self, data):
        pass

    def transform(self, data):
        pass 

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def __repr__(self):
        repr = self.desc + ' for '
        
        if not self.trained:
            repr += str(self.sel)
        else:
            repr += str(self.columns) if len(self.columns) < 3 else str(self.columns[:2] + ['...']) # FIXME: remove brackets
            repr += ' [trained]'

        return repr

class StepImputeFill(Step):
    def __init__(self, sel=all_predictors(), value=None, method=None, limit=None):
        super().__init__(sel)
        self.desc = f'Impute with {method if method else value}'
        self.value = value
        self.method = method
        self.limit = limit
        
    def fit(self, data):
        self.columns = self.sel(data.obj)
        self._trained = True

    def transform(self, data):
        new_data = data.obj # FIXME: also deal with ungrouped DataFrames
        new_data[self.columns] = \
            data[self.columns].fillna(self.value, method=self.method, axis=0, limit=self.limit)
        return new_data

class StepScale(Step):
    def __init__(self, sel=all_predictors(), with_mean=True, with_std=True):
        super().__init__(sel)
        self.desc = f'Scale with mean ({with_mean}) and std ({with_std})'
        self.with_mean = with_mean
        self.with_std = with_std
        self._group = False
        
    def fit(self, data):
        self.columns = self.sel(data)
        self.scalers = {
            c: StandardScaler(copy=True, with_mean=self.with_mean, with_std=self.with_std).fit(data[c].values[:, None])
            for c in self.columns
        }
        self._trained = True

    def transform(self, data):
        new_data = data
        for c, sclr in self.scalers.items():
            new_data[c] = sclr.transform(data[c].values[:, None])
        return new_data

class StepHistorical(Step):
    def __init__(self, sel=all_predictors(), fun='max', suffix=None, role='predictor'):
        super().__init__(sel)
        self.desc = f'Create historical {fun}'
        self.fun = fun
        if suffix is None:
            suffix = fun
        self.suffix = suffix
        self.role = role
        
    def fit(self, data):
        self.columns = self.sel(data.obj)
        self._trained = True

    def transform(self, data):
        new_data = data.obj # FIXME: also deal with ungrouped DataFrames
        new_columns = [c + '_' + self.suffix for c in self.columns]

        if self.fun == 'max':
            res = data[self.columns].cummax(skipna=True)
        elif self.fun == 'min':
            res = data[self.columns].cummin(skipna=True)
        new_data[new_columns] = res

        for nc in new_columns:
            new_data.add_role(nc, self.role)
            
        return new_data


if __name__ == "__main__":
    df = pd.read_csv('/Users/patrick/datasets/benchmark/sepsis/mimic/dyn.csv.gz', compression='gzip')
    df = df[['stay_id', 'time', 'hr', 'resp', 'temp', 'sbp', 'dbp', 'map']]

    rec = Recipe(df)
    rec.add_role('stay_id', 'group')
    rec.add_role(['hr', 'resp', 'temp', 'sbp', 'dbp', 'map'], 'predictor')

    rec.add_step(StepScale())
    rec.add_step(StepHistorical(fun='max'))
    rec.add_step(StepImputeFill(method='ffill'))
    rec.add_step(StepImputeFill(value=0))
    
    rec.prep()
    rec.bake()

