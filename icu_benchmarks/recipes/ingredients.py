from copy import copy
from typing import final

import pandas as pd


class Ingredients(pd.DataFrame):
    _metadata = ["roles"]

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=None,):
        super().__init__(data, index, columns, dtype, )
        self.roles = {}

    @property
    def _constructor(self):
        return Ingredients

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
