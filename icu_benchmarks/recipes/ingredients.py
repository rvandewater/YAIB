from copy import deepcopy
import numpy as np
import pandas as pd


class Ingredients(pd.DataFrame):
    _metadata = ["roles"]

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=None, roles=None):
        super().__init__(data, index, columns, dtype, copy, )
        
        if isinstance(data, Ingredients) and roles is None:
            if copy is None or copy is True:
                self.roles = deepcopy(data.roles)
            else:
                self.roles = data.roles
        elif roles is None:
            self.roles = {}
        elif not isinstance(roles, dict):
            raise TypeError(f'expected dict object for roles, got {roles.__class__}')
        elif not np.all([k in self.columns for k in roles]):
            raise ValueError(f'roles contains variable name that is not in the data.')
        else:
            if copy is None or copy is True:
                self.roles = deepcopy(roles)
            else:
                self.roles = roles
                
    @property
    def _constructor(self):
        return Ingredients

    def _check_column(self, column):
        if not isinstance(column, str):
            raise ValueError(f'Expected string, got {column}')
        if not column in self.columns:
            raise ValueError(f'{column} does not exist in this Data object')

    def _check_role(self, new_role):
        if not isinstance(new_role, str):
            raise TypeError(f'new_role must be string, was {new_role.__class__}')

    def add_role(self, column: str, new_role: str):
        """Adds an additional role for a column that already has roles.
        
        Args:
            column (str): The column to receive additional roles.
            new_role (str): The role to add to the column.

        Raises:
            RuntimeError: If the column has no role yet.
        """
        self._check_column(column)
        self._check_role(new_role)
        if column not in self.roles.keys():
            raise RuntimeError(f'{column} has no roles yet, use update_role instead.')
        self.roles[column] += [new_role]

    def update_role(self, column: str, new_role: str, old_role: str = None):
        """Adds a new role for a column without roles or changes an existing role to a different one.
        
        Args:
            column (str): The column to receive additional roles.
            new_role (str): The role to add to the column.

        Raises:
            ValueError: If old_role is given but column has no roles.
                If old_role is given but column has no role old_role.
                If no old_role is given but column has multiple roles already.
        """
        self._check_column(column)
        self._check_role(new_role)
        if old_role is not None:
            if column not in self.roles.keys():
                raise ValueError(
                    f'Attempted to update role of {column} from {old_role} to {new_role} '
                    f'but {column} does not have a role yet.'
                )
            elif old_role not in self.roles[column]:
                raise ValueError(
                    f'Attempted to set role of {column} from {old_role} to {new_role} '
                    f'but {old_role} not among current roles: {self.roles[column]}.'
                )
            self.roles[column].remove(old_role)
            self.roles[column].append(new_role)
        else:
            if column not in self.roles.keys() or len(self.roles[column]) == 1:
                self.roles[column] = [new_role]
            else:
                raise ValueError(
                    f'Attempted to update role of {column} to {new_role} but '
                    f'{column} has more than one current roles: {self.roles[column]}'
                )
