from __future__ import annotations
from collections import Counter
from copy import copy
from itertools import chain
from typing import Union

import pandas as pd

from icu_benchmarks.recipes.ingredients import Ingredients
from icu_benchmarks.recipes.selector import groups
from icu_benchmarks.recipes.step import Step


class Recipe():
    """Recipe for preprocessing data

    A Recipe object combines a pandas-like Ingredients object with one or more 
    sklearn-inspired transformation Steps to turn into a model-ready input. 

    Args:
        data (Union[pd.DataFrame, Ingredients]): 
    """
    def __init__(
        self, 
        data: Union[pd.DataFrame, Ingredients], 
        outcomes=None, 
        predictors=None, 
        groups=None, 
        sequences=None
    ) -> None:
        self.data = Ingredients(data)
        self.steps = []

        if outcomes:
            self.update_roles(outcomes, 'outcome')
        if predictors:
            self.update_roles(predictors, 'predictor')
        if groups:
            self.update_roles(groups, 'group')
        if sequences:
            self.update_roles(sequences, 'sequence')

    def add_roles(self, vars: Union[str, list[str]], new_role: str = 'predictor') -> Recipe:
        """Adds an additional role for one or more columns of the Recipe's Ingredients.

        Args:
            vars (Union[str, list[str]]): The column to receive additional roles.
            new_role (str, optional): Defaults to predictor'. The role to add to the column.

        See also:
            Ingredients.add_role()

        Returns:
            Recipe: self
        """
        if isinstance(vars, str):
            vars = [vars]
        for v in vars:
            self.data.add_role(v, new_role)
        return self

    def update_roles(self, vars: Union[str, list[str]], new_role: str = 'predictor', old_role=None) -> Recipe:
        """Adds a new role for one or more columns of the Recipe's Ingredients without roles or changes an existing role to a different one.

        Args:
            vars (Union[str, list[str]]): The column to receive additional roles.
            new_role (str, optional): Defaults to predictor'. The role to add or change to.
            old_role (str, optional): Defaults to None. The role to be changed.

        See also:
            Ingredients.update_role()

        Returns:
            Recipe: self
        """
        if isinstance(vars, str):
            vars = [vars]
        for v in vars:
            self.data.update_role(v, new_role, old_role)
        return self

    def add_step(self, step: Step) -> Recipe:
        """Adds a new step to the Recipe

        Args:
            step (Step): a transformation step that should be applied to the Ingredients during prep() and bake()

        Returns:
            Recipe: self
        """
        self.steps.append(step)
        return self

    def _check_data(self, data):
        if data is None:
            data = self.data
        elif data.__class__ == pd.DataFrame:
            data = Ingredients(data, roles=self.data.roles)
        if not data.columns.equals(self.data.columns):
            raise ValueError('Columns of data argument differs from recipe data.')
        return data

    def _apply_group(self, data, step):
        if step.group:
            group_vars = groups()(data)
            data = data.groupby(group_vars)
        return data

    def prep(self, data=None, refit=False):
        """
        Fits and transforms, in other words preps, the data.
        @param data:
        @param refit: Refit all columns
        @return:
        """
        data = self._check_data(data)
        data = copy(data)
        self._apply_fit_transform(data, refit)
        return pd.DataFrame(data)

    def bake(self, data=None):
        """
        Transforms, or bakes, the data if it has been prepped.
        @param data:
        @return:
        """
        data = self._check_data(data)
        data = copy(data)
        self._apply_fit_transform(data)
        return pd.DataFrame(data)

    def _apply_fit_transform(self, data=None, refit=False):
        # applies transform or fit and transform (when refit or not trained yet)
        for step in self.steps:
            data = self._apply_group(data, step)
            if refit or not step.trained:
                data = step.fit_transform(data)
            else:
                data = step.transform(data)
        return self

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
