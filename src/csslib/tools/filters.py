"""Module with predefined functions for the configurations_filter in the csslib.tools.dataloader.{DataLoader, VaspLoader} class."""

__all__ = [
    "load_function",
    "random_sampling",
    "split_by_spacegroup"
]

import pandas as pd
import random
from functools import partial
from typing import Callable


def load_function(function: Callable, **kwargs) -> Callable:
    """
        Loads filter function and tranfer parameters to it. Using functools.partial method.

        Args:
            function (Callable): function which should be loaded.
            kwargs (dict, optional): args which should be transfer to the function.

        Return:
            Callable: preinitialized function.
    """
    return partial(function, **kwargs)


def split_by_spacegroup(df: pd.DataFrame, space_group_no: int, greater_than_or_equal_to: bool = True) -> pd.DataFrame:
    """
        Splits the dataset with configurations by spacegroup number. Selects only configurations that has
        a number more than/lower than the specified value. It is convinient to use load_function for preinitialization.

        Args:
            df (pandas.DataFrame): dataframe with structures to select.
            space_group_no (int): space group number.
            greater_than_or_equal_to (bool, optional): if True selects only configurations with a spacegroup
            number greater than or equal to space_group_no. If False selects lower than or equal to 
            space_group_no. Defaults to True.

        Return:
            pandas.DataFrame: dataframe with selected structures.
    """
    return df.loc[df.space_group_no >= space_group_no if greater_than_or_equal_to else df.space_group_no <= space_group_no]


def random_sampling(df: pd.DataFrame, structures_number: int = 1, seed: int = 8888):
    """
        Randomly selects from dataset a defined number of structures. It is convinient to use load_function for preinitialization.

        Args:
            df (pandas.DataFrame): dataframe with structures to select.
            structures_number (int, optional): number of structures to select. Defaults to 1.
            seed (int, optional): seed to be used by random number generator.

        Return:
            pandas.DataFrame: dataframe with selected structures.
    """
    random.seed(seed)
    max_index = len(df) - 1
    indexes = []
    while len(indexes) != structures_number and len(indexes) != len(df):
        number = random.randint(0, max_index)
        if number not in indexes:
            indexes.append(number)
    indexes = pd.Index(indexes)
    return df.loc[indexes]
