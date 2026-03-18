""" Module with predefined functions for the transformation_func in the csslib.tools.dataloader.{DataLoader, ResultsLoader} class."""

__all__ = [
    "get_spacegroup_distribution"
]


import pandas as pd


def get_spacegroup_distribution(df: pd.DataFrame):
    """
        Groups configurations in the css dataframe and evaluates the count of inequivalent and all structures in the css.

        Args:
            df (pandas.DataFrame): dataframe concatenated from dataframes located in the css_structures_metadata folder.
    """
    return df.groupby(['space_group_no', 'space_group_symbol', 
                       *[column for column in df.columns if column.endswith('concentration')]]).agg(cfgs_count=('structure_filename', 'count'), 
                                                                                                    all_cfgs_count=('weight', 'sum')).reset_index()