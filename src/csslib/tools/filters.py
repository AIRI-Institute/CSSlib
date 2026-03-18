"""Module with predefined functions for the configurations_filter in the csslib.tools.dataloader.{DataLoader, VaspLoader} class."""

__all__ = [
    "split_by_spacegroup"
]

import pandas as pd


def split_by_spacegroup(df: pd.DataFrame, max_structures: int, space_group_no: int, 
                        greater_than_or_equal_to: bool = True) -> list[pd.DataFrame]:
    """
        Splits the dataset with configurations by spacegroup number. Selects only configurations that has
        a number more than/lower than the specified value.

        Args:
            df (pandas.DataFrame): dataframe with structures to select.
            max_structures (int): maximal number of structures to select.
            space_group_no (int): space group number.
            greater_than_or_equal_to (bool, optional): if True selects only configurations with a spacegroup
            number greater than or equal to space_group_no. If False selects lower than or equal to 
            space_group_no. Defaults to True.

        Return:
            list[pandas.DataFrame]: dataframes with selected, unselected and rest structures. rest_df is empty,
            if max_structures is None or max_structures more or equal to the number of available for the selection
            structures.
    """
    selected_df = pd.DataFrame([], columns=df.columns)
    unselected_df = pd.DataFrame([], columns=df.columns)
    rest_df = pd.DataFrame([], columns=df.columns)

    selected_structures_no = 0
    for _, row in df.iterrows():
        if max_structures is None or selected_structures_no < max_structures:
            if (row.space_group_no >= space_group_no if greater_than_or_equal_to else row.space_group_no <= space_group_no):
                selected_df = pd.concat([selected_df, row.to_frame().T], ignore_index=True)
                selected_structures_no += 1
            else:
                unselected_df = pd.concat([unselected_df, row.to_frame().T], ignore_index=True)
        else:
            rest_df = pd.concat([rest_df, row.to_frame().T], ignore_index=True)
    
    return selected_df, unselected_df, rest_df