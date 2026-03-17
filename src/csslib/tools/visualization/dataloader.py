"""Module for parsing of the csslib results."""


import gzip
import os
import pandas as pd
import pickle
from concurrent.futures import ProcessPoolExecutor
from csslib.exceptions import ResultsFolderNotFoundError
from tqdm import tqdm


def _results_parser_unit(file_name: str) -> pd.DataFrame:
    """
        A unit worker for ProcessPoolExecutor class used by get_spacegroup_distribution function. Reads an instanse of .pkl.gz.

        Args:
            file_name (str): full or relative path to the .pkl.gz file.

        Return:
            pandas.DataFrame: table with information about spacegroup distribution for the one composition.
    """
    df = None
    with gzip.open(file_name, 'rb') as archive:
        df = pickle.load(archive)

    concentration_columns = [column for column in df.columns if column.endswith('concentration')]
    df = df.groupby(['space_group_no', 'space_group_symbol', *concentration_columns]).agg(cfgs_count=('structure_filename', 'count'),
                                                                                          all_cfgs_count=('weight', 'sum')).reset_index()
    return df


def get_spacegroup_distribution(folder_path: str, num_workers: int = 1) -> pd.DataFrame:
    """
        Reads all css metadata files in the given folder and calculates the number of inequivalent and 
        all structures for each space group and substitution rate.

        Args:
            folder_path (str): path to results directory with .pkl.gz files.
            num_workers (int): number of available parallel processes.

        Raises:
            csslib.exceptions.ResultsFolderNotFoundError while results folder is not found or do not contain .pkl.gz files.

        Return:
            pandas.DataFrame: table with information about spacegroup distribution.
    """
    if not os.path.isdir(folder_path):
        raise ResultsFolderNotFoundError(f'Directory {folder_path} is not exists.')
    pkl_gz_files = [file for file in os.listdir(folder_path) if file.endswith('.pkl.gz')]
    if not pkl_gz_files:
        raise ResultsFolderNotFoundError(f'.pkl.gz files are not found at {folder_path}.')

    dataframes = []
    with (tqdm(range(len(pkl_gz_files)), desc="Collecting metadata of CSS structures", unit=" pkl", ncols=200) as pbar,
          ProcessPoolExecutor(max_workers=num_workers,) as pool):
        for pkl in pkl_gz_files:
            future = pool.submit(_results_parser_unit, os.path.join(folder_path, pkl))
            dataframes.append(future.result())
            future.add_done_callback(lambda _: pbar.update())
    return pd.concat(dataframes).sort_values(by=['space_group_no', 'space_group_symbol', *[column for column in dataframes[0].columns if column.endswith('concentration')]]).reset_index()