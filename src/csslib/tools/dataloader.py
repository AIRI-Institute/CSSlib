"""Module with the DataLoader class. Can be used to generate training/test samples, transform data, and prepare it for postprocessing."""


import gzip
import os
import pandas as pd
import pickle
from csslib.exceptions import ResultsFolderNotFoundError, DataLoaderError
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Callable


def _parse_worker(file_path: str, configurations_filter: Callable | None = None, transformation_func: Callable | None = None):
    """
        A worker for ProcessPoolExecutor class used by __call__ method of the DataLoader class. Reads configurations from .pkl.gz.

        Args:
            file_path (str): full or relative path to the .pkl.gz file.
            configurations_filter (function | None, optional): filter to apply to the configurations set. As the input
            function must get pandas.Dataframe object and return transformed pandas.Dataframe object. Defaults to None.
            transformation_func (function | None, optional): transformation function to be applied to the configurations set.
            As the input function must get pandas.Dataframe object and return transformed pandas.Dataframe object.
            Defaults to None.

        Return:
            pandas.DataFrame: table with information about css structures.
    """
    with gzip.open(file_path, "rb") as archive:
        df = pickle.load(archive)

    if configurations_filter is not None:
        df = configurations_filter(df)

    if transformation_func is not None:
        df = transformation_func(df)

    return df


class DataLoader:
    """
        Data loader of structures for performing VASP calculations and realization of the postprocessing.
    """
    def __init__(self, folder_path: str, max_structures: int | None = None, num_workers: int = 1,
                 configurations_filter: Callable | None = None, transformation_func: Callable | None = None):
        """
            Initialization method for the DataLoader class.

            Args:
                folder_path (str): path to folder with .pkl.gz archives with structures for loading/processing.
                max_structures (int | None, optional): number of the configuration to load during one call of the DataLoader, 
                e.g. configurations batch. If None, all configurations will be load. Defaults to None.
                num_workers (int, optional): number of workers for parallel execution of the DataLoader. Defaults to 1.
                configurations_filter (function | None, optional): filter to apply to the configurations set. As the input
                function must get pandas.Dataframe object and return transformed pandas.Dataframe object. Defaults to None.
                transformation_func (function | None, optional): transformation function to be applied to the configurations set.
                As the input function must get pandas.Dataframe object and return transformed pandas.Dataframe object.
                Defaults to None.

            Raise:
                csslib.exceptions.ResultsFolderNotFoundError: while results folder is not found or do not contain .pkl.gz files.
        """
        if not os.path.isdir(folder_path):
            raise ResultsFolderNotFoundError(f'Directory {folder_path} is not found.')

        self.__pkl_gz = [os.path.join(folder_path, archive) for archive in os.listdir(folder_path) if archive.endswith('.pkl.gz')]
        if not self.__pkl_gz:
            raise ResultsFolderNotFoundError(f'There are no .pkl.gz files in the {folder_path} directory.')

        self.max_structures = max_structures
        self.num_workers = num_workers
        self.__configurations_filter = configurations_filter
        self.__transformation_func = transformation_func
        self.__structures_df = None
        
    def apply(self, configurations_filter: Callable | None = None, transformation_func: Callable | None = None,
              apply_configurations_filter: bool = True, apply_transformation_func: bool = True):
        """
            Sets and applies or only applies configurations filter and transformation function if they are specified.

            Args:
                configurations_filter (function | None, optional): filter to apply to the configurations set. As the input
                function must get pandas.Dataframe object and return transformed pandas.Dataframe object. Defaults to None.
                transformation_func (function | None, optional): transformation function to be applied to the configurations set.
                As the input function must get pandas.Dataframe object and return transformed pandas.Dataframe object.
                Defaults to None.

            Raise:
                csslib.exceptions.DataLoaderError: if DataLoader.__structures_df is not defined.
        """
        if self.__structures_df is None:
            raise DataLoaderError('DataLoader.__structures_df attribute is not defined.')

        self.__configurations_filter = configurations_filter if configurations_filter is not None else self.__configurations_filter
        self.__transformation_func = transformation_func if transformation_func is not None else self.__transformation_func
        
        if self.__configurations_filter is not None and apply_configurations_filter:
            self.__structures_df = self.__configurations_filter(self.__structures_df)
        if self.__transformation_func is not None and apply_transformation_func:
            self.__structures_df = self.__transformation_func(self.__structures_df)
    
    def get_structures_df(self):
        """
            Getter method for the __structures_df attribute.
        """
        return self.__structures_df

    def __call__(self):
        results = Parallel(n_jobs=self.num_workers, backend="loky")(
            delayed(_parse_worker)(pkl, configurations_filter=self.__configurations_filter, transformation_func=self.__transformation_func)
            for pkl in tqdm(self.__pkl_gz, desc="Collecting metadata of CSS structures", unit=" .pkl.gz", ncols=200)
        )
        self.__structures_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
