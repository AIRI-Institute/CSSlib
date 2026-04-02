"""Module with the Loader class. Can be used to generate training/test samples, transform data, and prepare it for postprocessing."""

__all__ = [
    'DataLoader',
]


import gzip
import os
import pandas as pd
import pickle
from csslib.exceptions import DataLoaderError
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Any, Callable, Iterator


class DataLoader:
    """
        Data loader of structures for performing VASP calculations and realization of the postprocessing.
    """
    def __init__(self, path: str, num_workers: int = 1, transformation_function: Callable | None = None,
                 save_loaded_data: bool = True, save_loaded_data_filepath: str = "<path>/selected.pkl.gz",
                 copy_unloaded_data: bool = True, copy_unloaded_data_path: str = "<path>/css_unselected"):
        """
            Initialization method for the DataLoader class.

            Args:
                path (str): path to folder/file with .pkl.gz archive/s with structures for loading/processing.
                num_workers (int, optional): number of workers for parallel execution of the DataLoader. Defaults to 1.
                transformation_function (Callable | None, optional): transformation function to be applied to the configurations set.
                As the input function must get pandas.Dataframe object and return transformed pandas.Dataframe object.
                Defaults to None.
                save_loaded_data (bool, optional): if True stores loaded data in the save_loaded_data_filepath.
                Uses only when path variable points on the directory. Otherwise, the flag is always changes to False. Defaults to True.
                save_loaded_data_path (str, optional): path to the file where loaded data will be stored. Defaults to '<path>/selected.pkl.gz'.
                copy_unloaded_data (bool, optional): if True copies unloaded data to the copy_unloaded_data_path. Useful when
                configurations will be iteratively selected from the full css dataset. Uses only when path variable points on the directory.
                Otherwise, the flag is always changes to False. Defaults to True.
                copy_unloaded_data_path (str, optional): path to the folder where unloaded data will be stored. Defaults to '<path>/css_unselected'.

            Raise:
                csslib.exceptions.DataLoaderError: while results folder do not contain .pkl.gz files.
        """
        self.__pkl_gz = None
        self.__unselected_configuration_were_saved = False
        if os.path.isdir(path):
            self.__pkl_gz = [os.path.join(path, archive) for archive in os.listdir(path) if archive.endswith('.pkl.gz')]
            if not self.__pkl_gz:
                raise DataLoaderError(f'There are no .pkl.gz files in the {path} directory.')
            self.__save_loaded_data = save_loaded_data
            self.__copy_unloaded_data = copy_unloaded_data
        elif os.path.isfile(path) and path.endswith('.pkl.gz'):
            self.__pkl_gz = [path]
            self.__save_loaded_data = False
            self.__copy_unloaded_data = False
        else:
            raise DataLoaderError('path variable should be path to the results folder with .pkl.gz files or to the .pkl.gz file!')

        self.num_workers = num_workers
        self.transformation_function = transformation_function
        self.save_loaded_data_filepath = os.path.join(os.path.dirname(path), save_loaded_data_filepath.split('/')[-1]) if save_loaded_data_filepath == "<path>/selected.pkl.gz" else save_loaded_data_filepath
        self.copy_unloaded_data_path = os.path.join(os.path.dirname(path), copy_unloaded_data_path.split('/')[-1]) if copy_unloaded_data_path == "<path>/css_unselected" else copy_unloaded_data_path

        self.__df = self.__load()

    @staticmethod
    def __parse_worker(file_path: str, transformation_function: Callable | None = None,
                       copy_unloaded_data_path: str | None = None) -> pd.DataFrame:
        """
            A worker for ProcessPoolExecutor class used by __load method of the DataLoader class. Reads configurations from .pkl.gz.

            Args:
                file_path (str): full or relative path to the .pkl.gz file.
                transformation_function (Callable | None, optional): transformation function to be applied to the configurations set.
                As the input function must get pandas.Dataframe object and return transformed pandas.Dataframe object.
                Defaults to None.
                copy_unloaded_data_path (str, optional): path to the folder where unloaded data will be stored. Defaults to None.

            Return:
                pandas.DataFrame: table with information about css structures.
        """
        with gzip.open(file_path, "rb") as archive:
            df = pickle.load(archive)

        transformed_df = transformation_function(df) if transformation_function is not None else df
        if transformation_function is not None and copy_unloaded_data_path is not None:
            unloaded_df = df.drop(index=transformed_df.index).reset_index(drop=True)
            if len(unloaded_df):
                unloaded_df.to_pickle(os.path.join(copy_unloaded_data_path, os.path.basename(file_path)))
            else:
                try:
                    os.remove(os.path.join(copy_unloaded_data_path, os.path.basename(file_path)))
                except FileNotFoundError:
                    pass
                except PermissionError:
                    raise DataLoaderError(f'Permission error occured while tried to remove {os.path.join(copy_unloaded_data_path, os.path.basename(file_path))}. Remove it manually.')
        return transformed_df

    def __load(self) -> pd.DataFrame:
        """
            Loads .pkl.gz file/s in serial/parallel mode and stores data in _structures_df attribute.
            If self.

            Return:
                pandas.DataFrame: loaded .pkl.gz table.
        """
        if self.__copy_unloaded_data:
            os.makedirs(self.copy_unloaded_data_path, exist_ok=True)

        if self.num_workers == 1 or len(self.__pkl_gz) == 1:
            results = [self.__parse_worker(pkl, self.transformation_function, self.copy_unloaded_data_path if self.__copy_unloaded_data else None)
                       for pkl in tqdm(self.__pkl_gz, desc="Collecting metadata of CSS structures", unit=" .pkl.gz", ncols=200)]
        else:
            results = Parallel(n_jobs=self.num_workers, backend="loky")(
                delayed(self.__parse_worker)(pkl, self.transformation_function, self.copy_unloaded_data_path if self.__copy_unloaded_data else None)
                for pkl in tqdm(self.__pkl_gz, desc="Collecting metadata of CSS structures", unit=" .pkl.gz", ncols=200)
            )

        df = pd.concat(results, ignore_index=True).reset_index(drop=True) if results else pd.DataFrame()
        if self.__save_loaded_data:
            df.to_pickle(self.save_loaded_data_filepath)
        if self.__copy_unloaded_data:
            self.__unselected_configuration_were_saved = True
        return df

    def apply(self, transformation_function: Callable):
        """
            Sets and applies transformation function.

            Args:
                transformation_function (Callable): transformation function to be applied to the configurations set.
                As the input function must get pandas.Dataframe object and return transformed pandas.Dataframe object.
                Defaults to None.

            Raise:
                csslib.exceptions.DataLoaderError: if _structures_df attribute is not defined.
        """
        if self.__df is None:
            raise DataLoaderError('Data from .pkl.gz is not read yet.')

        self.transformation_function = transformation_function
        self.__df = self.transformation_function(self.__df)

    def save_df(self, filepath: str):
        """
            Saves dataframe in the .pkl.gz file. Useful when df attribute is changed by apply method and new
            dataframe should be saved.

            Args:
                filepath (str): full or relative path to the archive.
        """
        self.__df.to_pickle(filepath)

    def select_add(self, select_path: str | None = None, transformation_function: Callable | None = None,
                   save_merged_df_to: str | None = None):
        """
            Additionaly selects and adds configurations to the df attribute. Can be used in the active learning procedure.

            Args:
                select_path (str | None, optional): path to the folder with .pkl.gz files from which configurations will be selected.
                Can be empty if path parameter in __init__ method was a path to the directory. Defaults to None.
                transformation_function (Callable | None, optional): transformation function to be applied to the loaded data.
                Can be empty if transformation_function attribute was already setup by other methods. Defaults to None.
                save_merged_df_to (str | None, optional): path to the .pkl.gz file in which dataset will be saved. Defaults to None.

            Raise:
                csslib.exceptions.DataLoaderError: if unselected configuration were not saved earlier and select_path parameter is None or
                transformation_function attribute is None and transformation_function parameter is None.
        """
        if not self.__unselected_configuration_were_saved and select_path is None:
            raise DataLoaderError("select_path attribute must be filled when unselected data was not saved earlier.")
        if self.transformation_function is None and transformation_function is None:
            raise DataLoaderError("transformation_function parameter must be filled when it was not set earlier.")

        if not self.__unselected_configuration_were_saved or select_path:
            self.copy_unloaded_data_path = select_path
        self.__pkl_gz = [os.path.join(self.copy_unloaded_data_path, archive) for archive in os.listdir(self.copy_unloaded_data_path) if archive.endswith('.pkl.gz')]

        if transformation_function is not None:
            self.transformation_function = transformation_function

        self.__copy_unloaded_data = True
        self.__save_loaded_data = False

        added_df = self.__load()
        self.__df = pd.concat([self.__df, added_df], ignore_index=True)
        self.save_df(save_merged_df_to if save_merged_df_to is not None else self.save_loaded_data_filepath)

    def set_transformation_function(self, transformation_function: Callable):
        """
            Sets transformation_function attribute.

            Args:
                transformation_function (Callable): transformation function to be applied to the configurations set.
                As the input function must get pandas.Dataframe object and return transformed pandas.Dataframe object.
        """
        self.transformation_function = transformation_function

    def get_df(self) -> pd.DataFrame:
        """
            Getter method for the __structures_df attribute.

            Return:
                pandas.DataFrame: df attribute with information about the css.
        """
        return self.__df
    
    def __getitem__(self, key: Any) -> pd.Series:
        """
            Getitem magic method of the DataLoader class.
            
            Args:
                key (Any): key for the df pandas.DataFrame.
            
            Return:
                pandas.Series: selected column of the df attribute.
        """
        return self.__df[key]
    
    def __setitem__(self, key: Any, value: Any):
        """
            Setitem magic method of the DataLoader class.
            
            Args:
                key (Any): key for the df pandas.DataFrame object.
                value (Any): value to be set in the df pandas.DataFrame object.
        """
        self.__df[key] = value
    
    def __iter__(self) -> Iterator:
        """
            Iter magic method for iteration over df pandas.DataFrame object. Uses pandas.DataFrame.itertuples method.
            
            Return:
                Iterator: pandas named tuple object with index=True.
        """
        return self.__df.itertuples(index=True, name='CSSData')

    def __getattr__(self, name: str) -> Any:
        """
            Getattr magic method of the DataLoader class. Redirects all unknown attributes to the df object.
            
            Args:
                name (str): name of the attribute which should be found.
                
            Raise:
                csslib.exceptions.DataLoaderError: if attribute is not found in the df attribute.
        """
        try:
            return getattr(self.__df, name)
        except AttributeError:
            raise DataLoaderError(f'Attribute {name} is not found for the df protected attribute.')
        
    def __repr__(self):
        """
            Repr magic method of the DataLoader class. Outputs information from the df.describe() method. 
        """
        message = f'DataLoader object located at {hex(id(self))}. Stores the pandas.DataFrame object with the following statistics:\n'
        message += str(self.__df.describe())
        return message
