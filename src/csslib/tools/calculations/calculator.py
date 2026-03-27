"""Module with the Calculator class for DFT calculations of css structures."""

__all__ = [
    "calculator"
]

import os
from csslib.exceptions import CalculatorError
from csslib.tools.dataloader import DataLoader
from csslib.tools.calculations.inputs import InputSet
from csslib.tools.calculations.worker import Worker, RemoteWorker
from pymatgen.io.vasp import Incar, Poscar, Kpoints, Potcar
from typing import Callable, Any


class Calculator: # add calculation_dir, calculation_inputs, cmd
    """
        Class for automatization of DFT calculations for css dataset. 
    """
    def __init__(self, data: str | DataLoader, inputs: InputSet | Any, calculations_path: str, cmd: str | list[str], server_ip: str | list[str] | None = None, 
                 username: str | list[str] | None = None, password: str | list[str] | None = None, max_workers: int | list[int] | None = None, **dataloader_fields):
        """
            Initialization method for the Calculator class.

            Args:
                data (str | DataLoader): path to file with .pkl.gz archive or DataLoader class object with structures for calculations.
                inputs (InputSet | Any): set of inputs file to be used by workers. Can be any class or list object which contains objects with write_file method.
                calculations_path (str): path to the folder, where calculation files will be placed and read.
                cmd (str | list[str]): command/ list of commands which will be executed by worker.
                server_ip (str | list[str] | None, optional): ip adress/adresses to the remote servers, where calculations will be performed. If server_ip is None
                calculations will be performed on the local machine. Defaults to None.
                username (str | list[str], optional): username/usernames on the remote machine/machines. Must be filled if server_ip is not None. 
                If only one username is passed and server_ip is a list, then this username will be passed to all remote servers. Defaults to None.
                password (str | list[str], optional): paswword/passwords for the user on the remote machine/machines. Must be filled if server_ip is not None
                and server requires password. If only one password is passed and server_ip is a list, then this password will be passed to all remote servers. 
                For passwords storage it is recommended to create .env file and load this password using python dotenv library. Defaults to None.
                max_workers (int | list[int] | None, optional): maximal number of workers on the local machine or server/servers. If None then number of workers
                will be calculated as max_cpu_number (available on machine) - cpu_count (from cmd string) * max_workers. Defaults to None.
            
            Kwargs:
                dataloader_fields: contains information about dataloader fields which should be passed to the DataLoader class if data is str.
            
            Raise:
                csslib.exceptions.CalculatorError: if __init__ method parameters are not fullfilled. For instance, when server_ip is not None and username is None.
        """
        if server_ip is not None and username is None:
            raise CalculatorError('Username must be non-empty if server_ip parameter is passed.')
        
        if isinstance(data, str):
            data = DataLoader(data, **dataloader_fields)
        self.__data = data
        self.__inputs = inputs
        self.__path = calculations_path
        self.__cmd = cmd
        
        self.__worker_class = Worker if server_ip is None else RemoteWorker
        self.__server_ip = server_ip
        self.__username = username
        self.__password = password
        self.__max_workers = max_workers        
