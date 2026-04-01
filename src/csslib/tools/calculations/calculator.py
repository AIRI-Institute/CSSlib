"""Module with the Calculator class for DFT calculations of css structures."""

__all__ = [
    "Calculator"
]

import os
from csslib.exceptions import CalculatorError
from csslib.tools.dataloader import DataLoader
from csslib.tools.calculations.inputs import InputSet
from csslib.tools.calculations.remote import RemoteConnection, ConnectionStatus
from csslib.tools.calculations.scheduler import Scheduler
from csslib.tools.calculations.worker import Worker, RemoteWorker
from enum import Enum
from pymatgen.io.vasp import Incar, Poscar, Kpoints, Potcar
from typing import Callable, Any


class CalculationStatus(Enum):
    """
        Class for calculation status trace.

        Attributes:
            PENDING: if calculation is in the waiting list.
            STARTED: if calculation is started.
            COMPLETED: if calculation is completed.
    """
    PENDING = 0
    STARTED = 1
    COMPLETED = 2


class Calculator:
    """
        Class for automatization of DFT calculations for css dataset.
    """
    def __init__(self, data: str | DataLoader, inputs: InputSet | Any, calculations_path: str, cmd: str | list[str],
                 server_ip: str | list[str] | None = None, port: int | list[int] = 22, username: str | list[str] | None = None, password: str | list[str] | None = None,
                 host_keys_path: str | None = None, max_workers: int | list[int] | None = None, max_connection_attempts: int = 5,
                 use_sftp: bool = False, **dataloader_fields):
        """
            Initialization method for the Calculator class.

            Args:
                data (str | DataLoader): path to file with .pkl.gz archive or DataLoader class object with structures for calculations.
                inputs (InputSet | Any): set of inputs file to be used by workers. Can be any class or list object which contains objects with write_file method.
                calculations_path (str): path to the folder, where calculation files will be placed and read.
                cmd (str | list[str]): command/ list of commands which will be executed by worker. Currently, CSSlib supports cmd which starts with:
                mpi, mpirun, mpirun.openmpi, mpiexec, mpiexec.hydra, srun, sbatch. If calculations will perform on one core and will start with binary name
                use '#' symbol before the cmd.
                server_ip (str | list[str] | None, optional): ip adress/adresses of the remote servers, where calculations will be performed. If server_ip is None
                calculations will be performed on the local machine. Defaults to None.
                port (int | list[int]): port for which connection will be set. Defaults to 22.
                username (str | list[str], optional): username/usernames on the remote machine/machines. Must be filled if server_ip is not None.
                If only one username is passed and server_ip is a list, then this username will be passed to all remote servers. Defaults to None.
                password (str | list[str], optional): password/passwords for the user on the remote machine/machines. Must be filled if server_ip is not None
                and server requires password. If only one password is passed and server_ip is a list, then this password will be passed to all remote servers.
                For passwords storage it is recommended to create .env file and load this password using python dotenv library. Defaults to None.
                host_keys_path (str | None, optional): path to the folder with hostkeys. Defaults to None.
                max_workers (int | list[int] | None, optional): maximal number of workers on the local machine or server/servers. If None then number of workers
                will be calculated as max_cpu_number (available on machine) - cpu_count (from cmd string) * max_workers. Defaults to None.
                max_connection_attempts (int, optional): number of connection attempts of each RemoteConnection instances. Defaults to 5.
                use_sftp (bool): if True sftp tunnel will be used for files sharing with remote machines else scp will be used. Defaults to False.

            Kwargs:
                dataloader_fields: contains information about dataloader fields which should be passed to the DataLoader class if data is str.
        """
        self.__data = data
        self.__inputs = inputs
        self.__path = calculations_path
        self.__cmd = cmd

        self.__host_keys_path = host_keys_path
        self.__server_ip = server_ip
        self.__port = port
        self.__username = username
        self.__password = password
        self.__max_workers = max_workers
        self.__max_connection_attempts = max_connection_attempts
        self.__use_sftp = use_sftp
        self.__validate_init_parameters()
        self.__prepare_init_parameters()

        if isinstance(data, str):
            self.__data = DataLoader(data, **dataloader_fields)
        self.__connections = dict()
        self.__scheduler = Scheduler(cmd=self.__cmd, max_workers=self.__max_workers)
        self.__worker_class = Worker if self.__server_ip is None else RemoteWorker
        self.__workers_distribution = dict()
        self.__workers = dict()

    def _connect(self):
        """
            Creates RemoteConnection class instances and saves it as a list of connections.

            Raise:
                csslib.exceptions.CalculatorError: if no one connection was successful.
        """
        self.__connections = []
        for connection_info in zip(self.__server_ip, self.__port, self.__username, self.__password):
            connection = RemoteConnection(*connection_info, self.__host_keys_path, self.__max_connection_attempts, self.__use_sftp)
            if connection.connection_status == ConnectionStatus.CONNECTED:
                self.__connections[connection_info[0]] = connection
        if not self.__connections:
            raise CalculatorError('No one connection was successful. Please verify your internet connection or connection settings.')

    def __validate_init_parameters(self):
        """
            Validates data about connection to the remote server/servers.

            Raise:
                csslib.exceptions.CalculatorError: if __init__ method parameters are not fullfilled. For instance, when server_ip is not None and username is None.
        """
        if isinstance(self.__server_ip, str):
            if not isinstance(self.__port, int):
                raise CalculatorError('Only one port should be written when one server_ip is passed.')
            if not isinstance(self.__username, str):
                raise CalculatorError('Only one username should be written when one server_ip is passed.')
            if isinstance(self.__password, list):
                raise CalculatorError('One password or nothing should be written when one server_ip is passed.')
            if isinstance(self.__max_workers, list):
                raise CalculatorError('Max workers parameter should be int or None if only one server_ip is passed.')
            if isinstance(self.__cmd, list):
                raise CalculatorError('Cmd command parameter should be str if only one server_ip is passed.')
        elif isinstance(self.__server_ip, list):
            if isinstance(self.__port, list) and len(self.__port) != len(self.__server_ip):
                raise CalculatorError('Port should be int or list[int] with the size equal to the size of server_ip list.')
            if not isinstance(self.__username, str) or (isinstance(self.__username, list) and len(self.__username) != len(self.__server_ip)):
                raise CalculatorError('Username should be str or list[str] with the size equal to the size of server_ip list.')
            if isinstance(self.__password, list) and len(self.__password) != len(self.__server_ip):
                raise CalculatorError('Password should be str, list[str] with the size equal to the size of server_ip list or None.')
            if isinstance(self.__max_workers, list) and len(self.__max_workers) != len(self.__server_ip):
                raise CalculatorError('Max workers parameter should be int or list[int] with the size equal to the size of server_ip list or None.')
            if isinstance(self.__cmd, list) and len(self.__cmd) != len(self.__server_ip):
                raise CalculatorError('Cmd command parameter should be str or list[str] with the size equal to the size of server_ip list.')

    def __prepare_init_parameters(self):
        """
            Transformes data about connections to list objects for the next convinient use of the connect method.
        """
        if isinstance(self.__server_ip, list):
            if isinstance(self.__username, str):
                self.__username = [self.__username for _ in self.__server_ip]
            if isinstance(self.__port, int):
                self.__port = [self.__port for _ in self.__server_ip]
            if self.__password is None or isinstance(self.__password, str):
                self.__password = [self.__password for _ in self.__password]
            if isinstance(self.__max_workers, list):
                self.__max_workers = {server_ip: self.__max_workers[indx] for indx, server_ip in enumerate(self.__server_ip)}
            if isinstance(self.__cmd, list):
                self.__cmd = {server_ip: self.__cmd[indx] for indx, server_ip in enumerate(self.__server_ip)}
        elif isinstance(self.__server_ip, str):
            self.__server_ip = [self.__server_ip]
            self.__username = [self.__username]
            self.__port = [self.__port]
            self.__password = [self.__password]

    def run(self):
        """
            Starts the cycle of DFT calculations for the data stored in DataFrame located at the protected df class attribure.
        """
        self._connect()
        self.__scheduler.load_connections(self.__connections)
        self.__workers_distribution = self.__scheduler.distribute_workers()
