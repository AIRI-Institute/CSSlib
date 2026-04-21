"""Module with the RemoteConnection class for interaction with the remote server. Contains methods for establish SSH and SFTP connections."""

__all__ = [
    "RemoteConfiguration",
]


import paramiko
import scp
import socket
from csslib.exceptions import RemoteConnectionError
from csslib.logging_ import get_tools_logger
from dataclasses import dataclass
from enum import Enum

logger = get_tools_logger("calculations.remote")


@dataclass
class RemoteConfiguration:
    """
        Dataclass with an information about the remote connection.

        Attributes:
            server_ip (str): ip adress of the remote server, where calculations will be performed.
            port (int): port for which connection will be set. Defaults to 22.
            username (str): username on the remote machine.
            password (str | None): password for the user on the remote machine. Defaults to None.
            passphrase (str | None): passphrase used for decrypting private keys. Defaults to None.
            pkey (str | None): an optional private key to use for authentication. Defaults to None.
            key_filename (str | None): the filename of optional private key and certs to try for authentication. Defaults to None.
            host_keys_path (str | None): path to the folder with hostkeys. Defaults to None.
            connection_attempts (int): number of connnection attempts for situations when connection cannot be established by several reasons. Defaults to 5.
            use_sftp (bool): if True sftp tunnel will be used for files sharing with remote machines else scp will be used. Defaults to False.
    """
    server_ip: str
    username: str
    port: int = 22
    password: str | None = None
    passphrase: str | None = None
    pkey: str | None = None
    key_filename: str | None = None
    host_keys_path: str | None = None
    connection_attempts: int = 5
    use_sftp: bool = False


class ConnectionStatus(Enum):
    NONINITIALIZED = 0
    FAILED = 1
    CONNECTED = 2


class RemoteConnection:
    """
        Internal class of the CSSlib for interaction with the remote server. Opens one instance of SSH and SFTP connections.
    """
    def __init__(self, configuration: RemoteConfiguration):
        """
            Initialization method for the RemoteConnection class.

            Args:
                configuration (RemoteConfiguration): configuration dataclass with the information about RemoteConnection.
        """
        self.__configuration = configuration

        self.__ssh_client = paramiko.SSHClient()
        self.__file_sharing_client = None

        if self.__configuration.host_keys_path is None:
            self.__ssh_client.load_system_host_keys()
        else:
            self.__ssh_client.load_host_keys(self.__configuration.host_keys_path)

        self.connection_status = ConnectionStatus.NONINITIALIZED
        self.has_slurm = False
        while (self.connection_status != ConnectionStatus.CONNECTED and self.__configuration.connection_attempts != 0):
            logger.info("Trying to connect to %s:%s as %s.", self.__configuration.server_ip, self.__configuration.port, self.__configuration.username)
            self.connect()
            self.__configuration.connection_attempts -= 1

    def connect(self):
        """
            Method for establishing a connection to a remote server.

            Raise:
                csslib.exception.RemoteConnectionError: when different unresolvable error types occurs.
        """
        try:
            self.__ssh_client.connect(
                hostname=self.__configuration.server_ip, 
                username=self.__configuration.username, 
                port=self.__configuration.port, 
                password=self.__configuration.password,
                passphrase=self.__configuration.passphrase,
                pkey=self.__configuration.pkey,
                key_filename=self.__configuration.key_filename
            )
            self.connection_status = ConnectionStatus.CONNECTED
            self.__open_sftp() if self.__configuration.use_sftp else self.__open_scp()
            self.__check_slurm()
            logger.info("Connection to %s is established. SLURM=%s. IS_SFTP=%s", self.__configuration.server_ip, self.has_slurm, self.__configuration.use_sftp)
        except paramiko.AuthenticationException:
            message = f'Authentication failed when tried to connect to {self.__configuration.server_ip} on port {self.__configuration.port} for user - {self.__configuration.username}. '
            message += 'Incorrect username/password/private key or key format (e.g., PuTTY .ppk instead of OpenSSH format). '
            message += 'Server expects a different auth method!'
            logger.exception("Authentication failed for %s:%s.", self.__configuration.server_ip, self.__configuration.port)
            raise RemoteConnectionError(message)
        except paramiko.BadHostKeyException:
            logger.exception("Host key mismatch for %s:%s.", self.__configuration.server_ip, self.__configuration.port)
            raise RemoteConnectionError(f"Host key mismatch when tried to connect to {self.__configuration.server_ip} on port {self.__configuration.port} for user - {self.__configuration.username}. The server's host key is unknown or has changed.")
        except (paramiko.SSHException, socket.error, socket.gaierror) as e:
            self.connection_status = ConnectionStatus.FAILED
            logger.warning("Connection attempt to %s:%s failed: %s", self.__configuration.server_ip, self.__configuration.port, e)

    def __open_sftp(self):
        """
            Opens sftp connection to the remote server.
        """
        self.__file_sharing_client = self.__ssh_client.open_sftp()
    def __open_scp(self):
        """
            Opens scp connection to the remote server.
        """
        self.__file_sharing_client = scp.SCPClient(self.__ssh_client.get_transport())

    def __check_slurm(self):
        """
            Checks whether server has SLURM system.
        """
        _, _, stderr = self.__ssh_client.exec_command('squeue')
        if b'not found' not in stderr.read():
            self.has_slurm = True

    def __call__(self):
        """
            Returns instances of SSH and SFTP/SCP clients when the class is called.

            Return:
                tuple[SSHClient, SFTPClient | SCPClient]: SSH and SFTP/SCP clients.
        """
        return {'SSH': self.__ssh_client, 'SFTP/SCP': self.__file_sharing_client}

    def __del__(self):
        """
            Closes SSH and SFTP/SCP connections to the remote server when RemoteConnection object destroys.
        """
        try:
            if self.__file_sharing_client is not None:
                self.__file_sharing_client.close()
        finally:
            self.__ssh_client.close()
