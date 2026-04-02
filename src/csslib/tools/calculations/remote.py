"""Module with the RemoteConnection class for interaction with the remote server. Contains methods for establish SSH and SFTP connections."""

__all__ = []


import paramiko
import scp
import socket
from csslib.exceptions import RemoteConnectionError
from enum import Enum


class ConnectionStatus(Enum):
    NONINITIALIZED = 0
    FAILED = 1
    CONNECTED = 2


class RemoteConnection:
    """
        Internal class of the CSSlib for interaction with the remote server. Opens one instance of SSH and SFTP connections.
    """
    def __init__(self, server_ip: str, port: int, username: str, password: str | None, host_keys_path: str | None,
                 connection_attempts: int, use_sftp: bool = False):
        """
            Initialization method for the RemoteConnection class.

            Args:
                server_ip (str): ip adress of the remote server, where calculations will be performed.
                port (int): port for which connection will be set.
                username (str): username on the remote machine.
                password (str | None): password for the user on the remote machine.
                host_keys_path (str | None): path to the folder with hostkeys.
                connection_attempts (int): number of connnection attempts for situations when connection cannot be established by several reasons.
                use_sftp (bool): if True sftp tunnel will be used for files sharing with remote machines else scp will be used. Defaults to False.
        """
        self.__hostname = server_ip
        self.__port = port
        self.__username = username
        self.__password = password
        self.__use_sftp = use_sftp

        self.__ssh_client = paramiko.SSHClient()
        self.__file_sharing_client = None

        if host_keys_path is None:
            self.__ssh_client.load_system_host_keys()
        else:
            self.__ssh_client.load_host_keys(host_keys_path)

        self.connection_status = ConnectionStatus.NONINITIALIZED
        self.has_slurm = False
        while (self.connection_status != ConnectionStatus.CONNECTED and connection_attempts != 0):
            self.connect()
            connection_attempts -= 1

    def connect(self):
        """
            Method for establishing a connection to a remote server.

            Raise:
                csslib.exception.RemoteConnectionError: when different unresolvable error types occurs.
        """
        try:
            self.__ssh_client.connect(hostname=self.__hostname, username=self.__username, port=self.__port, password=self.__password)
            self.connection_status = ConnectionStatus.CONNECTED
            self.__open_sftp() if self.__use_sftp else self.__open_scp()
            self.__check_slurm()
        except paramiko.AuthenticationException:
            message = f'Authentication failed when tried to connect to {self.__hostname} on port {self.__port} for user - {self.__username}. '
            message += 'Incorrect username/password/private key or key format (e.g., PuTTY .ppk instead of OpenSSH format). '
            message += 'Server expects a different auth method!'
            # self.logget.exception("Authentication failed")
            raise RemoteConnectionError(message)
        except paramiko.BadHostKeyException:
            # self.logger.exception(f"Host key mismatch.")
            raise RemoteConnectionError(f"Host key mismatch when tried to connect to {self.__hostname} on port {self.__port} for user - {self.__username}. The server's host key is unknown or has changed.")
        except (paramiko.SSHException, socket.error, socket.gaierror) as e:
            # self.logger.warning(f"SSH issue: {e}")
            self.connection_status = ConnectionStatus.FAILED
            # self.logger.warning(f"Socket/Network error: {e}")

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
        self.__file_sharing_client.close()
        self.__ssh_client.close()
