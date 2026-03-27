"""Module with the Scheduler class, which helps to distribute evaluation resources between remote machines."""

__all__ = []


class Scheduler:
    """Class which plans how to distribute resources between remote machines. Only for internal use in the Calculator class."""
    def __init__(self, server_ip: str | list[str] | None = None, username: str | list[str] | None = None, 
                 password: str | list[str] | None = None, max_workers: int | list[int] | None = None):
        """
            Initialization method for the Scheduler class.

            Args:
                server_ip (str | list[str] | None, optional): ip adress/adresses to the remote servers, where calculations will be performed. If server_ip is None
                calculations will be performed on the local machine. Defaults to None.
                username (str | list[str], optional): username/usernames on the remote machine/machines. Must be filled if server_ip is not None. 
                If only one username is passed and server_ip is a list, then this username will be passed to all remote servers. Defaults to None.
                password (str | list[str], optional): paswword/passwords for the user on the remote machine/machines. Must be filled if server_ip is not None
                and server requires password. If only one password is passed and server_ip is a list, then this password will be passed to all remote servers. 
                For passwords storage it is recommended to create .env file and load this password using python dotenv library. Defaults to None.
                max_workers (int | list[int] | None, optional): maximal number of workers on the local machine or server/servers. If None then number of workers
                will be calculated as max_cpu_number (available on machine) - cpu_count (from cmd string) * max_workers. Defaults to None.
        """
        self.__server_ip = server_ip
        self.__username = username
        self.__password = password
        self.__max_workers = max_workers
        
    def __call__(self): # MAX-NODES - sacctmgr show user -s | awk '{sum+=$9} END {print sum}' MAX-CPUss - sacctmgr show user -s | awk '{sum+=$10} END {print sum}' 
        ...