"""Module with the Scheduler class, which helps to distribute evaluation resources between remote machines."""

__all__ = []


class Scheduler:
    """Class which plans how to distribute resources between remote machines. Only for internal use in the Calculator class."""
    def __init__(self, max_workers: int | dict | None = None):
        """
            Initialization method for the Scheduler class.

            Args:
                max_workers (int | dict | None, optional): maximal number of workers on the local machine or server/servers. If None then number of workers
                will be calculated as max_cpu_number (available on machine) - cpu_count (from cmd string) * max_workers. Defaults to None.
        """
        self.__max_workers = max_workers
        self.__connections = None    
        
    def load_connections(self, connections):
        """
            Loads connections dict and saves it in the connections protected attribute.
            
            Args:
                connections (dict): dictionary with RemoteConnection objects.
        """
        self.__connections = connections
    
    def __call__(self): # MAX-NODES - sacctmgr show user -s | awk '{sum+=$9} END {print sum}' MAX-CPUss - sacctmgr show user -s | awk '{sum+=$10} END {print sum}' 
        ...