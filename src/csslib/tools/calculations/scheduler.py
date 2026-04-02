"""
    Module with the Scheduler class, which helps to distribute evaluation resources between remote machines.
    Supports servers with the slurm system. IMPORTANT! For slurm systems Scheduler finds maximal submit number (MaxSubmit)
    on the partition from the cmd and verifies that number of cores lower then MaxCPUs value and number of nodes lower than
    MaxNodes value. Therefore, the cmd must contain -p partition flag, otherwise EXCEPTION WILL BE RAISED!
"""

__all__ = []


import paramiko
import subprocess
from csslib.exceptions import SchedulerError, RemoteConnectionError


class Scheduler:
    """Class which plans how to distribute resources between remote machines. Only for internal use in the Calculator class."""
    SUPPORTING_CMD_PREFIXES_MPI = ["mpi", "mpirun", "mpirun.mpich", "mpirun.openmpi", "mpiexec", "mpiexec.hydra", "mpiexec.mpich", "mpiexec.openmpi"]
    SUPPORTING_CMD_PREFIXES_SLURM = ["srun", "sbatch"]
    PARALLEL_FLAGS_MPI = ['-c', '--c', '-n', '--n', '-np', '--np']
    PARALLEL_FLAGS_SLURM_NTASKS = ['-n', '--ntasks']
    PARALLEL_FLAGS_SLURM_CPU_PER_TASK = ['-c', '--cpus-per-task']
    SLURM_PARTITION_FLAGS = ['-p', '--partition']
    SLURM_NODE_FLAGS = ['-N', '--nodes']

    def __init__(self, cmd: str | dict, max_workers: int | dict | None = None, use_local: bool = False):
        """
            Initialization method for the Scheduler class.

            Args:
                cmd (str | dict): cmd command or commands which will be parsed by Scheduler for the resources distribution.
                max_workers (int | dict | None): maximal number of workers on the local machine or server/servers. If None then number of workers
                will be calculated as max_cpu_number (available on machine) - cpu_count (from cmd string) * max_workers.
        """
        self.__cmd = cmd
        self.__max_workers = max_workers
        self.__use_local = use_local
        self.__local_has_slurm = False
        
        self.__cores = None
        self.__workers = dict()
        self.__connections = dict()

        self.__parse_cmd()

    def __verify_prefix(self, prefix: str):
        """
            Verifies the first element of the splitted cmd and raises an exception if this prefix is not in the Scheduler.SUPPORTING_CMD_PREFIXES.

            Args:
                prefix (str): first element of the splitted cmd string.

            Raise:
                csslib.exceptions.SchedulerError: if prefix not in the SUPPORTING_CMD_PREFIXES attribute.
        """
        if prefix not in Scheduler.SUPPORTING_CMD_PREFIXES_MPI and prefix not in Scheduler.SUPPORTING_CMD_PREFIXES_SLURM:
            message = f'Cmd prefix is not in the list of supporting cmd prefixes. Supporting cmd prefixes are: {Scheduler.SUPPORTING_CMD_PREFIXES_MPI} and {Scheduler.SUPPORTING_CMD_PREFIXES_SLURM}. '
            message += "If serial calculation is required use '#' symbol at the start of the cmd."
            raise SchedulerError(message)

    def __analyse_tokens(self, tokens: list) -> int:
        """
            Analyses the given list of tokens and returns the number of cores per server.

            Return:
                int: cores per server.
        """
        cores = None
        if tokens[0][0] == '#':
            cores = 1
        else:
            self.__verify_prefix(tokens[0])
            if tokens[0] in Scheduler.SUPPORTING_CMD_PREFIXES_MPI:
                for indx, token in enumerate(tokens):
                    if token in Scheduler.PARALLEL_FLAGS_MPI:
                        cores = int(tokens[indx + 1])
                if cores is None:
                    cores = -1
            else:
                ntasks, cpu_per_task = 1, 1
                for indx, token in enumerate(tokens):
                    if any([flag in token for flag in Scheduler.PARALLEL_FLAGS_SLURM_NTASKS]):
                        ntasks = int(tokens[indx + 1])
                    elif any([flag in token for flag in Scheduler.PARALLEL_FLAGS_SLURM_CPU_PER_TASK]):
                        cpu_per_task = int(tokens[indx + 1])
                cores = ntasks * cpu_per_task
        return cores

    def __parse_cmd(self):
        """
            Parses str or dict cmd object and extractes number of cores on the one task per server_ip. If cmd contains '#' symbol verification step will be skiped.

            Raise:
                csslib.exceptions.SchedulerError: if the use_local parameter at the __init__ method was set to True and cmd is dict, but the 'local' server name is not exist.  
        """
        if isinstance(self.__cmd, str):
            tokens = self.__cmd.split()
            self.__cores = self.__analyse_tokens(tokens)
        else:
            self.__cores = dict()
            for server in self.__cmd:
                tokens = self.__cmd[server].split()
                self.__cores[server] = self.__analyse_tokens(tokens)
            if 'local' not in self.__cores:
                raise SchedulerError("If use_local is True and cmd is a dictionary, then 'local' key in cmd must exist.")

    def load_connections(self, connections: dict):
        """
            Loads connections dict and saves it in the connections protected attribute.

            Args:
                connections (dict): dictionary with RemoteConnection objects.

            Raise:
                csslib.exceptions.SchedulerError: if at least one loaded server has not contain the SLURM system, but the cmd command for the SLURM system was passed.
        """
        self.__connections = connections
        for server in self.__connections:
            has_slurm = self.__connections[server].has_slurm
            cmd_str_for_slurm = (isinstance(self.__cmd, str) and any(flag in self.__cmd for flag in Scheduler.SUPPORTING_CMD_PREFIXES_SLURM))
            cmd_dict_for_slurm = (isinstance(self.__cmd, dict) and any(flag in self.__cmd[server] for flag in Scheduler.SUPPORTING_CMD_PREFIXES_SLURM))
            if (has_slurm and not (cmd_str_for_slurm or cmd_dict_for_slurm)) or (not has_slurm and (cmd_str_for_slurm or cmd_dict_for_slurm)):
                raise SchedulerError('Mismatch between the cmd and connection is found. If connection has the SLURM system you must use the cmd for SLURM system. If connection has the MPI system you must use the cmd for MPI system.')

    def __check_mpi(self, server: str):
        """
            Checks that MPI command passed through the cmd works on the remote/local machine. Otherwise, raises SchedulerError.
            
            Args:
                server (str): adress of the server which should be checked.
            
            Raise:
                csslib.exceptions.SchedulerError: if MPI command passed through the cmd does not work on the target machine. 
        """
        if (isinstance(self.__cores, dict) and self.__cores[server] != 1) or (isinstance(self.__cores, int) and self.__cores != 1):
            if server == 'local':
                command = self.__cmd.split()[0] if isinstance(self.__cmd, str) else self.__cmd[server].split()[0]
                result = subprocess.run(command, capture_output=True, text=True, shell=True)
                if 'not found' in result.stderr:
                    raise SchedulerError(f'Command {command} is not found on the local. Please double check your cmd!')
            else:
                ssh = self.__connections[server]()['SSH']
                command = self.__cmd.split()[0] if isinstance(self.__cmd, str) else self.__cmd[server].split()[0]
                _, _, stderr = ssh.exec_command(command)
                if b'not found' in stderr.read():
                    raise SchedulerError(f'Command {command} is not found on the server - {server}. Please double check your cmd!')

    def __check_slurm(self, server: str, stdout: paramiko.ChannelFile | str) -> int:
        """
            Checks SLURM system partitions for MaxCPUs, MaxNodes and MaxSubmit parameters. Verifies that MaxCPUs greater or equal to the number of
            cores given in the cmd command. Analogously for the MaxNodes parameter.

            Args:
                server (str): name of the server.
                stdout (paramiko.ChannelFile | str): return of the 'sacctmgr show user -s -P -n format=Partition,MaxNodes,MaxCPUs,MaxSubmit' commnand.

            Raise:
                csslib.exceptions.SchedulerError: if partition flag is not set for the SLURM system or partition from the cmd is not found on the remote/local server.
                Also raises if MaxCPUs or MaxNodes parameters overestimated in the cmd.

            Return:
                int: MaxSubmit parameter if all checks are passed.
        """
        table = dict()
        if not isinstance(stdout, str):
            while True:
                line = stdout.readline().rstrip()
                if not line:
                    break
                else:
                    splitted_line = line.split('|')
                    for indx, el in enumerate(splitted_line):
                        if indx == 0:
                            table[el] = {'MaxNodes': None, 'MaxCPUs': None, 'MaxSubmit': None}
                        else:
                            try:
                                match indx:
                                    case 1:
                                        table[splitted_line[0]]['MaxNodes'] = int(el)
                                    case 2:
                                        table[splitted_line[0]]['MaxCPUs'] = int(el)
                                    case 3:
                                        table[splitted_line[0]]['MaxSubmit'] = int(el)
                            except ValueError:
                                pass
        else:
            data = map(lambda x: x.split('|'), stdout.split('\n'))
            for line in data:
                table[line[0]] = {'MaxNodes': None, 'MaxCPUs': None, 'MaxSubmit': None}
                for indx, el in enumerate(line[1:]):
                    try:
                        match indx:
                            case 0:
                                table[splitted_line[0]]['MaxNodes'] = int(el)
                            case 1:
                                    table[splitted_line[0]]['MaxCPUs'] = int(el)
                            case 2:
                                table[splitted_line[0]]['MaxSubmit'] = int(el)
                    except ValueError:
                        pass
        cmd = self.__cmd if isinstance(self.__cmd, str) else self.__cmd[server]
        if not any([flag in cmd for flag in Scheduler.SLURM_PARTITION_FLAGS]):
            raise SchedulerError('For SLURM systems partition flag is mandatory. Please rewrite your sbatch or srun command with -p or --partition flags!')
        nodes_number, nodes_indx = None, -1
        for flag in Scheduler.SLURM_NODE_FLAGS:
            nodes_pos = cmd.index(flag) + len(flag) + 1 if flag in cmd else nodes_pos
            if flag in cmd and nodes_pos > nodes_indx:
                nodes_indx = nodes_pos
                nodes_number = int(cmd[nodes_indx:].split()[0])
        partition = cmd[cmd.index('--partition') + len('--partition') + 1:].split()[0] if '--partition' in cmd else cmd[cmd.index('-p') + len('-p') + 1:].split()[0]
        if partition not in table:
            raise SchedulerError('Partition from the cmd is not found on the remote/local server.')
        if table[partition]['MaxCPUs'] is not None and ((isinstance(self.__cores, dict) and table[partition]['MaxCPUs'] < self.__cores[server]) or (isinstance(self.__cores, int) and table[partition]['MaxCPUs'] < self.__cores)):
            raise SchedulerError('MaxCPUs parameter on the SLURM system is lower than passed through the cmd cores number.')
        if table[partition]['MaxNodes'] is not None and table[partition]['MaxNodes'] < nodes_number:
            raise SchedulerError('MaxNodes parameter on the SLURM system is lower than passed through the cmd nodes number.')
        return table[partition]['MaxSubmit']

    @staticmethod
    def __uniform_reduce(workers: dict, remain_workers_to_remove: int) -> tuple[int, dict]:
        """
            Performes uniform reduce of the workers dictionaries with equal number of cores.

            Args:
                workers (dict): dictionary with number of workers per server.
                remain_workers_to_remove (int): number of workers which should be removed after removing cycles.

            Return:

        """
        workers_number = sum(workers.values())
        if remain_workers_to_remove >= workers_number:
            for server in workers:
                workers[server] = 0
            remain_workers_to_remove -= workers_number
        else:
            while remain_workers_to_remove:
                max_workers = max(workers.values())
                for server in workers:
                    if workers[server] == max_workers and workers[server] > 0:
                        workers[server] -= 1
                        remain_workers_to_remove -= 1
                        break
        return remain_workers_to_remove, workers

    def __reduce_workers_number(self):
        """
            Reduces number of workers if max_workers attribute lower than the sum of workers evaluated for max_workers = None.
            Makes priority for servers where the cmd command contains maximal number of cores.
        """
        if isinstance(self.__cores, int):
            _, self.__workers = self.__uniform_reduce(self.__workers, sum(self.__workers.values()) - self.__max_workers)
            [self.__workers.pop(server) for server in self.__workers if self.__workers[server] == 0]
        elif isinstance(self.__cores, dict):
            workers_number = sum(self.__workers.values())
            remain_workers_to_remove = workers_number - self.__max_workers
            while remain_workers_to_remove:
                min_cores = min([self.__cores[server] for server in self.__cores if (server in self.__workers and self.__cores[server] != -1)])
                workers_to_reduce = {server: value for server, value in self.__workers.items() if self.__cores[server] == min_cores}
                remain_workers_to_remove, workers = self.__uniform_reduce(workers_to_reduce, remain_workers_to_remove)
                for server in workers:
                    if workers[server]:
                        self.__workers[server] = workers[server]
                    else:
                        self.__workers.pop(server)

    def distribute_workers(self):
        """
            Distributes workers between the given connections dictionary. Sets a number of workers per server.

            Raise:
                csslib.exceptions.SchedulerError: if method was called before load_connections.
        """
        available_resources = dict()
        if self.__use_local:
            result = subprocess.run('squeue', capture_output=True, text=True, shell=True) 
            if 'not found' in result.stderr:
                result = subprocess.run('nproc', capture_output=True, text=True, shell=True)
                self.__check_mpi('local')
                available_resources['local'] = int(result.stdout)
            else:
                result = subprocess.run('sacctmgr show user -s -P -n format=Partition,MaxNodes,MaxCPUs,MaxSubmit', capture_output=True, text=True, shell=True)
                available_resources['local'] = self.__check_slurm('local', result.stdout)
                self.__local_has_slurm = True
            cmd_str_for_slurm = (isinstance(self.__cmd, str) and any(flag in self.__cmd for flag in Scheduler.SUPPORTING_CMD_PREFIXES_SLURM))
            cmd_dict_for_slurm = (isinstance(self.__cmd, dict) and any(flag in self.__cmd['local'] for flag in Scheduler.SUPPORTING_CMD_PREFIXES_SLURM))
            if (self.__local_has_slurm and not (cmd_str_for_slurm or cmd_dict_for_slurm)) or (not self.__local_has_slurm and (cmd_str_for_slurm or cmd_dict_for_slurm)):
                raise SchedulerError('Mismatch between the cmd and connection is found. If connection has the SLURM system you must use the cmd for SLURM system. If connection has the MPI system you must use the cmd for MPI system.')

        
        for server in self.__connections:
            connection = self.__connections.get(server)
            ssh = connection()['SSH']
            if not connection.has_slurm:
                try:
                    _, stdout, _ = ssh.exec_command('nproc')
                    self.__check_mpi(server)
                except EOFError:
                    raise RemoteConnectionError('EOFError. This error can awake if server which you entered is unknown. Before the use of the csslib please append all servers to known hosts.')
                nproc = int(stdout.read())
                available_resources[server] = nproc
            else:
                try:
                    _, stdout, _ = ssh.exec_command('sacctmgr show user -s -P -n format=Partition,MaxNodes,MaxCPUs,MaxSubmit')
                except EOFError:
                    raise RemoteConnectionError('EOFError. This error can awake if server which you entered is unknown. Before the use of the csslib please append all servers to known hosts.')
                available_resources[server] = self.__check_slurm(server, stdout)
        if self.__max_workers is None or isinstance(self.__max_workers, int):
            for server in available_resources:
                if (server == 'local' and not self.__local_has_slurm) or not self.__connections[server].has_slurm:
                    workers = 0
                    cores = self.__cores[server] if isinstance(self.__cores, dict) else self.__cores
                    if cores == -1:
                        self.__workers[server] = 1
                    else:
                        while cores * workers <= available_resources[server]:
                            self.__workers[server] = workers
                            workers += 1
                else:
                    self.__workers[server] = available_resources[server]
            if isinstance(self.__max_workers, int):
                sum_workers = sum(self.__workers.values())
                if sum_workers > self.__max_workers:
                    self.__reduce_workers_number()
        elif isinstance(self.__max_workers, dict):
            for server in available_resources:
                if (server == 'local' and not self.__local_has_slurm) or not self.__connections[server].has_slurm:
                    workers = 0
                    cores = self.__cores[server] if isinstance(self.__cores, dict) else self.__cores
                    if cores == -1 and self.__max_workers[server]:
                        self.__workers[server] = 1
                    else:
                        while cores * workers <= available_resources[server] and workers <= self.__max_workers[server]:
                            self.__workers[server] = workers
                            workers += 1
                else:
                    self.__workers[server] = available_resources[server] if available_resources[server] <= self.__max_workers[server] else self.__max_workers[server]

    def __call__(self, connections: dict | None = None):
        """
            Call method of the Scheduler class. Loads connections if they are passed through __call__, distributes workers and returns the resources distribution.

            Args:
                connections (dict | None, optional): dictionary with connections to load into Scheduler. Defaults to None.
        """
        if connections is not None:
            self.load_connections(connections)
        self.distribute_workers()
        return self.__workers
