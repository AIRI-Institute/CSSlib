"""
    Module with the Scheduler class, which helps to distribute evaluation resources between remote machines.
    Supports MPI-style launchers and SLURM clusters.
"""

__all__ = []


import getpass
import os
import shlex
import subprocess

from csslib.exceptions import RemoteConnectionError, SchedulerError


class Scheduler:
    """Class which plans how to distribute resources between remote machines. Only for internal use in the Calculator class."""

    SUPPORTING_CMD_PREFIXES_MPI = [
        "mpi",
        "mpirun",
        "mpirun.mpich",
        "mpirun.openmpi",
        "mpiexec",
        "mpiexec.hydra",
        "mpiexec.mpich",
        "mpiexec.openmpi",
    ]
    SUPPORTING_CMD_PREFIXES_SLURM = ["srun", "sbatch"]
    PARALLEL_FLAGS_MPI = ["-c", "--c", "-n", "--n", "-np", "--np"]
    PARALLEL_FLAGS_SLURM_NTASKS = ["-n", "--ntasks"]
    PARALLEL_FLAGS_SLURM_CPU_PER_TASK = ["-c", "--cpus-per-task"]
    SLURM_PARTITION_FLAGS = ["-p", "--partition"]
    SLURM_NODE_FLAGS = ["-N", "--nodes"]
    SLURM_ACCOUNTING_NONE_MAX_SUBMIT = 9999
    SLURM_UNAVAILABLE_NODE_STATES = ("down", "drain", "drng", "fail", "maint", "resv")

    def __init__(self, cmd: str | dict, structures_number: int, max_workers: int | dict | None = None, use_local: bool = False):
        """
            Initialization method for the Scheduler class.

            Args:
                cmd (str | dict): cmd command or commands which will be parsed by Scheduler for the resources distribution.
                structures_number (int): number of structures to calculate.
                max_workers (int | dict | None): maximal number of workers on the local machine or server/servers. If None then max_workers
                will be set to structures_number.
                use_local (bool, optional): if True the local machine will be used for calculations else only remote servers will be used. Defaults to False.
        """
        self.__cmd = cmd
        self.__structures_number = structures_number
        self.__max_workers = max_workers if max_workers is not None else self.__structures_number
        self.__use_local = use_local
        self.__local_has_slurm = False

        self.__cores = None
        self.__workers = {}
        self.__connections = {}

        self.__parse_cmd()

    @staticmethod
    def __split_cmd(cmd: str) -> list[str]:
        return shlex.split(cmd)

    @staticmethod
    def __get_option_value(tokens: list[str], flags: list[str], default: int | str | None = None) -> int | str | None:
        for indx, token in enumerate(tokens):
            for flag in flags:
                if token == flag:
                    if indx + 1 < len(tokens):
                        return tokens[indx + 1]
                elif token.startswith(f"{flag}="):
                    return token.split("=", maxsplit=1)[1]
                elif len(flag) == 2 and token.startswith(flag) and token != flag:
                    return token[len(flag):]
        return default

    @staticmethod
    def __run_local(command: str) -> tuple[str, str, int]:
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        return result.stdout.strip(), result.stderr.strip(), result.returncode

    def __run_remote(self, server: str, command: str) -> tuple[str, str, int]:
        ssh = self.__connections[server]()["SSH"]
        try:
            _, stdout, stderr = ssh.exec_command(command)
        except EOFError as exc:
            message = "EOFError. This error can awake if server which you entered is unknown. "
            message += "Before the use of the csslib please append all servers to known hosts."
            raise RemoteConnectionError(message) from exc
        stdout_text = stdout.read().decode("utf-8", errors="ignore").strip()
        stderr_text = stderr.read().decode("utf-8", errors="ignore").strip()
        return stdout_text, stderr_text, stdout.channel.recv_exit_status()

    def __run_command(self, server: str, command: str) -> tuple[str, str, int]:
        if server == "local":
            return self.__run_local(command)
        return self.__run_remote(server, command)

    def __get_cmd(self, server: str) -> str:
        return self.__cmd if isinstance(self.__cmd, str) else self.__cmd[server]

    @staticmethod
    def __get_prefix(tokens: list[str]) -> str:
        if not tokens:
            raise SchedulerError("Cmd must not be empty.")
        return tokens[0][1:] if tokens[0].startswith("#") else tokens[0]

    def __verify_prefix(self, prefix: str):
        """
            Verifies the first element of the splitted cmd and raises an exception if this prefix is not in the Scheduler.SUPPORTING_CMD_PREFIXES.

            Args:
                prefix (str): first element of the splitted cmd string.

            Raise:
                csslib.exceptions.SchedulerError: if prefix not in the SUPPORTING_CMD_PREFIXES attribute.
        """
        if prefix not in Scheduler.SUPPORTING_CMD_PREFIXES_MPI and prefix not in Scheduler.SUPPORTING_CMD_PREFIXES_SLURM:
            message = "Cmd prefix is not in the list of supporting cmd prefixes. "
            message += f"Supporting cmd prefixes are: {Scheduler.SUPPORTING_CMD_PREFIXES_MPI} and {Scheduler.SUPPORTING_CMD_PREFIXES_SLURM}. "
            message += "If serial calculation is required use '#' symbol at the start of the cmd."
            raise SchedulerError(message)

    def __analyse_tokens(self, tokens: list[str]) -> int:
        """
            Analyses the given list of tokens and returns the number of cores per server.

            Return:
                int: cores per server.
        """
        if not tokens:
            raise SchedulerError("Cmd must not be empty.")
        if tokens[0].startswith("#"):
            return 1

        prefix = self.__get_prefix(tokens)
        self.__verify_prefix(prefix)
        if prefix in Scheduler.SUPPORTING_CMD_PREFIXES_MPI:
            cores = self.__get_option_value(tokens, Scheduler.PARALLEL_FLAGS_MPI)
            return int(cores) if cores is not None else -1

        ntasks = int(self.__get_option_value(tokens, Scheduler.PARALLEL_FLAGS_SLURM_NTASKS, 1))
        cpu_per_task = int(self.__get_option_value(tokens, Scheduler.PARALLEL_FLAGS_SLURM_CPU_PER_TASK, 1))
        return ntasks * cpu_per_task

    @staticmethod
    def __parse_slurm_limits(stdout: str) -> dict:
        table = {}
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            splitted_line = line.split("|")
            partition = splitted_line[0]
            table[partition] = {"MaxNodes": None, "MaxCPUs": None, "MaxSubmit": None}
            for key, value in zip(("MaxNodes", "MaxCPUs", "MaxSubmit"), splitted_line[1:4]):
                try:
                    table[partition][key] = int(value)
                except (TypeError, ValueError):
                    continue
        return table

    def __parse_cmd(self):
        """
            Parses str or dict cmd object and extracts number of cores on the one task per server.

            Raise:
                csslib.exceptions.SchedulerError: if use_local=True and cmd is dict, but the 'local' server name does not exist.
        """
        if isinstance(self.__cmd, str):
            self.__cores = self.__analyse_tokens(self.__split_cmd(self.__cmd))
            return

        self.__cores = {}
        for server, command in self.__cmd.items():
            self.__cores[server] = self.__analyse_tokens(self.__split_cmd(command))
        if self.__use_local and "local" not in self.__cores:
            raise SchedulerError("If use_local is True and cmd is a dictionary, then 'local' key in the cmd must exist.")

    def load_connections(self, connections: dict):
        """
            Loads connections dict and saves it in the connections protected attribute.

            Args:
                connections (dict): dictionary with RemoteConnection objects.

            Raise:
                csslib.exceptions.SchedulerError: if at least one loaded server has not contain the SLURM system, but the cmd command for the SLURM system was passed.
        """
        self.__connections = connections or {}
        for server, connection in self.__connections.items():
            has_slurm = connection.has_slurm
            cmd_for_server = self.__get_cmd(server)
            cmd_is_slurm = self.__get_prefix(self.__split_cmd(cmd_for_server)) in Scheduler.SUPPORTING_CMD_PREFIXES_SLURM
            if has_slurm != cmd_is_slurm:
                message = "Mismatch between the cmd and connection is found. "
                message += "If connection has the SLURM system you must use the cmd for SLURM system. "
                message += "If connection has the MPI system you must use the cmd for MPI system."
                raise SchedulerError(message)

    def __check_mpi(self, server: str):
        """
            Checks that MPI command passed through the cmd works on the remote/local machine. Otherwise, raises SchedulerError.

            Args:
                server (str): address of the server which should be checked.
        """
        cores = self.__cores[server] if isinstance(self.__cores, dict) else self.__cores
        if cores == 1:
            return

        command = self.__get_prefix(self.__split_cmd(self.__get_cmd(server)))
        if server == "local":
            probe_command = f"where.exe {command}" if os.name == "nt" else f"command -v {command}"
            _, _, returncode = self.__run_local(probe_command)
        else:
            _, _, returncode = self.__run_remote(server, f"command -v {command}")
        if returncode != 0:
            place = "local" if server == "local" else f"server - {server}"
            raise SchedulerError(f"Command {command} is not found on the {place}. Please double check your cmd!")

    def __get_slurm_partition(self, server: str) -> str:
        cmd = self.__get_cmd(server)
        tokens = self.__split_cmd(cmd)
        partition = self.__get_option_value(tokens, Scheduler.SLURM_PARTITION_FLAGS)
        if partition is None:
            raise SchedulerError("For SLURM systems partition flag is mandatory. Please rewrite your sbatch or srun command with -p or --partition flags!")
        return str(partition)

    def __get_slurm_nodes(self, server: str) -> int:
        tokens = self.__split_cmd(self.__get_cmd(server))
        nodes_number = self.__get_option_value(tokens, Scheduler.SLURM_NODE_FLAGS, 1)
        return int(nodes_number)

    def __get_slurm_limits(self, server: str) -> tuple[dict, bool]:
        command = "sacctmgr show user -s -P -n format=Partition,MaxNodes,MaxCPUs,MaxSubmit"
        stdout, stderr, returncode = self.__run_command(server, command)
        accounting_storage_none = "accounting_storage/none" in stdout or "accounting_storage/none" in stderr
        if accounting_storage_none:
            return {}, True
        if returncode != 0 or not stdout:
            return {}, False
        return self.__parse_slurm_limits(stdout), False

    def __get_slurm_partition_resources(self, server: str, partition: str) -> tuple[int | None, int | None]:
        cpu_stdout, cpu_stderr, cpu_returncode = self.__run_command(server, f'sinfo -h -p {partition} -o "%C"')
        if cpu_returncode != 0 or cpu_stderr:
            raise SchedulerError(f"Partition {partition} from the cmd is not found on the remote/local server.")

        idle_cpus = 0
        has_cpu_data = False
        for line in cpu_stdout.splitlines():
            parts = line.strip().split("/")
            if len(parts) != 4:
                continue
            try:
                idle_cpus += int(parts[1])
                has_cpu_data = True
            except ValueError:
                continue

        node_stdout, _, node_returncode = self.__run_command(server, f'sinfo -N -h -p {partition} -o "%t"')
        available_nodes = None
        if node_returncode == 0 and node_stdout:
            available_nodes = 0
            for line in node_stdout.splitlines():
                state = line.strip().lower()
                if not state or state.startswith(Scheduler.SLURM_UNAVAILABLE_NODE_STATES):
                    continue
                available_nodes += 1

        return (idle_cpus if has_cpu_data else None), available_nodes

    def __get_username(self, server: str) -> str:
        if server == "local":
            return getpass.getuser()
        stdout, _, _ = self.__run_remote(server, "whoami")
        return stdout.strip()

    def __get_active_slurm_jobs(self, server: str, partition: str) -> int:
        username = self.__get_username(server)
        stdout, _, returncode = self.__run_command(server, f"squeue -h -u {username} -p {partition} -o %i")
        if returncode != 0 or not stdout:
            return 0
        return len([line for line in stdout.splitlines() if line.strip()])

    def __check_slurm(self, server: str) -> int:
        """
            Checks SLURM system limits and estimates the number of jobs that can be started now.

            For clusters without accounting storage we fall back to the default SLURM submit ceiling and rely on the current
            partition state (`sinfo`) plus current user's queue length (`squeue`) to avoid flooding the queue with long-pending jobs.

            Return:
                int: currently available worker count for the target SLURM partition.
        """
        partition = self.__get_slurm_partition(server)
        nodes_number = self.__get_slurm_nodes(server)
        job_cores = self.__cores[server] if isinstance(self.__cores, dict) else self.__cores
        job_cores = 1 if job_cores in (-1, None) else job_cores

        limits_table, accounting_storage_none = self.__get_slurm_limits(server)
        partition_limits = limits_table.get(partition, {})
        raw_max_submit = partition_limits.get("MaxSubmit")
        max_submit = raw_max_submit if raw_max_submit is not None else Scheduler.SLURM_ACCOUNTING_NONE_MAX_SUBMIT

        max_cpus = partition_limits.get("MaxCPUs")
        if max_cpus is not None and max_cpus < job_cores:
            raise SchedulerError("MaxCPUs parameter on the SLURM system is lower than passed through the cmd cores number.")

        max_nodes = partition_limits.get("MaxNodes")
        if max_nodes is not None and max_nodes < nodes_number:
            raise SchedulerError("MaxNodes parameter on the SLURM system is lower than passed through the cmd nodes number.")

        idle_cpus, available_nodes = self.__get_slurm_partition_resources(server, partition)
        available_by_cpus = None if idle_cpus is None else idle_cpus // max(job_cores, 1)
        available_by_nodes = None
        if available_nodes is not None and nodes_number > 1:
            available_by_nodes = available_nodes // max(nodes_number, 1)

        if accounting_storage_none:
            candidates = []
            if available_by_cpus is not None:
                candidates.append(max(available_by_cpus, 0))
            if available_by_nodes is not None:
                candidates.append(max(available_by_nodes, 0))
            if not candidates:
                return Scheduler.SLURM_ACCOUNTING_NONE_MAX_SUBMIT
            return max(min(candidates), 0)

        active_jobs = self.__get_active_slurm_jobs(server, partition)
        available_by_submit = max(max_submit - active_jobs, 0)

        candidates = []
        submit_limit_looks_reliable = accounting_storage_none or raw_max_submit is None or raw_max_submit > 1
        if submit_limit_looks_reliable:
            candidates.append(available_by_submit)
        if available_by_cpus is not None:
            candidates.append(max(available_by_cpus, 0))
        if available_by_nodes is not None:
            candidates.append(max(available_by_nodes, 0))
        if not candidates:
            candidates.append(available_by_submit)
        return max(min(candidates), 0)

    @staticmethod
    def __uniform_reduce(workers: dict, remain_workers_to_remove: int) -> tuple[int, dict]:
        """
            Performs uniform reduction of the workers dictionaries with equal number of cores.
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

    def __reduce_workers_number(self, limit: int):
        """
            Reduces number of workers if the evaluated sum of workers exceeds the requested limit.
            Makes priority for servers where the cmd command contains maximal number of cores.
        """
        if limit < 0:
            limit = 0
        workers_number = sum(self.__workers.values())
        if workers_number <= limit:
            return

        remain_workers_to_remove = workers_number - limit
        if isinstance(self.__cores, int):
            _, self.__workers = self.__uniform_reduce(self.__workers, remain_workers_to_remove)
            self.__workers = {server: count for server, count in self.__workers.items() if count > 0}
            return

        while remain_workers_to_remove and self.__workers:
            active_servers = [server for server in self.__workers if self.__workers[server] > 0]
            non_negative_cores = [self.__cores[server] for server in active_servers if self.__cores[server] != -1]
            min_cores = min(non_negative_cores) if non_negative_cores else -1
            workers_to_reduce = {server: value for server, value in self.__workers.items() if self.__cores[server] == min_cores}
            remain_workers_to_remove, reduced = self.__uniform_reduce(workers_to_reduce, remain_workers_to_remove)
            for server, value in reduced.items():
                if value > 0:
                    self.__workers[server] = value
                elif server in self.__workers:
                    self.__workers.pop(server)

    def __limit_workers_by_resources(self, available_resources: dict):
        self.__workers = {}
        for server, resources in available_resources.items():
            server_has_slurm = (server == "local" and self.__local_has_slurm) or (server != "local" and self.__connections[server].has_slurm)
            if server_has_slurm:
                workers = max(resources, 0)
            else:
                cores = self.__cores[server] if isinstance(self.__cores, dict) else self.__cores
                if cores == -1:
                    workers = 1 if resources > 0 else 0
                else:
                    workers = max(resources // max(cores, 1), 0)

            if isinstance(self.__max_workers, dict):
                workers = min(workers, self.__max_workers.get(server, workers))
            self.__workers[server] = workers

        self.__workers = {server: count for server, count in self.__workers.items() if count > 0}

        hard_limit = self.__structures_number
        if isinstance(self.__max_workers, int):
            hard_limit = min(hard_limit, self.__max_workers)
        self.__reduce_workers_number(hard_limit)

    def distribute_workers(self):
        """
            Distributes workers between the given connections dictionary. Sets a number of workers per server.

            Raise:
                csslib.exceptions.SchedulerError: if method was called before load_connections.
        """
        if not self.__connections and not self.__use_local:
            raise SchedulerError("Connections must be loaded into Scheduler before this operation.")

        available_resources = {}
        if self.__use_local:
            stdout, stderr, returncode = self.__run_local("squeue --version")
            self.__local_has_slurm = returncode == 0 and "not found" not in stderr.lower() and "not recognized" not in stderr.lower()
            cmd_for_local = self.__get_cmd("local") if isinstance(self.__cmd, dict) else self.__cmd
            cmd_is_slurm = self.__get_prefix(self.__split_cmd(cmd_for_local)) in Scheduler.SUPPORTING_CMD_PREFIXES_SLURM
            if self.__local_has_slurm != cmd_is_slurm:
                message = "Mismatch between the cmd and connection is found. "
                message += "If connection has the SLURM system you must use the cmd for SLURM system. "
                message += "If connection has the MPI system you must use the cmd for MPI system."
                raise SchedulerError(message)

            if self.__local_has_slurm:
                available_resources["local"] = self.__check_slurm("local")
            else:
                self.__check_mpi("local")
                available_resources["local"] = os.cpu_count() or 1

        for server, connection in self.__connections.items():
            if connection.has_slurm:
                available_resources[server] = self.__check_slurm(server)
            else:
                self.__check_mpi(server)
                stdout, _, _ = self.__run_remote(server, "nproc")
                available_resources[server] = int(stdout.strip())

        self.__limit_workers_by_resources(available_resources)
        return self.__workers

    def __call__(self, connections: dict | None = None):
        """
            Call method of the Scheduler class. Loads connections if they are passed through __call__, distributes workers and returns the resources distribution.

            Args:
                connections (dict | None, optional): dictionary with connections to load into Scheduler. Defaults to None.
        """
        if connections is not None:
            self.load_connections(connections)
        return self.distribute_workers()
