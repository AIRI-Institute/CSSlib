"""Module with calculator classes for DFT calculations of css structures."""

__all__ = [
    "Calculator",
    "VaspCalculator",
    "EspressoCalculator",
    "make_espresso_pwxml_parser",
    "EspressoRunFallback",
]


import copy
import os
import re
import shlex
import threading
import time
import warnings
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from csslib.exceptions import CalculatorError
from csslib.tools.calculations.inputs import EspressoInputs, InputSet, VaspInputs
from csslib.tools.calculations.remote import ConnectionStatus, RemoteConnection
from csslib.tools.calculations.scheduler import Scheduler
from csslib.tools.calculations.worker import JobState, RemoteWorker, Worker
from csslib.tools.dataloader import DataLoader
from pymatgen.io.vasp import Vasprun


def _find_existing_file(workdir: str, candidate_names: list[str]) -> str | None:
    for candidate_name in candidate_names:
        candidate_path = os.path.join(workdir, candidate_name)
        if os.path.exists(candidate_path):
            return candidate_path
    for root, _, files in os.walk(workdir):
        for candidate_name in candidate_names:
            if candidate_name in files:
                return os.path.join(root, candidate_name)
    return None


def _default_vasp_parser(workdir: str):
    vasprun_path = _find_existing_file(workdir, ["vasprun.xml"])
    if vasprun_path is None:
        raise FileNotFoundError(f"vasprun.xml was not found in {workdir}")
    return Vasprun(vasprun_path, parse_potcar_file=False)


@dataclass
class EspressoRunFallback:
    """
        Lightweight fallback result for Quantum Espresso calculations when the optional PWxml parser is unavailable.
    """

    workdir: str
    xml_path: str | None
    output_path: str | None
    xml_tree: ET.ElementTree | None = None
    parser_name: str = "fallback"
    is_complete_parser: bool = False


def _fallback_espresso_parser(workdir: str) -> EspressoRunFallback:
    xml_path = _find_existing_file(workdir, ["pwscf.xml", "data-file-schema.xml"])
    output_path = _find_existing_file(workdir, ["pw.out", "espresso.out", "scf.out", "relax.out"])
    xml_tree = ET.parse(xml_path) if xml_path is not None else None
    return EspressoRunFallback(
        workdir=workdir,
        xml_path=xml_path,
        output_path=output_path,
        xml_tree=xml_tree,
    )


def make_espresso_pwxml_parser(allow_fallback: bool = True, **pwxml_kwargs) -> Callable[[str], Any]:
    def _parser(workdir: str):
        try:
            from pymatgen.io.espresso.outputs import PWxml
        except ImportError as exc:
            if not allow_fallback:
                message = "Full Quantum Espresso parsing requires the pymatgen-io-espresso package. "
                message += "Install it with: pip install git+https://github.com/Griffin-Group/pymatgen-io-espresso"
                raise ImportError(message) from exc
            warnings.warn(
                "pymatgen-io-espresso is not installed. Falling back to a lightweight Quantum Espresso result object. "
                "Install the optional dependency for full Vasprun-like parsing.",
                RuntimeWarning,
                stacklevel=2,
            )
            return _fallback_espresso_parser(workdir)

        xml_path = _find_existing_file(workdir, ["pwscf.xml", "data-file-schema.xml"])
        if xml_path is None:
            raise FileNotFoundError(f"Quantum Espresso XML output was not found in {workdir}")
        return PWxml(xml_path, **pwxml_kwargs)

    return _parser


def _default_espresso_parser(workdir: str):
    return make_espresso_pwxml_parser()(workdir)


@dataclass
class CalculationInfo:
    """
        Class for calculation info storage. Useful in cases, when session has been interrupted and synchronization or recovery is required.
    """

    server_ip: str | None = None
    username: str | None = None
    jobid: int | None = None
    local_path: str | None = None
    remote_path: str | None = None


class CalculationStatus(Enum):
    """
        Class for calculation status trace.
    """

    PENDING = 0
    STARTED = 1
    COMPLETED = 2


@dataclass
class ActiveCalculation:
    index: Any
    server: str
    worker: Worker | RemoteWorker
    submitted_at: float = field(default_factory=time.time)
    state: JobState = JobState.UNKNOWN


class Calculator:
    """
        Class for automatization of DFT calculations for css dataset.
    """

    INSPECTOR_INTERVAL_SECONDS = 10
    REPLAN_WAIT_SECONDS = 10

    def __init__(
        self,
        data: str | DataLoader,
        inputs: InputSet | Any,
        parser: Callable | Any,
        cmd: str | list[str],
        remote_path: str,
        local_path: str | None = None,
        loading_files: list[str] | None = None,
        server_ip: str | list[str] | None = None,
        port: int | list[int] = 22,
        username: str | list[str] | None = None,
        password: str | list[str] | None = None,
        host_keys_path: str | None = None,
        max_workers: int | list[int] | None = None,
        max_connection_attempts: int = 5,
        use_sftp: bool = False,
        use_local: bool = False,
        **dataloader_fields,
    ):
        self.__data = data
        self.__inputs = inputs
        self.__parser = parser
        self.__remote_path = remote_path
        self.__local_path = local_path
        self.__loading_files = loading_files
        self.__cmd = cmd

        self.__host_keys_path = host_keys_path
        self.__server_ip = server_ip
        self.__port = port
        self.__username = username
        self.__password = password
        self.__max_workers = max_workers
        self.__max_connection_attempts = max_connection_attempts
        self.__use_sftp = use_sftp
        self.__use_local = use_local or server_ip is None

        if self.__loading_files is None:
            raise CalculatorError("loading_files attribute must be non empty!")

        self.__validate_init_parameters()
        self.__prepare_init_parameters()

        if isinstance(data, str):
            self.__data = DataLoader(data, **dataloader_fields)

        if self.__local_path is None:
            self.__local_path = os.path.join(self.__data.base_path, "css_calculated")

        self.__connections = {}
        self.__scheduler = Scheduler(cmd=self.__cmd, structures_number=len(self.__data), max_workers=self.__max_workers, use_local=self.__use_local)
        self.__workers_distribution = {}

        self.__state_lock = threading.RLock()
        self.__change_event = threading.Event()
        self.__stop_event = threading.Event()
        self.__active_calculations: dict[Any, ActiveCalculation] = {}
        self.__priority_pending_indices = deque()
        self.__errors: list[tuple[Any, str, str]] = []

    def _connect(self):
        self.__connections = {}
        for server, config in self.__server_configs.items():
            connection = RemoteConnection(
                server,
                config["port"],
                config["username"],
                config["password"],
                self.__host_keys_path,
                self.__max_connection_attempts,
                self.__use_sftp,
            )
            if connection.connection_status == ConnectionStatus.CONNECTED:
                self.__connections[server] = connection
        if self.__server_configs and not self.__connections:
            raise CalculatorError("No one connection was successful. Please verify your internet connection or connection settings.")

    def __validate_init_parameters(self):
        if self.__server_ip is None:
            return

        if isinstance(self.__server_ip, str):
            if not isinstance(self.__port, int):
                raise CalculatorError("Only one port should be written when one server_ip is passed.")
            if not isinstance(self.__username, str):
                raise CalculatorError("Only one username should be written when one server_ip is passed.")
            if isinstance(self.__password, list):
                raise CalculatorError("One password or nothing should be written when one server_ip is passed.")
            if isinstance(self.__max_workers, list) and len(self.__max_workers) not in ({1, 2} if self.__use_local else {1}):
                raise CalculatorError("Max workers parameter should be int or a short list compatible with the selected resources.")
            if isinstance(self.__cmd, list) and len(self.__cmd) not in ({1, 2} if self.__use_local else {1}):
                raise CalculatorError("Cmd command parameter should be str or a short list compatible with the selected resources.")
            return

        if isinstance(self.__port, list) and len(self.__port) != len(self.__server_ip):
            raise CalculatorError("Port should be int or list[int] with the size equal to the size of server_ip list.")
        if not isinstance(self.__username, str) and (not isinstance(self.__username, list) or len(self.__username) != len(self.__server_ip)):
            raise CalculatorError("Username should be str or list[str] with the size equal to the size of server_ip list.")
        if isinstance(self.__password, list) and len(self.__password) != len(self.__server_ip):
            raise CalculatorError("Password should be str, list[str] with the size equal to the size of server_ip list or None.")
        if isinstance(self.__max_workers, list):
            valid_lengths = {len(self.__server_ip)}
            if self.__use_local:
                valid_lengths.add(len(self.__server_ip) + 1)
            if len(self.__max_workers) not in valid_lengths:
                raise CalculatorError("Max workers parameter should be int or list[int] with the size equal to the size of server_ip list.")
        if isinstance(self.__cmd, list):
            valid_lengths = {len(self.__server_ip)}
            if self.__use_local:
                valid_lengths.add(len(self.__server_ip) + 1)
            if len(self.__cmd) not in valid_lengths:
                raise CalculatorError("Cmd command parameter should be str or list[str] with the size equal to the size of server_ip list.")

    def __prepare_init_parameters(self):
        self.__server_configs = {}
        if self.__server_ip is None:
            self.__server_ip = []
            self.__username = []
            self.__port = []
            self.__password = []
            if isinstance(self.__cmd, list):
                if len(self.__cmd) != 1:
                    raise CalculatorError("For local-only calculations cmd list should contain exactly one command.")
                self.__cmd = self.__cmd[0]
            if isinstance(self.__max_workers, list):
                if len(self.__max_workers) != 1:
                    raise CalculatorError("For local-only calculations max_workers list should contain exactly one value.")
                self.__max_workers = self.__max_workers[0]
            return

        if isinstance(self.__server_ip, str):
            self.__server_ip = [self.__server_ip]
            self.__username = [self.__username]
            self.__port = [self.__port]
            self.__password = [self.__password]
        else:
            if isinstance(self.__username, str):
                self.__username = [self.__username for _ in self.__server_ip]
            if isinstance(self.__port, int):
                self.__port = [self.__port for _ in self.__server_ip]
            if self.__password is None or isinstance(self.__password, str):
                self.__password = [self.__password for _ in self.__server_ip]

        self.__server_configs = {
            server_ip: {"port": self.__port[indx], "username": self.__username[indx], "password": self.__password[indx]}
            for indx, server_ip in enumerate(self.__server_ip)
        }

        if isinstance(self.__cmd, list):
            cmd_list = self.__cmd
            remote_cmd = cmd_list[: len(self.__server_ip)]
            self.__cmd = {server_ip: remote_cmd[indx] for indx, server_ip in enumerate(self.__server_ip)}
            if self.__use_local:
                if len(cmd_list) == len(self.__server_ip):
                    raise CalculatorError("If use_local is True and cmd is list, then the last command must be associated with local machine.")
                self.__cmd["local"] = cmd_list[-1]

        if isinstance(self.__max_workers, list):
            max_workers_list = self.__max_workers
            self.__max_workers = {
                server_ip: max_workers_list[indx]
                for indx, server_ip in enumerate(self.__server_ip)
            }
            if self.__use_local and len(max_workers_list) > len(self.__server_ip):
                self.__max_workers["local"] = max_workers_list[-1]

        if isinstance(self.__cmd, dict) and self.__use_local and "local" not in self.__cmd:
            raise CalculatorError("Local cmd must exist when use_local is True.")

    def _prepare_dataset(self):
        df = self.__data.get_df()
        if "calculation_status" not in df.columns:
            df["calculation_status"] = [CalculationStatus.PENDING for _ in range(len(df))]
        if "calculation_info" not in df.columns:
            df["calculation_info"] = [CalculationInfo() for _ in range(len(df))]
        if "calculation_output" not in df.columns:
            df["calculation_output"] = [None for _ in range(len(df))]

    def __clone_inputs(self):
        try:
            return copy.deepcopy(self.__inputs)
        except Exception:
            return copy.copy(self.__inputs)

    @staticmethod
    def __sanitize_name(value: Any) -> str:
        return re.sub(r"[^0-9A-Za-z._-]+", "_", str(value))

    def __get_cmd(self, server: str) -> str:
        return self.__cmd if isinstance(self.__cmd, str) else self.__cmd[server]

    @staticmethod
    def __get_cmd_prefix(cmd: str) -> str:
        tokens = shlex.split(cmd)
        if not tokens:
            raise CalculatorError("Cmd must not be empty.")
        token = tokens[0]
        return token[1:] if token.startswith("#") else token

    def __get_server_mode(self, server: str) -> str:
        prefix = self.__get_cmd_prefix(self.__get_cmd(server))
        return "slurm" if prefix in Scheduler.SUPPORTING_CMD_PREFIXES_SLURM else "mpi"

    def __get_server_cores(self, server: str) -> int:
        tokens = shlex.split(self.__get_cmd(server))
        if not tokens:
            return 1
        if tokens[0].startswith("#"):
            return 1
        prefix = self.__get_cmd_prefix(self.__get_cmd(server))
        if prefix in Scheduler.SUPPORTING_CMD_PREFIXES_MPI:
            for indx, token in enumerate(tokens[:-1]):
                if token in Scheduler.PARALLEL_FLAGS_MPI:
                    return int(tokens[indx + 1])
                if any(token.startswith(f"{flag}=") for flag in Scheduler.PARALLEL_FLAGS_MPI):
                    return int(token.split("=", maxsplit=1)[1])
            return 1
        ntasks = 1
        cpu_per_task = 1
        for indx, token in enumerate(tokens):
            if token in Scheduler.PARALLEL_FLAGS_SLURM_NTASKS and indx + 1 < len(tokens):
                ntasks = int(tokens[indx + 1])
            elif any(token.startswith(f"{flag}=") for flag in Scheduler.PARALLEL_FLAGS_SLURM_NTASKS):
                ntasks = int(token.split("=", maxsplit=1)[1])
            elif token in Scheduler.PARALLEL_FLAGS_SLURM_CPU_PER_TASK and indx + 1 < len(tokens):
                cpu_per_task = int(tokens[indx + 1])
            elif any(token.startswith(f"{flag}=") for flag in Scheduler.PARALLEL_FLAGS_SLURM_CPU_PER_TASK):
                cpu_per_task = int(token.split("=", maxsplit=1)[1])
        return ntasks * cpu_per_task

    def __build_worker(self, server: str, index: Any, cif_data: Any, connection: RemoteConnection | None):
        calculation_name = f"structure_{self.__sanitize_name(index)}"
        inputs = self.__clone_inputs()
        if server == "local":
            return Worker(
                cmd=self.__get_cmd("local") if isinstance(self.__cmd, dict) else self.__get_cmd(server),
                inputs=inputs,
                parser=self.__parser,
                structure=cif_data,
                loading_files=self.__loading_files,
                calculation_name=calculation_name,
                work_root=self.__remote_path,
                local_path=self.__local_path,
            )

        return RemoteWorker(
            ssh=connection()["SSH"],
            file_sharing=connection()["SFTP/SCP"],
            connection_owner=connection,
            server_ip=server,
            username=self.__server_configs[server]["username"],
            cmd=self.__get_cmd(server),
            inputs=inputs,
            parser=self.__parser,
            structure=cif_data,
            loading_files=self.__loading_files,
            calculation_name=calculation_name,
            remote_root=self.__remote_path,
            local_path=self.__local_path,
        )

    def __build_slot_connection(self, server: str) -> RemoteConnection | None:
        if server == "local":
            return None
        config = self.__server_configs[server]
        connection = RemoteConnection(
            server,
            config["port"],
            config["username"],
            config["password"],
            self.__host_keys_path,
            self.__max_connection_attempts,
            self.__use_sftp,
        )
        if connection.connection_status != ConnectionStatus.CONNECTED:
            raise CalculatorError(f"Unable to create worker connection for server {server}.")
        return connection

    def __inspection_loop(self):
        dataframe = self.__data.get_df()
        while not self.__stop_event.is_set():
            with self.__state_lock:
                active_snapshot = list(self.__active_calculations.values())

            has_changes = False
            for active in active_snapshot:
                inspection = active.worker.inspect()
                if inspection.state != active.state:
                    active.state = inspection.state
                    has_changes = True

                if inspection.state != JobState.COMPLETED:
                    if inspection.state == JobState.FAILED:
                        with self.__state_lock:
                            dataframe.at[active.index, "calculation_status"] = CalculationStatus.PENDING
                            self.__active_calculations.pop(active.index, None)
                            self.__errors.append((active.index, active.server, inspection.message or "Calculation failed."))
                        has_changes = True
                    continue

                try:
                    calculation_info_dict, calculation_output = active.worker.finalize()
                    info = CalculationInfo(**calculation_info_dict)
                except Exception as exc:
                    with self.__state_lock:
                        dataframe.at[active.index, "calculation_status"] = CalculationStatus.PENDING
                        self.__active_calculations.pop(active.index, None)
                        self.__errors.append((active.index, active.server, str(exc)))
                    has_changes = True
                    continue

                with self.__state_lock:
                    dataframe.at[active.index, "calculation_info"] = info
                    dataframe.at[active.index, "calculation_output"] = calculation_output
                    dataframe.at[active.index, "calculation_status"] = CalculationStatus.COMPLETED
                    self.__active_calculations.pop(active.index, None)
                has_changes = True

            if has_changes:
                self.__change_event.set()
            self.__stop_event.wait(self.INSPECTOR_INTERVAL_SECONDS)

    def __raise_if_errors(self):
        with self.__state_lock:
            if not self.__errors:
                return
            index, server, message = self.__errors[0]
        raise CalculatorError(f"{len(self.__errors)} calculations failed. First error: index={index}, server={server}, error={message}")

    def __cancel_active_calculations(self):
        with self.__state_lock:
            active_snapshot = list(self.__active_calculations.values())
        for active in active_snapshot:
            try:
                active.worker.cancel()
            except Exception:
                continue

    def __active_mpi_counts(self) -> dict[str, int]:
        with self.__state_lock:
            active_snapshot = list(self.__active_calculations.values())
        counts = {}
        for active in active_snapshot:
            if active.worker.resource_manager == "slurm":
                continue
            counts[active.server] = counts.get(active.server, 0) + 1
        return counts

    def __compute_launch_slots(self, desired_distribution: dict[str, int]) -> dict[str, int]:
        active_mpi = self.__active_mpi_counts()
        launch_slots = {}
        for server, desired_workers in desired_distribution.items():
            if self.__get_server_mode(server) == "slurm":
                slots = max(desired_workers, 0)
            else:
                slots = max(desired_workers - active_mpi.get(server, 0), 0)
            if slots > 0:
                launch_slots[server] = slots
        return launch_slots

    def __ordered_pending_indices(self) -> list[Any]:
        dataframe = self.__data.get_df()
        ordered = []
        seen = set()

        while self.__priority_pending_indices:
            index = self.__priority_pending_indices.popleft()
            if index in seen:
                continue
            if dataframe.at[index, "calculation_status"] == CalculationStatus.PENDING:
                ordered.append(index)
                seen.add(index)

        for index, row in dataframe.iterrows():
            if index in seen:
                continue
            if row["calculation_status"] == CalculationStatus.PENDING:
                ordered.append(index)
        return ordered

    def __remove_fictive_atoms(self, cif_data: str):
        fictive_atoms = [column.split('_', maxsplit=1)[0] for column in self.__data.columns if 'fictive' in column]
        if fictive_atoms:
            removed_lines = 0
            lines = cif_data.splitlines()
            for indx, line in enumerate(lines[::-1]):
                splitted_line = line.split()
                if len(splitted_line) > 1 and splitted_line[1] in fictive_atoms:
                    lines.pop(len(lines) - 1 - indx + removed_lines)
                    removed_lines += 1
                elif len(splitted_line) == 1:
                    break
            return '\n'.join(lines)
        return cif_data

    def __launch_calculation(self, index: Any, server: str):
        dataframe = self.__data.get_df()
        cif_data = self.__remove_fictive_atoms(dataframe.at[index, "cif_data"])
        connection = self.__build_slot_connection(server)
        worker = self.__build_worker(server, index, cif_data, connection)
        info = worker.start()
        active = ActiveCalculation(
            index=index,
            server=server,
            worker=worker,
            state=JobState.PENDING if worker.launch_mode == "slurm-job" else JobState.RUNNING,
        )
        with self.__state_lock:
            self.__active_calculations[index] = active
            dataframe.at[index, "calculation_status"] = CalculationStatus.STARTED
            dataframe.at[index, "calculation_info"] = CalculationInfo(**info)

    def __server_sort_key(self, server: str) -> tuple[int, int, str]:
        mode = self.__get_server_mode(server)
        return (0 if mode == "mpi" else 1, -self.__get_server_cores(server), str(server))

    def __migrate_pending_slurm_to_mpi(self, launch_slots: dict[str, int]) -> bool:
        mpi_free_slots = sum(slots for server, slots in launch_slots.items() if self.__get_server_mode(server) == "mpi")
        if mpi_free_slots <= 0:
            return False

        with self.__state_lock:
            active_snapshot = list(self.__active_calculations.values())
        migratable = [active for active in active_snapshot if active.worker.can_migrate_pending()]
        if not migratable:
            return False

        dataframe = self.__data.get_df()
        has_changes = False
        for active in sorted(migratable, key=lambda item: item.submitted_at)[:mpi_free_slots]:
            if not active.worker.cancel():
                continue
            with self.__state_lock:
                self.__active_calculations.pop(active.index, None)
                dataframe.at[active.index, "calculation_status"] = CalculationStatus.PENDING
                dataframe.at[active.index, "calculation_info"] = CalculationInfo()
                self.__priority_pending_indices.appendleft(active.index)
            has_changes = True
        return has_changes

    def __dispatch_pending(self):
        desired_distribution = self.__scheduler()
        self.__workers_distribution = desired_distribution
        launch_slots = self.__compute_launch_slots(desired_distribution)

        if self.__migrate_pending_slurm_to_mpi(launch_slots):
            desired_distribution = self.__scheduler()
            self.__workers_distribution = desired_distribution
            launch_slots = self.__compute_launch_slots(desired_distribution)

        pending_indices = self.__ordered_pending_indices()
        if not pending_indices:
            return

        server_sequence = []
        for server in sorted(launch_slots, key=self.__server_sort_key):
            server_sequence.extend([server] * launch_slots[server])

        for server, index in zip(server_sequence, pending_indices):
            self.__launch_calculation(index, server)

    def __is_finished(self) -> bool:
        dataframe = self.__data.get_df()
        if any(status != CalculationStatus.COMPLETED for status in dataframe["calculation_status"]):
            return False
        return not self.__active_calculations

    def run(self): # TODO: if calculation was started, then we need to check calculation before it will be restarted. Also we can add parse method after each calculation.
        """
            Starts the dispatch loop of DFT calculations for the data stored in DataFrame.
        """
        self._prepare_dataset()
        os.makedirs(self.__local_path, exist_ok=True)
        if self.__use_local:
            os.makedirs(self.__remote_path, exist_ok=True)

        if self.__server_configs:
            self._connect()
            self.__scheduler.load_connections(self.__connections)

        self.__workers_distribution = self.__scheduler()
        if sum(self.__workers_distribution.values()) <= 0:
            raise CalculatorError("No available workers were detected for the current set of resources.")

        self.__stop_event.clear()
        self.__change_event.clear()
        self.__errors.clear()
        with self.__state_lock:
            self.__active_calculations.clear()
            self.__priority_pending_indices.clear()

        inspector = threading.Thread(target=self.__inspection_loop, name="csslib-calculation-inspector", daemon=True)
        inspector.start()
        try:
            while True:
                self.__raise_if_errors()
                with self.__state_lock:
                    if self.__is_finished():
                        break
                self.__dispatch_pending()
                self.__raise_if_errors()
                with self.__state_lock:
                    if self.__is_finished():
                        break
                self.__change_event.wait(self.REPLAN_WAIT_SECONDS)
                self.__change_event.clear()
        finally:
            self.__stop_event.set()
            inspector.join(timeout=self.INSPECTOR_INTERVAL_SECONDS + 1)
            if self.__errors:
                self.__cancel_active_calculations()

        self.__raise_if_errors()
        return self.__data


class VaspCalculator(Calculator):
    """
        Partially preinitialized Calculator for VASP calculations.
    """

    DEFAULT_LOADING_FILES = ["vasprun.xml", "OUTCAR", "CONTCAR", "OSZICAR"]

    def __init__(
        self,
        data: str | DataLoader,
        cmd: str | list[str],
        remote_path: str,
        local_path: str | None = None,
        input_paths: str | os.PathLike | list[str | os.PathLike] | None = None,
        inputs: VaspInputs | None = None,
        parser: Callable | None = None,
        loading_files: list[str] | None = None,
        transform: Callable[[Any], Any] | None = None,
        potcar_dir: str | os.PathLike | None = None,
        potcar_map: dict[str, str] | None = None,
        assemble_potcar: bool = False,
        **kwargs,
    ):
        inputs_object = inputs if inputs is not None else VaspInputs(
            input_paths=input_paths,
            transform=transform,
            potcar_dir=potcar_dir,
            potcar_map=potcar_map,
            assemble_potcar=assemble_potcar,
        )
        super().__init__(
            data=data,
            inputs=inputs_object,
            parser=parser or _default_vasp_parser,
            cmd=cmd,
            remote_path=remote_path,
            local_path=local_path,
            loading_files=loading_files or self.DEFAULT_LOADING_FILES,
            **kwargs,
        )


class EspressoCalculator(Calculator):
    """
        Partially preinitialized Calculator for Quantum Espresso calculations.
    """

    DEFAULT_LOADING_FILES = ["pw.in", "pw.out", "espresso.out", "data-file-schema.xml", "pwscf.xml"]

    def __init__(
        self,
        data: str | DataLoader,
        cmd: str | list[str],
        remote_path: str,
        local_path: str | None = None,
        input_paths: str | os.PathLike | list[str | os.PathLike] | None = None,
        inputs: EspressoInputs | None = None,
        parser: Callable | None = None,
        loading_files: list[str] | None = None,
        transform: Callable[[Any], Any] | None = None,
        pseudopotentials: dict[str, str] | None = None,
        pseudo_dir: str | None = None,
        control: dict[str, Any] | None = None,
        system: dict[str, Any] | None = None,
        electrons: dict[str, Any] | None = None,
        ions: dict[str, Any] | None = None,
        cell: dict[str, Any] | None = None,
        kpoints_grid: tuple[int, int, int] = (1, 1, 1),
        kpoints_shift: tuple[int, int, int] = (0, 0, 0),
        use_primitive: bool = False,
        symprec: float = 1e-3,
        pwxml_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        inputs_object = inputs if inputs is not None else EspressoInputs(
            input_paths=input_paths,
            transform=transform,
            pseudopotentials=pseudopotentials,
            pseudo_dir=pseudo_dir,
            control=control,
            system=system,
            electrons=electrons,
            ions=ions,
            cell=cell,
            kpoints_grid=kpoints_grid,
            kpoints_shift=kpoints_shift,
            use_primitive=use_primitive,
            symprec=symprec,
        )
        super().__init__(
            data=data,
            inputs=inputs_object,
            parser=parser or make_espresso_pwxml_parser(**(pwxml_kwargs or {})),
            cmd=cmd,
            remote_path=remote_path,
            local_path=local_path,
            loading_files=loading_files or self.DEFAULT_LOADING_FILES,
            **kwargs,
        )
