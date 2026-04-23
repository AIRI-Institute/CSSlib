"""Module with calculator classes for DFT calculations of css structures."""

__all__ = [
    "Calculator",
    "VaspCalculator",
    "EspressoCalculator",
]


import copy
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable

from csslib.exceptions import CalculatorError
from csslib.logging_ import get_tools_logger
from csslib.tools.calculations.cmd import MPI, SLURM
from csslib.tools.calculations.inputs import EspressoInputs, InputSet, VaspInputs
from csslib.tools.calculations.parser import default_espresso_parser, default_vasp_parser
from csslib.tools.calculations.remote import ConnectionStatus, RemoteConfiguration, RemoteConnection
from csslib.tools.calculations.scheduler import Scheduler
from csslib.tools.calculations.worker import JobState, RemoteWorker, Worker
from csslib.tools.dataloader import DataLoader

logger = get_tools_logger("calculations.calculator")


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


@dataclass
class ActiveCalculation:
    """
        Stores runtime metadata for a calculation that is currently tracked by the inspector.
    """
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
        cmd: MPI | SLURM | list[MPI] | list[SLURM],
        remote_path: str,
        local_path: str | None = None,
        loading_files: list[str] | None = None,
        connection_config: RemoteConfiguration | list[RemoteConfiguration] | None = None,
        max_workers: int | list[int] | None = None,
        use_local: bool = False,
        **dataloader_fields,
    ):
        """
            Initializes the calculator, dataset, resource configuration and runtime state.

            Args:
                data (str | DataLoader): path to the dataset or an already initialized DataLoader object.
                inputs (InputSet | Any): input-set object that prepares package-specific input files for one structure.
                parser (Callable | Any): parser callable that converts finished calculation files into a result object.
                cmd (str | list[str]): launch command or commands for local and/or remote resources.
                remote_path (str): path to the root directory where calculation working folders will be created.
                local_path (str | None, optional): path to the local directory where selected output files will be stored.
                Defaults to None.
                loading_files (list[str] | None, optional): relative paths to files that should be copied after the calculation finishes.
                Defaults to None.
                connection_config (RemoteConfiguration | list[RemoteConfiguration] | None, optional): instance/instances of the RemoteConfiguration class. Should be used for remote connection setup. Defaults to None.
                max_workers (int | list[int] | None, optional): maximal number of workers allowed on each resource. Defaults to None.
                max_connection_attempts (int, optional): number of reconnection attempts for remote resources. Defaults to 5.
                **dataloader_fields: extra keyword arguments forwarded to DataLoader when data is passed as a path.

            Raise:
                CalculatorError: if loading_files is empty or initialization parameters are inconsistent.
        """
        self.__data = data
        self.__inputs = inputs
        self.__parser = parser
        self.__remote_path = remote_path
        self.__local_path = local_path
        self.__loading_files = loading_files
        self.__cmd = cmd

        self.__configurations = connection_config
        self.__max_workers = max_workers
        self.__use_local = use_local or self.__configurations is None

        if self.__loading_files is None:
            raise CalculatorError("loading_files attribute must be non empty!")

        self.__validate_init_parameters()
        self.__prepare_init_parameters()

        if isinstance(data, str):
            self.__data = DataLoader(data, **dataloader_fields)

        if self.__local_path is None:
            self.__local_path = os.path.join(self.__data.base_path, "css_calculated")

        self.__connections = {}
        self.__single_mode = True
        self.__scheduler = Scheduler(cmd=self.__cmd, structures_number=len(self.__data), max_workers=self.__max_workers, use_local=self.__use_local)
        self.__workers_distribution = {}

        self.__state_lock = threading.RLock()
        self.__change_event = threading.Event()
        self.__stop_event = threading.Event()
        
        self.__inspector: threading.Thread | None = None
        self.__looper: threading.Thread | None = None
        
        self.__active_calculations: dict[Any, ActiveCalculation] = {}
        self.__priority_pending_indices = deque()
        self.__errors: list[tuple[Any, str, str]] = []
        logger.debug(
            "Calculator initialized: structures=%d, use_local=%s, remote_servers=%d, local_path=%s, remote_path=%s.",
            len(self.__data),
            self.__use_local,
            len(self.__configurations),
            self.__local_path,
            self.__remote_path,
        )

    def _connect(self):
        """
            Establishes all configured remote connections and keeps only successfully connected servers.

            Raise:
                CalculatorError: if remote servers were configured but none of the connections was established successfully.
        """
        self.__connections = {}
        for server_ip, config in self.__configurations.items():
            logger.info("Connecting to remote server %s as %s.", server_ip, config.username)
            connection = RemoteConnection(config)
            if connection.connection_status == ConnectionStatus.CONNECTED:
                self.__connections[server_ip] = connection
                logger.info("Remote server %s is connected.", server_ip)
            else:
                logger.warning("Remote server %s is unavailable and will be skipped.", server_ip)
        if self.__configurations is not None and not self.__connections:
            raise CalculatorError("No one connection was successful. Please verify your internet connection or connection settings.")

    def __validate_init_parameters(self):
        """
            Validates combinations of initialization arguments for local and remote execution modes.

            Raise:
                CalculatorError: if provided ports, usernames, passwords, commands or worker limits are inconsistent with server_ip.
        """
        if self.__configurations is None:
            return

        if isinstance(self.__configurations, RemoteConfiguration):
            if isinstance(self.__max_workers, list) and len(self.__max_workers) not in ({1, 2} if self.__use_local else {1}):
                raise CalculatorError("Max workers parameter should be int or a short list compatible with the selected resources.")
            if isinstance(self.__cmd, list) and len(self.__cmd) not in ({1, 2} if self.__use_local else {1}):
                raise CalculatorError("Cmd command parameter should be str or a short list compatible with the selected resources.")
            return

        if isinstance(self.__max_workers, list):
            valid_lengths = {len(self.__configurations)}
            if self.__use_local:
                valid_lengths.add(len(self.__configurations) + 1)
            if len(self.__max_workers) not in valid_lengths:
                raise CalculatorError("Max workers parameter should be int or list[int] with the size equal to the size of server_ip list.")
        if isinstance(self.__cmd, list):
            valid_lengths = {len(self.__configurations)}
            if self.__use_local:
                valid_lengths.add(len(self.__configurations) + 1)
            if len(self.__cmd) not in valid_lengths:
                raise CalculatorError("Cmd command parameter should be str or list[str] with the size equal to the size of server_ip list.")

    def __prepare_init_parameters(self):
        """
            Normalizes server-dependent initialization arguments and converts list-like settings into internal dictionaries.

            Raise:
                CalculatorError: if local execution is requested but the corresponding local command or worker count is missing.
        """
        
        if self.__configurations is not None and not isinstance(self.__configurations, list):
            self.__configurations = [self.__configurations]
        if not isinstance(self.__cmd, list):
            self.__cmd = [self.__cmd]
        if not isinstance(self.__max_workers, list):
            self.__max_workers = [self.__max_workers]

        cmd_list = self.__cmd
        remote_cmd = cmd_list[: len(self.__configurations)]
        self.__cmd = {config.server_ip: remote_cmd[indx] for indx, config in enumerate(self.__configurations)}
        if self.__use_local:
            if len(cmd_list) <= len(self.__configurations) and len(self.__cmd) != 1:
                raise CalculatorError("If use_local is True, then the last command must be associated with local machine.")
            self.__cmd["local"] = cmd_list[-1]

        max_workers_list = self.__max_workers
        self.__max_workers = {
            config.server_ip: max_workers_list[indx]
            for indx, config in enumerate(self.__configurations)
        }
        if self.__use_local and (len(max_workers_list) > len(self.__configurations) or len(max_workers_list) == 1):
            self.__max_workers["local"] = max_workers_list[-1]
        elif self.__use_local:
            raise CalculatorError("If use_local is True, then the last max_workers parameter must be associated with local machine.")

        if self.__configurations is not None:
            self.__configurations = {config.server_ip: config for config in self.__configurations}

    def _prepare_dataset(self, force: bool = False):
        """
            Adds calculation status, metadata and parsed output columns to the dataset when they are missing.

            Args:
                force (bool, optional): starts calculations with JobState.RUNNING and JobState.UNKNOWN. Defaults to False.
        """
        
        if "calculation_status" not in self.__data.columns:
            self.__data["calculation_status"] = [JobState.PENDING for _ in range(len(self.__data))]
        else:
            self.__data["calculation_status"][(self.__data["calculation_status"] == JobState.FAILED) | (self.__data["calculation_status"] == JobState.CANCELLED)] = JobState.PENDING
            self.__data["calculation_status"][(self.__data["calculation_status"] == JobState.RUNNING) | (self.__data["calculation_status"] == JobState.UNKNOWN)] = JobState.UNKNOWN if not force else JobState.PENDING
        if "calculation_info" not in self.__data.columns:
            self.__data["calculation_info"] = [CalculationInfo() for _ in range(len(self.__data))]
        if "calculation_output" not in self.__data.columns:
            self.__data["calculation_output"] = [None for _ in range(len(self.__data))]
        logger.debug("Calculator dataset is prepared with status, info and output columns.")

    def __clone_inputs(self):
        """
            Creates an isolated copy of the input-set object for one worker.

            Return:
                InputSet | Any: deep or shallow copy of the current input-set object for isolated worker execution.
        """
        try:
            return copy.deepcopy(self.__inputs)
        except Exception:
            return copy.copy(self.__inputs)

    @staticmethod
    def __sanitize_name(value: Any) -> str:
        """
            Converts an arbitrary value into a filesystem-safe calculation name.

            Args:
                value (Any): value that should be transformed into a safe directory name.

            Return:
                str: sanitized string suitable for filesystem paths.
        """
        return re.sub(r"[^0-9A-Za-z._-]+", "_", str(value))

    def __build_worker(self, server: str, index: Any, cif_data: Any, connection: RemoteConnection | None):
        """
            Builds a local or remote worker instance for one dataset row.

            Args:
                server (str): target server name.
                index (Any): dataset index of the structure.
                cif_data (Any): structure representation that should be converted into input files.
                connection (RemoteConnection | None): remote connection object for the selected server or None for local runs.

            Return:
                Worker | RemoteWorker: initialized worker object ready for non-blocking launch.
        """
        
        calculation_name = f"structure_{self.__sanitize_name(index)}"
        inputs = self.__clone_inputs()
        if server == "local":
            return Worker(
                cmd=self.__cmd["local"],
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
            username=self.__configurations[server].username,
            cmd=self.__cmd[server],
            inputs=inputs,
            parser=self.__parser,
            structure=cif_data,
            loading_files=self.__loading_files,
            calculation_name=calculation_name,
            remote_root=self.__remote_path,
            local_path=self.__local_path,
        )

    def __build_slot_connection(self, server: str) -> RemoteConnection | None:
        """
            Creates a dedicated remote connection for a worker slot when remote execution is used or returns already created connection if single_mode attribute is True.

            Args:
                server (str): target server name for which a dedicated worker connection should be created.

            Raise:
                CalculatorError: if a remote worker connection could not be established.

            Return:
                RemoteConnection | None: connected remote resource handle or None for local execution.
        """
        if server == "local":
            return None
        elif self.__single_mode:
            return self.__connections[server]

        config = self.__configurations[server]
        connection = RemoteConnection(config)
        if connection.connection_status != ConnectionStatus.CONNECTED:
            raise CalculatorError(f"Unable to create worker connection for server {server}.")
        return connection

    def __record_failure(self, active: ActiveCalculation, message: str) -> bool:
        """
            Stores a failed calculation in the dataframe and error list.

            Args:
                active (ActiveCalculation): tracked calculation descriptor.
                message (str): failure description.

            Return:
                bool: True if the active calculation was recorded as failed, False if it was already processed.
        """

        with self.__state_lock:
            if self.__active_calculations.pop(active.index, None) is None:
                return False
            self.__data.at[active.index, "calculation_status"] = JobState.FAILED
            self.__errors.append((active.index, active.server, message))
        logger.error("Calculation index=%s on %s failed: %s", active.index, active.server, message)
        return True

    def __store_state(self, active: ActiveCalculation, state: JobState):
        """
            Updates dataframe status for an active calculation.

            Args:
                active (ActiveCalculation): tracked calculation descriptor.
                state (JobState): new runtime state.
        """

        with self.__state_lock:
            if active.index in self.__active_calculations:
                self.__data.at[active.index, "calculation_status"] = state

    def __finalize_completed_calculation(self, active: ActiveCalculation) -> bool:
        """
            Finalizes a completed calculation, collects outputs and stores parser results.

            Args:
                active (ActiveCalculation): tracked completed calculation.

            Return:
                bool: True if new results were stored, False if the calculation was already processed.
        """

        try:
            calculation_info_dict, calculation_output = active.worker.finalize(verify_completion=False)
            info = CalculationInfo(**calculation_info_dict)
        except Exception as exc:
            logger.exception("Unable to finalize calculation index=%s on %s.", active.index, active.server)
            return self.__record_failure(active, str(exc))

        with self.__state_lock:
            if self.__active_calculations.pop(active.index, None) is None:
                return False
            self.__data.at[active.index, "calculation_info"] = info
            self.__data.at[active.index, "calculation_output"] = calculation_output
            self.__data.at[active.index, "calculation_status"] = JobState.COMPLETED
        logger.info("Calculation index=%s on %s is completed and parsed.", active.index, active.server)
        return True

    def __synchronize_active_calculations(self) -> bool:
        """
            Performs one synchronization pass over active calculations.

            Return:
                bool: True if dataframe state changed during synchronization.
        """

        with self.__state_lock:
            active_snapshot = list(self.__active_calculations.values())

        has_changes = False
        for active in active_snapshot:
            try:
                inspection = active.worker.inspect()
            except Exception as exc:
                logger.exception("Unable to inspect calculation index=%s on %s.", active.index, active.server)
                has_changes = self.__record_failure(active, str(exc)) or has_changes
                continue

            if inspection.state != active.state:
                logger.info(
                    "Calculation index=%s on %s changed state: %s -> %s.",
                    active.index,
                    active.server,
                    active.state.value,
                    inspection.state.value,
                )
                active.state = inspection.state
                has_changes = True

            if inspection.state == JobState.COMPLETED:
                has_changes = self.__finalize_completed_calculation(active) or has_changes
                continue

            if inspection.state == JobState.FAILED:
                message = inspection.message or "Calculation failed."
                has_changes = self.__record_failure(active, message) or has_changes
                continue

            self.__store_state(active, inspection.state)

        if has_changes:
            self.__change_event.set()
        return has_changes

    def __inspection_loop(self):
        """
            Polls all active calculations, finalizes completed jobs and runs parsers.
        """
        
        while not self.__stop_event.is_set():
            self.__synchronize_active_calculations()
            self.__stop_event.wait(self.INSPECTOR_INTERVAL_SECONDS)

    def __raise_if_errors(self):
        """
            Raises a CalculatorError if at least one calculation failure has already been recorded.

            Raise:
                CalculatorError: if at least one calculation failure was already recorded by the inspector thread.
        """
        with self.__state_lock:
            if not self.__errors:
                return
            index, server, message = self.__errors[0]
        raise CalculatorError(f"{len(self.__errors)} calculations failed. First error: index={index}, server={server}, error={message}")

    def __cancel_active_calculations(self):
        """
            Attempts to cancel all calculations that are still marked as active.
        """
        with self.__state_lock:
            active_snapshot = list(self.__active_calculations.values())
        for active in active_snapshot:
            try:
                if active.worker.cancel():
                    logger.warning("Cancelled active calculation index=%s on %s.", active.index, active.server)
            except Exception:
                logger.exception("Unable to cancel calculation index=%s on %s.", active.index, active.server)
                continue
            self.__data.at[active.index, "calculation_status"] = JobState.FAILED

    def __active_mpi_counts(self) -> dict[str, int]:
        """
            Counts currently active non-SLURM calculations per server.

            Return:
                dict[str, int]: number of currently active non-SLURM calculations per server.
        """
        with self.__state_lock:
            active_snapshot = list(self.__active_calculations.values())
        counts = {}
        for active in active_snapshot:
            if isinstance(active.worker._cmd, SLURM):
                continue
            counts[active.server] = counts.get(active.server, 0) + 1
        return counts

    def __compute_launch_slots(self, desired_distribution: dict[str, int]) -> dict[str, int]:
        """
            Computes how many new calculations can be launched on each server right now.

            Args:
                desired_distribution (dict[str, int]): scheduler output describing desired worker counts per server.

            Return:
                dict[str, int]: number of new calculations that may be launched right now on each server.
        """
        
        active_mpi = self.__active_mpi_counts()
        launch_slots = {}
        for server, desired_workers in desired_distribution.items():
            if isinstance(self.__cmd[server], SLURM):
                slots = max(desired_workers, 0)
            else:
                slots = max(desired_workers - active_mpi.get(server, 0), 0)
            if slots > 0:
                launch_slots[server] = slots
        return launch_slots

    def __ordered_pending_indices(self) -> list[Any]:
        """
            Builds the ordered list of dataset indices that are still pending.

            Return:
                list[Any]: ordered list of dataset indices that are still waiting for calculation.
        """
        ordered = []
        seen = set()

        while self.__priority_pending_indices:
            index = self.__priority_pending_indices.popleft()
            if index in seen:
                continue
            if self.__data.at[index, "calculation_status"] == JobState.PENDING:
                ordered.append(index)
                seen.add(index)

        for row in self.__data:
            if row.Index in seen:
                continue
            if row.calculation_status == JobState.PENDING:
                ordered.append(row.Index)
        return ordered

    def __remove_fictive_atoms(self, cif_data: str):
        """
            Removes fictive atoms from CIF text before input generation when such atoms are present in the dataset.

            Args:
                cif_data (str): CIF text that may contain fictive atoms which should not be used in calculations.

            Return:
                str: CIF text with trailing fictive atom records removed when such columns are present in the dataset.
        """
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
        """
            Starts one calculation on the selected server and stores its initial metadata.

            Args:
                index (Any): dataset index of the structure that should be launched.
                server (str): target server name for the new calculation.
        """
        cif_data = self.__remove_fictive_atoms(self.__data.at[index, "cif_data"])
        connection = self.__build_slot_connection(server)
        worker = self.__build_worker(server, index, cif_data, connection)
        logger.info("Launching calculation index=%s on %s.", index, server)
        info = worker.start()
        active = ActiveCalculation(
            index=index,
            server=server,
            worker=worker,
            state=JobState.PENDING if isinstance(self.__cmd[server], SLURM) else JobState.RUNNING,
        )
        with self.__state_lock:
            self.__active_calculations[index] = active
            self.__data.at[index, "calculation_status"] = JobState.RUNNING
            self.__data.at[index, "calculation_info"] = CalculationInfo(**info)
        logger.debug("Calculation index=%s on %s started with info=%s.", index, server, info)

    def __server_sort_key(self, server: str) -> tuple[int, int, str]:
        """
            Builds a stable priority key used to order servers during dispatch.

            Args:
                server (str): server name for which priority key should be built.

            Return:
                tuple[int, int, str]: tuple used for stable dispatch ordering between MPI and SLURM resources.
        """
        
        cmd = self.__cmd[server]
        return (0 if isinstance(cmd, MPI) else 1, -cmd.cores_number if isinstance(cmd, MPI) else -cmd.ntasks * cmd.cpu_per_task, str(server))

    def __migrate_pending_slurm_to_mpi(self, launch_slots: dict[str, int]) -> bool:
        """
        Moves cancelable pending SLURM jobs back to the queue when new MPI slots become available.

        Args:
            launch_slots (dict[str, int]): currently available launch slots on each server.

        Return:
            bool: True if at least one pending SLURM job was cancelled and returned to the pending queue.
        """
        
        mpi_free_slots = sum(slots for server, slots in launch_slots.items() if isinstance(self.__cmd[server], MPI))
        if mpi_free_slots <= 0:
            return False

        with self.__state_lock:
            active_snapshot = list(self.__active_calculations.values())
        migratable = [active for active in active_snapshot if active.worker.can_migrate_pending()]
        if not migratable:
            return False

        has_changes = False
        for active in sorted(migratable, key=lambda item: item.submitted_at)[:mpi_free_slots]:
            if not active.worker.cancel():
                continue
            with self.__state_lock:
                self.__active_calculations.pop(active.index, None)
                self.__data.at[active.index, "calculation_status"] = JobState.PENDING
                self.__data.at[active.index, "calculation_info"] = CalculationInfo()
                self.__priority_pending_indices.appendleft(active.index)
            has_changes = True
        return has_changes

    def __dispatch_pending(self):
        """
            Recomputes resource availability and launches pending structures into free execution slots.
        """
        
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
        """
            Checks whether all dataset rows are completed and no active calculations remain.

            Return:
                bool: True if all dataset rows are completed and no active calculations remain.
        """
        if any(status != JobState.COMPLETED for status in self.__data.calculation_status):
            return False
        return not self.__active_calculations

    def __loop(self):
        """
            Method representing loop of the calculator run.
        """
        
        interrupted = False
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
        except KeyboardInterrupt:
            interrupted = True
            logger.warning("Calculator run was interrupted by user. Synchronizing completed calculations before exit.")
            raise
        finally:
            self.__stop_event.set()
            if self.__inspector is not None and not interrupted:
                self.__inspector.join(timeout=self.INSPECTOR_INTERVAL_SECONDS + 1)
            elif self.__inspector is not None and interrupted:
                logger.warning("Skipping wait for inspector thread because the run was interrupted by user.")
            if interrupted:
                try:
                    self.__synchronize_active_calculations()
                except KeyboardInterrupt:
                    logger.warning("Synchronization of completed calculations was interrupted again by user.")
            else:
                self.__synchronize_active_calculations()
            if self.__errors and not interrupted:
                self.__cancel_active_calculations()

        self.__raise_if_errors()

    def run(self, /, nonblocking: bool = False, force: bool = False, one_connection_per_server: bool = True):
        """
            Starts the calculation dispatch loop.

            Args:
                nonblocking (bool, optional): starts calculator in the nonblocking mode. Defaults to False.
                force (bool, optional): starts calculations with JobState.RUNNING and JobState.UNKNOWN. Defaults to False.
                one_connection_per_server (bool, optional): use only one SSHClient and filesharing object per server. SLOWER BUT MORE STABLE. Defaults to True.

            Return:
                DataLoader: updated DataLoader object with calculation statuses, metadata and parsed outputs.
        """
        self.__single_mode = one_connection_per_server
        self._prepare_dataset(force=force)
        os.makedirs(self.__local_path, exist_ok=True)
        if self.__use_local:
            os.makedirs(self.__remote_path, exist_ok=True)
        logger.info("Calculator run started. Structures=%d, nonblocking=%s.", len(self.__data), nonblocking)

        if self.__configurations is not None:
            self._connect()
            self.__scheduler.load_connections(self.__connections)

        self.__workers_distribution = self.__scheduler()
        if sum(self.__workers_distribution.values()) <= 0:
            raise CalculatorError("No available workers were detected for the current set of resources.")
        logger.info("Initial workers distribution: %s", self.__workers_distribution)

        self.__stop_event.clear()
        self.__change_event.clear()
        self.__errors.clear()
        with self.__state_lock:
            self.__active_calculations.clear()
            self.__priority_pending_indices.clear()

        self.__inspector = threading.Thread(target=self.__inspection_loop, name="csslib-calculation-inspector", daemon=True)
        self.__inspector.start()
        if not nonblocking:
            self.__loop()
        else:
            self.__looper = threading.Thread(target=self.__loop, name="csslib-calculator-looper", daemon=True)
            self.__looper.start()
            logger.info("Calculator loop is started in nonblocking mode.")
        
        logger.info("Calculator run finished.")
        return self.__data


class VaspCalculator(Calculator):
    """
        Partially preinitialized Calculator for VASP calculations.
    """

    DEFAULT_LOADING_FILES = ["vasprun.xml", "OUTCAR", "CONTCAR", "OSZICAR"]

    def __init__(
        self,
        data: str | DataLoader,
        cmd: MPI | SLURM | list[MPI] | list[SLURM],
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
        connection_config: RemoteConfiguration | list[RemoteConfiguration] | None = None,
        use_local: bool = False,
        **kwargs,
    ):
        """
            Initializes a partially preconfigured VASP calculator instance.

            Args:
                data (str | DataLoader): path to the dataset or initialized DataLoader object.
                cmd (MPI | SLURM | list[MPI] | list[SLURM]): command or commands used for VASP execution.
                remote_path (str): root directory where calculation work folders will be created.
                local_path (str | None, optional): local directory for copied calculation outputs. Defaults to None.
                input_paths (str | os.PathLike | list[str | os.PathLike] | None, optional): path or paths to static VASP input templates. Defaults to None.
                inputs (VaspInputs | None, optional): preconfigured VaspInputs object. Defaults to None.
                parser (Callable | None, optional): custom VASP result parser. Defaults to None.
                loading_files (list[str] | None, optional): files that should be copied after calculation. Defaults to None.
                transform (Callable[[Any], Any] | None, optional): transformation applied to CIF data before POSCAR generation. Defaults to None.
                potcar_dir (str | os.PathLike | None, optional): directory with POTCAR fragments. Defaults to None.
                potcar_map (dict[str, str] | None, optional): mapping from element symbols to POTCAR fragment names. Defaults to None.
                assemble_potcar (bool, optional): if True POTCAR is assembled dynamically for each structure. Defaults to False.
                connection_config (RemoteConfiguration | list[RemoteConfiguration] | None, optional): instance/instances of the RemoteConfiguration class. Should be used for remote connection setup. Defaults to None.
                use_local (bool, optional): use the local computer for calculations. Defaults to False.
                **kwargs: additional keyword arguments forwarded to Calculator.
        """
        
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
            parser=parser or default_vasp_parser,
            cmd=cmd,
            remote_path=remote_path,
            local_path=local_path,
            loading_files=loading_files or self.DEFAULT_LOADING_FILES,
            connection_config=connection_config,
            use_local=use_local,
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
        cmd: MPI | SLURM | list[MPI] | list[SLURM],
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
        connection_config: RemoteConfiguration | list[RemoteConfiguration] | None = None,
        use_local: bool = False,
        **kwargs,
    ):
        """
            Initializes a partially preconfigured Quantum Espresso calculator instance.

            Args:
                data (str | DataLoader): path to the dataset or initialized DataLoader object.
                cmd (str | list[str]): command or commands used for Quantum Espresso execution.
                remote_path (str): root directory where calculation work folders will be created.
                local_path (str | None, optional): local directory for copied calculation outputs. Defaults to None.
                input_paths (str | os.PathLike | list[str | os.PathLike] | None, optional): path or paths to static QE input templates. Defaults to None.
                inputs (EspressoInputs | None, optional): preconfigured EspressoInputs object. Defaults to None.
                parser (Callable | None, optional): custom QE result parser. Defaults to None.
                loading_files (list[str] | None, optional): files that should be copied after calculation. Defaults to None.
                transform (Callable[[Any], Any] | None, optional): transformation applied to CIF data before input generation. Defaults to None.
                pseudopotentials (dict[str, str] | None, optional): explicit pseudopotential mapping. Defaults to None.
                pseudo_dir (str | None, optional): relative pseudo directory written into pw.in. Defaults to None.
                control (dict[str, Any] | None, optional): Quantum Espresso CONTROL namelist. Defaults to None.
                system (dict[str, Any] | None, optional): Quantum Espresso SYSTEM namelist. Defaults to None.
                electrons (dict[str, Any] | None, optional): Quantum Espresso ELECTRONS namelist. Defaults to None.
                ions (dict[str, Any] | None, optional): Quantum Espresso IONS namelist. Defaults to None.
                cell (dict[str, Any] | None, optional): Quantum Espresso CELL namelist. Defaults to None.
                kpoints_grid (tuple[int, int, int], optional): Monkhorst-Pack k-point grid. Defaults to (1, 1, 1).
                kpoints_shift (tuple[int, int, int], optional): k-point grid shift. Defaults to (0, 0, 0).
                use_primitive (bool, optional): if True primitive standardized structure is used. Defaults to False.
                symprec (float, optional): tolerance used during symmetry analysis. Defaults to 1e-3.
                pwxml_kwargs (dict[str, Any] | None, optional): keyword arguments forwarded to PWxml parser. Defaults to None.
                connection_config (RemoteConfiguration | list[RemoteConfiguration] | None, optional): instance/instances of the RemoteConfiguration class. Should be used for remote connection setup. Defaults to None.
                use_local (bool, optional): use the local computer for calculations. Defaults to False.
                **kwargs: additional keyword arguments forwarded to Calculator.
        """
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
            parser=parser or partial(default_espresso_parser, pwxml_kwargs=pwxml_kwargs),
            cmd=cmd,
            remote_path=remote_path,
            local_path=local_path,
            loading_files=loading_files or self.DEFAULT_LOADING_FILES,
            connection_config=connection_config,
            use_local=use_local,
            **kwargs,
        )
