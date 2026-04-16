"""Module with Worker and RemoteWorker classes for non-blocking calculations control."""

__all__ = [
    "JobState",
    "Worker",
    "RemoteWorker",
]


import os
import posixpath
import re
import shlex
import shutil
import stat
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import paramiko
import scp
from csslib.tools.calculations.cmd import MPI, SLURM
from csslib.logging_ import get_tools_logger

logger = get_tools_logger("calculations.worker")


class JobState(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


@dataclass
class JobInspection:
    state: JobState
    message: str | None = None


class _BaseWorker:
    """Common functionality for local and remote workers."""

    FAILED_SLURM_STATES = {"BOOT_FAIL", "CANCELLED", "DEADLINE", "FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "TIMEOUT"}
    PENDING_SLURM_STATES = {"CONFIGURING", "PENDING", "RESV_DEL_HOLD", "REQUEUE_FED", "REQUEUE_HOLD", "REQUEUED", "RESIZING", "SIGNALING", "SPECIAL_EXIT", "STAGE_OUT", "SUSPENDED"}
    RUNNING_SLURM_STATES = {"COMPLETING", "RUNNING"}
    
    GENERATED_SBATCH_SCRIPT = "csslib.sh"
    MAX_UNKNOWN_SLURM_POLLS = 12
    COMPLETED_SLURM_STATES = {"COMPLETED"}
    KNOWN_SLURM_STATES = FAILED_SLURM_STATES | PENDING_SLURM_STATES | RUNNING_SLURM_STATES | COMPLETED_SLURM_STATES

    def __init__(
        self,
        cmd: MPI | SLURM,
        inputs: Any,
        parser: Callable[[str], object] | Any,
        structure: str,
        loading_files: list[str],
        calculation_name: str,
        local_path: str,
    ):
        """
        Initializes shared worker state, paths and execution metadata.

        Args:
            cmd (MPI | SLURM): launch command for the calculation.
            inputs (object): input-set object responsible for writing calculation inputs.
            parser (object): parser callable applied after successful calculation completion.
            structure (str): CIF-like structure representation passed into the input-set object.
            loading_files (list[str]): relative paths that should be copied after the calculation finishes.
            calculation_name (str): unique calculation folder name.
            local_path (str): root directory for locally stored copied outputs.
        """
        
        self._cmd = cmd
        self._inputs = inputs
        self._parser = parser
        self._structure = structure
        self._loading_files = loading_files
        self._calculation_name = calculation_name
        self._local_result_path = os.path.join(local_path, calculation_name)

        self._started = False
        self._jobid = None
        self._pid = None
        self._process = None
        self._last_state = JobState.UNKNOWN
        self._stderr_cache = None
        self._unknown_slurm_polls = 0
        self._exitcode_filename = ".csslib.exitcode"

    @staticmethod
    def _normalize_slurm_state(state: str) -> str:
        """
            Normalizes a raw SLURM state string for comparisons.

            Args:
                state (str): raw SLURM state string.

            Return:
                str: normalized uppercase SLURM state without suffixes.
        """
        
        return state.strip().upper().split("+", maxsplit=1)[0]

    @staticmethod
    def _extract_jobid(stdout: str, stderr: str) -> int | None:
        """
            Extracts a numeric job identifier from scheduler output text.

            Args:
                stdout (str): stdout text produced by a SLURM submission command.
                stderr (str): stderr text produced by a SLURM submission command.

            Return:
                int | None: extracted SLURM job id or None when no integer token was found.
        """
        
        combined_text = f"{stdout}\n{stderr}"
        last_line = _BaseWorker._last_nonempty_line(combined_text)
        match = re.search(r"\b(\d+)\b", last_line)
        if match:
            return int(match.group(1))
        match = re.search(r"\b(\d+)\b", combined_text)
        return int(match.group(1)) if match else None

    @staticmethod
    @contextmanager
    def _pushd(path: str):
        """
            Temporarily switches the current working directory.

            Args:
                path (str): directory that should be used temporarily as the current working directory.
        """
        previous_dir = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(previous_dir)

    def _write_inputs(self, target_dir: str):
        """
            Loads the structure into the input set and writes prepared input files.

            Args:
                target_dir (str): path to the directory where prepared input files should be written.
        """
        self._inputs.load_cif(self._structure)
        writer = self._inputs.write
        try:
            writer(target_dir)
            return
        except TypeError:
            pass
        try:
            writer(path=target_dir)
            return
        except TypeError:
            pass
        with self._pushd(target_dir):
            writer()

    @staticmethod
    def _copy_path(source: str, destination: str):
        """
            Copies a file or directory between local filesystem paths.

            Args:
                source (str): source file or directory path.
                destination (str): destination file or directory path.
        """
        if os.path.abspath(source) == os.path.abspath(destination):
            return
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        if os.path.isdir(source):
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)

    def _path_exists_in_input_dir(self, candidate: str) -> bool:
        """
            Checks whether a file already exists in the prepared input directory.

            Args:
                candidate (str): path or file name that should be checked inside the prepared input directory.

            Raise:
                NotImplementedError: if the concrete worker does not implement filesystem-specific existence checks.

            Return:
                bool: True if the candidate already exists as a file in the input directory.
        """
        raise NotImplementedError

    def _write_generated_sbatch_script(self, payload_command: str):
        """
            Writes the generated wrapper script into the worker-specific filesystem.

            Args:
                payload_command (str): full shell script text that should be written into the generated launch script.

            Raise:
                NotImplementedError: if the concrete worker does not implement script writing.
        """
        raise NotImplementedError

    def _mark_unknown_slurm_poll(self, message: str) -> JobInspection:
        """
            Tracks repeated unknown SLURM polls and eventually turns them into a failure.

            Args:
                message (str): fallback message describing why SLURM state is currently unknown.

            Return:
                JobInspection: inspection result that stays UNKNOWN for a grace period and turns into FAILED afterwards.
        """
        self._unknown_slurm_polls += 1
        if self._unknown_slurm_polls >= self.MAX_UNKNOWN_SLURM_POLLS:
            stderr_text = self._read_job_streams_for_error()
            if stderr_text:
                return JobInspection(JobState.FAILED, stderr_text)
            return JobInspection(JobState.FAILED, message)
        return JobInspection(JobState.UNKNOWN, message)

    def _reset_unknown_slurm_polls(self):
        """
            Resets the counter of consecutive unknown SLURM polls.
        """
        self._unknown_slurm_polls = 0

    @staticmethod
    def _tail_text(text: str | None, max_lines: int = 40) -> str | None:
        """
            Returns either the full text or a shortened tail of a long log.

            Args:
                text (str | None): full text that should be shortened.
                max_lines (int, optional): maximal number of tail lines that should be preserved. Defaults to 40.

            Return:
                str | None: original text, shortened tail or None when the input text is empty.
        """
        
        if not text:
            return None
        lines = text.splitlines()
        if len(lines) <= max_lines:
            return text.strip()
        return "\n".join(lines[-max_lines:]).strip()

    @staticmethod
    def _last_nonempty_line(text: str | None) -> str:
        """
            Returns the last non-empty line from a text block.

            Args:
                text (str | None): raw text that may contain banners or empty lines.

            Return:
                str: last non-empty stripped line or an empty string.
        """

        if not text:
            return ""
        for line in reversed(text.splitlines()):
            stripped_line = line.strip()
            if stripped_line:
                return stripped_line
        return ""

    @classmethod
    def _extract_slurm_states(cls, text: str | None) -> list[str]:
        """
            Extracts normalized SLURM states from a raw command output.

            Args:
                text (str | None): raw output of squeue or sacct.

            Return:
                list[str]: recognized normalized SLURM states.
        """

        if not text:
            return []
        states = []
        for line in text.splitlines():
            normalized_state = cls._normalize_slurm_state(line)
            if normalized_state in cls.KNOWN_SLURM_STATES:
                states.append(normalized_state)
        return states

    @staticmethod
    def _collect_local_files(source_root: str, destination_root: str, loading_files: list[str]):
        """
            Copies selected output files from a work directory into persistent local storage.

            Args:
                source_root (str): root directory containing produced calculation files.
                destination_root (str): root directory where selected output files should be copied.
                loading_files (list[str]): relative paths that should be copied if they exist.
        """
        
        os.makedirs(destination_root, exist_ok=True)
        for relative_path in loading_files:
            source_path = os.path.join(source_root, relative_path)
            if not os.path.exists(source_path):
                continue
            destination_path = os.path.join(destination_root, relative_path)
            _BaseWorker._copy_path(source_path, destination_path)

    def _get_failure_message(self) -> str:
        """
            Returns cached stderr text or a generic worker failure message.

            Return:
                str: cached stderr text or a generic failure message.
        """
        
        return self._stderr_cache or "Calculation failed."

    def start(self) -> dict:
        """
            Prepares the workspace and starts the calculation in non-blocking mode.

            Return:
                dict: bookkeeping information about the started calculation.
        """
        
        logger.debug("Preparing worker workspace for %s.", self._calculation_name)
        self._prepare_workspace()
        self._prepare_inputs()
        self._launch_nonblocking()
        self._started = True
        logger.info("Worker %s is started.", self._calculation_name)
        return self.get_calculation_info()

    def inspect(self) -> JobInspection:
        """
            Inspects the current state of a previously started calculation.

            Return:
                JobInspection: current state of the started calculation.
        """
        if not self._started:
            return JobInspection(JobState.UNKNOWN, "Calculation was not started.")
        inspection = self._inspect_slurm_job() if isinstance(self._cmd, SLURM) else self._inspect_process_job()
        self._last_state = inspection.state
        return inspection

    def finalize(self, verify_completion: bool = True) -> tuple[dict, object]:
        """
            Collects outputs and parses the result of a completed calculation.

            Raise:
                RuntimeError: if finalize is called before the calculation reaches COMPLETED state.

            Return:
                tuple[dict, object]: calculation info dictionary and parser output object.
        """
        if verify_completion:
            inspection = self.inspect()
            if inspection.state != JobState.COMPLETED:
                raise RuntimeError(inspection.message or f"Cannot finalize calculation in state {inspection.state.value}.")
        logger.info("Collecting outputs for worker %s.", self._calculation_name)
        self._collect_outputs()
        logger.info("Parsing outputs for worker %s from %s.", self._calculation_name, self._local_result_path)
        calculation_output = self._parser(self._local_result_path)
        logger.info("Worker %s is finalized successfully.", self._calculation_name)
        return self.get_calculation_info(), calculation_output

    def cancel(self) -> bool:
        """
            Cancels a started calculation when the underlying launcher supports it.

            Return:
                bool: True if the running or pending calculation was cancelled successfully.
        """
        if not self._started:
            return False
        cancelled = self._cancel_slurm_job() if isinstance(self._cmd, SLURM) else self._cancel_process_job()
        if cancelled:
            self._last_state = JobState.CANCELLED
            logger.warning("Worker %s was cancelled.", self._calculation_name)
        return cancelled

    def get_calculation_info(self) -> dict:
        """
            Returns bookkeeping metadata for the current calculation.

            Return:
                dict: dictionary with server, username, job identifier and local/remote paths of the calculation.
        """
        
        return {
            "server_ip": self._get_server_ip(),
            "username": self._get_username(),
            "jobid": self._jobid if self._jobid is not None else self._pid,
            "local_path": self._local_result_path,
            "remote_path": self._get_execution_path(),
        }

    def can_migrate_pending(self) -> bool:
        """
            Checks whether a pending SLURM job can be migrated to another resource.

            Return:
                bool: True if the calculation is a pending sbatch job that can be cancelled and relaunched elsewhere.
        """
        
        if isinstance(self._cmd, MPI) or self._jobid is None:
            return False
        return self.inspect().state == JobState.PENDING

    def _prepare_workspace(self):
        """
            Prepares filesystem state required before starting a calculation.

            Raise:
                NotImplementedError: if the concrete worker does not implement workspace preparation.
        """
        raise NotImplementedError

    def _launch_nonblocking(self):
        """
            Starts calculation execution without blocking the caller.

            Raise:
                NotImplementedError: if the concrete worker does not implement non-blocking launch.
        """
        raise NotImplementedError

    def _inspect_process_job(self) -> JobInspection:
        """
            Inspects the state of a non-SLURM process.

            Raise:
                NotImplementedError: if the concrete worker does not implement process-state inspection.

            Return:
                JobInspection: current inspection result for a non-SLURM process.
        """
        raise NotImplementedError

    def _inspect_slurm_job(self) -> JobInspection:
        """
            Inspects the state of a SLURM-managed job.

            Raise:
                NotImplementedError: if the concrete worker does not implement SLURM-state inspection.

            Return:
                JobInspection: current inspection result for a SLURM-submitted job.
        """
        raise NotImplementedError

    def _cancel_process_job(self) -> bool:
        """
            Cancels a non-SLURM process.

            Raise:
                NotImplementedError: if the concrete worker does not implement process cancellation.

            Return:
                bool: True if cancellation succeeded.
        """
        raise NotImplementedError

    def _cancel_slurm_job(self) -> bool:
        """
            Cancels a SLURM-managed job.

            Raise:
                NotImplementedError: if the concrete worker does not implement SLURM job cancellation.

            Return:
                bool: True if cancellation succeeded.
        """
        raise NotImplementedError

    def _collect_outputs(self):
        """
            Collects requested calculation outputs after completion.

            Raise:
                NotImplementedError: if the concrete worker does not implement output collection.
        """
        raise NotImplementedError

    def _prepare_inputs(self):
        """
            Writes calculation inputs and prepares the command and script for sbatch or srun.
        """
        
        path = self._get_input_write_path()
        self._write_inputs(path)
        self._write_generated_sbatch_script(self._cmd.get_script(path))

    def _get_input_write_path(self) -> str:
        """
            Returns the path where input files should be written.

            Raise:
                NotImplementedError: if the concrete worker does not define where input files should be written.

            Return:
                str: path where input files should be created before launch.
        """
        raise NotImplementedError

    def _get_execution_path(self) -> str:
        """
            Returns the path where the calculation should be executed.

            Raise:
                NotImplementedError: if the concrete worker does not define its execution directory.

            Return:
                str: path where the calculation will be executed.
        """
        raise NotImplementedError

    def _get_server_ip(self) -> str | None:
        """
            Returns the server identifier used in calculation bookkeeping.

            Raise:
                NotImplementedError: if the concrete worker does not define its server identifier.

            Return:
                str | None: server IP or local marker for the calculation.
        """
        raise NotImplementedError

    def _get_username(self) -> str | None:
        """
            Returns the username associated with the calculation.

            Raise:
                NotImplementedError: if the concrete worker does not define its username.

            Return:
                str | None: username used for the calculation or None when not applicable.
        """
        raise NotImplementedError

    def _read_job_streams_for_error(self) -> str | None:
        """
            Returns the best available error text from stderr or stdout.

            Raise:
                NotImplementedError: if the concrete worker does not implement stderr/stdout reading.

            Return:
                str | None: best available error text for the current calculation.
        """
        raise NotImplementedError

    def _read_exitcode_value(self) -> int | None:
        """
            Returns the stored process or job exit code when it is available.

            Raise:
                NotImplementedError: if the concrete worker does not implement exit code reading.

            Return:
                int | None: stored process or job exit code when available.
        """
        raise NotImplementedError



class Worker(_BaseWorker):
    """
        Class for managing CSS structure calculations on the local computer.
    """

    def __init__(
        self,
        cmd: MPI | SLURM,
        inputs: Any,
        parser: Callable[[str], object] | Any,
        structure: str,
        loading_files: list[str],
        calculation_name: str,
        work_root: str,
        local_path: str,
    ):
        """
        Initializes local worker paths and state for one calculation.

        Args:
            cmd (MPI | SLURM): launch command for the local calculation.
            inputs (object): input-set object responsible for writing local inputs.
            parser (object): parser callable applied after successful completion.
            structure (str): CIF-like structure representation.
            loading_files (list[str]): relative paths that should be copied after completion.
            calculation_name (str): unique calculation folder name.
            work_root (str): root directory where the calculation working directory will be created.
            local_path (str): root directory for copied outputs.
        """
        
        super().__init__(cmd, inputs, parser, structure, loading_files, calculation_name, local_path)
        self._work_path = os.path.join(work_root, calculation_name)
        self._stdout_path = os.path.join(self._work_path, self._cmd.output)
        self._stderr_path = os.path.join(self._work_path, self._cmd.error)

    def _prepare_workspace(self):
        """
            Creates local working and result directories for the calculation.
        """
        
        os.makedirs(self._work_path, exist_ok=True)
        os.makedirs(self._local_result_path, exist_ok=True)

    def _launch_nonblocking(self):
        """
            Launches the local calculation or submits a local sbatch job in non-blocking mode.

            Raise:
                RuntimeError: if local sbatch submission fails or the SLURM job id cannot be extracted.
        """
        
        if isinstance(self._cmd, SLURM) and self._cmd.prefix == "sbatch":
            result = subprocess.run(self._cmd.get_cmd(), cwd=self._work_path, capture_output=True, text=True, shell=True)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"SLURM command failed in {self._work_path}.")
            self._jobid = self._extract_jobid(result.stdout, result.stderr)
            if self._jobid is None:
                raise RuntimeError("Unable to extract SLURM jobid from sbatch output.")
            return

        with open(self._stdout_path, "w", encoding="utf-8") as stdout_file, open(self._stderr_path, "w", encoding="utf-8") as stderr_file:
            self._process = subprocess.Popen(
                self._cmd.get_cmd(),
                cwd=self._work_path,
                shell=True,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
            )
        self._pid = self._process.pid

    def _inspect_process_job(self) -> JobInspection:
        """
            Inspects the state of a local non-SLURM process.

            Return:
                JobInspection: current state of the local non-SLURM process.
        """
        
        if self._process is None:
            return JobInspection(JobState.UNKNOWN, "Local process handle is missing.")
        returncode = self._process.poll()
        if returncode is None:
            return JobInspection(JobState.RUNNING)
        if returncode == 0:
            return JobInspection(JobState.COMPLETED)
        self._stderr_cache = self._read_local_file(self._stderr_path)
        return JobInspection(JobState.FAILED, self._stderr_cache or f"Process exited with code {returncode}.")

    def _inspect_slurm_job(self) -> JobInspection:
        """
            Inspects the state of a local SLURM job.

            Return:
                JobInspection: current state of the local SLURM job.
        """
        
        exitcode = self._read_exitcode_value()
        if exitcode is not None:
            self._reset_unknown_slurm_polls()
            if exitcode == 0:
                return JobInspection(JobState.COMPLETED)
            return JobInspection(JobState.FAILED, self._read_job_streams_for_error() or f"SLURM job exited with code {exitcode}.")

        squeue = subprocess.run(f"squeue -h -j {self._jobid} -o %T", cwd=self._work_path, capture_output=True, text=True, shell=True)
        states = self._extract_slurm_states(squeue.stdout)
        if states:
            self._reset_unknown_slurm_polls()
            return self._inspection_from_slurm_states(states)

        scontrol = subprocess.run(f"scontrol show job {self._jobid}", cwd=self._work_path, capture_output=True, text=True, shell=True)
        if scontrol.returncode == 0:
            states = re.findall(r"JOBSTATE=([A-Z_]+)", scontrol.stdout.upper())
            if states:
                self._reset_unknown_slurm_polls()
                return self._inspection_from_slurm_states(states)

        sacct = subprocess.run(f"sacct -n -P -j {self._jobid} -o State", cwd=self._work_path, capture_output=True, text=True, shell=True)
        states = self._extract_slurm_states(sacct.stdout)
        if states:
            self._reset_unknown_slurm_polls()
            return self._inspection_from_slurm_states(states)
        return self._mark_unknown_slurm_poll(f"SLURM job {self._jobid} state could not be determined yet.")

    def _inspection_from_slurm_states(self, states: list[str]) -> JobInspection:
        """
            Converts observed local SLURM states into one inspection result.

            Args:
                states (list[str]): normalized SLURM states observed for the current job.

            Return:
                JobInspection: consolidated inspection result derived from the provided SLURM states.
        """
        
        if any(state in self.FAILED_SLURM_STATES for state in states):
            return JobInspection(JobState.FAILED, self._read_job_streams_for_error() or f"SLURM job {self._jobid} failed with state: {', '.join(states)}.")
        if any(state in self.PENDING_SLURM_STATES for state in states):
            return JobInspection(JobState.PENDING)
        if any(state in self.RUNNING_SLURM_STATES for state in states):
            return JobInspection(JobState.RUNNING)
        if all(state == "COMPLETED" for state in states):
            return JobInspection(JobState.COMPLETED)
        return JobInspection(JobState.RUNNING)

    def _cancel_process_job(self) -> bool:
        """
            Terminates local process execution if it is still running.

            Return:
                bool: True if the local process was still running and was terminated.
        """
        
        if self._process is None:
            return False
        if self._process.poll() is None:
            self._process.terminate()
            return True
        return False

    def _cancel_slurm_job(self) -> bool:
        """
            Cancels a local SLURM job through scancel.

            Return:
                bool: True if local scancel finished successfully.
        """
        
        if self._jobid is None:
            return False
        result = subprocess.run(f"scancel {self._jobid}", cwd=self._work_path, capture_output=True, text=True, shell=True)
        return result.returncode == 0

    def _collect_outputs(self):
        """
            Copies selected local calculation outputs into the persistent local result directory.
        """
        
        logger.debug("Copying local outputs for worker %s from %s to %s.", self._calculation_name, self._work_path, self._local_result_path)
        self._collect_local_files(self._work_path, self._local_result_path, self._loading_files)

    def _prepare_inputs(self):
        """
            Delegates input preparation to the shared base implementation for local execution.
        """
        
        super()._prepare_inputs()

    def _get_input_write_path(self) -> str:
        """
            Returns the local working directory used for input generation.

            Return:
                str: local working directory where input files should be written.
        """
        return self._work_path

    def _path_exists_in_input_dir(self, candidate: str) -> bool:
        """
            Checks whether a file already exists in the local working directory.

            Args:
                candidate (str): file name or path that should be checked inside the local work directory.

            Return:
                bool: True if the candidate file already exists in the local work directory.
        """
        
        candidate_path = candidate if os.path.isabs(candidate) else os.path.join(self._work_path, candidate)
        return os.path.isfile(candidate_path)

    def _write_generated_sbatch_script(self, payload_command: str):
        """
            Writes the generated wrapper script into the local working directory.

            Args:
                payload_command (str): generated shell script text for sbatch or srun execution.
        """
        
        script_path = os.path.join(self._work_path, self.GENERATED_SBATCH_SCRIPT)
        with open(script_path, "w", encoding="utf-8", newline="\n") as file:
            file.write(payload_command)
        os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    def _get_execution_path(self) -> str:
        """
            Returns the local execution directory.

            Return:
                str: local working directory used for calculation execution.
        """
        
        return self._work_path

    def _get_server_ip(self) -> str:
        """
            Returns the local server marker for bookkeeping.

            Return:
                str: local server marker used by calculator bookkeeping.
        """
        
        return "local"

    def _get_username(self) -> None:
        """
            Returns no username for local execution.

            Return:
                None: local worker does not track remote username.
        """
        
        return None

    @staticmethod
    def _read_local_file(path: str) -> str | None:
        """
            Reads a local text file when it exists.

            Args:
                path (str): local file path that should be read.

            Return:
                str | None: file content or None when the file does not exist.
        """
        
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8", errors="ignore") as file:
            return file.read().strip()

    def _read_job_streams_for_error(self) -> str | None:
        """
            Returns the best available local error text.

            Return:
                str | None: stderr text or a shortened stdout tail for the local calculation.
        """
        
        stderr_text = self._read_local_file(self._stderr_path)
        if stderr_text:
            return stderr_text
        return self._tail_text(self._read_local_file(self._stdout_path))

    def _read_exitcode_value(self) -> int | None:
        """
            Reads the stored local exit code from the wrapper file.

            Return:
                int | None: stored local exit code when it was written by the wrapper script.
        """
        
        exitcode_path = os.path.join(self._work_path, self._exitcode_filename)
        exitcode_text = self._read_local_file(exitcode_path)
        if not exitcode_text:
            return None
        try:
            return int(exitcode_text)
        except ValueError:
            return None


class RemoteWorker(_BaseWorker):
    """
        Class for managing CSS structure calculations on the remote server.
    """

    def __init__(
        self,
        ssh: paramiko.SSHClient,
        file_sharing: paramiko.SFTPClient | scp.SCPClient,
        server_ip: str,
        username: str,
        cmd: MPI | SLURM,
        inputs: Any,
        parser: Callable[[str], object] | Any,
        structure: str,
        loading_files: list[str],
        calculation_name: str,
        remote_root: str,
        local_path: str,
        connection_owner: Any | None = None,
    ):
        """
        Initializes remote worker paths, staging directories and connection handles.

        Args:
            ssh (object): SSH client used for remote command execution.
            file_sharing (object): SFTP or SCP client used for file transfer.
            server_ip (str): remote server address.
            username (str): username used on the remote server.
            cmd (str): launch command for the remote calculation.
            inputs (object): input-set object responsible for writing staged inputs.
            parser (object): parser callable applied after successful completion.
            structure (object): CIF-like structure representation.
            loading_files (list[str]): relative paths that should be downloaded after completion.
            calculation_name (str): unique calculation folder name.
            remote_root (str): root directory for remote working folders.
            local_path (str): root directory for copied outputs and local staging.
            connection_owner (object, optional): remote connection wrapper capable of refreshing SSH and file-sharing clients.
            Defaults to None.
        """
        
        super().__init__(cmd, inputs, parser, structure, loading_files, calculation_name, local_path)
        self._connection_owner = connection_owner
        self._ssh = ssh
        self._file_sharing = file_sharing
        self._server_ip = server_ip
        self._username = username
        self._remote_path = posixpath.join(remote_root, calculation_name)
        self._staging_path = os.path.join(local_path, f".staging_{calculation_name}")
        self._stdout_filename = self._cmd.output
        self._stderr_filename = self._cmd.error

    def _prepare_workspace(self):
        """
            Creates local staging and result directories for the remote calculation.
        """
        
        os.makedirs(self._staging_path, exist_ok=True)
        os.makedirs(self._local_result_path, exist_ok=True)

    def _exec_remote(
        self,
        command: str,
        *,
        login_shell: bool = True,
        interactive_shell: bool = False,
        allocate_pty: bool = False,
    ) -> tuple[str, str, int]:
        """
            Executes one remote shell command and returns stdout, stderr and exit code.

            Args:
                command (str): shell command that should be executed on the remote machine.

            Return:
                tuple[str, str, int]: stdout, stderr and return code of the finished remote command.
        """
        
        self._refresh_remote_clients()
        shell_flags = []
        if login_shell:
            shell_flags.append("-l")
        if interactive_shell:
            shell_flags.append("-i")
        shell_flags.append("-c")
        wrapped_command = f"bash {' '.join(shell_flags)} {shlex.quote(command)}"
        _, stdout, stderr = self._ssh.exec_command(wrapped_command, get_pty=allocate_pty)
        stdout_text = stdout.read().decode("utf-8", errors="ignore").strip()
        stderr_text = stderr.read().decode("utf-8", errors="ignore").strip()
        return stdout_text, stderr_text, stdout.channel.recv_exit_status()

    def _refresh_remote_clients(self): # TODO: change function behaviour and reconnect when some troubles occured.
        """
            Refreshes SSH and file-sharing clients from the owning connection object.
        """
        
        if self._connection_owner is None:
            return
        clients = self._connection_owner()
        self._ssh = clients["SSH"]
        self._file_sharing = clients["SFTP/SCP"]

    def _launch_nonblocking(self):
        """
            Uploads inputs and launches the remote calculation in non-blocking mode.

            Raise:
                RuntimeError: if remote job submission or non-blocking process launch fails.
        """
        
        self._upload_inputs()
        if isinstance(self._cmd, SLURM) and self._cmd.prefix == "sbatch":
            stdout, stderr, returncode = self._exec_remote(
                f"cd {shlex.quote(self._remote_path)} && {self._cmd.get_cmd()}",
                login_shell=True,
                interactive_shell=True,
                allocate_pty=True,
            )
            shutil.rmtree(self._staging_path, ignore_errors=True)
            if returncode != 0:
                raise RuntimeError(stderr or stdout or f"Remote SLURM command failed in {self._remote_path}.")
            self._jobid = self._extract_jobid(stdout, stderr)
            if self._jobid is None:
                raise RuntimeError("Unable to extract SLURM jobid from remote sbatch output.")
            return

        command = (
            f"cd {shlex.quote(self._remote_path)} && "
            f"( bash -lc {shlex.quote(self._cmd.get_cmd())} "
            f"> {shlex.quote(self._stdout_filename)} "
            f"2> {shlex.quote(self._stderr_filename)} "
            f"< /dev/null; printf '%s' $? > {shlex.quote(self._exitcode_filename)} ) & echo $!"
        )
        stdout, stderr, returncode = self._exec_remote(
            command,
            login_shell=True,
            interactive_shell=True,
            allocate_pty=True,
        )
        shutil.rmtree(self._staging_path, ignore_errors=True)
        if returncode != 0:
            raise RuntimeError(stderr or stdout or f"Remote command failed to start in {self._remote_path}.")
        try:
            self._pid = int(stdout.splitlines()[-1].strip())
        except (IndexError, ValueError) as exc:
            raise RuntimeError(f"Unable to extract remote pid from command output: {stdout}") from exc

    def _upload_inputs(self):
        """
            Uploads staged input files from the local staging directory to the remote work directory.
        """
        
        if isinstance(self._file_sharing, scp.SCPClient):
            self._mkdir_remote(self._remote_path)
            for entry in os.listdir(self._staging_path):
                local_item = os.path.join(self._staging_path, entry)
                self._file_sharing.put(local_item, remote_path=self._remote_path, recursive=os.path.isdir(local_item))
            return

        self._mkdir_remote(self._remote_path)
        for root, _, files in os.walk(self._staging_path):
            relative_root = os.path.relpath(root, self._staging_path)
            remote_root = self._remote_path if relative_root == "." else posixpath.join(self._remote_path, relative_root.replace("\\", "/"))
            self._mkdir_remote(remote_root)
            for file_name in files:
                local_file = os.path.join(root, file_name)
                remote_file = posixpath.join(remote_root, file_name)
                self._file_sharing.put(local_file, remote_file)

    def _mkdir_remote(self, remote_path: str):
        """
            Creates a directory on the remote machine.

            Args:
                remote_path (str): remote directory path that should be created.
        """
        
        self._exec_remote(f"mkdir -p {shlex.quote(remote_path)}")

    def _path_exists_in_input_dir(self, candidate: str) -> bool:
        """
            Checks whether a file already exists in the local staging directory.

            Args:
                candidate (str): file name or path that should be checked inside the local staging directory.

            Return:
                bool: True if the candidate file exists in the staging directory.
        """
        
        candidate_path = candidate if os.path.isabs(candidate) else os.path.join(self._staging_path, candidate)
        return os.path.isfile(candidate_path)

    def _write_generated_sbatch_script(self, payload_command: str):
        """
            Writes the generated wrapper script into the local staging directory.

            Args:
                payload_command (str): generated shell script text for sbatch or srun execution.
        """
        
        script_path = os.path.join(self._staging_path, self.GENERATED_SBATCH_SCRIPT)
        with open(script_path, "w", encoding="utf-8", newline="\n") as file:
            file.write(payload_command)

    def _inspect_process_job(self) -> JobInspection:
        """
            Inspects the state of a remote non-SLURM process.

            Return:
                JobInspection: current state of the remote non-SLURM process.
        """
        
        command = (
            f"cd {shlex.quote(self._remote_path)} && "
            f"if [ -f {shlex.quote(self._exitcode_filename)} ]; then "
            f"printf 'DONE %s' \"$(cat {shlex.quote(self._exitcode_filename)})\"; "
            f"elif kill -0 {self._pid} 2>/dev/null; then echo RUNNING; "
            f"else echo UNKNOWN; fi"
        )
        stdout, stderr, returncode = self._exec_remote(command, login_shell=True, interactive_shell=False, allocate_pty=False)
        if returncode != 0 and stderr:
            return JobInspection(JobState.UNKNOWN, stderr)
        marker_line = self._last_nonempty_line(stdout)
        if marker_line.startswith("DONE"):
            exitcode = int(marker_line.split(maxsplit=1)[1])
            if exitcode == 0:
                return JobInspection(JobState.COMPLETED)
            self._stderr_cache = self._read_remote_file(self._stderr_filename)
            return JobInspection(JobState.FAILED, self._stderr_cache or f"Remote process exited with code {exitcode}.")
        if marker_line == "RUNNING":
            return JobInspection(JobState.RUNNING)
        return JobInspection(JobState.UNKNOWN, self._tail_text(stdout) or "Remote process state could not be determined.")

    def _inspect_slurm_job(self) -> JobInspection:
        """
            Inspects the state of a remote SLURM job.

            Return:
                JobInspection: current state of the remote SLURM job.
        """
        
        exitcode = self._read_exitcode_value()
        if exitcode is not None:
            self._reset_unknown_slurm_polls()
            if exitcode == 0:
                return JobInspection(JobState.COMPLETED)
            return JobInspection(JobState.FAILED, self._read_job_streams_for_error() or f"Remote SLURM job exited with code {exitcode}.")

        stdout, _, _ = self._exec_remote(f"squeue -h -j {self._jobid} -o %T", login_shell=True, interactive_shell=False, allocate_pty=False)
        states = self._extract_slurm_states(stdout)
        if states:
            self._reset_unknown_slurm_polls()
            return self._inspection_from_slurm_states(states)

        stdout, _, returncode = self._exec_remote(f"scontrol show job {self._jobid}", login_shell=True, interactive_shell=False, allocate_pty=False)
        if returncode == 0:
            states = re.findall(r"JOBSTATE=([A-Z_]+)", stdout.upper())
            if states:
                self._reset_unknown_slurm_polls()
                return self._inspection_from_slurm_states(states)

        stdout, _, _ = self._exec_remote(f"sacct -n -P -j {self._jobid} -o State", login_shell=True, interactive_shell=False, allocate_pty=False)
        states = self._extract_slurm_states(stdout)
        if states:
            self._reset_unknown_slurm_polls()
            return self._inspection_from_slurm_states(states)
        return self._mark_unknown_slurm_poll(f"Remote SLURM job {self._jobid} state could not be determined yet.")

    def _inspection_from_slurm_states(self, states: list[str]) -> JobInspection:
        """
            Converts observed remote SLURM states into one inspection result.

            Args:
                states (list[str]): normalized SLURM states observed for the current remote job.

            Return:
                JobInspection: consolidated inspection result derived from the provided SLURM states.
        """
        
        if any(state in self.FAILED_SLURM_STATES for state in states):
            return JobInspection(JobState.FAILED, f"Remote SLURM job {self._jobid} failed with state: {', '.join(states)}.")
        if any(state in self.PENDING_SLURM_STATES for state in states):
            return JobInspection(JobState.PENDING)
        if any(state in self.RUNNING_SLURM_STATES for state in states):
            return JobInspection(JobState.RUNNING)
        if all(state == "COMPLETED" for state in states):
            return JobInspection(JobState.COMPLETED)
        return JobInspection(JobState.RUNNING)

    def _cancel_process_job(self) -> bool:
        """
            Cancels remote process execution.

            Return:
                bool: True if remote kill command finished successfully.
        """
        if self._pid is None:
            return False
        _, _, returncode = self._exec_remote(f"kill {self._pid}")
        return returncode == 0

    def _cancel_slurm_job(self) -> bool:
        """
            Cancels a remote SLURM job through scancel.

            Return:
                bool: True if remote scancel finished successfully.
        """
        if self._jobid is None:
            return False
        _, _, returncode = self._exec_remote(f"scancel {self._jobid}")
        return returncode == 0

    def _remote_path_type(self, remote_path: str) -> str | None:
        """
            Determines whether a remote path points to a file, a directory or nothing.

            Args:
                remote_path (str): remote file or directory path that should be inspected.

            Return:
                str | None: 'file', 'dir' or None when the path does not exist.
        """
        
        stdout, _, _ = self._exec_remote(
            f"if [ -d {shlex.quote(remote_path)} ]; then echo dir; "
            f"elif [ -f {shlex.quote(remote_path)} ]; then echo file; else echo missing; fi"
        )
        path_type = stdout.strip()
        return None if path_type == "missing" else path_type

    def _download_sftp_tree(self, remote_path: str, local_path: str):
        """
            Recursively downloads a remote directory tree over SFTP.

            Args:
                remote_path (str): remote directory path that should be downloaded recursively over SFTP.
            local_path (str): local destination directory.
        """
        
        os.makedirs(local_path, exist_ok=True)
        for entry in self._file_sharing.listdir_attr(remote_path):
            remote_child = posixpath.join(remote_path, entry.filename)
            local_child = os.path.join(local_path, entry.filename)
            if stat.S_ISDIR(entry.st_mode):
                self._download_sftp_tree(remote_child, local_child)
            else:
                os.makedirs(os.path.dirname(local_child), exist_ok=True)
                self._file_sharing.get(remote_child, local_child)

    def _download_sftp(self, remote_path: str, local_path: str):
        """
            Downloads one remote file or directory over SFTP.

            Args:
                remote_path (str): remote file or directory path that should be downloaded over SFTP.
                local_path (str): local destination path.
        """
        
        remote_stat = self._file_sharing.stat(remote_path)
        if stat.S_ISDIR(remote_stat.st_mode):
            self._download_sftp_tree(remote_path, local_path)
        else:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self._file_sharing.get(remote_path, local_path)

    def _download_scp(self, remote_path: str, local_path: str, recursive: bool):
        """
            Downloads one remote file or directory over SCP.

            Args:
                remote_path (str): remote file or directory path that should be downloaded over SCP.
                local_path (str): local destination path.
                recursive (bool): if True the remote path is downloaded recursively as a directory tree.
        """
        
        target_path = os.path.dirname(local_path) if recursive else local_path
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self._file_sharing.get(remote_path, local_path=target_path, recursive=recursive)

    def _collect_outputs(self):
        """
            Downloads selected remote calculation outputs into the persistent local result directory.
        """
        
        os.makedirs(self._local_result_path, exist_ok=True)
        for relative_path in self._loading_files:
            remote_item = posixpath.join(self._remote_path, relative_path.replace("\\", "/"))
            local_item = os.path.join(self._local_result_path, relative_path)
            path_type = self._remote_path_type(remote_item)
            if path_type is None:
                logger.debug("Remote output %s for worker %s is absent.", remote_item, self._calculation_name)
                continue
            logger.debug("Downloading remote output %s for worker %s to %s.", remote_item, self._calculation_name, local_item)
            if isinstance(self._file_sharing, scp.SCPClient):
                self._download_scp(remote_item, local_item, recursive=(path_type == "dir"))
            else:
                self._download_sftp(remote_item, local_item)

    def _prepare_inputs(self):
        """
            Delegates remote input preparation to the shared base implementation.
        """
        
        super()._prepare_inputs()

    def _get_input_write_path(self) -> str:
        """
            Returns the local staging directory used for remote input generation.

            Return:
                str: local staging directory where remote input files should be prepared.
        """
        
        return self._staging_path

    def _get_execution_path(self) -> str:
        """
            Returns the remote execution directory.

            Return:
                str: remote working directory used for calculation execution.
        """
        
        return self._remote_path

    def _get_server_ip(self) -> str:
        """
            Returns the remote server address for bookkeeping.

            Return:
                str: remote server address used by calculator bookkeeping.
        """
        
        return self._server_ip

    def _get_username(self) -> str:
        """
            Returns the remote username for bookkeeping.

            Return:
                str: username used on the remote server.
        """
        
        return self._username

    def _read_remote_file(self, relative_path: str) -> str | None:
        """
            Reads a text file from the remote working directory.

            Args:
                relative_path (str): relative path to the file inside the remote working directory.

            Return:
                str | None: remote file content or None when the file does not exist or cannot be read.
        """
        
        stdout, _, returncode = self._exec_remote(
            f"cd {shlex.quote(self._remote_path)} && cat {shlex.quote(relative_path)}",
            login_shell=True,
            interactive_shell=False,
            allocate_pty=False,
        )
        if returncode != 0:
            return None
        return stdout.strip()

    def _read_job_streams_for_error(self) -> str | None:
        """
            Returns the best available remote error text.

            Return:
                str | None: stderr text or a shortened stdout tail for the remote calculation.
        """
        
        stderr_text = self._read_remote_file(self._stderr_filename)
        if stderr_text:
            return stderr_text
        return self._tail_text(self._read_remote_file(self._stdout_filename))

    def _read_exitcode_value(self) -> int | None:
        """
            Reads the stored remote exit code from the wrapper file.

            Return:
                int | None: stored remote exit code when it was written by the wrapper script.
        """
        
        exitcode_text = self._read_remote_file(self._exitcode_filename)
        if not exitcode_text:
            return None
        try:
            return int(exitcode_text)
        except ValueError:
            return None
