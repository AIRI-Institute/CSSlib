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

import scp


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

    SUPPORTING_CMD_PREFIXES_SLURM = {"srun", "sbatch"}
    SUPPORTING_CMD_PREFIXES_MPI = {
        "mpi",
        "mpirun",
        "mpirun.mpich",
        "mpirun.openmpi",
        "mpiexec",
        "mpiexec.hydra",
        "mpiexec.mpich",
        "mpiexec.openmpi",
    }
    FAILED_SLURM_STATES = {"BOOT_FAIL", "CANCELLED", "DEADLINE", "FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "TIMEOUT"}
    PENDING_SLURM_STATES = {"CONFIGURING", "PENDING", "RESV_DEL_HOLD", "REQUEUE_FED", "REQUEUE_HOLD", "REQUEUED", "RESIZING", "SIGNALING", "SPECIAL_EXIT", "STAGE_OUT", "SUSPENDED"}
    RUNNING_SLURM_STATES = {"COMPLETING", "RUNNING"}
    STDOUT_FILENAME = ".csslib.stdout"
    STDERR_FILENAME = ".csslib.stderr"
    EXITCODE_FILENAME = ".csslib.exitcode"
    GENERATED_SBATCH_SCRIPT = "csslib.sh"
    MAX_UNKNOWN_SLURM_POLLS = 12
    SBATCH_OPTIONS_WITH_VALUE = {
        "-A",
        "-a",
        "-b",
        "-c",
        "-C",
        "-D",
        "-e",
        "-J",
        "-L",
        "-M",
        "-m",
        "-N",
        "-n",
        "-o",
        "-p",
        "-q",
        "-t",
        "-w",
        "-x",
        "--account",
        "--array",
        "--batch",
        "--chdir",
        "--clusters",
        "--comment",
        "--constraint",
        "--container",
        "--container-id",
        "--contiguous",
        "--cpus-per-gpu",
        "--cpus-per-task",
        "--deadline",
        "--distribution",
        "--error",
        "--exclude",
        "--export",
        "--gpus",
        "--gpus-per-node",
        "--gpus-per-socket",
        "--gpus-per-task",
        "--job-name",
        "--licenses",
        "--mail-type",
        "--mail-user",
        "--mem",
        "--mem-bind",
        "--mem-per-cpu",
        "--mem-per-gpu",
        "--nodelist",
        "--nodes",
        "--ntasks",
        "--ntasks-per-core",
        "--ntasks-per-gpu",
        "--ntasks-per-node",
        "--ntasks-per-socket",
        "--open-mode",
        "--output",
        "--partition",
        "--profile",
        "--qos",
        "--reservation",
        "--signal",
        "--sockets-per-node",
        "--spread-job",
        "--switches",
        "--thread-spec",
        "--threads-per-core",
        "--time",
        "--tmp",
        "--uid",
        "--wait",
        "--wckey",
        "--wrap",
    }
    SRUN_OPTIONS_WITH_VALUE = {
        "-A",
        "-c",
        "-C",
        "-D",
        "-e",
        "-J",
        "-M",
        "-N",
        "-n",
        "-o",
        "-p",
        "-t",
        "-w",
        "--account",
        "--chdir",
        "--constraint",
        "--cpus-per-task",
        "--distribution",
        "--error",
        "--exclude",
        "--export",
        "--job-name",
        "--mem",
        "--mpi",
        "--nodes",
        "--nodelist",
        "--ntasks",
        "--output",
        "--partition",
        "--qos",
        "--reservation",
        "--time",
    }

    def __init__(self, cmd, inputs, parser, structure, loading_files, calculation_name, local_path):
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
        self._effective_cmd = cmd
        self._unknown_slurm_polls = 0

        prefix = self._get_cmd_prefix(self._cmd)
        self.resource_manager = "slurm" if prefix in self.SUPPORTING_CMD_PREFIXES_SLURM else "mpi"
        self.launch_mode = "slurm-job" if prefix == "sbatch" else "process"

    @staticmethod
    def _get_cmd_prefix(cmd: str) -> str:
        tokens = shlex.split(cmd)
        if not tokens:
            raise ValueError("cmd must not be empty.")
        token = tokens[0]
        return token[1:] if token.startswith("#") else token

    @staticmethod
    def _normalize_slurm_state(state: str) -> str:
        return state.strip().upper().split("+", maxsplit=1)[0]

    @staticmethod
    def _extract_jobid(stdout: str, stderr: str) -> int | None:
        match = re.search(r"\b(\d+)\b", f"{stdout}\n{stderr}")
        return int(match.group(1)) if match else None

    @staticmethod
    @contextmanager
    def _pushd(path: str):
        previous_dir = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(previous_dir)

    def _write_inputs(self, target_dir: str):
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
        if os.path.abspath(source) == os.path.abspath(destination):
            return
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        if os.path.isdir(source):
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)

    @staticmethod
    def _quote_shell_tokens(tokens: list[str]) -> str:
        return " ".join(shlex.quote(token) for token in tokens)

    @staticmethod
    def _get_option_value(tokens: list[str], flags: set[str] | list[str], default: str | None = None) -> str | None:
        for index, token in enumerate(tokens):
            for flag in flags:
                if token == flag:
                    if index + 1 < len(tokens):
                        return tokens[index + 1]
                elif token.startswith(f"{flag}="):
                    return token.split("=", maxsplit=1)[1]
                elif len(flag) == 2 and token.startswith(flag) and token != flag:
                    return token[len(flag):]
        return default

    @classmethod
    def _split_sbatch_command(cls, cmd: str) -> tuple[list[str], list[str], bool]:
        tokens = shlex.split(cmd)
        if not tokens:
            raise ValueError("cmd must not be empty.")
        if cls._get_cmd_prefix(cmd) != "sbatch":
            return tokens, [], False

        sbatch_tokens = [tokens[0]]
        payload_tokens: list[str] = []
        index = 1
        wrap_already_present = False
        while index < len(tokens):
            token = tokens[index]
            if token == "--":
                payload_tokens = tokens[index + 1 :]
                break
            if token == "--wrap":
                wrap_already_present = True
                sbatch_tokens.append(token)
                if index + 1 < len(tokens):
                    sbatch_tokens.append(tokens[index + 1])
                break
            if token.startswith("--wrap="):
                wrap_already_present = True
                sbatch_tokens.append(token)
                break
            if token.startswith("-"):
                sbatch_tokens.append(token)
                if "=" in token:
                    index += 1
                    continue
                if token in cls.SBATCH_OPTIONS_WITH_VALUE and index + 1 < len(tokens):
                    sbatch_tokens.append(tokens[index + 1])
                    index += 2
                    continue
                index += 1
                continue

            payload_tokens = tokens[index:]
            break

        return sbatch_tokens, payload_tokens, wrap_already_present

    @classmethod
    def _split_srun_command(cls, cmd: str) -> tuple[list[str], list[str]]:
        tokens = shlex.split(cmd)
        if not tokens:
            raise ValueError("cmd must not be empty.")
        if cls._get_cmd_prefix(cmd) != "srun":
            return tokens, []

        srun_tokens = [tokens[0]]
        payload_tokens: list[str] = []
        index = 1
        while index < len(tokens):
            token = tokens[index]
            if token == "--":
                payload_tokens = tokens[index + 1 :]
                break
            if token.startswith("-"):
                srun_tokens.append(token)
                if "=" in token:
                    index += 1
                    continue
                if token in cls.SRUN_OPTIONS_WITH_VALUE and index + 1 < len(tokens):
                    srun_tokens.append(tokens[index + 1])
                    index += 2
                    continue
                index += 1
                continue

            payload_tokens = tokens[index:]
            break

        return srun_tokens, payload_tokens

    def _path_exists_in_input_dir(self, candidate: str) -> bool:
        raise NotImplementedError

    def _write_generated_sbatch_script(self, payload_command: str):
        raise NotImplementedError

    @staticmethod
    def _sbatch_has_chdir_option(tokens: list[str]) -> bool:
        for token in tokens:
            if token in {"-D", "--chdir"}:
                return True
            if token.startswith("--chdir="):
                return True
            if token.startswith("-D") and token != "-D":
                return True
        return False

    @staticmethod
    def _sbatch_has_stream_option(tokens: list[str], short_flag: str, long_flag: str) -> bool:
        for token in tokens:
            if token in {short_flag, long_flag}:
                return True
            if token.startswith(f"{long_flag}="):
                return True
            if token.startswith(short_flag) and token != short_flag:
                return True
        return False

    @classmethod
    def _rewrite_mpi_launcher_to_srun(cls, payload_tokens: list[str], default_ntasks: int) -> list[str]:
        if not payload_tokens or payload_tokens[0] not in cls.SUPPORTING_CMD_PREFIXES_MPI:
            return payload_tokens

        launcher_parallel_flags = {"-n", "--n", "-np", "--np"}
        ntasks = default_ntasks
        index = 1
        executable_tokens: list[str] | None = None
        while index < len(payload_tokens):
            token = payload_tokens[index]
            if token in launcher_parallel_flags:
                if index + 1 >= len(payload_tokens):
                    return payload_tokens
                ntasks = int(payload_tokens[index + 1])
                index += 2
                continue
            if any(token.startswith(f"{flag}=") for flag in launcher_parallel_flags):
                ntasks = int(token.split("=", maxsplit=1)[1])
                index += 1
                continue
            if token.startswith("-"):
                return payload_tokens
            executable_tokens = payload_tokens[index:]
            break

        if not executable_tokens:
            return payload_tokens

        rewritten = ["srun"]
        if ntasks > 1:
            rewritten.extend(["-n", str(ntasks)])
        rewritten.extend(executable_tokens)
        return rewritten

    def _normalize_sbatch_payload_tokens(self, payload_tokens: list[str]) -> list[str]:
        if not payload_tokens:
            return payload_tokens

        ntasks = int(self._get_option_value(shlex.split(self._cmd), {"-n", "--ntasks"}, "1") or "1")
        payload_tokens = self._rewrite_mpi_launcher_to_srun(payload_tokens, ntasks)

        first_token = payload_tokens[0]
        if ntasks > 1 and first_token != "srun":
            return ["srun", "-n", str(ntasks)] + payload_tokens
        return payload_tokens

    def _prepare_sbatch_command(self) -> str:
        sbatch_tokens, payload_tokens, wrap_already_present = self._split_sbatch_command(self._cmd)
        if wrap_already_present:
            return self._cmd
        if not payload_tokens:
            raise RuntimeError("sbatch command must contain payload or a script path.")

        candidate_script = payload_tokens[0]
        if self._path_exists_in_input_dir(candidate_script):
            return self._cmd

        ntasks = int(self._get_option_value(sbatch_tokens, {"-n", "--ntasks"}, "1") or "1")
        if payload_tokens and payload_tokens[0] in self.SUPPORTING_CMD_PREFIXES_MPI:
            launcher_has_parallel_flag = self._get_option_value(payload_tokens, {"-n", "--n", "-np", "--np"}) is not None
            if ntasks > 1 and not launcher_has_parallel_flag:
                payload_tokens = [payload_tokens[0], "-n", str(ntasks)] + payload_tokens[1:]
        elif ntasks > 1:
            payload_tokens = ["mpirun", "-n", str(ntasks)] + payload_tokens

        script_lines = ["#!/bin/bash -l"]
        for directive in self._render_sbatch_directives(sbatch_tokens[1:]):
            script_lines.append(directive)
        if not self._sbatch_has_export_all(sbatch_tokens):
            script_lines.append("#SBATCH --export=ALL")
        if not self._sbatch_has_stream_option(sbatch_tokens, "-o", "--output"):
            script_lines.append(f"#SBATCH --output={self.STDOUT_FILENAME}")
        if not self._sbatch_has_stream_option(sbatch_tokens, "-e", "--error"):
            script_lines.append(f"#SBATCH --error={self.STDERR_FILENAME}")
        script_lines.extend(
            [
                "",
                "set +e",
                *self._shell_prelude_lines(),
                f"cd {shlex.quote(self._get_execution_path())}",
                f"bash -ilc {shlex.quote(self._quote_shell_tokens(payload_tokens))}",
                "status=$?",
                f"printf '%s' \"$status\" > {shlex.quote(self.EXITCODE_FILENAME)}",
                "exit \"$status\"",
                "",
            ]
        )
        self._write_generated_sbatch_script("\n".join(script_lines))
        return f"sbatch {shlex.quote(self.GENERATED_SBATCH_SCRIPT)}"

    def _prepare_srun_command(self) -> str:
        srun_tokens, payload_tokens = self._split_srun_command(self._cmd)
        if not payload_tokens:
            raise RuntimeError("srun command must contain payload after SLURM options.")

        if payload_tokens[0] in self.SUPPORTING_CMD_PREFIXES_MPI:
            index = 1
            while index < len(payload_tokens):
                token = payload_tokens[index]
                if token in {"-n", "--n", "-np", "--np"} and index + 1 < len(payload_tokens):
                    index += 2
                    continue
                if any(token.startswith(f"{flag}=") for flag in {"-n", "--n", "-np", "--np"}):
                    index += 1
                    continue
                if token.startswith("-"):
                    index += 1
                    continue
                payload_tokens = payload_tokens[index:]
                break

        if not self._slurm_tokens_have_export_all(srun_tokens):
            srun_tokens.extend(["--export", "ALL"])

        script_lines = [
            "#!/bin/bash -l",
            "set +e",
            *self._shell_prelude_lines(),
            f"cd {shlex.quote(self._get_execution_path())}",
            f"bash -ilc {shlex.quote(self._quote_shell_tokens(srun_tokens + payload_tokens))}",
            "status=$?",
            f"printf '%s' \"$status\" > {shlex.quote(self.EXITCODE_FILENAME)}",
            "exit \"$status\"",
            "",
        ]
        self._write_generated_sbatch_script("\n".join(script_lines))
        return f"bash {shlex.quote(self.GENERATED_SBATCH_SCRIPT)}"

    @classmethod
    def _render_sbatch_directives(cls, sbatch_argument_tokens: list[str]) -> list[str]:
        directives = []
        index = 0
        while index < len(sbatch_argument_tokens):
            token = sbatch_argument_tokens[index]
            if token == "--wrap" and index + 1 < len(sbatch_argument_tokens):
                directives.append(f"#SBATCH {token}={sbatch_argument_tokens[index + 1]}")
                index += 2
                continue
            if token.startswith("-") and "=" not in token and token in cls.SBATCH_OPTIONS_WITH_VALUE and index + 1 < len(sbatch_argument_tokens):
                directives.append(f"#SBATCH {token} {sbatch_argument_tokens[index + 1]}")
                index += 2
                continue
            directives.append(f"#SBATCH {token}")
            index += 1
        return directives

    def _mark_unknown_slurm_poll(self, message: str) -> JobInspection:
        self._unknown_slurm_polls += 1
        if self._unknown_slurm_polls >= self.MAX_UNKNOWN_SLURM_POLLS:
            stderr_text = self._read_job_streams_for_error()
            if stderr_text:
                return JobInspection(JobState.FAILED, stderr_text)
            return JobInspection(JobState.FAILED, message)
        return JobInspection(JobState.UNKNOWN, message)

    def _reset_unknown_slurm_polls(self):
        self._unknown_slurm_polls = 0

    @staticmethod
    def _slurm_tokens_have_export_all(tokens: list[str]) -> bool:
        for index, token in enumerate(tokens):
            if token == "--export" and index + 1 < len(tokens) and tokens[index + 1].upper() == "ALL":
                return True
            if token.startswith("--export=") and token.split("=", maxsplit=1)[1].upper() == "ALL":
                return True
        return False

    @classmethod
    def _sbatch_has_export_all(cls, sbatch_tokens: list[str]) -> bool:
        return cls._slurm_tokens_have_export_all(sbatch_tokens)

    @staticmethod
    def _shell_prelude_lines() -> list[str]:
        return [
            'if [ -f /etc/profile ]; then . /etc/profile; fi',
            'if [ -f /etc/profile.d/modules.sh ]; then . /etc/profile.d/modules.sh; fi',
            'if [ -f ~/.bash_profile ]; then . ~/.bash_profile; elif [ -f ~/.bash_login ]; then . ~/.bash_login; elif [ -f ~/.profile ]; then . ~/.profile; fi',
            'if [ -f ~/.bashrc ]; then . ~/.bashrc; fi',
        ]

    @staticmethod
    def _tail_text(text: str | None, max_lines: int = 40) -> str | None:
        if not text:
            return None
        lines = text.splitlines()
        if len(lines) <= max_lines:
            return text.strip()
        return "\n".join(lines[-max_lines:]).strip()

    @staticmethod
    def _collect_local_files(source_root: str, destination_root: str, loading_files: list[str]):
        os.makedirs(destination_root, exist_ok=True)
        for relative_path in loading_files:
            source_path = os.path.join(source_root, relative_path)
            if not os.path.exists(source_path):
                continue
            destination_path = os.path.join(destination_root, relative_path)
            _BaseWorker._copy_path(source_path, destination_path)

    def _get_failure_message(self) -> str:
        return self._stderr_cache or "Calculation failed."

    def start(self) -> dict:
        self._prepare_workspace()
        self._prepare_inputs()
        self._launch_nonblocking()
        self._started = True
        return self.get_calculation_info()

    def inspect(self) -> JobInspection:
        if not self._started:
            return JobInspection(JobState.UNKNOWN, "Calculation was not started.")
        inspection = self._inspect_slurm_job() if self.launch_mode == "slurm-job" else self._inspect_process_job()
        self._last_state = inspection.state
        return inspection

    def finalize(self) -> tuple[dict, object]:
        inspection = self.inspect()
        if inspection.state != JobState.COMPLETED:
            raise RuntimeError(inspection.message or f"Cannot finalize calculation in state {inspection.state.value}.")
        self._collect_outputs()
        calculation_output = self._parser(self._local_result_path)
        return self.get_calculation_info(), calculation_output

    def cancel(self) -> bool:
        if not self._started:
            return False
        cancelled = self._cancel_slurm_job() if self.launch_mode == "slurm-job" else self._cancel_process_job()
        if cancelled:
            self._last_state = JobState.CANCELLED
        return cancelled

    def get_calculation_info(self) -> dict:
        return {
            "server_ip": self._get_server_ip(),
            "username": self._get_username(),
            "jobid": self._jobid if self._jobid is not None else self._pid,
            "local_path": self._local_result_path,
            "remote_path": self._get_execution_path(),
        }

    def can_migrate_pending(self) -> bool:
        if self.launch_mode != "slurm-job" or self._jobid is None:
            return False
        return self.inspect().state == JobState.PENDING

    def _prepare_workspace(self):
        raise NotImplementedError

    def _launch_nonblocking(self):
        raise NotImplementedError

    def _inspect_process_job(self) -> JobInspection:
        raise NotImplementedError

    def _inspect_slurm_job(self) -> JobInspection:
        raise NotImplementedError

    def _cancel_process_job(self) -> bool:
        raise NotImplementedError

    def _cancel_slurm_job(self) -> bool:
        raise NotImplementedError

    def _collect_outputs(self):
        raise NotImplementedError

    def _prepare_inputs(self):
        self._write_inputs(self._get_input_write_path())
        if self.launch_mode == "slurm-job":
            self._effective_cmd = self._prepare_sbatch_command()
        elif self.resource_manager == "slurm":
            self._effective_cmd = self._prepare_srun_command()

    def _get_input_write_path(self) -> str:
        raise NotImplementedError

    def _get_execution_path(self) -> str:
        raise NotImplementedError

    def _get_server_ip(self) -> str | None:
        raise NotImplementedError

    def _get_username(self) -> str | None:
        raise NotImplementedError

    def _read_job_streams_for_error(self) -> str | None:
        raise NotImplementedError

    def _read_exitcode_value(self) -> int | None:
        raise NotImplementedError


class Worker(_BaseWorker):
    """
        Class for managing CSS structure calculations on the local computer.
    """

    def __init__(self, cmd, inputs, parser, structure, loading_files, calculation_name, work_root, local_path):
        super().__init__(cmd, inputs, parser, structure, loading_files, calculation_name, local_path)
        self._work_path = os.path.join(work_root, calculation_name)
        self._stdout_path = os.path.join(self._work_path, self.STDOUT_FILENAME)
        self._stderr_path = os.path.join(self._work_path, self.STDERR_FILENAME)

    def _prepare_workspace(self):
        os.makedirs(self._work_path, exist_ok=True)
        os.makedirs(self._local_result_path, exist_ok=True)

    def _launch_nonblocking(self):
        if self.launch_mode == "slurm-job":
            result = subprocess.run(self._effective_cmd, cwd=self._work_path, capture_output=True, text=True, shell=True)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"SLURM command failed in {self._work_path}.")
            self._jobid = self._extract_jobid(result.stdout, result.stderr)
            if self._jobid is None:
                raise RuntimeError("Unable to extract SLURM jobid from sbatch output.")
            return

        with open(self._stdout_path, "w", encoding="utf-8") as stdout_file, open(self._stderr_path, "w", encoding="utf-8") as stderr_file:
            self._process = subprocess.Popen(
                self._effective_cmd if self.resource_manager == "slurm" else self._cmd,
                cwd=self._work_path,
                shell=True,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
            )
        self._pid = self._process.pid

    def _inspect_process_job(self) -> JobInspection:
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
        exitcode = self._read_exitcode_value()
        if exitcode is not None:
            self._reset_unknown_slurm_polls()
            if exitcode == 0:
                return JobInspection(JobState.COMPLETED)
            return JobInspection(JobState.FAILED, self._read_job_streams_for_error() or f"SLURM job exited with code {exitcode}.")

        squeue = subprocess.run(f"squeue -h -j {self._jobid} -o %T", cwd=self._work_path, capture_output=True, text=True, shell=True)
        states = [self._normalize_slurm_state(state) for state in squeue.stdout.splitlines() if state.strip()]
        if states:
            self._reset_unknown_slurm_polls()
            return self._inspection_from_slurm_states(states)

        scontrol = subprocess.run(f"scontrol show job {self._jobid}", cwd=self._work_path, capture_output=True, text=True, shell=True)
        if scontrol.returncode == 0:
            states = re.findall(r"JobState=([A-Z_]+)", scontrol.stdout.upper())
            if states:
                self._reset_unknown_slurm_polls()
                return self._inspection_from_slurm_states(states)

        sacct = subprocess.run(f"sacct -n -P -j {self._jobid} -o State", cwd=self._work_path, capture_output=True, text=True, shell=True)
        states = [self._normalize_slurm_state(state) for state in sacct.stdout.splitlines() if state.strip()]
        if states:
            self._reset_unknown_slurm_polls()
            return self._inspection_from_slurm_states(states)
        return self._mark_unknown_slurm_poll(f"SLURM job {self._jobid} state could not be determined yet.")

    def _inspection_from_slurm_states(self, states: list[str]) -> JobInspection:
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
        if self._process is None:
            return False
        if self._process.poll() is None:
            self._process.terminate()
            return True
        return False

    def _cancel_slurm_job(self) -> bool:
        if self._jobid is None:
            return False
        result = subprocess.run(f"scancel {self._jobid}", cwd=self._work_path, capture_output=True, text=True, shell=True)
        return result.returncode == 0

    def _collect_outputs(self):
        self._collect_local_files(self._work_path, self._local_result_path, self._loading_files)

    def _prepare_inputs(self):
        super()._prepare_inputs()

    def _get_input_write_path(self) -> str:
        return self._work_path

    def _path_exists_in_input_dir(self, candidate: str) -> bool:
        candidate_path = candidate if os.path.isabs(candidate) else os.path.join(self._work_path, candidate)
        return os.path.isfile(candidate_path)

    def _write_generated_sbatch_script(self, payload_command: str):
        script_path = os.path.join(self._work_path, self.GENERATED_SBATCH_SCRIPT)
        with open(script_path, "w", encoding="utf-8", newline="\n") as file:
            file.write(payload_command)
        os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    def _get_execution_path(self) -> str:
        return self._work_path

    def _get_server_ip(self) -> str:
        return "local"

    def _get_username(self) -> None:
        return None

    @staticmethod
    def _read_local_file(path: str) -> str | None:
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8", errors="ignore") as file:
            return file.read().strip()

    def _read_job_streams_for_error(self) -> str | None:
        stderr_text = self._read_local_file(self._stderr_path)
        if stderr_text:
            return stderr_text
        return self._tail_text(self._read_local_file(self._stdout_path))

    def _read_exitcode_value(self) -> int | None:
        exitcode_path = os.path.join(self._work_path, self.EXITCODE_FILENAME)
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

    def __init__(self, ssh, file_sharing, server_ip, username, cmd, inputs, parser, structure, loading_files, calculation_name, remote_root, local_path, connection_owner=None):
        super().__init__(cmd, inputs, parser, structure, loading_files, calculation_name, local_path)
        self._connection_owner = connection_owner
        self._ssh = ssh
        self._file_sharing = file_sharing
        self._server_ip = server_ip
        self._username = username
        self._remote_path = posixpath.join(remote_root, calculation_name)
        self._staging_path = os.path.join(local_path, f".staging_{calculation_name}")
        self._stdout_filename = self.STDOUT_FILENAME
        self._stderr_filename = self.STDERR_FILENAME
        self._exitcode_filename = self.EXITCODE_FILENAME

    def _prepare_workspace(self):
        os.makedirs(self._staging_path, exist_ok=True)
        os.makedirs(self._local_result_path, exist_ok=True)

    def _exec_remote(self, command: str) -> tuple[str, str, int]:
        self._refresh_remote_clients()
        wrapped_command = f"bash -ilc {shlex.quote(command)}"
        _, stdout, stderr = self._ssh.exec_command(wrapped_command, get_pty=True)
        stdout_text = stdout.read().decode("utf-8", errors="ignore").strip()
        stderr_text = stderr.read().decode("utf-8", errors="ignore").strip()
        return stdout_text, stderr_text, stdout.channel.recv_exit_status()

    def _refresh_remote_clients(self):
        if self._connection_owner is None:
            return
        clients = self._connection_owner()
        self._ssh = clients["SSH"]
        self._file_sharing = clients["SFTP/SCP"]

    def _launch_nonblocking(self):
        self._upload_inputs()
        if self.launch_mode == "slurm-job":
            stdout, stderr, returncode = self._exec_remote(f"cd {shlex.quote(self._remote_path)} && {self._effective_cmd}")
            shutil.rmtree(self._staging_path, ignore_errors=True)
            if returncode != 0:
                raise RuntimeError(stderr or stdout or f"Remote SLURM command failed in {self._remote_path}.")
            self._jobid = self._extract_jobid(stdout, stderr)
            if self._jobid is None:
                raise RuntimeError("Unable to extract SLURM jobid from remote sbatch output.")
            return

        command = (
            f"cd {shlex.quote(self._remote_path)} && "
            f"( bash -lc {shlex.quote(self._effective_cmd if self.resource_manager == 'slurm' else self._cmd)} "
            f"> {shlex.quote(self._stdout_filename)} "
            f"2> {shlex.quote(self._stderr_filename)} "
            f"< /dev/null; printf '%s' $? > {shlex.quote(self._exitcode_filename)} ) & echo $!"
        )
        stdout, stderr, returncode = self._exec_remote(command)
        shutil.rmtree(self._staging_path, ignore_errors=True)
        if returncode != 0:
            raise RuntimeError(stderr or stdout or f"Remote command failed to start in {self._remote_path}.")
        try:
            self._pid = int(stdout.splitlines()[-1].strip())
        except (IndexError, ValueError) as exc:
            raise RuntimeError(f"Unable to extract remote pid from command output: {stdout}") from exc

    def _upload_inputs(self):
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
        self._exec_remote(f"mkdir -p {shlex.quote(remote_path)}")

    def _path_exists_in_input_dir(self, candidate: str) -> bool:
        candidate_path = candidate if os.path.isabs(candidate) else os.path.join(self._staging_path, candidate)
        return os.path.isfile(candidate_path)

    def _write_generated_sbatch_script(self, payload_command: str):
        script_path = os.path.join(self._staging_path, self.GENERATED_SBATCH_SCRIPT)
        with open(script_path, "w", encoding="utf-8", newline="\n") as file:
            file.write(payload_command)

    def _inspect_process_job(self) -> JobInspection:
        command = (
            f"cd {shlex.quote(self._remote_path)} && "
            f"if [ -f {shlex.quote(self._exitcode_filename)} ]; then "
            f"printf 'DONE %s' \"$(cat {shlex.quote(self._exitcode_filename)})\"; "
            f"elif kill -0 {self._pid} 2>/dev/null; then echo RUNNING; "
            f"else echo UNKNOWN; fi"
        )
        stdout, stderr, returncode = self._exec_remote(command)
        if returncode != 0 and stderr:
            return JobInspection(JobState.UNKNOWN, stderr)
        if stdout.startswith("DONE"):
            exitcode = int(stdout.split(maxsplit=1)[1])
            if exitcode == 0:
                return JobInspection(JobState.COMPLETED)
            self._stderr_cache = self._read_remote_file(self._stderr_filename)
            return JobInspection(JobState.FAILED, self._stderr_cache or f"Remote process exited with code {exitcode}.")
        if stdout.strip() == "RUNNING":
            return JobInspection(JobState.RUNNING)
        return JobInspection(JobState.UNKNOWN, "Remote process state could not be determined.")

    def _inspect_slurm_job(self) -> JobInspection:
        exitcode = self._read_exitcode_value()
        if exitcode is not None:
            self._reset_unknown_slurm_polls()
            if exitcode == 0:
                return JobInspection(JobState.COMPLETED)
            return JobInspection(JobState.FAILED, self._read_job_streams_for_error() or f"Remote SLURM job exited with code {exitcode}.")

        stdout, _, _ = self._exec_remote(f"squeue -h -j {self._jobid} -o %T")
        states = [self._normalize_slurm_state(state) for state in stdout.splitlines() if state.strip()]
        if states:
            self._reset_unknown_slurm_polls()
            return self._inspection_from_slurm_states(states)

        stdout, _, returncode = self._exec_remote(f"scontrol show job {self._jobid}")
        if returncode == 0:
            states = re.findall(r"JOBSTATE=([A-Z_]+)", stdout.upper())
            if states:
                self._reset_unknown_slurm_polls()
                return self._inspection_from_slurm_states(states)

        stdout, _, _ = self._exec_remote(f"sacct -n -P -j {self._jobid} -o State")
        states = [self._normalize_slurm_state(state) for state in stdout.splitlines() if state.strip()]
        if states:
            self._reset_unknown_slurm_polls()
            return self._inspection_from_slurm_states(states)
        return self._mark_unknown_slurm_poll(f"Remote SLURM job {self._jobid} state could not be determined yet.")

    def _inspection_from_slurm_states(self, states: list[str]) -> JobInspection:
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
        if self._pid is None:
            return False
        _, _, returncode = self._exec_remote(f"kill {self._pid}")
        return returncode == 0

    def _cancel_slurm_job(self) -> bool:
        if self._jobid is None:
            return False
        _, _, returncode = self._exec_remote(f"scancel {self._jobid}")
        return returncode == 0

    def _remote_path_type(self, remote_path: str) -> str | None:
        stdout, _, _ = self._exec_remote(
            f"if [ -d {shlex.quote(remote_path)} ]; then echo dir; "
            f"elif [ -f {shlex.quote(remote_path)} ]; then echo file; else echo missing; fi"
        )
        path_type = stdout.strip()
        return None if path_type == "missing" else path_type

    def _download_sftp_tree(self, remote_path: str, local_path: str):
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
        remote_stat = self._file_sharing.stat(remote_path)
        if stat.S_ISDIR(remote_stat.st_mode):
            self._download_sftp_tree(remote_path, local_path)
        else:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self._file_sharing.get(remote_path, local_path)

    def _download_scp(self, remote_path: str, local_path: str, recursive: bool):
        target_path = os.path.dirname(local_path) if recursive else local_path
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self._file_sharing.get(remote_path, local_path=target_path, recursive=recursive)

    def _collect_outputs(self):
        os.makedirs(self._local_result_path, exist_ok=True)
        for relative_path in self._loading_files:
            remote_item = posixpath.join(self._remote_path, relative_path.replace("\\", "/"))
            local_item = os.path.join(self._local_result_path, relative_path)
            path_type = self._remote_path_type(remote_item)
            if path_type is None:
                continue
            if isinstance(self._file_sharing, scp.SCPClient):
                self._download_scp(remote_item, local_item, recursive=(path_type == "dir"))
            else:
                self._download_sftp(remote_item, local_item)

    def _prepare_inputs(self):
        super()._prepare_inputs()

    def _get_input_write_path(self) -> str:
        return self._staging_path

    def _get_execution_path(self) -> str:
        return self._remote_path

    def _get_server_ip(self) -> str:
        return self._server_ip

    def _get_username(self) -> str:
        return self._username

    def _read_remote_file(self, relative_path: str) -> str | None:
        stdout, _, returncode = self._exec_remote(f"cd {shlex.quote(self._remote_path)} && cat {shlex.quote(relative_path)}")
        if returncode != 0:
            return None
        return stdout.strip()

    def _read_job_streams_for_error(self) -> str | None:
        stderr_text = self._read_remote_file(self._stderr_filename)
        if stderr_text:
            return stderr_text
        return self._tail_text(self._read_remote_file(self._stdout_filename))

    def _read_exitcode_value(self) -> int | None:
        exitcode_text = self._read_remote_file(self.EXITCODE_FILENAME)
        if not exitcode_text:
            return None
        try:
            return int(exitcode_text)
        except ValueError:
            return None
