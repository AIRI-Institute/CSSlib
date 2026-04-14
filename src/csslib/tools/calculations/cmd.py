"""Module with classes for the cmd preparation."""

__all__ = [
    "MPI",
    "SLURM",
]


from dataclasses import dataclass

from csslib.exceptions import ConfigurationError


@dataclass
class MPI:
    """
        Dataclass with data for the calculation start.
        
        Attributes:
            prefix (str, optional): type of the mpi binary to execute. Defaults to mpirun.
            cores_number (int | None, optional): number of cores to use. Defaults to None, e.g. all cores are used.
            binary (str | None, optional): name of the binary file to be used. If None the error will be raised. Defaults to None.
            output (str, optional): name of the output file. Defaults to ".csslib.stdout".
            error (str, optional): name of the stderr file. Defaults to ".csslib.stderr".
    """
    
    __SUPPORTING_PREFIXES_MPI = {
        "mpi",
        "mpirun",
        "mpirun.mpich",
        "mpirun.openmpi",
        "mpiexec",
        "mpiexec.hydra",
        "mpiexec.mpich",
        "mpiexec.openmpi",
    }
    
    prefix: str = "mpirun"
    cores_number: int | None = None
    binary: str | None = None
    output: str = ".csslib.stdout"
    error: str = ".csslib.stderr"
    
    def __post_init__(self):
        if self.prefix not in self.__SUPPORTING_CMD_PREFIXES_MPI:
            raise ConfigurationError(f"prefix attribute should be chosen from the supporting prefixes. They are: {str(self.__SUPPORTING_PREFIXES_MPI)}")
    
        if self.binary is None:
            raise ConfigurationError("binary attribute should be filled.")
        
    def get_cmd(self) -> str:
        """
            Returns the cmd to be used by the worker.
            
            Return:
                str: cmd.
        """
        
        return self.prefix + f" -np {self.cores_number} " if self.cores_number is not None else " " + self.binary
        

@dataclass
class SLURM:
    """
        Dataclass with data for the calculation start.
        
        Attributes:
            prefix (str, optional): type of the slurm command to execute. Defaults to sbatch.
            partition (str): name of the partition to be used. Must be non-empty for the resources distribution. Defaults to None. MUST BE FILLED.
            mpitype (str, optional): name of the mpi binary to be used. If prefix is srun, this parameter will be ignored. Defaults to mpirun.
            binary (str | None, optional): name of the binary file to be used. If None the error will be raised. Defaults to None.
            job_name (str, optional): name of the job. Defaults to csslib.
            nodes (int, optional): nodes number. Defaults to 1.
            ntasks (int | None, optional): number of tasks. If None, the field will be ignored. Defaults to None.
            cpu_per_task (int | None, optional): cpus number per task. If None, the field will be ignored. Defaults to None.
            mem (str | None, optional): memory to be used by the job. If None, the field will be ignored. Defaults to None. Example: "16G".
            mem_per_cpu (str | None, optional): memory to by used by the each core of the job. If None, the field will be ignored. Defaults to None. Exaple: "2G".
            timelimit (str | None, optional): timelimit for the job. If None, the field will be ignored. Defaults to None. Example: "02:00:00".
            output (str, optional): name of the output file. Defaults to ".csslib.stdout". Example: "slurm-%j.out".
            error (str, optional): name of the stderr file. Defaults to ".csslib.stderr". Example: "slurm-%j.err".
            nodelist (str | None, optional): string with nodes which should be used for the job. If None, the field will be ignored. Defaults to None. Example: "node_02" or "node_[02,54]" or "node_[103-105]".
            exclude (str | None, optional): string with nodes which should be excluded for the job. If None, the field will be ignored. Defaults to None. Example: "node_02" or "node_[02,54]" or "node_[103-105]".
    """
    
    __SUPPORTING_PREFIXES_SLURM = {"srun", "sbatch"}
    __SUPPORTING_PREFIXES_MPI = {
        "mpi",
        "mpirun",
        "mpirun.mpich",
        "mpirun.openmpi",
        "mpiexec",
        "mpiexec.hydra",
        "mpiexec.mpich",
        "mpiexec.openmpi",
    }
    
    prefix: str = "sbatch"
    partition: str | None = None
    mpitype: str = "mpirun"
    binary: str | None = None
    job_name: str = "csslib"
    nodes: int = 1
    ntasks: int | None = None
    cpu_per_task: int | None = None
    mem: str | None = None
    mem_per_cpu: str | None = None
    timelimit: str | None = None
    output: str = ".csslib.stdout"
    error: str = ".csslib.stderr"
    nodelist: str | None = None
    exclude: str | None = None
    
    def __post_init__(self):
        if self.partition is None:
            raise ConfigurationError("partition must be filled!")
        
        if self.prefix not in self.__SUPPORTING_PREFIXES_SLURM:
            raise ConfigurationError(f"prefix attribute should be chosen from the supporting prefixes. They are: {str(self.__SUPPORTING_PREFIXES_SLURM)}")

        if self.mpitype not in self.__SUPPORTING_PREFIXES_MPI:
            raise ConfigurationError(f"mpitype attribute should be chosen from the supporting prefixes. They are: {str(self.__SUPPORTING_PREFIXES_MPI)}")
            
        if self.binary is None:
            raise ConfigurationError("binary attribute should be filled.")
        
        if self.ntasks is None and self.cpu_per_task is not None:
            self.ntasks = 1
        elif self.ntasks is not None and self.cpu_per_task is None:
            self.cpu_per_task = 1
        
    def get_cmd(self) -> str:
        """
            Returns the cmd to be used by the worker.
            
            Return:
                str: cmd.
        """
        
        return self.prefix + " csslib.sh" if self.prefix == "sbatch" else "bash csslib.sh"
    
    def get_script(self, remote_dir: str | None = None) -> str:
        """
            Returns the script to be used by the worker.
            
            Args:
                remote_dir (str | None, optional): remote directory, where calculation should be performed.
            
            Return:
                str: script lines.
        """
        
        script = ""
        if self.prefix == "sbatch":
            script += "#!/bin/bash -l\n"
            script += f"#SBATCH --partition={self.partition}\n"
            script += f"#SBATCH --job-name={self.job_name}\n"
            script += f"#SBATCH --nodes={self.nodes}\n"
            script += f"#SBATCH --ntasks={self.ntasks}\n" if self.ntasks is not None else ""
            script += f"#SBATCH --cpus-per-task={self.cpu_per_task}\n" if self.cpu_per_task is not None else ""
            script += f"#SBATCH --mem={self.mem}\n" if self.mem is not None else ""
            script += f"#SBATCH --mem_per_cpu={self.mem_per_cpu}\n" if self.mem_per_cpu is not None else ""
            script += f"#SBATCH --time={self.timelimit}\n" if self.timelimit is not None else ""
            script += f"#SBATCH --output={self.output}\n"
            script += f"#SBATCH --error={self.error}\n"
            script += f"#SBATCH --nodelist={self.nodelist}\n" if self.nodelist is not None else ""
            script += f"#SBATCH --exclude={self.error}\n" if self.exclude is not None else ""
            script += "#SBATCH --export=ALL\n\n"
            script += "set +e\n"
            script += "cd $SLURM_SUBMIT_DIR\n"
            script += f"{self.mpitype} {self.binary}\n"
            script += "status=$?\n"
            script += "printf '%s' \"$status\" > .csslib.exitcode\n"
            script += "exit \"$status\""
        else:
            srun_command = "srun "
            srun_command += f"--partition={self.partition} "
            srun_command += f"--job-name={self.job_name} "
            srun_command += f"--nodes={self.nodes} "
            srun_command += f"--ntasks={self.ntasks} " if self.ntasks is not None else ""
            srun_command += f"--cpus-per-task={self.cpu_per_task} " if self.cpu_per_task is not None else ""
            srun_command += f"--mem={self.mem} " if self.mem is not None else ""
            srun_command += f"--mem_per_cpu={self.mem_per_cpu} " if self.mem_per_cpu is not None else ""
            srun_command += f"--time={self.timelimit} " if self.timelimit is not None else ""
            srun_command += f"--output={self.output} "
            srun_command += f"--error={self.error} "
            srun_command += f"--nodelist={self.nodelist} " if self.nodelist is not None else ""
            srun_command += f"--exclude={self.error} " if self.exclude is not None else ""
            srun_command += self.binary
            
            script += "#!/bin/bash -l\n"
            script += "set +e\n"
            script += f"cd {remote_dir}\n"
            script += srun_command + "\n"
            script += "status=$?\n"
            script += "printf '%s' \"$status\" > .csslib.exitcode\n"
            script += "exit \"$status\""
        return script
    
    
"""
#NOTE:
            'if [ -f /etc/profile ]; then . /etc/profile; fi',
            'if [ -f /etc/profile.d/modules.sh ]; then . /etc/profile.d/modules.sh; fi',
            'if [ -f ~/.bash_profile ]; then . ~/.bash_profile; elif [ -f ~/.bash_login ]; then . ~/.bash_login; elif [ -f ~/.profile ]; then . ~/.profile; fi',
            'if [ -f ~/.bashrc ]; then . ~/.bashrc; fi',
"""