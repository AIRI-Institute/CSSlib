"""
    Module with predefined functions for parsing calculation results.
    
    All functions in this module obtains workdir as an input parameter and parses all required data from it.
"""


__all__ = []


import os
from pymatgen.io.vasp import Vasprun


def _find_existing_file(workdir: str, candidate_names: list[str]) -> str | None:
    """
        Finds the specified name in the given workdir.
        
        Args:
            workdir (str): name of the working directory.
            candidate_names (list[str]): list of files that should be found. Function returns the first found candidate.
            
        Return:
            str | None: path of the candidate file. If file is not found returns None.
    """

    for candidate_name in candidate_names:
        candidate_path = os.path.join(workdir, candidate_name)
        if os.path.exists(candidate_path):
            return candidate_path
    for root, _, files in os.walk(workdir):
        for candidate_name in candidate_names:
            if candidate_name in files:
                return os.path.join(root, candidate_name)
    return None


def default_vasp_parser(workdir: str, **vasp_kwargs):
    """
        Default vasp parser function. Parses file into the pymatgen.io.vasp.Vasprun object and ignores potcar file part in the output file.
        
        Args:
            workdir (str): name of the working directory.
            
        Kwargs
            vasp_kwargs (dict): kwargs to be passed to the pymatgen.io.vasp.Vasprun function.
    """

    vasprun_path = _find_existing_file(workdir, ["vasprun.xml"])
    if vasprun_path is None:
        raise FileNotFoundError(f"vasprun.xml was not found in {workdir}")
    return Vasprun(vasprun_path, parse_potcar_file=False, **vasp_kwargs)


def default_espresso_parser(workdir: str, **pwxml_kwargs):
    """
        Default espresso parser function. Parses file into pymatgen.io.espresso.outputs.PWxml object.
        
        Args:
            workdir (str): name of the working directory.
            
        Kwargs
            pwxml_kwargs (dict): kwargs to be passed to the pymatgen.io.espresso.outputs.PWxml function.
    """
    try:
        from pymatgen.io.espresso.outputs import PWxml # TODO: try to implement pymatgen.io.pwscf.PWoutput
        xml_path = _find_existing_file(workdir, ["pwscf.xml", "data-file-schema.xml"])
        if xml_path is None:
            raise FileNotFoundError(f"Quantum Espresso XML output was not found in {workdir}")
        return PWxml(xml_path, **pwxml_kwargs)
    except ImportError as exc:
        message = "Full Quantum Espresso parsing requires the pymatgen-io-espresso package. "
        message += "Install it with: reinstall CSSlib as pip install csslib[espresso] or "
        message += "install only extention as pip install git+https://github.com/Griffin-Group/pymatgen-io-espresso"
        raise ImportError(message) from exc
