import sys
from pydantic import BaseModel, ConfigDict


class Substitutions(BaseModel):
    """
        Substitution field pydantic scheme class of the config.
        
        Mandatory fields:
          specie_to_substitute (str): the chemical element that must be replaced
          substitute_with (str): the chemical element that will replace the original chemical element
          substitution_low_limit (float): lower boundary of the substitution. Restrictions: >= 0 and < 1. Example: 0.05
          substitution_high_limit (float): higher boundary of the substitution. Restrictions: > 0, <= 1 and > substitution_low_limit. Example: 0.5
        
        Fields that must be empty:
          substitution_low_limit_natoms (int): the minimum number of atoms that must be replaced 
          substitution_high_limit_natoms (int): the maximum number of atoms that must be replaced
          indices_to_substitute (list[int]): the list of atom indices that must be replaced
    """
    model_config = ConfigDict(extra="forbid")
    specie_to_substitute: str
    substitute_with: str
    substitution_low_limit: float
    substitution_high_limit: float
    substitution_low_limit_natoms: int | None = None
    substitution_high_limit_natoms: int | None = None
    indices_to_substitute: list[int] | None = None
    substitute_with_labels: list[str] | None = None
    labels_to_substitute: list[str] | None = None


class Config(BaseModel):
    """
        Config pydantic scheme class.
        
        Mandatory fields:
          result_dir (str): a full or relative path to the results directory
          structure_filename (str): a full or relative path to the .cif initial structure file
        
        Optional fields:
          supercell (str): system replication numbers in the following format: "2x1x1", "2x2x1", ... (default value - "1x1x1")
          num_workers (int): the number of parallel processes (default value - 1)
          substitution (list[Substitutions]): list of required substitutions for the system
    """
    model_config = ConfigDict(extra="forbid")
    result_dir: str
    structure_filename: str
    supercell: str = "1x1x1"
    num_workers: int = 1
    substitution: list[Substitutions] | None = None


def get_available_config_fields(output=sys.stdout):
    """
        Prints all possible config field names.
        
        Args:
          output (TextIO | Any): an output stream
        
        Returns:
          None
    """
    message = 'List of available config fields:\n'
    message += '  - "result_dir" - full or relative path to the results directory (mandatory)\n'
    message += '  - "structure_filename" - full or relative path to the .cif initial structure file (mandatory)\n'
    message += '  - "supercell" - system replication numbers in the following format: "2x1x1", "2x2x1", ... (optional, default value - "1x1x1")\n'
    message += '  - "num_workers" - number of parallel processes (optional, default value - 1)\n'
    message += '  - "substitution" - list of required substitutions for the system (optional). The field has the following subfields:\n'
    message += '    - "specie_to_substitute" - string with a chemical element which should be replaced (mandatory)\n'
    message += '    - "substitute_with" - string with with a chemical element that will replace the original chemical element (mandatory)\n'
    message += '    - "substitution_low_limit" - float value containing lower boundary of the substitution (mandatory). Example: 0.05\n'
    message += '    - "substitution_high_limit" - float value containing higher boundary of the substitution (mandatory). Example: 0.5\n'
    message += '    Internal parameters:\n'
    message += '    - "substitution_low_limit_natoms"\n'
    message += '    - "substitution_high_limit_natoms"\n'
    message += '    - "indices_to_substitute"\n'
    message += 'Example config can be obtained by the `get_example_config` function of `csslib.config`.\n'
    print(message, file=output)


def get_example_config(output=sys.stdout):
    """
        Prints example config.
        
        Args:
          output (TextIO | Any): an output stream
        
        Returns:
          None
    """
    message = '''{
  "result_dir": "path-to-directory-with-results",
  "structure_filename": "path-to-cif-file",
  "supercell": "1x1x1",
  "num_workers":  1,
  "substitution": [
    {
      "specie_to_substitute": "X",
      "substitute_with": "Y",
      "substitution_low_limit": 0.0,
      "substitution_high_limit": 0.05
    },
    {
      "specie_to_substitute": "E",
      "substitute_with": "N",
      "substitution_low_limit": 0.0,
      "substitution_high_limit": 0.06
    }
  ]
}'''
    print(message, file=output)
