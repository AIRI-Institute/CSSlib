import sys
from pydantic import BaseModel, ConfigDict
from typing import Optional



class Substitutions(BaseModel):
    '''
        Substitution field pydantic scheme class.
        
        Fields:
        - "specie_to_substitute" - string with a chemical element which should be replaced (mandatory)
        - "substitute_with" - string with with a chemical element that will replace the original chemical element (mandatory)
        - "substitution_low_limit" - float value containing lower boundary of the substitution (mandatory). Example: 0.05
        - "substitution_high_limit" - float value containing higher boundary of the substitution (mandatory). Example: 0.5
        
        Internal parameters:
        - "substitution_low_limit_natoms"
        - "substitution_high_limit_natoms"
        - "indices_to_substitute"
    '''
    model_config = ConfigDict(extra="forbid")
    specie_to_substitute: str
    substitute_with: str
    substitution_low_limit: float
    substitution_high_limit: float
    substitution_low_limit_natoms: int | None = None
    substitution_high_limit_natoms: int | None = None
    indices_to_substitute: list[int] | None = None


class Config(BaseModel):
    '''
        Config pydantic scheme class.
        
        Fields:
        - "result_dir" - full or relative path to the results directory (mandatory)
        - "structure_filename" - full or relative path to the .cif initial structure file (mandatory)
        - "supercell" - system replication numbers in the following format: "2x1x1", "2x2x1", ... (optional, default value - "1x1x1")
        - "num_workers" - number of parallel processes (optional, default value - 1)
        - "substitution" - list of required substitutions for the system (optional)
    '''
    model_config = ConfigDict(extra="forbid")
    result_dir: str
    structure_filename: str
    supercell: str = "1x1x1"
    num_workers: int = 1
    substitution: list[Substitutions] | None = None


def get_available_config_fields(output=sys.stdout):
    '''
        Prints all possible config field names.
    '''
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
    '''
        Prints example config.
    '''
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
