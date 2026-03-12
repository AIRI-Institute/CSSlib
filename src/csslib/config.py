import sys
from pydantic import BaseModel, ConfigDict


class Substitution(BaseModel):
    '''
        Substitution field pydantic scheme class of the config.
        
        Fields:
          specie_to_substitute (str): the chemical element that must be replaced.
          substitute_with (str): the chemical element that will replace the original chemical element.
          substitution_low_limit (float): lower boundary of the substitution. Restrictions: >= 0 and < 1. Example: 0.05.
          substitution_high_limit (float): higher boundary of the substitution. Restrictions: > 0, <= 1 and > substitution_low_limit. Example: 0.5.
        
        Fields that must be empty:
          substitution_low_limit_natoms (int): the minimum number of atoms that must be replaced.
          substitution_high_limit_natoms (int): the maximum number of atoms that must be replaced.
          indices_to_substitute (list[int]): the list of atom indices that must be replaced.
    '''
    model_config = ConfigDict(extra="forbid")
    specie_to_substitute: str
    substitute_with: str
    substitution_low_limit: float
    substitution_high_limit: float
    substitution_low_limit_natoms: int | None = None
    substitution_high_limit_natoms: int | None = None
    indices_to_substitute: list[int] | None = None
    
    def __repr__(self):
        message = '\n    Substitution(\n'
        message += f'      specie_to_substitute="{self.specie_to_substitute}",\n'
        message += f'      substitute_with="{self.substitute_with}",\n'
        message += f'      substitution_low_limit={self.substitution_low_limit},\n'
        message += f'      substitution_high_limit={self.substitution_high_limit},\n    )'
        return message
      
    def __str__(self):
        return self.__repr__()


class Config(BaseModel):
    '''
        Config pydantic scheme class.
        
        Fields:
          result_dir (str): a full or relative path to the results directory.
          structure_filename (str): a full or relative path to the .cif initial structure file.
          supercell (str, optional): system replication numbers in the following format: "2x1x1", "2x2x1", ... Defaults to "1x1x1".
          num_workers (int, optional): the number of parallel processes. Defaults to 1.
          substitution (list[Substitutions], optional): list of required substitutions for the system.
    '''
    model_config = ConfigDict(extra="forbid")
    result_dir: str
    structure_filename: str
    supercell: str = "1x1x1"
    num_workers: int = 1
    substitution: list[Substitution] | None = None
    
    def __repr__(self):
        message = 'Config(\n'
        message += f'  result_dir="{self.result_dir}",\n'
        message += f'  structure_filename="{self.structure_filename}",\n'
        message += f'  supercell="{self.supercell}",\n'
        message += f'  num_workers={self.num_workers},\n'
        message += f'  substitution=['
        if self.substitution is not None:
            for indx, subst in enumerate(self.substitution):
                message += subst.__repr__()
                if len(self.substitution) > 1 and indx != len(self.substitution) - 1:
                    message += ','
                if indx == len(self.substitution) - 1:
                    message += '\n  '
        message += f']\n)\n'
        return message
      
    def __str__(self):
        return self.__repr__()


def get_available_config_fields(output=sys.stdout):
    '''
        Prints all possible config field names.
        
        Args:
          output (TextIO | Any): an output stream.
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
        
        Args:
          output (TextIO | Any): an output stream.
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
