from pydantic import BaseModel, ConfigDict
from typing import Optional


class Substitutions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    specie_to_substitute: str
    substitute_with: str
    substitution_low_limit: float
    substitution_high_limit: float
    substitution_low_limit_natoms: int | None = None
    substitution_high_limit_natoms: int | None = None
    indices_to_substitute: list[int] | None = None


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    result_dir: str
    structure_filename: str
    supercell: str = "1x1x1"
    num_workers: int = 1
    substitution: list[Substitutions] | None = None
