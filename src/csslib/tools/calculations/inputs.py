"""Module with input-set classes for different electronic structure packages."""

__all__ = [
    "InputSet",
    "VaspInputs",
    "EspressoInputs",
]


import copy
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from pymatgen.core import Structure
from pymatgen.io.pwscf import PWInput
from pymatgen.io.vasp import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


class InputSet:
    """
        Base class for package-specific input sets.

        The class stores static launch files and knows how to transform a CIF string or a pymatgen-compatible
        object into a Structure that can be later written into the target calculation directory.
    """

    def __init__(self, input_paths: str | os.PathLike | Sequence[str | os.PathLike] | None = None, transform: Callable[[Any], Structure] | None = None):
        self._input_paths = self._normalize_paths(input_paths)
        self._static_files = self._collect_static_files(self._input_paths)
        self._transform = transform or self._default_transform

        self.cif_data = None
        self.structure = None

    @staticmethod
    def _normalize_paths(input_paths: str | os.PathLike | Sequence[str | os.PathLike] | None) -> list[str]:
        if input_paths is None:
            return []
        if isinstance(input_paths, (str, os.PathLike)):
            return [os.fspath(input_paths)]
        return [os.fspath(path) for path in input_paths]

    @staticmethod
    def _collect_static_files(paths: Sequence[str]) -> dict[str, str]:
        static_files = {}
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Input path does not exist: {path}")
            if os.path.isfile(path):
                static_files[os.path.basename(path)] = os.path.abspath(path)
                continue

            base_path = os.path.abspath(path)
            for root, _, files in os.walk(base_path):
                for file_name in files:
                    source_path = os.path.join(root, file_name)
                    relative_path = os.path.relpath(source_path, base_path)
                    static_files[relative_path] = source_path
        return static_files

    @staticmethod
    def _structure_from_any(cif_data: Any) -> Structure:
        if isinstance(cif_data, Structure):
            return cif_data.copy()

        if hasattr(cif_data, "structure") and isinstance(cif_data.structure, Structure):
            return cif_data.structure.copy()

        if hasattr(cif_data, "final_structure") and isinstance(cif_data.final_structure, Structure):
            return cif_data.final_structure.copy()

        if isinstance(cif_data, (str, os.PathLike)):
            cif_text_or_path = os.fspath(cif_data)
            if os.path.exists(cif_text_or_path):
                return Structure.from_file(cif_text_or_path)
            return Structure.from_str(cif_text_or_path, fmt="cif")

        if hasattr(cif_data, "read"):
            return Structure.from_str(cif_data.read(), fmt="cif")

        raise TypeError("cif_data must be a CIF string/path or a pymatgen Structure-compatible object.")

    def _default_transform(self, cif_data: Any) -> Structure:
        return self._structure_from_any(cif_data)

    def load_cif(self, cif_data: Any) -> Structure:
        """
            Transforms and stores the current structure.
        """
        self.cif_data = cif_data
        self.structure = self._transform(cif_data)
        return self.structure

    def _copy_static_files(self, target_dir: str, exclude: Iterable[str] | None = None):
        excluded_paths = {path.replace("\\", "/") for path in (exclude or [])}
        for relative_path, source_path in self._static_files.items():
            normalized_relative_path = relative_path.replace("\\", "/")
            if normalized_relative_path in excluded_paths or Path(relative_path).name in excluded_paths:
                continue
            destination_path = os.path.join(target_dir, relative_path)
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            shutil.copy2(source_path, destination_path)

    def _write_structure_files(self, target_dir: str):
        self._copy_static_files(target_dir)

    def write(self, path: str | None = None):
        """
            Writes all input files into the target directory.
        """
        if self.structure is None:
            raise ValueError("load_cif must be called before write.")

        target_dir = os.getcwd() if path is None else path
        os.makedirs(target_dir, exist_ok=True)
        self._write_structure_files(target_dir)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, copy.deepcopy(value, memo))
        return result


class VaspInputs(InputSet):
    """
        Input set for VASP calculations.

        Static files such as INCAR, KPOINTS, POTCAR and job scripts can be placed into `input_paths`.
        POSCAR is always regenerated from the loaded structure.
    """

    def __init__(
        self,
        input_paths: str | os.PathLike | Sequence[str | os.PathLike] | None = None,
        transform: Callable[[Any], Structure] | None = None,
        poscar_name: str = "POSCAR",
        potcar_name: str = "POTCAR",
        potcar_dir: str | os.PathLike | None = None,
        potcar_map: dict[str, str] | None = None,
        assemble_potcar: bool = False,
    ):
        super().__init__(input_paths=input_paths, transform=transform)
        self.poscar_name = poscar_name
        self.potcar_name = potcar_name
        self.potcar_dir = os.fspath(potcar_dir) if potcar_dir is not None else None
        self.potcar_map = potcar_map or {}
        self.assemble_potcar = assemble_potcar

    def _get_vasp_species_order(self) -> list[str]:
        if self.structure is None:
            raise ValueError("load_cif must be called before VASP file generation.")

        species_order = []
        for site in self.structure:
            symbol = site.specie.symbol
            if symbol not in species_order:
                species_order.append(symbol)
        return species_order

    def _resolve_potcar_fragment_path(self, symbol: str) -> str:
        if self.potcar_dir is None:
            raise FileNotFoundError("potcar_dir must be specified when assemble_potcar=True.")

        requested_name = self.potcar_map.get(symbol, symbol)
        candidates = []

        requested_path = Path(requested_name)
        if requested_path.is_absolute():
            candidates.append(requested_path)
        else:
            potcar_root = Path(self.potcar_dir)
            candidates.extend(
                [
                    potcar_root / requested_name,
                    potcar_root / requested_name / self.potcar_name,
                    potcar_root / f"{self.potcar_name}.{requested_name}",
                    potcar_root / symbol,
                    potcar_root / symbol / self.potcar_name,
                    potcar_root / f"{self.potcar_name}.{symbol}",
                ]
            )

        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)

        message = f"POTCAR fragment for element {symbol} was not found. "
        message += f"Requested label: {requested_name}. Checked inside: {self.potcar_dir}"
        raise FileNotFoundError(message)

    def _assemble_potcar(self, target_dir: str):
        potcar_path = os.path.join(target_dir, self.potcar_name)
        species_order = self._get_vasp_species_order()
        with open(potcar_path, "wb") as output_file:
            for symbol in species_order:
                fragment_path = self._resolve_potcar_fragment_path(symbol)
                with open(fragment_path, "rb") as input_file:
                    shutil.copyfileobj(input_file, output_file)

    def _write_structure_files(self, target_dir: str):
        excluded_files = {self.poscar_name}
        if self.assemble_potcar:
            excluded_files.add(self.potcar_name)
        self._copy_static_files(target_dir, exclude=excluded_files)
        Poscar(self.structure).write_file(os.path.join(target_dir, self.poscar_name))
        if self.assemble_potcar:
            self._assemble_potcar(target_dir)


class EspressoInputs(InputSet):
    """
        Input set for Quantum Espresso calculations.

        In addition to simple CIF conversion, the structure is standardized and sorted before `pw.in` generation.
        This usually produces a cleaner cell description and stable species ordering for pseudopotentials.
    """

    PSEUDOPOTENTIAL_SUFFIXES = (".upf", ".psp8", ".psml", ".usp")

    def __init__(
        self,
        input_paths: str | os.PathLike | Sequence[str | os.PathLike] | None = None,
        transform: Callable[[Any], Structure] | None = None,
        input_filename: str = "pw.in",
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
    ):
        super().__init__(input_paths=input_paths, transform=transform or self._espresso_transform)
        self.input_filename = input_filename
        self.pseudopotentials = pseudopotentials or {}
        self.pseudo_dir = pseudo_dir
        self.control = control or {}
        self.system = system or {}
        self.electrons = electrons or {}
        self.ions = ions or {}
        self.cell = cell or {}
        self.kpoints_grid = kpoints_grid
        self.kpoints_shift = kpoints_shift
        self.use_primitive = use_primitive
        self.symprec = symprec

    def _espresso_transform(self, cif_data: Any) -> Structure:
        structure = self._structure_from_any(cif_data)
        try:
            analyzer = SpacegroupAnalyzer(structure, symprec=self.symprec)
            structure = analyzer.get_primitive_standard_structure() if self.use_primitive else analyzer.get_conventional_standard_structure()
        except Exception:
            structure = structure.copy()
        return structure.get_sorted_structure()

    def _build_pseudopotentials(self) -> dict[str, str]:
        pseudo_map = dict(self.pseudopotentials)
        pseudo_candidates = {}
        detected_pseudo_dir = self.pseudo_dir

        for relative_path in self._static_files:
            suffix = Path(relative_path).suffix.lower()
            if suffix not in self.PSEUDOPOTENTIAL_SUFFIXES:
                continue
            file_name = os.path.basename(relative_path)
            relative_parent = Path(relative_path).parent.as_posix()
            if detected_pseudo_dir is None and relative_parent not in ("", "."):
                detected_pseudo_dir = relative_parent

            stem = Path(file_name).stem
            symbol = stem.split(".")[0].split("_")[0].split("-")[0]
            if symbol:
                pseudo_candidates[symbol.capitalize()] = file_name

        if self.structure is None:
            raise ValueError("load_cif must be called before pseudopotential resolution.")

        for specie in self.structure.composition.elements:
            symbol = specie.symbol
            if symbol in pseudo_map:
                continue
            if symbol in pseudo_candidates:
                pseudo_map[symbol] = pseudo_candidates[symbol]
            else:
                pseudo_map[symbol] = f"{symbol}.UPF"

        if detected_pseudo_dir is not None:
            self.pseudo_dir = detected_pseudo_dir

        return pseudo_map

    def _write_structure_files(self, target_dir: str):
        self._copy_static_files(target_dir, exclude={self.input_filename})
        pseudopotentials = self._build_pseudopotentials()

        control = dict(self.control)
        if self.pseudo_dir is not None:
            control.setdefault("pseudo_dir", self.pseudo_dir)

        system = dict(self.system)
        system.setdefault("ibrav", 0)
        system.setdefault("nat", len(self.structure))
        system.setdefault("ntyp", len(self.structure.composition.elements))

        pw_input = PWInput(
            structure=self.structure,
            pseudo=pseudopotentials,
            control=control,
            system=system,
            electrons=dict(self.electrons),
            ions=dict(self.ions),
            cell=dict(self.cell),
            kpoints_grid=self.kpoints_grid,
            kpoints_shift=self.kpoints_shift,
        )
        pw_input.write_file(os.path.join(target_dir, self.input_filename))
