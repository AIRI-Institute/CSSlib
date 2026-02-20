from math import prod
from itertools import product
import os
import json
from tqdm import tqdm
import pandas as pd
from copy import deepcopy
import zipfile
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifParser, CifBlock
from pymatgen.analysis.defects.generators import VoronoiInterstitialGenerator
import warnings
from config_logging import get_main_logger, get_supercell_worker_logger, get_collect_worker_logger
import sys

warnings.filterwarnings("ignore")


class CSS:
    _RESULTS_DIR = "results"
    _SUPERCELL_INPUT_CIFS_DIR = "disordered_structures"
    _SUPERCELL_OUTPUT_DIR = "ordered_representations"
    _ORDERED_REPRESENTATIONS_METADATA_DIR = "ordered_representations_metadata"

    def __init__(self, config_filename: str) -> None:
        with open(config_filename) as f:
            self.config = json.load(f)
        self._result_path = os.path.join(self._RESULTS_DIR, self.config["result_dir"])
        self._supercell_input_cifs_path = os.path.join(self._result_path, self._SUPERCELL_INPUT_CIFS_DIR)
        self._supercell_output_path = os.path.join(self._result_path, self._SUPERCELL_OUTPUT_DIR)
        self._ordered_representations_metadata_path = os.path.join(self._result_path,
                                                                   self._ORDERED_REPRESENTATIONS_METADATA_DIR)
        os.makedirs(self._RESULTS_DIR, exist_ok=True)
        os.makedirs(self._result_path)
        self._parser_data = None
        self._structure_sym = None
        self._scale_factor = 0
        self.logger = get_main_logger(self._result_path)

    def read_structure(self) -> None:
        """
        Read an initial structure from a cif-file.
        :return: None.
        """

        structure = Structure.from_file(self.config["structure_filename"])
        self.logger.info("Initial structure is read.")
        finder = SpacegroupAnalyzer(structure)
        self._structure_sym = finder.get_symmetrized_structure()
        self._structure_sym.to(os.path.join(self._result_path, "css_temp.cif"),
                               fmt="cif",
                               symprec=True,
                               refine_struct=True)
        parser = CifParser(os.path.join(self._result_path, "css_temp.cif"))
        self._parser_data = next(iter(parser._cif.data.values()))
        os.remove(os.path.join(self._result_path, "css_temp.cif"))

    def generate_interstitial_structure(self) -> None:
        """
        Generate interstitial structure using Voronoi algorithm and save it to a cif-file.
        Interstitial sites are filled by Neptunium species.
        :return: None.
        """

        self.logger.info("Preparing to generate interstitial structure.")
        interstitial_generator = VoronoiInterstitialGenerator()
        for i, interstitial in enumerate(interstitial_generator.generate(self._structure_sym, {"Np", })):
            self._parser_data["_atom_site_type_symbol"].append("Np")
            self._parser_data["_atom_site_label"].append(f"Np{i}")
            self._parser_data["_atom_site_symmetry_multiplicity"].append(str(interstitial.multiplicity))
            self._parser_data["_atom_site_fract_x"].append(f"{interstitial.site.frac_coords[0]:.7f}")
            self._parser_data["_atom_site_fract_y"].append(f"{interstitial.site.frac_coords[1]:.7f}")
            self._parser_data["_atom_site_fract_z"].append(f"{interstitial.site.frac_coords[2]:.7f}")
            self._parser_data["_atom_site_occupancy"].append("1.0")

        interstitial_structure_filename = self._create_interstitial_structure_filename()
        self._save_structure(self._parser_data, interstitial_structure_filename)
        self.logger.info("Interstitial structure is generated and saved at %s.", self._result_path)

    def generate_substituted_disordered_structures(self) -> None:
        """
        Generate substituted disordered structures (with partial occupancies) according to the configuration file.
        Save them to a cif-files.
        :return: None.
        """

        self.logger.info("Preparing to generate disordered structures (with partial occupancies) ...")
        cell_natoms = sum(map(int, self._parser_data["_atom_site_symmetry_multiplicity"]))
        self._scale_factor = prod(map(int, self.config["supercell"].split("x")))
        supercell_natoms = cell_natoms * self._scale_factor

        for subst in self.config["substitution"]:
            subst["substitution_low_limit_natoms"] = (int(subst["substitution_low_limit"] *
                                                          supercell_natoms + 0.001))
            subst["substitution_high_limit_natoms"] = (int(subst["substitution_high_limit"] *
                                                           supercell_natoms + 0.001))
            subst["indices_to_substitute"] = [j for j in range(len(self._parser_data["_atom_site_type_symbol"]))
                                              if self._parser_data["_atom_site_type_symbol"][j] ==
                                              subst["specie_to_substitute"]]

        subst_natoms_list = []
        product_range = range(1 + max([subst["substitution_high_limit_natoms"]
                                       for subst in self.config["substitution"]]))
        product_repeat = sum([len(subst["indices_to_substitute"]) for subst in self.config["substitution"]])

        for subst_natoms in product(product_range, repeat=product_repeat):
            idx_right = 0
            for subst in self.config["substitution"]:
                idx_left = idx_right
                idx_right += len(subst["indices_to_substitute"])
                if sum(subst_natoms[idx_left: idx_right]) > subst["substitution_high_limit_natoms"]:
                    break
            else:
                subst_natoms_list.append(subst_natoms)

        os.makedirs(self._supercell_input_cifs_path)

        for subst_natoms in subst_natoms_list:
            p_data = deepcopy(self._parser_data)
            k = 0
            indices_to_substitute_occup = {i: 1.0 for subst in self.config["substitution"]
                                           for i in subst["indices_to_substitute"]}
            for subst in self.config["substitution"]:
                for j in range(len(subst["indices_to_substitute"])):
                    p_data["_atom_site_type_symbol"].append(subst["substitute_with"])
                    p_data["_atom_site_label"].append(subst["substitute_with"] + str(k))
                    p_data["_atom_site_symmetry_multiplicity"].append(
                        p_data["_atom_site_symmetry_multiplicity"][subst["indices_to_substitute"][j]])
                    p_data["_atom_site_fract_x"].append(
                        p_data["_atom_site_fract_x"][subst["indices_to_substitute"][j]])
                    p_data["_atom_site_fract_y"].append(
                        p_data["_atom_site_fract_y"][subst["indices_to_substitute"][j]])
                    p_data["_atom_site_fract_z"].append(
                        p_data["_atom_site_fract_z"][subst["indices_to_substitute"][j]])
                    new_atom_site_occupancy = (subst_natoms[k] / self._scale_factor /
                                               int(p_data['_atom_site_symmetry_multiplicity'][subst['indices_to_substitute'][j]]))
                    p_data["_atom_site_occupancy"].append(f"{new_atom_site_occupancy:.7f}")
                    indices_to_substitute_occup[subst["indices_to_substitute"][j]] -= new_atom_site_occupancy
                    k += 1

            for idx, occup in indices_to_substitute_occup.items():
                if occup >= 0.:
                    p_data["_atom_site_occupancy"][idx] = f"{occup:.7f}"
                else:
                    break
            else:
                supercell_structure_filename = self._create_supercell_structure_filename(p_data)
                self._save_structure(p_data, self._SUPERCELL_INPUT_CIFS_DIR, supercell_structure_filename)
                self.logger.debug("%s disordered structure (with partial occupancies) is generated and saved.",
                                  supercell_structure_filename)
        self.logger.info("%d disordered structures (with partial occupancies) are generated and saved at %s.",
                         len(os.listdir(self._supercell_input_cifs_path)), self._supercell_input_cifs_path)

    def _create_supercell_structure_filename(self, parser_data: CifBlock) -> str:
        """
        Create a filename for disordered structure (with partial occupancies).
        :param parser_data: Structural data to save.
        :return: Filename.
        """

        supercell_structure_filename = ""
        for idx in range(len(parser_data["_atom_site_type_symbol"])):
            supercell_structure_filename += parser_data["_atom_site_type_symbol"][idx]
            supercell_structure_filename += f"{int(float(parser_data['_atom_site_occupancy'][idx]) * int(parser_data['_atom_site_symmetry_multiplicity'][idx]) * self._scale_factor + 0.001)}"
        supercell_structure_filename += ".cif"
        return supercell_structure_filename

    def _create_interstitial_structure_filename(self) -> str:
        """
        Create a filename for interstitial structure.
        :return: Filename.
        """

        return os.path.splitext(os.path.split(self.config["structure_filename"])[1])[0] + "_interstitial" + ".cif"

    def _save_structure(self, parser_data: CifBlock, *args: str) -> None:
        """
        Save a structure to a cif-file.
        :param parser_data: Structural data to save.
        :param args: Path to the directory to save the structure.
        :return: None.
        """

        with open(os.path.join(self._result_path, *args), "w") as f:
            f.write(str(parser_data))

    @staticmethod
    def _init_supercell_worker(result_path: str) -> None:
        """
        Configure logger for supercell worker.
        :param result_path: Path to the results' directory.
        :return: None.
        """

        logger_ = get_supercell_worker_logger(result_path)
        global supercell_worker_logger
        supercell_worker_logger = logger_

    @staticmethod
    def _supercell_worker(cmd: str, compound: str) -> int:
        """
        Run a Supercell worker (a process with Supercell software instance).
        :param cmd: Command to run.
        :return: None.
        """

        worker_result = subprocess.run(cmd, shell=True, text=True, encoding="utf-8", capture_output=True)
        if worker_result.returncode == 0:
            num_struct_before = re.search(r"The total number of combinations is (\d+)", worker_result.stdout).group(1)
            num_struct_after = re.search(r"Combinations after merge: (\d+)", worker_result.stdout).group(1)
            supercell_worker_logger.info("%s - DONE! - The total number of structures: %s - Symmetrically inequivalent structures: %s",
                                         compound, num_struct_before, num_struct_after)
        else:
            supercell_worker_logger.info("%s - FAILED! - %s", compound, worker_result.stderr)
        return worker_result.returncode

    def run_supercell(self) -> None:
        """
        Run Supercell software to convert disordered structures (with partial occupancies)
        to ordered representations (supercell structures).
        :return: None.
        """

        self.logger.info("Preparing to check out possibility of creation ordered representations of disordered structures ...")
        if (error_message := self._dry_run_supercell()) is not None:
            self.logger.error("%s Change config-file to simplify CSS and try again.", error_message.rstrip())
            sys.exit(1)
        self.logger.info("Checking out possibility of creation ordered representations of disordered structures is finished successfully!")
        self.logger.info("Preparing to generate ordered representations of disordered structures ...")
        os.makedirs(self._supercell_output_path)
        futures = []
        with (tqdm(range(len(os.listdir(self._supercell_input_cifs_path))),
                   desc="Creating ordered representations of disordered structures",
                   unit=" composition",
                   ncols=200)
              as pbar,
              ProcessPoolExecutor(max_workers=self.config["num_workers"],
                                  initializer=self._init_supercell_worker,
                                  initargs=(self._result_path,))
              as pool):
            for supercell_structure_filename in os.listdir(self._supercell_input_cifs_path):
                cmd = f"supercell -i {os.path.join(self._supercell_input_cifs_path, supercell_structure_filename)} -m "\
                      f"-s {self.config['supercell']} "\
                      f"-a {os.path.join(self._supercell_output_path, supercell_structure_filename.replace('.cif', ''))}.zip "\
                      f"-o {supercell_structure_filename.replace('.cif', '')}"
                compound = supercell_structure_filename.replace('.cif', '')
                future = pool.submit(self._supercell_worker, cmd, compound)
                future.add_done_callback(lambda p: pbar.update())
                futures.append(future)

            num_failed_tasks = 0
            for future in as_completed(futures):
                num_failed_tasks += future.result() != 0
        self.logger.info("Ordered representations of disordered structures are generated and saved at %s.",
                         self._supercell_output_path)
        if num_failed_tasks:
            self.logger.info("Generation of ordered representations of disordered structures is failed for %d compound(s)!", num_failed_tasks)
        else:
            self.logger.info("Generation of ordered representations of disordered structures is finished successfully!")

    @staticmethod
    def _dry_supercell_worker(cmd: str) -> str | None:
        """
        Run a Supercell worker (a process with Supercell software instance) in dry-run mode
        to check the possibility of creation ordered representations of disordered structures.
        :param cmd: Command to run.
        :return: Error message if something went wrong, None otherwise.
        """

        worker_result = subprocess.run(cmd, shell=True, text=True, encoding="utf-8", capture_output=True)
        if worker_result.returncode == 0:
            return None
        return worker_result.stderr

    def _dry_run_supercell(self) -> str | None:
        """
        Check the possibility of creation ordered representations of disordered structures.
        :return: None if the possibility exists, error message otherwise.
        """

        futures = []
        with tqdm(range(len(os.listdir(self._supercell_input_cifs_path))),
                  desc="Checking out possibility of creation ordered representations of disordered structures",
                  unit=" composition",
                  ncols=200) as pbar:
            pool = ProcessPoolExecutor(max_workers=self.config["num_workers"])
            for supercell_structure_filename in os.listdir(self._supercell_input_cifs_path):
                cmd = f"supercell -i {os.path.join(self._supercell_input_cifs_path, supercell_structure_filename)} "\
                      f"-s {self.config['supercell']} -d -v 0"
                future = pool.submit(self._dry_supercell_worker, cmd)
                future.add_done_callback(lambda p: pbar.update())
                futures.append(future)
            for future in as_completed(futures):
                if (error_message := future.result()) is not None:
                    pool.shutdown(wait=True, cancel_futures=True)
                    return error_message
            else:
                pool.shutdown(wait=True, cancel_futures=False)
                return None

    @staticmethod
    def _init_collect_worker(fields: tuple, substitute_with_species: tuple,
                             ordered_representations_metadata_path: str, result_path: str) -> None:
        """
        Initialize collect workers.
        :param fields: Names of dataframe columns where metadata collected.
        :param substitute_with_species: Species that were used as substitutes.
        :param ordered_representations_metadata_path: Path to ordered representations of disordered structures.
        :param result_path: Path to the results' directory.
        :return: None.
        """

        global fields_, substitute_with_species_, ordered_representations_metadata_path_, collect_worker_logger_
        fields_ = fields
        substitute_with_species_ = substitute_with_species
        ordered_representations_metadata_path_ = ordered_representations_metadata_path
        logger = get_collect_worker_logger(result_path)
        collect_worker_logger_ = logger

    @staticmethod
    def _collect_data_one_composition(archive_path: str) -> None:
        """
        Collect meta-information about one particular composition.
        :param archive_path: Path to archive containing ordered representations of disordered structure.
        :return: None.
        """

        ordered_representations_metadata = {key: [] for key in fields_}
        with zipfile.ZipFile(archive_path, "r") as archive:
            for structure_filename in archive.namelist():
                with archive.open(structure_filename, "r") as file:
                    file_data = file.read().decode("utf-8")
                    structure = CifParser.from_str(file_data).get_structures(primitive=False)[0]
                    finder = SpacegroupAnalyzer(structure)
                    specie_counter = Counter(map(str, structure.species))
                    ordered_representations_metadata["cif_data"].append(file_data)
                    ordered_representations_metadata["structure_filename"].append(structure_filename.replace(".zip", ""))
                    ordered_representations_metadata["composition"].append(str(structure.composition))
                    ordered_representations_metadata["space_group_no"].append(int(finder.get_space_group_number()))
                    ordered_representations_metadata["space_group_symbol"].append(finder.get_space_group_symbol())
                    ordered_representations_metadata["weight"].append(int(re.search(r"_w(.*?).cif", structure_filename).group(1)))
                    for specie in substitute_with_species_:
                        ordered_representations_metadata[f"{specie}_concentration"].append(specie_counter[specie] / len(structure))
        ordered_representations_metadata_df = pd.DataFrame.from_dict(ordered_representations_metadata)
        ordered_representations_metadata_path = os.path.join(ordered_representations_metadata_path_,
                                                             os.path.splitext(os.path.split(archive_path)[1])[0] + ".pkl.gz")
        ordered_representations_metadata_df.to_pickle(ordered_representations_metadata_path)
        collect_worker_logger_.info(
            "%s - DONE! - The total number of structures: %d",
            os.path.splitext(os.path.split(archive_path)[1])[0], ordered_representations_metadata_df.shape[0])

    def collect_data_mp(self) -> None:
        """
        Collect meta-information about all ordered representations of disordered structures
        and save it to pandas dataframes.
        :return: None.
        """

        self.logger.info("Preparing to collect ordered representations' metadata ...")
        substitute_with_species = tuple({subst['substitute_with'] for subst in self.config["substitution"]})
        fields = ["cif_data", "structure_filename", "composition", "space_group_no", "space_group_symbol", "weight"]
        for specie in substitute_with_species:
            fields.append(f"{specie}_concentration")
        fields = tuple(fields)
        os.makedirs(self._ordered_representations_metadata_path)
        archive_paths = [os.path.join(self._supercell_output_path, archive_filename)
                         for archive_filename in os.listdir(self._supercell_output_path)]
        with (tqdm(range(len(os.listdir(self._supercell_output_path))),
                   desc="Collecting ordered representations' metadata of disordered structures",
                   unit=" composition",
                   ncols=200)
              as pbar,
              ProcessPoolExecutor(max_workers=self.config["num_workers"],
                                  initializer=self._init_collect_worker,
                                  initargs=(fields, substitute_with_species, self._ordered_representations_metadata_path, self._result_path))
              as pool):
            for archive_path in archive_paths:
                future = pool.submit(self._collect_data_one_composition, archive_path)
                future.add_done_callback(lambda p: pbar.update())
        self.logger.info("Ordered representations' metadata are collected and saved at %s.",
                         self._ordered_representations_metadata_path)
