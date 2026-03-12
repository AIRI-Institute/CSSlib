"""Main library file with CSS class."""
import csslib.exceptions as css_exc
import os
import pandas as pd
import re
import subprocess
import sys
import warnings
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from csslib.config_logging import get_main_logger, get_supercell_worker_logger, get_collect_worker_logger
from csslib.config import Config
from itertools import product
from math import prod
from pymatgen.analysis.defects.generators import VoronoiInterstitialGenerator
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser, CifBlock
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm

warnings.filterwarnings("ignore")


class CSS:
    _RESULTS_DIR = "results"
    _CSS_STRUCTURES_DIR = "css_structures"
    _CSS_STRUCTURES_METADATA_DIR = "css_structures_metadata"
    _CIF_FIELDS = ["_atom_site_type_symbol", "_atom_site_label",
                   "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z",
                   "_atom_site_occupancy"]

    def __init__(self, config_filename: str, rewrite_results: bool = False) -> None:
        if not os.path.isfile(config_filename):
            raise css_exc.ConfigurationNotFoundError(f'{config_filename} is not found.')

        json_data, val_err = None, None
        with open(config_filename, 'rb') as f:
            json_data = f.read()
        try:
            self.config = Config.model_validate_json(json_data)
        except css_exc.ValidationError as e:
            val_err = css_exc.catch_config_errors(e)
        if val_err is not None:
            raise css_exc.ConfigurationError(val_err)

        self._result_path = os.path.join(self._RESULTS_DIR, self.config.result_dir)
        os.makedirs(self._RESULTS_DIR, exist_ok=True)
        try:
            os.makedirs(self._result_path)
        except FileExistsError:
            if not rewrite_results:
                raise css_exc.ResultsFolderExistError

        self._css_structures_path = os.path.join(self._result_path, self._CSS_STRUCTURES_DIR)
        self._css_structures_metadata_path = os.path.join(self._result_path, self._CSS_STRUCTURES_METADATA_DIR)
        self._substitution_template_filename = f"substitution_template_{os.path.basename(self.config.structure_filename)}"
        self._structure = None
        self._parser_data = None
        self._scale_factor = 0
        self._substitution_labels_natoms_list = []
        self.logger = get_main_logger(self._result_path)

    def generate_css(self):
        """
        Generate css structures.
        """
        self._read_structure()
        self._evaluate_substitution_parameters()
        self._create_and_save_substitution_template()
        self._dry_run_supercell()
        self._run_supercell()
        self._collect_data()

    def _read_structure(self) -> None:
        """
        Read an initial structure from a cif-file.
        """
        try:
            self._structure = Structure.from_file(self.config.structure_filename)
        except FileNotFoundError:
            raise css_exc.StructureNotFoundError(f'Structure file `{self.config.structure_filename}` is not found.')
        parser = CifParser(self.config.structure_filename)
        self._parser_data = next(iter(parser._cif.data.values()))
        self.logger.info("Initial structure is read at %s.", self.config.structure_filename)

    # def generate_interstitial_structure(self) -> None:  # TODO: implement smart-exceptions
    #     """
    #     Generate interstitial structure using Voronoi algorithm and save it to a cif-file.
    #     Interstitial sites are filled by Neptunium species.
    #     :return: None.
    #     """
    #
    #     self.logger.info("Preparing to generate interstitial structure.")
    #     interstitial_generator = VoronoiInterstitialGenerator()
    #     for i, interstitial in enumerate(interstitial_generator.generate(self._structure_sym, {"Np", })):
    #         self._parser_data["_atom_site_type_symbol"].append("Np")
    #         self._parser_data["_atom_site_label"].append(f"Np{i}")
    #         self._parser_data["_atom_site_symmetry_multiplicity"].append(str(interstitial.multiplicity))
    #         self._parser_data["_atom_site_fract_x"].append(f"{interstitial.site.frac_coords[0]:.7f}")
    #         self._parser_data["_atom_site_fract_y"].append(f"{interstitial.site.frac_coords[1]:.7f}")
    #         self._parser_data["_atom_site_fract_z"].append(f"{interstitial.site.frac_coords[2]:.7f}")
    #         self._parser_data["_atom_site_occupancy"].append("1.0")
    #
    #     interstitial_structure_filename = self._create_interstitial_structure_filename()
    #     self._save_structure(self._parser_data, interstitial_structure_filename)
    #     self.logger.info("Interstitial structure is generated and saved at %s.", self._result_path)

    def _evaluate_substitution_parameters(self) -> None:
        """
        Evaluate all possible combinations of substituted species' numbers.
        """
        self.logger.info("Preparing to evaluate substitution parameters ...")
        cell_natoms = len(self._structure)
        self._scale_factor = prod(map(int, self.config.supercell.split("x")))
        supercell_natoms = cell_natoms * self._scale_factor

        substitute_with_labels_list = []
        labels_to_substitute_list = []
        for subst in self.config.substitution:
            subst.substitution_low_limit_natoms = (int(subst.substitution_low_limit * supercell_natoms + 0.0001))
            subst.substitution_high_limit_natoms = (int(subst.substitution_high_limit * supercell_natoms + 0.0001))
            subst.indices_to_substitute = [j for j in range(len(self._parser_data["_atom_site_type_symbol"]))
                                           if self._parser_data["_atom_site_type_symbol"][j] ==
                                           subst.specie_to_substitute]
            subst.substitute_with_labels = []
            subst.labels_to_substitute = []
            label_idx = 0
            label_to_substitute = ""
            for j in range(len(subst.indices_to_substitute)):
                while ((label := subst.substitute_with + str(label_idx)) in
                       self._parser_data["_atom_site_label"] + substitute_with_labels_list) and \
                        label_to_substitute != self._parser_data["_atom_site_label"][subst.indices_to_substitute[j]]:
                    label_idx += 1
                label_to_substitute = self._parser_data["_atom_site_label"][subst.indices_to_substitute[j]]
                subst.substitute_with_labels.append(label)
                subst.labels_to_substitute.append(label_to_substitute)
                if substitute_with_labels_list and substitute_with_labels_list[-1] == label:
                    continue
                substitute_with_labels_list.append(subst.substitute_with_labels[-1])
                labels_to_substitute_list.append(subst.labels_to_substitute[-1])

        substitute_with_natoms_list = []
        product_range = range(1 + max([subst.substitution_high_limit_natoms for subst in self.config.substitution]))
        product_repeat = len(substitute_with_labels_list)
        for substitute_with_natoms in product(product_range, repeat=product_repeat):
            idx_right = 0
            for subst in self.config.substitution:
                idx_left = idx_right
                idx_right += len(set(subst.substitute_with_labels))
                substitute_with_natoms_sum = sum(substitute_with_natoms[idx_left: idx_right])
                if (substitute_with_natoms_sum > subst.substitution_high_limit_natoms or
                        substitute_with_natoms_sum < subst.substitution_low_limit_natoms):
                    break
            else:
                substitute_with_natoms_list.append(substitute_with_natoms)

        labels_to_substitute_natoms = {}
        labels_counter = Counter(self._structure.labels)
        for label_to_substitute in labels_to_substitute_list:
            labels_to_substitute_natoms[label_to_substitute] = labels_counter[label_to_substitute] * self._scale_factor

        for substitute_with_natoms in substitute_with_natoms_list:
            labels_natoms = {}
            for i in range(len(substitute_with_natoms)):
                labels_natoms[substitute_with_labels_list[i]] = substitute_with_natoms[i]
                label_to_substitute = labels_to_substitute_list[i]
                labels_natoms[label_to_substitute] = (labels_natoms.get(label_to_substitute,
                                                                        labels_to_substitute_natoms[label_to_substitute])
                                                      - substitute_with_natoms[i])
                if labels_natoms[label_to_substitute] < 0:
                    break
            else:
                self._substitution_labels_natoms_list.append(labels_natoms)
        self.logger.info("Substitution parameters are evaluated successfully!")

    def _create_and_save_substitution_template(self) -> None:
        """
        Create and save template for subsequent substitution runs, i.e. for css structures generation.
        """
        for subst in self.config.substitution:
            for j in range(len(subst.indices_to_substitute)):
                self._parser_data["_atom_site_type_symbol"].append(subst.substitute_with)
                self._parser_data["_atom_site_label"].append(subst.substitute_with_labels[j])
                self._parser_data["_atom_site_fract_x"].append(
                    self._parser_data["_atom_site_fract_x"][subst.indices_to_substitute[j]])
                self._parser_data["_atom_site_fract_y"].append(
                    self._parser_data["_atom_site_fract_y"][subst.indices_to_substitute[j]])
                self._parser_data["_atom_site_fract_z"].append(
                    self._parser_data["_atom_site_fract_z"][subst.indices_to_substitute[j]])
                self._parser_data["_atom_site_occupancy"].append("0.0")
        for i in range(len(self._parser_data.loops)):
            if "_atom_site_type_symbol" in self._parser_data.loops[i]:
                self._parser_data.loops[i] = self._CIF_FIELDS
        self._save_structure(self._parser_data, self._substitution_template_filename)
        self.logger.info("Substitution template is created and saved at %s.",
                         os.path.join(self._result_path, self._substitution_template_filename))

    @staticmethod
    def _create_css_structures_filename(substitution_labels_natoms: dict) -> str:
        """
        Create a filename for archive with css structures.

        Args:
            substitution_labels_natoms (dict): Amount of substituted species in the supercell structure.

        Returns:
            str: Filename.
        """
        return "-".join([f"{k}_{v}" for k, v in substitution_labels_natoms.items()]) + ".zip"

    # def _create_interstitial_structure_filename(self) -> str:
    #     """
    #     Create a filename for interstitial structure.
    #     :return: Filename.
    #     """
    #
    #     return os.path.splitext(os.path.split(self.config.structure_filename)[1])[0] + "_interstitial" + ".cif"

    def _save_structure(self, parser_data: CifBlock, *args: str) -> None:
        """
        Save a structure to a cif-file.

        Args:
            parser_data (CifBlock): Structural data to save.
            args (str): Path to the directory to save the structure.
        """
        with open(os.path.join(self._result_path, *args), "w") as f:
            f.write(str(parser_data))

    @staticmethod
    def _init_supercell_worker(result_path: str) -> None:
        """
        Initialize supercell workers.

        Args:
            result_path (str): Path to the results' directory.
        """
        logger_ = get_supercell_worker_logger(result_path)
        global supercell_worker_logger
        supercell_worker_logger = logger_

    @staticmethod
    def _supercell_worker(cmd: str, css_structures_filename: str) -> int:
        """
        Run a Supercell worker (a process with Supercell software instance).

        Args:
            cmd (str): Command to run.

        Returns:
            int: Return code.
        """
        worker_output = subprocess.run(cmd, shell=True, text=True, encoding="utf-8", capture_output=True)
        if worker_output.returncode == 0:
            num_struct_before = re.search(r"The total number of combinations is (\d+)", worker_output.stdout).group(1)
            num_struct_after = re.search(r"Combinations after merge: (\d+)", worker_output.stdout).group(1)
            supercell_worker_logger.info("%s - DONE! - The total number of structures: %s - Symmetrically inequivalent structures: %s",
                                         css_structures_filename, num_struct_before, num_struct_after)
        else:
            supercell_worker_logger.info("%s - FAILED! - %s", css_structures_filename, worker_output.stderr)
        return worker_output.returncode

    def _run_supercell(self) -> None:
        """
        Run Supercell software to create css structures.
        """
        self.logger.info("Preparing to generate CSS structures ...")
        os.makedirs(self._css_structures_path)
        futures = []
        with (tqdm(range(len(self._substitution_labels_natoms_list)),
                   desc="Creating CSS structures",
                   unit=" composition",
                   ncols=200)
              as pbar,
              ProcessPoolExecutor(max_workers=self.config.num_workers,
                                  initializer=self._init_supercell_worker,
                                  initargs=(self._result_path,))
              as pool):
            for substitution_labels_natoms in self._substitution_labels_natoms_list:
                css_structures_filename = self._create_css_structures_filename(substitution_labels_natoms)
                cmd = f"supercell -m " \
                      f"-i {os.path.join(self._result_path, self._substitution_template_filename)} "\
                      f"-s {self.config.supercell} "\
                      f"-a {os.path.join(self._css_structures_path, css_structures_filename)} "\
                      f"-o {os.path.splitext(css_structures_filename)[0]} "\
                      f"-t 0.7"
                for label, natoms in substitution_labels_natoms.items():
                    cmd += f' -p "{label}:p={natoms}"'
                future = pool.submit(self._supercell_worker, cmd, css_structures_filename)
                future.add_done_callback(lambda p: pbar.update())
                futures.append(future)

            num_failed_tasks = 0
            for future in as_completed(futures):
                num_failed_tasks += future.result() != 0
        self.logger.info("CSS structures are generated and saved at %s.", self._css_structures_path)
        if num_failed_tasks:
            self.logger.info("Generation of CSS structures is failed for %d compound(s)!", num_failed_tasks)
        else:
            self.logger.info("Generation of CSS structures is finished successfully!")

    @staticmethod
    def _dry_supercell_worker(cmd: str) -> str | None:
        """
        Run a Supercell worker (a process with Supercell software instance) in dry-run mode
        to check the possibility of css structures creation.

        Args:
            cmd (str): Command to run.

        Returns:
            str or None: Error message if something went wrong, None otherwise.
        """
        worker_output = subprocess.run(cmd, shell=True, text=True, encoding="utf-8", capture_output=True)
        if worker_output.returncode == 0:
            return None
        return worker_output.stderr

    def _dry_run_supercell(self) -> None:
        """
        Check the possibility of css structures creation.
        """
        self.logger.info("Preparing to check out possibility of CSS structures creation ...")
        futures = []
        with tqdm(range(len(self._substitution_labels_natoms_list)),
                  desc="Checking out possibility of CSS structures creation",
                  unit=" composition",
                  ncols=200) as pbar:
            pool = ProcessPoolExecutor(max_workers=self.config.num_workers)
            for substitution_labels_natoms in self._substitution_labels_natoms_list:
                cmd = f"supercell -d -v 0 " \
                      f"-i {os.path.join(self._result_path, self._substitution_template_filename)} " \
                      f"-s {self.config.supercell} -t 0.7"
                for label, natoms in substitution_labels_natoms.items():
                    cmd += f' -p "{label}:p={natoms}"'
                future = pool.submit(self._dry_supercell_worker, cmd)
                future.add_done_callback(lambda p: pbar.update())
                futures.append(future)
            for future in as_completed(futures):
                if (error_message := future.result()) is not None:
                    pool.shutdown(wait=True, cancel_futures=True)
                    self.logger.error("%s Change config-file to simplify CSS and try again.", error_message.rstrip())
                    sys.exit(1)
            pool.shutdown(wait=True, cancel_futures=False)
            self.logger.info("Checking out possibility of CSS structures creation is finished successfully!")

    @staticmethod
    def _init_collect_worker(fields: tuple, substitute_with_species: tuple,
                             css_structures_metadata_path: str, result_path: str) -> None:
        """
        Initialize collect workers.

        Args:
            fields (tuple): Names of dataframe columns where metadata collected.
            substitute_with_species (tuple): Species that were used as substitutes.
            css_structures_metadata_path (str): Path to archives with css structures.
            result_path (str): Path to the results' directory.
        """
        global fields_, substitute_with_species_, css_structures_metadata_path_, collect_worker_logger_
        fields_ = fields
        substitute_with_species_ = substitute_with_species
        css_structures_metadata_path_ = css_structures_metadata_path
        logger = get_collect_worker_logger(result_path)
        collect_worker_logger_ = logger

    @staticmethod
    def _collect_data_one_composition(archive_path: str) -> None:
        """
        Collect meta-information about one particular composition.

        Args:
             archive_path (str): Path to archive containing css structures with single composition.
        """
        css_structures_metadata = {key: [] for key in fields_}
        with zipfile.ZipFile(archive_path, "r") as archive:
            for structure_filename in archive.namelist():
                with archive.open(structure_filename, "r") as file:
                    file_data = file.read().decode("utf-8")
                    structure = CifParser.from_str(file_data).get_structures(primitive=False)[0]
                    finder = SpacegroupAnalyzer(structure)
                    specie_counter = Counter(map(str, structure.species))
                    css_structures_metadata["cif_data"].append(file_data)
                    css_structures_metadata["structure_filename"].append(os.path.splitext(structure_filename)[0])
                    css_structures_metadata["composition"].append(str(structure.composition))
                    css_structures_metadata["space_group_no"].append(int(finder.get_space_group_number()))
                    css_structures_metadata["space_group_symbol"].append(finder.get_space_group_symbol())
                    css_structures_metadata["weight"].append(int(re.search(r"_w(\d+).cif", structure_filename).group(1)))
                    for specie in substitute_with_species_:
                        css_structures_metadata[f"{specie}_concentration"].append(specie_counter[specie] / len(structure))
        css_structures_metadata_df = pd.DataFrame.from_dict(css_structures_metadata)
        css_structures_metadata_path = os.path.join(css_structures_metadata_path_,
                                                    os.path.splitext(os.path.basename(archive_path))[0] + ".pkl.gz")
        css_structures_metadata_df.to_pickle(css_structures_metadata_path)
        collect_worker_logger_.info("%s - DONE! - The total number of CSS structures: %d",
                                    os.path.splitext(os.path.basename(archive_path))[0],
                                    css_structures_metadata_df.shape[0])

    def _collect_data(self) -> None:
        """
        Collect meta-information about all css structures and save it to pandas dataframes.
        """
        self.logger.info("Preparing to collect metadata of CSS structures ...")
        substitute_with_species = tuple({subst.substitute_with for subst in self.config.substitution})
        fields = ["cif_data", "structure_filename", "composition", "space_group_no", "space_group_symbol", "weight"]
        for specie in substitute_with_species:
            fields.append(f"{specie}_concentration")
        fields = tuple(fields)
        os.makedirs(self._css_structures_metadata_path)
        archive_paths = [os.path.join(self._css_structures_path, archive_filename)
                         for archive_filename in os.listdir(self._css_structures_path)]
        with (tqdm(range(len(os.listdir(self._css_structures_path))),
                   desc="Collecting metadata of CSS structures",
                   unit=" composition",
                   ncols=200)
              as pbar,
              ProcessPoolExecutor(max_workers=self.config.num_workers,
                                  initializer=self._init_collect_worker,
                                  initargs=(fields, substitute_with_species,
                                            self._css_structures_metadata_path, self._result_path))
              as pool):
            for archive_path in archive_paths:
                future = pool.submit(self._collect_data_one_composition, archive_path)
                future.add_done_callback(lambda p: pbar.update())
        self.logger.info("Metadata of CSS structures is collected and saved at %s.", self._css_structures_metadata_path)

    def __repr__(self) -> str:
        """
        String representation of CSS class object.

        Returns:
            str: Representation string.
        """
        ...  # TODO: implement repr method for more informative output about class object
