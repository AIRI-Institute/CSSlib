from math import prod
from itertools import product
import os
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
from copy import deepcopy
import zipfile
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import subprocess
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifParser, CifBlock
from pymatgen.analysis.defects.generators import VoronoiInterstitialGenerator
from pymatgen.symmetry.groups import SpaceGroup
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import networkx as nx
import warnings

warnings.filterwarnings("ignore")


class CSS:
    _RESULTS_DIR = "results"
    _SUPERCELL_INPUT_CIFS_DIR = "supercell_input_cifs"
    _SUPERCELL_OUTPUT_DIR = "supercell_output"

    def __init__(self, config_filename: str) -> None:
        with open(config_filename) as f:
            self.config = json.load(f)
        os.makedirs(self._RESULTS_DIR, exist_ok=True)
        os.makedirs(os.path.join(self._RESULTS_DIR, self.config["result_dir"]))
        self._parser_data = None
        self._structure_sym = None
        self._scale_factor = 0

    def read_structure(self) -> None:
        """
        Read an initial structure from a cif-file.
        :return: None.
        """

        structure = Structure.from_file(self.config["structure_filename"])
        finder = SpacegroupAnalyzer(structure)
        self._structure_sym = finder.get_symmetrized_structure()
        self._structure_sym.to(os.path.join(self._RESULTS_DIR, self.config["result_dir"], "css_temp.cif"),
                               fmt="cif", symprec=True, refine_struct=True)
        parser = CifParser(os.path.join(self._RESULTS_DIR, self.config["result_dir"], "css_temp.cif"))
        self._parser_data = next(iter(parser._cif.data.values()))
        os.remove(os.path.join(self._RESULTS_DIR, self.config["result_dir"], "css_temp.cif"))

    def generate_interstitial_structure(self) -> None:
        """
        Generate interstitial structure using Voronoi algorithm and save it to a cif-file.
        Interstitial sites are filled by deuterium species.
        :return: None.
        """

        interstitial_generator = VoronoiInterstitialGenerator()
        for i, interstitial in enumerate(interstitial_generator.generate(self._structure_sym, {"D", })):
            self._parser_data["_atom_site_type_symbol"].append("D")
            self._parser_data["_atom_site_label"].append(f"D{i}")
            self._parser_data["_atom_site_symmetry_multiplicity"].append(str(interstitial.multiplicity))
            self._parser_data["_atom_site_fract_x"].append(f"{interstitial.site.frac_coords[0]:.7f}")
            self._parser_data["_atom_site_fract_y"].append(f"{interstitial.site.frac_coords[1]:.7f}")
            self._parser_data["_atom_site_fract_z"].append(f"{interstitial.site.frac_coords[2]:.7f}")
            self._parser_data["_atom_site_occupancy"].append("1.0")

        interstitial_structure_filename = self._create_interstitial_structure_filename()
        self._save_structure(self._parser_data, interstitial_structure_filename)

    def generate_substituted_disordered_structures(self) -> None:
        """
        Generate substituted disordered structures with partial occupancies according to the configuration file.
        Save them to a cif-files.
        :return: None.
        """

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

        os.makedirs(os.path.join(self._RESULTS_DIR, self.config["result_dir"], self._SUPERCELL_INPUT_CIFS_DIR))

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

    def _create_supercell_structure_filename(self, parser_data: CifBlock) -> str:
        """
        Create a filename for disordered supercell structure.
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

        return self.config["structure_filename"].replace(".cif", "") + "_interstitial" + ".cif"

    def _save_structure(self, parser_data: CifBlock, *args: str) -> None:
        """
        Save a structure to a cif-file.
        :param parser_data: Structural data to save.
        :param args: Path to the directory to save the structure.
        :return: None.
        """

        with open(os.path.join(self._RESULTS_DIR, self.config["result_dir"], *args), "w") as f:
            f.write(str(parser_data))

    @staticmethod
    def _supercell_worker(cmd: str) -> None:
        """
        Run a command (a process with Supercell software instance).
        :param cmd: Command to run.
        :return: None.
        """

        subprocess.run(cmd, shell=True)

    def run_supercell(self) -> None:
        """
        Run Supercell software to convert a crystallographic structures with partial occupancies and/or vacancies
        to ordinary supercell structures.
        :return: None.
        """

        commands = [f"supercell -i {os.path.join(self._RESULTS_DIR, self.config['result_dir'], self._SUPERCELL_INPUT_CIFS_DIR, supercell_structure_filename)} -m "
                    f"-s {self.config['supercell']} "
                    f"-a {os.path.join(self._RESULTS_DIR, self.config['result_dir'], self._SUPERCELL_OUTPUT_DIR, supercell_structure_filename.replace('.cif', ''))}.zip "
                    f"-o {supercell_structure_filename.replace('.cif', '')}" for supercell_structure_filename
                    in os.listdir(os.path.join(self._RESULTS_DIR, self.config["result_dir"],
                                               self._SUPERCELL_INPUT_CIFS_DIR))]
        os.makedirs(os.path.join(self._RESULTS_DIR, self.config["result_dir"], self._SUPERCELL_OUTPUT_DIR))
        with ProcessPoolExecutor(max_workers=self.config["num_workers"]) as pool:
            with tqdm(desc="Creating disordered structures", unit=" composition", ncols=100) as pbar:
                for cmd in commands:
                    future = pool.submit(self._supercell_worker, cmd)
                    future.add_done_callback(lambda p: pbar.update())

    def collect_data(self) -> None:
        """
        Collect data from the output of Supercell software to a pandas dataframe and save it to a pickle-file.
        :return: None.
        """

        data_dict = {"cif_data": [], "structure_filename": [], "composition": [],
                     "space_group_no": [], "space_group_symbol": [], "weight": []}
        substitute_with_species = {subst['substitute_with'] for subst in self.config["substitution"]}
        for specie in substitute_with_species:
            data_dict[f"{specie}_concentration"] = []
        for archive_filename in tqdm(os.listdir(os.path.join(self._RESULTS_DIR, self.config["result_dir"],
                                                             self._SUPERCELL_OUTPUT_DIR)),
                                     desc="Collecting disordered structures",
                                     unit=" composition",
                                     ncols=100):
            with zipfile.ZipFile(os.path.join(self._RESULTS_DIR, self.config["result_dir"],
                                              self._SUPERCELL_OUTPUT_DIR, archive_filename), "r") as archive:
                for structure_filename in archive.namelist():
                    with archive.open(structure_filename, "r") as file:
                        file_data = file.read().decode("utf-8")
                        structure = CifParser.from_str(file_data).get_structures(primitive=False)[0]
                        finder = SpacegroupAnalyzer(structure)
                        specie_counter = Counter(map(str, structure.species))
                        data_dict["cif_data"].append(file_data)
                        data_dict["structure_filename"].append(structure_filename.replace(".zip", ""))
                        data_dict["composition"].append(str(structure.composition))
                        data_dict["space_group_no"].append(int(finder.get_space_group_number()))
                        data_dict["space_group_symbol"].append(finder.get_space_group_symbol())
                        data_dict["weight"].append(int(re.search(r"_w(.*?).cif", structure_filename).group(1)))
                        for specie in substitute_with_species:
                            data_dict[f"{specie}_concentration"].append(specie_counter[specie] / len(structure))

        data_df = pd.DataFrame.from_dict(data_dict)
        data_df.reset_index(inplace=True, drop=True)
        data_df.to_pickle(os.path.join(self._RESULTS_DIR, self.config["result_dir"], "data.pkl.gz"))

    @staticmethod
    def _spaceGroupConventional(sg: str) -> str:
        """
        Convert a space group symbol to a conventional form.
        :param sg: Space group symbol.
        :return: Formatted space group symbol.
        """

        sg = re.sub("-[\d]", lambda x: "\\bar{" + x.group()[1:] + "}", sg)
        return f"${sg}$"

    def plot_group_subgroup_graph(self, node_size: int = 1200) -> None:
        """
        Plot the group-subgroup graph.
        :return: None.
        """

        css_df = pd.read_pickle(os.path.join(self._RESULTS_DIR, self.config["result_dir"], "data.pkl.gz"))
        with open("venv/Lib/site-packages/pymatgen/symmetry/symm_data.json", "r") as f:
            symm_data = json.load(f)
        symm_data_subg = symm_data["maximal_subgroups"]
        symm_data_abbr = {v: k for k, v in symm_data["abbreviated_spacegroup_symbols"].items()}

        sgs = sorted(css_df["space_group_no"].unique(), reverse=True)
        sg_info = {sg: ((css_df["space_group_no"] == sg).sum(),
                        symm_data_abbr.get(SpaceGroup.from_int_number(sg).symbol,
                                           SpaceGroup.from_int_number(sg).symbol))
                   for sg in sgs}
        sg_info2 = {i[1]: i[0] for i in sg_info.values()}
        label_map_black = {v[1]: self._spaceGroupConventional(v[1]) + f"\n({k})" for k, v in sg_info.items() if
                           np.log10(v[0]) >= np.log10(max(sg_info2.values())) / 3}
        label_map_white = {v[1]: self._spaceGroupConventional(v[1]) + f"\n({k})" for k, v in sg_info.items() if
                           np.log10(v[0]) < np.log10(max(sg_info2.values())) / 3}

        graph = nx.DiGraph()
        for i in range(len(sgs)):
            for j in range(len(sgs)):
                if sgs[j] in symm_data_subg[str(sgs[i])] and i != j:
                    graph.add_edge(sg_info[sgs[i]][1], sg_info[sgs[j]][1])

        not_connected_nodes = set(graph.nodes) - set([i[1] for i in graph.edges])
        for node2 in not_connected_nodes:
            for node1 in graph.nodes:
                if SpaceGroup(node2).is_subgroup(SpaceGroup(node1)):
                    graph.add_edge(node1, node2)
                    break

        nodes = [i for i in graph.nodes]
        orders = np.array([SpaceGroup(nodes[i]).order for i in range(len(nodes))])
        pos_x = [0] * len(nodes)
        unique, counts = np.unique(orders, return_counts=True)
        for count_pos in range(len(counts)):
            for i in range(counts[count_pos]):
                pos_x[np.where(orders == unique[count_pos])[0][i]] = (i + 1) / (counts[count_pos] + 1)
        orders_unique = np.sort(np.unique(orders))
        orders_dict = {orders_unique[i].item(): i for i in range(orders_unique.shape[0])}
        pos = {nodes[i]: (pos_x[i], orders_dict[orders[i]]) for i in range(len(nodes))}

        edges_curved = set()  # It can be happened that some edges are not shown because of
        # overlapping. One can curve them manually to avoid this.
        edges_straight = set(graph.edges) - edges_curved

        fig, ax = plt.subplots(figsize=(15, 10))
        cmap = "viridis"
        nx.draw_networkx_nodes(graph, pos, node_color=[np.log(sg_info2[i]) for i in graph.nodes], node_size=node_size,
                               edgecolors="black", linewidths=1, cmap=cmap, vmin=0,
                               vmax=np.log(max([i for i in sg_info2.values()])), ax=ax)
        nx.draw_networkx_labels(graph, pos, labels=label_map_black, font_size=6, font_color="black")
        nx.draw_networkx_labels(graph, pos, labels=label_map_white, font_size=6, font_color="white")
        nx.draw_networkx_edges(graph, pos, edgelist=edges_straight, edge_color="grey", node_size=node_size, width=1,
                               arrowsize=12, ax=ax)
        nx.draw_networkx_edges(graph, pos, edgelist=edges_curved, edge_color="grey", width=1, node_size=node_size,
                               arrowsize=12, connectionstyle='arc3, rad = -0.1', ax=ax)
        ax.tick_params(left=True, labelleft=True)
        ax.set_ylabel("Space group order", fontsize=12)
        ax.set_yticks(range(orders_unique.shape[0]))
        ax.set_yticklabels(list(map(str, orders_unique)), fontsize=12)
        ax.yaxis.set_tick_params(labelsize=12)

        cax = fig.add_axes([ax.get_position().x1 + 0.03, ax.get_position().y0 - 0.05, 0.02,
                            ax.get_position().y1 - ax.get_position().y0 + 0.1])

        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=0, vmax=np.log10(max([i for i in sg_info2.values()]))))
        sm.set_array([])
        cbar = fig.colorbar(sm, aspect=70, cax=cax)
        cbar.ax.set_ylabel("lg (number of inequivalent structures)", fontsize=12)
        cbar.ax.yaxis.set_tick_params(labelsize=12)
        cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        cbar.outline.set_visible(False)
        plt.tight_layout()
        plt.show()
