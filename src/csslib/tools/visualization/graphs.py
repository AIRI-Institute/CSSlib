"""
    Module for visualizing the obtained results. Contains functions for plotting the following graphs:
    - group to subgroup reduction with color mar
"""


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import re
from csslib.exceptions import VisualizationError
from csslib.tools.dataloader import DataLoader
from matplotlib.ticker import MaxNLocator
from pymatgen.symmetry.groups import SpaceGroup, SYMM_DATA


def _spaceGroupConventional(sg: str) -> str:
    """
        Converts a space group symbol to a conventional form.
        
        Args:
            sg (str): space group symbol.
    
        Return: 
            str: formatted space group symbol.
    """
    sg = re.sub(r"-\d", lambda x: "\\bar{" + x.group()[1:] + "}", sg)
    return f"${sg}$"


def group_subgroup_diagram(*, css_df: pd.DataFrame | None = None, results_path: str | None = None, 
                           num_workers: int = 1, figsize: tuple[int]=(15, 10), node_size: int = 1200, cmap: str = 'viridis', 
                           save_plot: bool = False, fig_path: str = '.', transparent: bool = False, 
                           dpi: int = 300, bbox_inches: str | None = None, pad_inches: float = 0.1) -> None:
    """
        Plots the group-subgroup diagram. Requires dataframe with counted number of inequivalent structures for the
        specified spacegroup number and substitution rates or path to the results directory for aitoparsing of full css info.
        If both css_df and results_path are empty raises csslib.exceptions.VisualizationError.

        Args: 
            css_df (pandas.Dataframe | None, optional): dataframe with information about spacegroup distribution. 
            Defaults to None.
            results_path (str | None, optional): path to the results folder with .pkl.gz files. Autonatically parses
            using csslib.tools.dataloader.DataLoader class using predefined transformation function. Defaults to None.
            num_workers (int, optional): number of workers for parallel processing using csslib.tools.dataloader.Dataloader
            class. Defaults to 1.
            figsize (tuple[int], optional): size of the output figure. Defaults to (15, 10).
            node_size (int, optional): size of nodes. Defaults to 1200.
            cmap (str, optional): defines the colormap for the figure. Defaults to 'viridis'.
            save_plot (bool, optional): if True, saves figure in the input directory. Defaults to False.
            fig_path (str, optional): path to place + filename, where figure will be saved. Defaults to '.'.
            transperent (bool, optional): if True, the figure background will be transperent. Defaults to False.
            dpi (int, optional): the figure dpi. Defaults to 300.
            bbox_inches (str | None, optional): bounding box in inches. Defaults to None.
            pad_inches (float, optional): amount of padding in inches around the figure when bbox_inches is 'tight'. Defaults to 0.1.
        
        Raise:

    """
    if css_df is None and results_path is None:
        raise VisualizationError('One of the following parameters must be defined:\n- css_df\n- results_path')
    if css_df is not None and results_path is not None:
        raise VisualizationError('Both parameters css_df and results_path are setup. Please leave one parameter as None.')
    
    if results_path is not None:
        dataloader = DataLoader(results_path, num_workers=num_workers, transformation_func=lambda df: 
                                df.groupby(['space_group_no', 'space_group_symbol', *[col for col in df.columns if col.endswith('concentration')]]).agg(cfgs_count=('structure_filename', 'count'), 
                                                                                                                                                        all_cfgs_count=('weight', 'sum')).reset_index())
        dataloader()
        css_df = dataloader.get_structures_df()
        css_df = css_df.sort_values(['space_group_no', 'space_group_symbol', *[col for col in css_df.columns if col.endswith('concentration')]]).reset_index()

    symm_data = SYMM_DATA
    symm_data_subg = symm_data["maximal_subgroups"]
    symm_data_abbr = {v: k for k, v in symm_data["abbreviated_spacegroup_symbols"].items()}

    sgs = sorted(css_df["space_group_no"].unique(), reverse=True)
    sg_info = {sg: (css_df.loc[(css_df["space_group_no"] == sg)]['cfgs_count'].sum(),
                    symm_data_abbr.get(SpaceGroup.from_int_number(sg).symbol,
                                       SpaceGroup.from_int_number(sg).symbol))
               for sg in sgs}
    sg_info2 = {i[1]: i[0] for i in sg_info.values()}
    label_map_black = {v[1]: _spaceGroupConventional(v[1]) + f"\n({k})" for k, v in sg_info.items() if
                       np.log10(v[0]) >= np.log10(max(sg_info2.values())) / 3}
    label_map_white = {v[1]: _spaceGroupConventional(v[1]) + f"\n({k})" for k, v in sg_info.items() if
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

    fig, ax = plt.subplots(figsize=figsize)
    cmap = cmap
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
    if save_plot:
        plt.savefig(fig_path, transparent, dpi, bbox_inches, pad_inches)
    else:
        plt.show()
