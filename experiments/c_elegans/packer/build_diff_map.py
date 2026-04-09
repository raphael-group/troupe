#!/usr/bin/env python3
"""
build_diff_map.py

Build a weighted differentiation map from the C. elegans Packer lineage tree.

For each labeled node in the tree, we find its nearest labeled ancestor and
record the transition (ancestor_label -> node_label).  Edge weights equal the
number of times each transition is observed.  Self-transitions are excluded.

The script then:
  1. Saves the adjacency matrix and idx2state mapping.
  2. Converts the graph to a DAG by greedily removing the minimum-weight edge
     from each remaining cycle.
  3. Plots the DAG.

Outputs (written next to this script):
  diff_map.npy       (N, N) adjacency matrix
  idx2state.json     {str(index): label} mapping
  diff_map_dag.pdf   plot of the DAG
"""

import json
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
from ete3 import Tree

BASE_DIR   = Path(__file__).parent
SCRIPTS_DIR = BASE_DIR.parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
from eval_utils import draw_weighted_graph

TREE_PATH  = BASE_DIR / "raw_data" / "lineage_tree.nwk"
LABEL_PATH = BASE_DIR / "raw_data" / "tree_node_labels.txt"


###### Lineage ABa settings ######
# This relabeling is based on potency reduction from the original diff map
RELABEL = {
    # 'Parent_of_hyp1V_and_ant_arc_V': 'hyp1V_and_ant_arc_V', 
    'ABarpaaa_lineage': 'Glia',
    '9': '8',
    'Parent_of_hyp1V_and_ant_arc_V': '8',
    'hyp1V_and_ant_arc_V': '8',
    'Pharyngeal_muscle': 'Pharyngeal_neuron',
    'Pharyngeal_marginal_cell': 'Pharyngeal_neuron', 
    'Ciliated_amphid_neuron': 'Ciliated_non_amphid_neuron',
    '6': 'Ciliated_non_amphid_neuron',
}
# These are terminal types that don't appear very much
REMOVE = {
    '13', '15', 'XXX', 'T'
}

EDGES_TO_REMOVE = {
    ('Ciliated_non_amphid_neuron', '3')
}
##################################


# Root type: the label that starts the whole differentiation process.
ROOT_LABEL = "0"

NODE_COLORS = {
    # Progenitor / numbered types – grays
    "0": "#555555", "1": "#6e6e6e", "2": "#7d7d7d", "3": "#8d8d8d",
    "4": "#666666", "6": "#9e9e9e", "7": "#777777", "8": "#acacac",
    "9": "#c0c0c0",
    # Named progenitors
    "Ciliated_amphid_neuron":     "#b0b0d0",
    "Ciliated_non_amphid_neuron": "#c0c0e0",
    "Hypodermis":                 "#a0c8c0",
    "Pharyngeal_muscle":          "#f0c0b0",
    "Pharyngeal_marginal_cell":   "#f0d0c0",
    "ABarpaaa_lineage":           "#d0d0d0",
    "Excretory_cell_parent":      "#b8d8b0",
    "Parent_of_exc_gland_AVK":    "#c0e0b8",
    "Parent_of_hyp1V_and_ant_arc_V": "#b0d8d0",
    # Terminal types
    "Body_wall_muscle":           "#f28e2b",
    "Coelomocyte":                "#fbbf72",
    "GLR":                        "#f5a623",
    "M_cell":                     "#fcd08a",
    "Z1_Z4":                      "#e8a060",
    "hmc":                        "#f0b870",
    "hmc_homolog":                "#f0c080",
    "Glia":                       "#4e79a7",
    "5":                          "#6a95bc",
    "11":                         "#7faecf",
    "13":                         "#94c5e0",
    "Pharyngeal_neuron":          "#e15759",
    "Pharyngeal_gland":           "#e87070",
    "Pharyngeal_intestinal_valve":"#ef9090",
    "Arcade_cell":                "#f5a8a8",
    "Seam_cell":                  "#76b7b2",
    "G2_and_W_blasts":            "#95ccc8",
    "hyp1V_and_ant_arc_V":        "#b0ddd8",
    "15":                         "#c5eeea",
    "Intestine":                  "#9c755f",
    "Germline":                   "#bab0ac",
    "Rectal_cell":                "#cfc8c4",
    "Rectal_gland":               "#ddd8d4",
    "Excretory_cell":             "#59a14f",
    "Excretory_duct_and_pore":    "#78bb6e",
    "Excretory_gland":            "#96cc8e",
    "10":                         "#b2daaa",
}

lineage = "ABa"
lineage_text = f"{lineage}_lineage_"
if lineage == "":
    lineage_text = ""

def get_lineage(tree, lineage_name):
    if lineage_name is None:
        return tree
    for node in tree.traverse():
        if node.name == lineage_name:
            tree = node
            tree.up = None
            return tree
    print(f"\t !!!FAILED TO FIND LINEAGE: {lineage_name}!!!")
    return None



# ---------------------------------------------------------------------------
# Step 1: build the raw differentiation map
# ---------------------------------------------------------------------------

def build_diff_map(tree_path: Path, label_path: Path) -> tuple[np.ndarray, dict]:
    """
    Returns
    -------
    adj : (N, N) float64 ndarray
        adj[i, j] = number of times a node of type i is the nearest labeled
        ancestor of a node of type j.  Diagonal is zero (self-transitions
        are excluded).
    idx2state : dict[int, str]
        Maps matrix row/column index to the state label string.
        The root label (ROOT_LABEL) is always index 0.
    """
    tree = Tree(str(tree_path), format=3)
    tree = get_lineage(tree, lineage)

    name2label: dict[str, str] = {}
    with open(label_path) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[1]:
                if parts[1] in RELABEL:
                    parts[1] = RELABEL[parts[1]]
                name2label[parts[0]] = parts[1]

    label2leafcount = Counter()
    for node in tree.traverse():
        label = name2label.get(node.name)
        node.add_feature("label", label)
        if label is not None and node.is_leaf():
            label2leafcount[label] += 1

    to_keep = []
    for node in tree.traverse():
        if node.label not in REMOVE:
            to_keep.append(node)
    
    print(f"Removed {len(list(tree.traverse())) - len(to_keep)} nodes")
    tree.prune(to_keep, preserve_branch_length=True)
    
    label_counts = list(label2leafcount.items())
    label_counts.sort(key = lambda el: el[1])
    print("Label counts at the leaves:")
    for label, count in label_counts:
        print(f"\t {count}\t {label}")
    print()
    

    # Count labeled-ancestor transitions
    edge_counter: Counter = Counter()
    for node in tree.traverse():
        if node.label is None or node.is_root() or node.up.label is None:
            continue
        anc = node.up
        edge_counter[(anc.label, node.label)] += 1

    # Build ordered state list: root first, then alphabetical
    all_labels = sorted(
        {lbl for pair in edge_counter for lbl in pair},
        key=lambda s: (s != ROOT_LABEL, s),
    )

    idx2state = {i: lbl for i, lbl in enumerate(all_labels)}
    state2idx = {lbl: i for i, lbl in idx2state.items()}

    N = len(all_labels)
    adj = np.zeros((N, N), dtype=np.float64)
    for (src, tgt), w in edge_counter.items():
        if (src, tgt) not in EDGES_TO_REMOVE:
            adj[state2idx[src], state2idx[tgt]] = float(w)

    return adj, idx2state


# ---------------------------------------------------------------------------
# Step 2: remove minimum-weight edges to produce a DAG
# ---------------------------------------------------------------------------

def make_dag(adj: np.ndarray, idx2state: dict) -> np.ndarray:
    """
    Greedily remove the minimum-weight edge from each cycle until the graph
    is a DAG.  Returns a modified copy of adj.

    At each iteration we find one simple cycle (via NetworkX) and remove the
    edge in that cycle with the lowest weight.  Ties broken by preferring
    edges whose reverse direction has higher weight (i.e. the clearer
    back-edge).
    """
    dag = adj.copy()
    # Set self-edges to be 0
    for i in range(len(dag)):
        dag[i, i] = 0
    state2idx = {v: k for k, v in idx2state.items()}

    def _to_nx(matrix):
        G = nx.DiGraph()
        N = matrix.shape[0]
        G.add_nodes_from(range(N))
        for i in range(N):
            for j in range(N):
                if matrix[i, j] > 0:
                    G.add_edge(i, j, weight=matrix[i, j])
        return G

    G = _to_nx(dag)
    removed = []

    while not nx.is_directed_acyclic_graph(G):
        try:
            cycle = nx.find_cycle(G)          # list of (u, v) edges
        except nx.NetworkXNoCycle:
            break

        # Pick the minimum-weight edge in this cycle; break ties by choosing
        # the edge whose reverse has higher weight (clearer back-edge).
        def edge_sort_key(e):
            u, v = e
            w_fwd = dag[u, v]
            w_rev = dag[v, u]
            return (w_fwd, -w_rev)

        u, v = min(cycle, key=edge_sort_key)
        src, tgt = idx2state[u], idx2state[v]
        removed.append((src, tgt, dag[u, v]))
        dag[u, v] = 0.0
        G.remove_edge(u, v)

    print(f"Removed {len(removed)} edge(s) to produce a DAG:")
    total = adj.sum()
    removed_weight = sum(w for _, _, w in removed)
    if removed_weight > 0:
        for src, tgt, w in removed:
            rev = adj[state2idx[tgt], state2idx[src]]
            print(f"  {src} -> ({w:.0f})-> {tgt}  [reverse weight: {rev:.0f}]")
        print(f"Weight removed: {removed_weight:.0f} / {total:.0f} "
            f"({removed_weight / total * 100:.1f}%)")

    return dag


# ---------------------------------------------------------------------------
# Step 3: plot
# ---------------------------------------------------------------------------

def plot_dag(dag: np.ndarray, idx2state: dict, outpath: Path, self_edges: np.ndarray = None) -> None:
    root_idx = next(i for i, s in idx2state.items() if s == ROOT_LABEL)
    node_labels = {i: s for i, s in idx2state.items()}
    draw_weighted_graph(
        dag,
        str(outpath),
        threshold=0,
        node_labels=node_labels,
        node_colors=NODE_COLORS,
        totipotent_state=root_idx,
        self_edges=self_edges
    )
    print(f"Plot saved to {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Building differentiation map ...")
    adj, idx2state = build_diff_map(TREE_PATH, LABEL_PATH)
    self_edges = np.diag(adj)

    N = adj.shape[0]
    n_edges = int((adj > 0).sum())
    print(f"  States: {N},  Edges: {n_edges},  Total weight: {adj.sum():.0f}")

    np.save(BASE_DIR / "diff_map.npy", adj)
    with open(BASE_DIR / "idx2state.json", "w") as fh:
        json.dump({str(k): v for k, v in idx2state.items()}, fh, indent=2)
    print(f"  Saved diff_map.npy and idx2state.json")

    print("\nConverting to DAG ...")
    dag = make_dag(adj, idx2state)

    print("\nPlotting DAG ...")
    plot_dag(dag, idx2state, BASE_DIR / f"figures/{lineage_text}diff_map_dag.pdf", self_edges)


if __name__ == "__main__":
    main()
