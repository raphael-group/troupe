#!/usr/bin/env python3

import json
import sys
from collections import Counter
from pathlib import Path
import copy
import os
import pickle

import networkx as nx
import numpy as np
from ete3 import Tree

BASE_DIR   = Path(__file__).parent
TREE_PATH  = BASE_DIR / "raw_data" / "lineage_tree.nwk"
LABEL_PATH = BASE_DIR / "raw_data" / "tree_node_labels.txt"


###### Lineage ABa settings ######
# This relabeling is based on potency reduction from the original diff map
RELABEL = {
    'ABarpaaa_lineage': 'Glia',
    '9': '8',
    'Parent_of_hyp1V_and_ant_arc_V': '8',
    'hyp1V_and_ant_arc_V': '8',
    'Pharyngeal_muscle': 'Pharyngeal_neuron',
    'Pharyngeal_marginal_cell': 'Pharyngeal_neuron', 
    'Ciliated_amphid_neuron': 'Ciliated_non_amphid_neuron',
    '6': 'Ciliated_non_amphid_neuron',
}

TERMINAL_LABELS = {
    '3',
    'Glia',
    'Ciliated_non_amphid_neuron',
    'Seam_cell',
    'Hypodermis',
    'Arcade_cell',
    'Pharyngeal_neuron'
}

TERMINAL_RELABEL = {
    '3': 'Misc.',
    'Glia': 'Glia',
    'Ciliated_non_amphid_neuron': 'Neuron',
    'Seam_cell': 'Seam',
    'Hypodermis': 'Hypodermis',
    'Arcade_cell': 'Arcade',
    'Pharyngeal_neuron': 'Pharynx'
}
##################################

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


def _subsample_then_remove_progenitors(tree, rng, sample_probability, terminal_labels):
    """Return a pruned copy of ``tree`` after subsampling then progenitor removal."""
    tree = copy.deepcopy(tree)
    leaves = tree.get_leaves()

    sampled = [leaf for leaf in leaves if rng.random() < sample_probability]
    if not sampled:
        return None

    tree.prune(sampled, preserve_branch_length=True)
    kept_after_subsample = tree.get_leaves()
    terminal_after_filter = [
        leaf for leaf in kept_after_subsample if leaf.state in terminal_labels
    ]
    if not terminal_after_filter:
        return None

    tree.prune(terminal_after_filter, preserve_branch_length=True)
    return tree

def preprocess_trees(tree_path, label_path, num_trees, sample_probability):
    """
    Returns a list of pre-processed trees
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

    for node in tree.traverse("preorder"):
        state = name2label.get(node.name)
        node.add_feature("state", TERMINAL_RELABEL[state])
    
    print("Tree leaves before and after sampling:")
    print(len(tree.get_leaves()))
    tree_list = []
    rng = np.random.default_rng()
    for _ in range(num_trees):
        sampled_tree = _subsample_then_remove_progenitors(tree, rng, sample_probability, TERMINAL_LABELS)
        tree_list.append(sampled_tree)
        print("\t", len(sampled_tree.get_leaves()))

    
    return tree_list


def main():
    print("Building differentiation map ...")
    num_trees = 10
    sample_probability = 0.5
    tree_list = preprocess_trees(TREE_PATH, LABEL_PATH, num_trees, sample_probability)

    processed_dir = f"{BASE_DIR}/processed_data/{num_trees}/{sample_probability}"
    os.makedirs(processed_dir, exist_ok=True)
    with open(f"{processed_dir}/trees.pkl", 'wb') as fp:
        pickle.dump(tree_list, fp)


if __name__ == "__main__":
    main()
