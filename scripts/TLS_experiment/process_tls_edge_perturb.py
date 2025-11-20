"""
Script for processesing TLS data located at cell-diff-via-ml/experiments/TLS/raw_data.

Example usage:
    python scripts/TLS_experiment/process_tls_edge_perturb.py
"""

import ete3
from ete3 import TreeNode
import pickle
import os
import csv
from collections import Counter
import sys
from utils import is_ultrametric, binarize_tree
import math

TLSC = [
    "Bar3",
    "Bar6",
    "Bar9",
    "Bar12",
    "Bar14",
    "Bar15",
    "Bar18",
    "Bar20",
    "Bar21",
    "Bar23",
    "Bar24"
]

labels = [
    "Endoderm",
    "Endothelial",
    "NMPs",
    "NeuralTube1",
    "NeuralTube2",
    "PCGLC",
    "Somite",
    "Somite0",
    "Somite-1",
    "SomiteDermo",
    "SomiteSclero",
    "Unknown",
    "aPSM",
    "pPSM"
]
# label2idx = {label: i for i, label in enumerate(labels)}

labels_coarse = [
    "NMPs",
    "NeuralTube",
    "Somite",
    "Endoderm",
    "Endothelial",
    "PCGLC"
]
# coarse_label2idx = {label: i for i, label in enumerate(labels_coarse)}

label2coarse = {
    "Endoderm": "Endoderm",
    "Endothelial": "Endothelial",
    "NMPs": "NMPs",
    "NeuralTube1": "NeuralTube",
    "NeuralTube2": "NeuralTube",
    "PCGLC": "PCGLC",
    "Somite": "Somite",
    "Somite0": "Somite",
    "Somite-1": "Somite",
    "SomiteDermo": "Somite",
    "SomiteSclero": "Somite",
    "Unknown": "Unknown",
    "aPSM": "Somite",
    "pPSM": "Somite"    # TODO: Based on the UMAP, maybe this should be NMP?
}

# Avoids Pickle Error: RecursionError: maximum recursion depth exceeded while pickling an object
sys.setrecursionlimit(5000) 


stem_extensions = [0.0, 0.001, 0.01, 0.1]           # Add to stem length
branch_scalings = [0.5, 1, 2, 4, 8, 16, 32]      # Scale all branches by this factor

var_name = "stem_extension" #"branch_scaling"

for var in stem_extensions:

    tls_type = f"TLSC_{var_name}"
    working_dir = "/n/fs/ragr-research/users/wh8114/projects/cell-diff-via-ml"

    if var_name == "stem_extension":
        out_dir = f"{working_dir}/experiments/{tls_type}/stem={var}/processed_data"
    elif var_name == "branch_scaling":
        out_dir = f"{working_dir}/experiments/{tls_type}/branches={var}/processed_data"
    else:
        assert False

    os.makedirs(out_dir, exist_ok=True)

    in_dir = f"{working_dir}/experiments/TLS/raw_data"
    tree_dir = f"{in_dir}/best_trees"
    leaf_label_dir = f"{in_dir}/cell_types"
    barcodes_file = f"{in_dir}/multiseq_barcodes.tsv"


    # Load in the tls ids
    id2tls = {}
    with open(barcodes_file, "r") as fp:
        rd = csv.reader(fp, delimiter="\t")
        for row in rd:
            if rd.line_num == 1:
                continue
            id2tls[row[0]] = row[1]

    # Load trees for each tls id
    tree_list = []
    for entry in os.listdir(tree_dir):
        if "collapsed.nwk" in entry:    # Only use binary trees
            continue
        if "AM-DNA" in entry:           # Skip big multifurcating trees
            continue
        id = entry.split('.')[0]

        if id not in TLSC:              # Use un-perturbed data
            continue

        with open(f"{tree_dir}/{entry}", "r") as fp:
            nwk = fp.readline()
            if nwk.startswith('[&R]'):
                nwk = nwk[4:]

        leaf2label = {}
        with open(f"{leaf_label_dir}/{id}_metadata.txt", "r") as fp:
            rd = csv.reader(fp, delimiter="\t")
            for row in rd:
                if rd.line_num == 1:
                    continue
                leaf2label[row[0]] = row[3][2:-2]   # Removing '[...]'

        tree = ete3.Tree(nwk, format=3)

        # Check that trees are ultrametric to start
        assert is_ultrametric(tree), "Tree is not ultrametric to start"

        # NOTE: Stem length = 0 here

        known_leaves = []
        leaves = tree.get_leaves()
        for leaf in leaves:
            cell_label = leaf2label[leaf.name]
            leaf.add_feature("state_fine", cell_label)
            # Apply coarse labeling
            if cell_label == "Unknown":
                continue
            known_leaves.append(leaf)
            cell_label_coarse = label2coarse[cell_label]
            leaf.add_feature("state_coarse", cell_label_coarse)
            leaf.add_feature("state", cell_label_coarse)
            

        print(f"removing {len(leaves) - len(known_leaves)} / {len(leaves)} leaves")

        tree.prune(known_leaves, preserve_branch_length=True)   # NOTE: Removes dummy root node
        binarized_tree = binarize_tree(tree)
        tree_list.append(binarized_tree)

        assert len(binarized_tree.children) >= 2
        if var_name == "stem_extension":
            binarized_tree.dist = binarized_tree.dist + var
        elif var_name == "branch_scaling":
            for node in binarized_tree.traverse():
                node.dist = node.dist * var

        # NOTE: Stem length > 0 here
        assert all([len(n.children) <= 2 for n in binarized_tree.traverse()])
        assert is_ultrametric(binarized_tree), "Processed tree is not ultrametric"

    for tree in tree_list:
        print(len(tree.get_leaves()), "leaves")
        leaf_types = set()
        for leaf in tree.get_leaves():
            leaf_types.add(leaf.state)
        print("\t leaf types:   ", len(leaf_types), leaf_types)

        child_counts = Counter()
        for node in tree.traverse():
            if not node.is_leaf():
                child_counts[len(node.children)] += 1
        print("\t child counts: ", child_counts)
        print("\t root len:     ", tree.dist)

    # TODO: Should we also be scaling the branch lengths -> x120 to make them in hours
    with open(f"{out_dir}/trees.pkl", "wb") as fp:
        pickle.dump(tree_list, fp)



