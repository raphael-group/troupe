"""
Script for processesing TLS data located at cell-diff-via-ml/experiments/TLS/raw_data.

Example usage:
    python scripts/TLS_experiment/process_tls.py
"""

import ete3
import pickle
import os
import csv
from collections import Counter
import sys

# Avoids Pickle Error: RecursionError: maximum recursion depth exceeded while pickling an object
sys.setrecursionlimit(5000) 


working_dir = "/n/fs/ragr-research/users/wh8114/projects/cell-diff-via-ml"

out_dir = f"{working_dir}/experiments/TLS/processed_data"
in_dir = f"{working_dir}/experiments/TLS/raw_data"
tree_dir = f"{in_dir}/best_trees"
leaf_label_dir = f"{in_dir}/cell_types"
barcodes_file = f"{in_dir}/multiseq_barcodes.tsv"

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


def main():
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

        if id in TLSC:              # Use un-perturbed data
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

        tree.prune(known_leaves, preserve_branch_length=True)
        binarized_tree = binarize_tree(tree)
        tree_list.append(binarized_tree)

    total_leaves = 0

    for tree in tree_list:
        leaves = tree.get_leaves()
        print(len(leaves), "leaves")
        total_leaves += len(leaves)
        leaf_types = set()
        for leaf in leaves:
            leaf_types.add(leaf.state)
        print("\t leaf types:   ", len(leaf_types), leaf_types)

        child_counts = Counter()
        
        for node in tree.traverse():
            if not node.is_leaf():
                child_counts[len(node.children)] += 1
        print("\t child counts: ", child_counts)
        print("\t root len:     ", tree.dist)

    # TODO: Should we also be scaling the branch lengths -> x120 to make them in hours
    print("Total leaves: ", total_leaves)
    print("Num trees:    ", len(tree_list))

    with open(f"{out_dir}/trees.pkl", "wb") as fp:
        pickle.dump(tree_list, fp)

def binarize_tree(t):
    """
    For every node with >2 children, iteratively group the last two children
    under a new internal node whose edge length from the parent is 0.
    Original child branch lengths are preserved.
    """
    # Post-order so we fix children before their parents.
    for node in t.traverse("postorder"):
        # Keep collapsing until binary
        while len(node.get_children()) > 2:
            # take two children
            a = node.get_children()[-1]
            b = node.get_children()[-2]
            # detach them from 'node'
            a.detach()
            b.detach()
            # create new internal node with 0-length edge from 'node'
            m = ete3.TreeNode(dist=0.0)
            node.add_child(m)
            # reattach a and b under the new node; their .dist are preserved
            m.add_child(a)
            m.add_child(b)
    return t




if __name__ == "__main__":
    main()

