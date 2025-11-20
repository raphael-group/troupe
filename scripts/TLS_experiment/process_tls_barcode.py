"""
Script for processesing TLS data located at cell-diff-via-ml/experiments/TLS/raw_data.

Example usage:
    python scripts/TLS_experiment/process_tls_barcode.py
"""

import ete3
import pickle
import os
import csv
from collections import Counter
import sys
import math
from utils import is_ultrametric, binarize_tree

# Avoids Pickle Error: RecursionError: maximum recursion depth exceeded while pickling an object
sys.setrecursionlimit(5000) 

tls_type = "TLSC_barcodes"
stem_epsilon = 0.0       # Add to stem length
branch_scaling = 1.0      # Scale all branches by this factor


working_dir = "/n/fs/ragr-research/users/wh8114/projects/cell-diff-via-ml"

if stem_epsilon == 0 and branch_scaling == 1.0:
    out_dir = f"{working_dir}/experiments/{tls_type}/processed_data"
elif stem_epsilon != 0:
    out_dir = f"{working_dir}/experiments/{tls_type}/stem={stem_epsilon}/processed_data"
else:
    out_dir = f"{working_dir}/experiments/{tls_type}/branches={branch_scaling}/processed_data"

os.makedirs(out_dir, exist_ok=True)


in_dir = f"{working_dir}/experiments/{tls_type}/raw_data"
tree_dir = f"{in_dir}"
leaf_label_dir = f"{in_dir}"

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

labels_coarse = [
    "NMPs",
    "NeuralTube",
    "Somite",
    "Endoderm",
    "Endothelial",
    "PCGLC"
]

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
    # Load trees for each tls id
    tree_list = []
    for entry in os.listdir(tree_dir):
        # if "collapsed.nwk" in entry:    # Only use binary trees
        #     continue
        if "AM-DNA" in entry:           # Skip big multifurcating trees
            continue
        if "metadata" in entry:
            continue
        id = entry.split('.')[0]

        with open(f"{tree_dir}/{entry}", "r") as fp:
            nwk = fp.readline()
            if nwk.startswith('[&R]'):
                nwk = nwk[4:]

        leaf2label = {}
        with open(f"{leaf_label_dir}/{id[:-6]}_metadata.txt", "r") as fp:
            rd = csv.reader(fp, delimiter="\t")
            for row in rd:
                if rd.line_num == 1:
                    continue
                leaf2label[row[0]] = row[3][2:-2]   # Removing '[...]'

        tree = ete3.Tree(nwk, format=3)
        # Check that trees are ultrametric to start
        assert is_ultrametric(tree), "Tree is not ultrametric to start"

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

        # Optional: lengthen stem
        assert len(tree.children) == 1
        child = tree.children[0]
        child.dist = child.dist + stem_epsilon

        # Optional: scale branch lengths
        if branch_scaling != 1.0:
            for node in tree.traverse():
                node.dist = node.dist * branch_scaling
            

        print(f"removing {len(leaves) - len(known_leaves)} / {len(leaves)} leaves")

        tree.prune(known_leaves, preserve_branch_length=True)
        binarized_tree = binarize_tree(tree)
        tree_list.append(binarized_tree)

        assert is_ultrametric(binarized_tree), "Processed tree is not ultrametric"

    # Get tree stats
    total_leaves = 0
    for tree in tree_list:
        num_leaves = len(tree.get_leaves())
        print(num_leaves, "leaves")
        total_leaves += num_leaves

        leaf_types = set()
        for leaf in tree.get_leaves():
            leaf_types.add(leaf.state)
        print("\t leaf types:   ", len(leaf_types), leaf_types)

        child_counts = Counter()

        print(tree.get_ascii(attributes=["dist"]))
        assert len(tree.children) == 1
        
        for node in tree.traverse():
            if not node.is_leaf():
                child_counts[len(node.children)] += 1
        print("\t child counts: ", child_counts)
        print("\t root len:     ", tree.dist)

    print(f"{len(tree_list)} trees w/ an average of {total_leaves / len(tree_list)} leaves per tree")

    with open(f"{out_dir}/trees.pkl", "wb") as fp:
        pickle.dump(tree_list, fp)





if __name__ == "__main__":
    main()

