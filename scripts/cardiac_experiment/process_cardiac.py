"""
Script for processesing cardiac data

Example usage:
    python scripts/cardiac_experiment/process_cardiac.py
"""

import ete3
import pickle
import os
import csv
from collections import Counter
import sys

# Avoids Pickle Error: RecursionError: maximum recursion depth exceeded while pickling an object
sys.setrecursionlimit(5000) 


working_dir = "/Users/william_hs/Desktop/Projects/troupe"
data_dir = "/Users/william_hs/Desktop/Projects/datasets/cardiac"

out_dir = f"{working_dir}/experiments/cardiac/processed_data"
in_dir = data_dir
tree_dir = f"{in_dir}/LAML_trees"
leaf_label_dir = f"{in_dir}/tree_meta"

cardiac = [
    'AZ114_1',
    'AZ132_5',
    'AZ92_2_MK3',
    'AZ114_2',
    'AZ132_2_MK1-9',
    'AZ132_2_MK1-6',
    'AZ132_6',
    'AZ92_3',
    'AZ132_3',
    'AZ92_1',
    'AZ92_4_MK1',
    'AZ132_1_MK1-9',
    'AZ132_1_MK1-6',
    'AZ132_4',
    'AZ92_2_MK1',
    'AZ92_4_MK3',
]

# label2coarse = {
#     'PGC': 'Epiblast',
#     'Epiblast': 'Epiblast',
#     'PS': 'PS',
#     'Node&Notochordal_plate': 'Node&Notochordal_plate',
#     'NMPs': 'NMPs',
#     'Neural1': 'Neural',
#     'Neural2': 'Neural',
#     'Floor&Basal_plate_26': 'low_quality',
#     'Floor&Basal_plate_30': 'low_quality',
#     'Hindgut': 'Endo',
#     'Foregut': 'Endo',
#     'PSM': 'Somites_sclerotome', #'PSM',
#     'Somites': 'Somites_sclerotome', # 'Somites',
#     'Somites_sclerotome': 'Somites_sclerotome',
#     'Meso1': 'Anterior_meso', #'Meso1',
#     'Meso4': 'Anterior_meso', #'Meso4',
#     'Meso5': 'Anterior_meso', #'Meso5',
#     'Endothelial': 'Endothelial',
#     'Meso2': 'Meso2',
#     'Meso3': 'Anterior_meso',
#     'SHF': 'Anterior_meso',
#     'Early_cardiomyocytes': 'Anterior_meso',
#     'Cardiomyocytes': 'Anterior_meso',
#     'low_quality': 'low_quality'
# }

label2coarse = {
    'PGC': 'Epiblast',
    'Epiblast': 'Epiblast',
    'PS': 'PS',
    'Node&Notochordal_plate': 'Node&Notochordal_plate',
    'NMPs': 'NMPs',
    'Neural1': 'Neural',
    'Neural2': 'Neural',
    'Floor&Basal_plate_26': 'low_quality',
    'Floor&Basal_plate_30': 'low_quality',
    'Hindgut': 'Endo',
    'Foregut': 'Endo',
    'PSM': 'PSM',
    'Somites': 'Somites',
    'Somites_sclerotome': 'Somites_sclerotome',
    'Meso1': 'Meso1',
    'Meso4': 'Anterior_meso', #'Meso4',
    'Meso5': 'Anterior_meso', #'Meso5',
    'Endothelial': 'Endothelial',
    'Meso2': 'Anterior_meso', # 'Meso2',
    'Meso3': 'Anterior_meso',
    'SHF': 'Anterior_meso',
    'Early_cardiomyocytes': 'Anterior_meso',
    'Cardiomyocytes': 'Anterior_meso',
    'low_quality': 'low_quality'
}

potencies = {
    'Anterior_meso': ['Anterior_meso'],
    'Endo': ['Endo'],
    'Endothelial': ['Endothelial'],
    'Epiblast': ['Epiblast'],
    'Meso1': ['Anterior_meso', 'Somites_sclerotome'], # ['Meso2', 'Meso4', 'Somites_sclerotome'],
    'Meso2': ['Anterior_meso'],
    'Meso4': ['Anterior_meso'],
    'Meso5': ['Anterior_meso'],
    'NMPs': ['Neural', 'Somites_sclerotome'],
    'Neural': ['Neural'],
    'Node&Notochordal_plate': ['Node&Notochordal_plate'],
    'PSM': ['PSM'],
    'PS': ['PS'],
    'Somites': ['Somites_sclerotome'],
    'Somites_sclerotome': ['Somites_sclerotome'],
    'low_quality': ['low_quality']
}

batches_to_include = [
    'AZ132'
]

times_to_include = [96] #, 120, 144, 168]

types_to_remove = [
    'low_quality'
]


def main():
    # Load trees
    tree_list = []
    tree_times = [] # in days
    observed_types = set()
    for entry in os.listdir(tree_dir):
        name_arr = entry.split('_')
        batch = name_arr[0]
        num = name_arr[1]

        if batch not in batches_to_include:
            continue

        with open(f"{tree_dir}/{entry}", "r") as fp:
            nwk = fp.readline()
            if nwk.startswith('[&R]'):
                nwk = nwk[4:]

        leaf2label = {}
        tree_time = None
        with open(f"{leaf_label_dir}/{batch}_{num}_cell_meta.txt", "r") as fp:
            rd = csv.reader(fp, delimiter="\t")
            for row in rd:
                if rd.line_num == 1:
                    continue
                tree_time_hours = float(row[3][:-1]) # append the num days
                tree_time = tree_time_hours / 24
                leaf2label[row[0]] = row[4]
        
        if tree_time_hours not in times_to_include:
            continue

        tree = ete3.Tree(nwk, format=3)

        known_leaves = []
        leaves = tree.get_leaves()
        for leaf in leaves:
            cell_label = leaf2label[leaf.name]
            cell_label_coarse = label2coarse[cell_label]
            leaf.add_feature("state", cell_label_coarse)
            
            # Apply coarse labeling
            if cell_label in types_to_remove:
                continue
            observed_types.add(leaf.state)
            known_leaves.append(leaf)
           
        print(f"Removing {len(leaves) - len(known_leaves)} / {len(leaves)} leaves")

        crown_dist_pre_prune = tree.get_children()[0].dist
        tree.prune(known_leaves, preserve_branch_length=True)
        crown_dist_post_prune = tree.dist

        assert crown_dist_post_prune == crown_dist_pre_prune

        root_children = tree.get_children() #[0].get_children()
        print(f"Tree {batch}-{num} has {len(root_children)}-way multifurcation at root")
        multifurcation_distirbution = Counter()
        for node in tree.traverse():
            if not node.is_leaf() and not node.is_root():
                multifurcation_distirbution[len(node.get_children())] += 1
        print("Multifurcation distribution:", multifurcation_distirbution)

        print(f"Adding {len(root_children)} trees...")
        print()
        for child in root_children:
            child.up = None
            binarized_tree = binarize_tree(child)
            tree_list.append(binarized_tree)
            tree_times.append(tree_time)

    leaf_count_distribution = Counter()
    total_leaves = 0
    for i, tree, in enumerate(tree_list):
        time = tree_times[i]
        for node in tree.traverse():
            node.dist = node.dist / 10 * time
        leaves = tree.get_leaves()
        # print(len(leaves), "leaves")
        leaf_count_distribution[len(leaves)] += 1
        total_leaves += len(leaves)
        leaf_types = set()
        for leaf in leaves:
            leaf_types.add(leaf.state)
        # print("\t leaf types:   ", len(leaf_types), leaf_types)

        child_counts = Counter()
        
        for node in tree.traverse():
            if not node.is_leaf():
                child_counts[len(node.children)] += 1
        # print("\t child counts: ", child_counts)
        # print("\t root len:     ", tree.dist)
        leaf_dists = [tree.get_distance(leaf) for leaf in leaves]
        # print(f"\t leaves are in {min(leaf_dists):1g} - {max(leaf_dists):1g}")

    print("Total leaves: ", total_leaves)
    print("Num trees:    ", len(tree_list))

    with open(f"{out_dir}/trees.pkl", "wb") as fp:
        pickle.dump(tree_list, fp)

    leaf_counts = list(leaf_count_distribution.keys())
    leaf_counts.sort()
    for count in leaf_counts:
        print(count, "\t", leaf_count_distribution[count])

    # TODO: Produce terminal potency file
    with open(f"{out_dir}/observed_potencies.txt", "w") as fp:
        lines = []
        for type in observed_types:
            text = f"{type}\t"
            text += ",".join(potencies[type])
            text += '\n'
            lines.append(text)
        fp.writelines(lines)

    with open(f"{out_dir}/terminal_labels.txt", "w") as fp:
        lines = []
        for type in observed_types:
            print(type, "vs", potencies[type])
            if len(potencies[type]) == 1:
                text = f"{type}\n"
                lines.append(text)
        fp.writelines(lines)


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

