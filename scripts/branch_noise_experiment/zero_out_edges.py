import ete3
import pickle
import random

p = 0.05
working_dir = "/Users/william_hs/Desktop/Projects/troupe/experiments/branch_noise_experiment"
for trial in range(10):
    path_to_trees = f"{working_dir}/trees_24/time_1.75/sample_0.05/trial_{trial}/trees.pkl"
    with open(path_to_trees, "rb") as fp:
        tree_list = pickle.load(fp)

    for tree in tree_list:
        # For each edge in the tree, 0 it out with 5% probability
        for node in tree_list:
            is_zeroed = random.random() < p
            if is_zeroed:
                node.dist = 0
    

    path_to_new_trees = f"{working_dir}/trees_24/time_1.75/sample_0.05/trial_{trial}/trees_zeroed_branches.pkl"
    with open(path_to_new_trees, "wb") as fp:
        pickle.dump(tree_list, fp)