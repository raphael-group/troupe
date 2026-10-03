
import pickle
import ete3
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os

tls_type = "TLSC_new"
# tls_type = "TLS_no_endothelial_new" 
working_dir = "/Users/william_hs/Desktop/Projects/troupe"
tree_path = f"{working_dir}/experiments/{tls_type}/processed_data/trees.pkl"

with open(tree_path, "rb") as fp:
    tree_list = pickle.load(fp)

for tree in tree_list:
    branch_lens = [node.dist for node in tree.traverse()]
    avg_len = sum(branch_lens) / len(branch_lens)

    print(f"Avg:   {avg_len:4f}")

    text = f"\tcrown: {tree.dist:4f}"
    children = tree.get_children()
    if len(children) > 0:
        text += f"\tc1:{children[0].dist:4f} \tc2:{children[1].dist:4f}"
    print(text)


# NOTE: The growth rate of the totipotent progenitor state will be severely affected by the crown length. The crown length is ~30 - 40x smaller than the average length 















# print(f"Number of leaves per tree across {len(tree_list)} trees")
# leaf_counts = []
# for tree in tree_list:
#     num_leaves = len(tree.get_leaves())
#     print("\t", num_leaves)
#     leaf_counts.append(num_leaves)

# print("Avg leaf count:", sum(leaf_counts) / len(tree_list))

# k = 8000
# p = 0.03
# data = {
#     'TLSC': [],
#     'TLS': [],
#     f'Binom (k={k}, p={p})': []
# }

# for key in data.keys():
#     if key.startswith('TLS'):
#         tls_type = key
#         tree_path = f"{working_dir}/experiments/{tls_type}/processed_data/trees.pkl"
#         with open(tree_path, "rb") as fp:
#             tree_list = pickle.load(fp)

#         leaf_counts = []
#         for tree in tree_list:
#             num_leaves = len(tree.get_leaves())
#             leaf_counts.append(num_leaves)
        
#         data[key] = leaf_counts
#     else:
#         rands = np.random.binomial(k, p, size=12)
#         data[key] = list(rands)


# sns.histplot(data=data, stat='density', multiple='dodge', common_norm=False)
# os.makedirs(f"{working_dir}/tmp/figures", exist_ok=True)
# plt.savefig(f"{working_dir}/tmp/figures/leaf_count_hist.png", dpi=400)
