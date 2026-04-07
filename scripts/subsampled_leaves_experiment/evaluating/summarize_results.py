import pickle
import numpy as np
import networkx as nx
import json
import os
import ete3
import csv

def get_total_leaves(tree_pth):
    with open(tree_pth, "rb") as fp:
        tree_list = pickle.load(fp)
    return sum(len(tree.get_leaves()) for tree in tree_list)

def get_model(model_pth):
    with open(model_pth, "rb") as fp:
        model_dict = pickle.load(fp)

    birth_kernel = np.asarray(model_dict["daughter_kernel"], dtype=float)
    growth_rates = np.asarray(model_dict["growth_rates"], dtype=float)
    idx2potency = model_dict["idx2potency"] # NOTE: May need to do special handling for potencys?

    return birth_kernel, growth_rates, idx2potency

def build_potency_model(birth_kernel, growth_rates, idx2potency):
    num_states = len(idx2potency)
    non_zero_idxs = [(i,j) for i in range(num_states) for j in range(num_states) if birth_kernel[i,j] > 0]
    potency_birth_kernel = {(idx2potency[i], idx2potency[j]): birth_kernel[i,j] for i,j in non_zero_idxs}
    potency_growth_rates = {idx2potency[i]: growth_rates[i] for i in range(num_states)}
    return potency_birth_kernel, potency_growth_rates

def get_reachable_idxs(adj_matrix, starting_state, threshold):
    G = nx.DiGraph()
    num_nodes = len(adj_matrix)
    for i in range(num_nodes):
        for j in range(num_nodes):
            weight = adj_matrix[i, j]
            if weight > threshold:
                G.add_edge(i, j, weight=weight)

    assert starting_state in G

    reachable_nodes = list(nx.descendants(G, starting_state))
    reachable_nodes.append(starting_state)
    return reachable_nodes

def get_dict_l1_error(d1, d2):
    keys = set(d1.keys()) | set(d2.keys())
    return sum(abs(float(d1.get(k, 0.0)) - float(d2.get(k, 0.0))) for k in keys)


def dict_l1_norm(d):
    return sum(abs(float(v)) for v in d.values())


working_dir = "/n/fs/ragr-research/users/wh8114/projects/troupe"
model_json_pth = f"{working_dir}/experiments/subsampled_leaves_4_terminals/model.json"

with open(model_json_pth, 'r') as fp:
    data_dict = json.load(fp)
birth_kernel_true = np.array(data_dict["birth_kernel"])
growth_rate_true = np.array(data_dict["growth_rates"])
n = len(growth_rate_true)
terminals = [str(i) for i in range(n) if birth_kernel_true[i, i] == 1.0]
idx2potency = {}
for i in range(n):
    reachable_idxs = get_reachable_idxs(birth_kernel_true, i, 0.00001)
    potency_list = [str(j) for j in reachable_idxs if str(j) in terminals]
    potency_list.sort()
    potency = tuple(potency_list)
    idx2potency[i] = potency
potency_birth_kernel_true, potency_growth_rates_true = build_potency_model(birth_kernel_true, growth_rate_true, idx2potency)

terminal_set = set(terminals)

num_trees_list = [8, 16, 32, 64]
subsample_list = [0.05, 0.1, 0.2]
trials = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
results_prefix="results/subsampled_leaves_4_terminals/classe/fundamental"
data_prefix="experiments/subsampled_leaves_4_terminals"

rows = []

for num_trees in num_trees_list:
    print(num_trees)
    for subsample in subsample_list:
        print("\t", subsample)
        for trial in trials:
            tree_pth = (
                f"{working_dir}/{data_prefix}/trees_{num_trees}/time_5.0/"
                f"sample_{subsample}/trial_{trial}/trees.pkl"
            )
            best_model_pth = (
                f"{working_dir}/{results_prefix}/trees_{num_trees}/time_5.0/"
                f"sample_{subsample}/trial_{trial}/best_model_dict.pkl"
            )

            if not os.path.exists(tree_pth):
                print(f"Missing tree file: {tree_pth}")
                continue
            if not os.path.exists(best_model_pth):
                print(f"Missing model file: {best_model_pth}")
                continue

            num_leaves = get_total_leaves(tree_pth)

            birth_kernel, growth_rates, idx2potency = get_model(best_model_pth)
            potency_birth_kernel, potency_growth_rates = build_potency_model(birth_kernel, growth_rates, idx2potency)

            birth_kernel_error = get_dict_l1_error(
                potency_birth_kernel, potency_birth_kernel_true
            )
            growth_rate_error = get_dict_l1_error(
                potency_growth_rates, potency_growth_rates_true
            )

            birth_kernel_rel_error = birth_kernel_error / (
                dict_l1_norm(potency_birth_kernel_true) + 1e-12
            )
            growth_rate_rel_error = growth_rate_error / (
                dict_l1_norm(potency_growth_rates_true) + 1e-12
            )

            rows.append({
                "num_trees": num_trees,
                "subsample": subsample,
                "trial": trial,
                "num_leaves": num_leaves,
                "birth_kernel_error": birth_kernel_error,
                "birth_kernel_rel_error": birth_kernel_rel_error,
                "growth_rate_error": growth_rate_error,
                "growth_rate_rel_error": growth_rate_rel_error
            })

# Save summary table
out_csv = f"{working_dir}/{results_prefix}/parameter_comparison_summary.csv"
os.makedirs(os.path.dirname(out_csv), exist_ok=True)

with open(out_csv, "w", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=rows[0].keys() if rows else [])
    if rows:
        writer.writeheader()
        writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {out_csv}")