import click
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
import os
import json

import numpy as np
from scipy.linalg import expm
import torch
import pickle
from collections import Counter

from itertools import combinations, permutations

from likelihood import log_vec_likelihood, _prep_log_tree
from optimizer import sparse_regularization
from utils import get_terminal_labels, get_observed_potencies, get_idx2potency, get_reachable_idxs
from eval_utils import draw_weighted_graph

from kneed import KneeLocator
import math

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="networkx backend defined more than once: nx-loopback")
    import networkx as nx

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Wedge
import seaborn as sns
sns.set_theme()
sns.set_style("white")
sns.set_palette("Dark2")

mpl.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 17,
    'axes.labelsize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'font.family': 'sans-serif',
    'font.weight': 600,
    'axes.labelweight': 600,
    'axes.titleweight': 600,
    'figure.autolayout': True
    })
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'


# color_list = [
#         '#e41a1c',
#         '#377eb8',
#         '#4daf4a',
#         '#984ea3',
#         '#ff7f00',
#         '#ffff33',
#         '#a65628',
#         '#f781bf'
#     ]

color_list = [
    '#1b9e77',
    '#d95f02',
    '#7570b3',
    '#e7298a',
    '#66a61e',
    '#e6ab02',
    '#a6761d',
    '#f781bf',
    '#999999'
]

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

script_dir = f"{base_dir}/scripts"

simulated_data_dir = f'{base_dir}/simulated_data'
initial_simulation_dir = f"{simulated_data_dir}/initial_experiment"
big_tree_simulation_dir = f"{simulated_data_dir}/big_tree_experiment"
transient_state_simulation_dir = f"{simulated_data_dir}/transient_state_experiment"
unobserved_state_simulation_dir = f"{simulated_data_dir}/unobserved_state_experiment"
model_error_simulation_dir = f"{simulated_data_dir}/model_error_experiment"

results_dir = f'{base_dir}/results'
initial_results_dir = f'{results_dir}/initial_experiment'
initial_lbfgs_results_dir = f'{results_dir}/initial_experiment_lbfgs'
big_tree_results_dir = f'{results_dir}/big_tree_experiment'
big_tree_lbfgs_results_dir = f'{results_dir}/big_tree_experiment_lbfgs'
unobserved_state_results_dir = f'{results_dir}/unobserved_state_experiment'
model_error_results_dir = f'{results_dir}/model_error_experiment'

###################################################################################################
#####   Helper methods                                                                        #####
###################################################################################################

# TODO: Put these in a utils.py file

def list_folder_names(path):
    folder_names = []
    for entry in os.listdir(path):
        if os.path.isdir(os.path.join(path, entry)):
            folder_names.append(entry)
    return folder_names

def load_model_params(model_dict_path):
    """
    Given the path to the model_dict pickle, unload relevant info to be used in `log_vec_likelihood`
    """
    with open(model_dict_path, "rb") as fp:
        model_dict = pickle.load(fp)
    inferred_rate_matrix = model_dict["rate_matrix"]
    inferred_root_distribution = model_dict["root_distribution"]
    # The '* 1e20' is done so that all x s.t. log(x) is called will be x > 0
    inferred_pi_params = torch.log(inferred_root_distribution * 1e20)
    if "growth_rates" in model_dict:
        inferred_growth_rates = model_dict["growth_rates"]
    else:
        inferred_growth_rates = None
    idx2state = model_dict["idx2state"]
    state2idx = {state: idx for idx, state in idx2state.items()}

    return inferred_rate_matrix, inferred_pi_params, inferred_growth_rates, state2idx

def isomorphic_error(mat1, mat2, obs_states=None):
    """
    Return minimum achievable error between numpy matrices mat1 and mat2 among all
    row/col permutations.

    NOTE: Assumes that obs_states are the first indices in the array
    """

    assert mat1.shape == mat2.shape

    num_states = len(mat1)
    idxs = set(range(num_states))

    if obs_states is None:
        num_observed = 0
        unobserved_states = idxs
    else:
        num_observed = len(obs_states)
        unobserved_states = idxs.difference(set(obs_states))
    num_unobserved = num_states - num_observed

    assert num_unobserved == len(unobserved_states)

    min_err = 1e20
    min_perm = None
    for perm in permutations(list(unobserved_states)):
        perm_list = list(perm)
        if obs_states is not None:
            perm_list += obs_states
        labels = np.array(perm_list)
        # Swap rows
        remixed = mat1[labels]
        # Then the cols
        remixed = remixed.T[labels].T
        err = np.sum(np.abs(remixed - mat2))
        if err < min_err:
            min_perm = labels
            min_err = err

    return min_err, min_perm



###################################################################################################
###################################################################################################

@click.group()
def cli():
    pass


@cli.command()
def evaluate_subsampling_experiment():
    """
    Usage example:
        python scripts/evaluate_results.py evaluate-subsampling-experiment
    """

    # TODO: make this a CLI argument
    remake_error_data = False

    # NOTE: This is specific to the simulation details
    num_terminal_states = 4
    num_observed_states = num_terminal_states

    working_dir = base_dir
    experiment_name = "subsampled_leaves"
    experiment_dir = f"{working_dir}/experiments/{experiment_name}"
    labels_are_ints = True
    terminal_labels = get_terminal_labels(f"{experiment_dir}/terminal_labels.txt", is_int_state=labels_are_ints)

    model_info_path = f"{working_dir}/scripts/branching_process_experiment/model_params/rate_matrix_11.json"
    with open(model_info_path, "r") as fp:
        info_dict = json.load(fp)
    ground_truth_np = np.array(info_dict["rate_matrix"])
    ground_truth_growth_rates_np = np.array(info_dict["growth_rates"])
    ground_truth_pi_params_np = np.array([-20, -20, -20, -20, 20, -20, -20])

    tree_counts = [128, 64] # [32, 64, 128]
    trials = [0, 1, 2, 3] #, 4]
    subsample_factors = [1, 2, 4, 8, 16]

    if remake_error_data:
        error_data = {
            "is_numerical_likelihood": [],
            "num_trees": [],
            "trial": [],
            "subsample_factor": [],
            "terminal_growth_rate_error": [],
            "progenitor_growth_rate_error": [],
            "growth_rates_error": [],
            "transition_rates_error": [],
            "total_cell_count": []
        }
        for num_trees in tree_counts:
            for trial in trials:
                for factor in subsample_factors:
                    for is_numerical_likelihood in [True, False]:
                        if is_numerical_likelihood and factor == 1:
                            continue
                        if is_numerical_likelihood:
                            is_numerical_str = "numerical_likelihood"
                        else:
                            is_numerical_str = "vanilla_likelihood"
                        trial_info_str = f"trees_{num_trees}/trial_{trial}/subsample_x{factor}"
                        
                        # Get data
                        trees_path = f"{experiment_dir}/processed_data/{trial_info_str}/trees.pkl"
                        with open(trees_path, "rb") as fp:
                            trees = pickle.load(fp)
                            total_cell_count = sum([len(tree.get_leaves()) for tree in trees])
                        
                        # Find the best potency and its inferred model
                        potency_dir_path = f"{experiment_dir}/results_{is_numerical_str}/{trial_info_str}"
                        idx2potency_loss = {}
                        if not os.path.isdir(potency_dir_path):
                            print("Skipping:", potency_dir_path)
                            continue
                        potency_folder_idxs = list_folder_names(potency_dir_path)
                        if "figures" in potency_folder_idxs:
                            potency_folder_idxs.remove("figures")
                        for idx in potency_folder_idxs:
                            loss_path = f"{potency_dir_path}/{idx}/loss.txt"
                            potency_path = f"{potency_dir_path}/{idx}/potency.pkl"
                            if not os.path.exists(loss_path):
                                print("=> Skipping", loss_path)
                                continue
                            with open(loss_path, "r") as fp:
                                loss = float(fp.readline())
                            with open(potency_path, "rb") as fp:
                                potency = pickle.load(fp)
                            idx2potency_loss[idx] = (potency, loss)
                        sorted_potencies = sorted(idx2potency_loss.items(), key=lambda item: item[1][1])  # Sort by potency size
                        best_potency = sorted_potencies[0][1][0]
                        best_idx = sorted_potencies[0][0]
                        best_potency_sets, best_loss = idx2potency_loss[best_idx]
                        best_potency_results_dir = f"{potency_dir_path}/{best_idx}"
                        model_dict_path = f"{best_potency_results_dir}/model_dict.pkl"
                        print("Best potency:")
                        print(best_potency)

                        # Get inferred parameters
                        with open(model_dict_path, "rb") as fp:
                            model_dict = pickle.load(fp)
                            inferred_rate_matrix = model_dict["rate_matrix"]
                            inferred_root_distribution = model_dict["root_distribution"]
                            inferred_pi_params = torch.log(inferred_root_distribution * 1e20)
                            inferred_growth_rates = model_dict["growth_rates"]
                            idx2state = model_dict["idx2state"]
                            state2idx = {state: idx for idx, state in idx2state.items()}
                            print("state2idx:")
                            print(state2idx)
                        prepped_trees = [_prep_log_tree(tree, len(inferred_rate_matrix), state2idx) for tree in trees]
                        inferred_log_lik = log_vec_likelihood(prepped_trees,
                                                            inferred_rate_matrix,
                                                            inferred_pi_params,
                                                            growth_rates=inferred_growth_rates,
                                                            state2idx=state2idx,
                                                            rho=1/factor).item()
                        true_log_lik = log_vec_likelihood(prepped_trees,
                                                        torch.tensor(ground_truth_np, dtype=torch.float64),
                                                        torch.tensor(ground_truth_pi_params_np, dtype=torch.float64),
                                                        growth_rates=torch.tensor(ground_truth_growth_rates_np, dtype=torch.float64),
                                                        rho=1/factor).item()
                        inferred_rate_matrix_np = inferred_rate_matrix.detach().numpy()
                        inferred_growth_rates_np = inferred_growth_rates.detach().numpy()
                        num_hidden_states = len(inferred_rate_matrix)-num_observed_states
                        num_terminal_states = len(terminal_labels)
                        print(f"num_hidden: {num_hidden_states} \t infer: {inferred_log_lik} \t true_lik: {true_log_lik}")
                        
                        node_labels = {i: str(idx2state[i]) for i in idx2state.keys()}
                        node_colors = {label: color_list[state2idx[label]] for label in terminal_labels}
                        for i in range(num_observed_states):
                            state = str(idx2state[i])
                            if state not in terminal_labels:
                                node_colors[state] = color_list[i]
                                node_labels[i] = state
                        for i in range(num_hidden_states):
                            idx = i+num_observed_states
                            label = "" #f"UnPr_{idx}"
                            node_colors[label] = "#FFFFFF"   # White
                            node_labels[idx] = label
                        state2potency = get_idx2potency(inferred_rate_matrix_np)

                        figure_dir = f"{potency_dir_path}/figures"
                        os.makedirs(figure_dir, exist_ok=True)
                        inferred_output_path = f"{figure_dir}/inferred_graph.pdf"
                        starting_idx = torch.argmax(inferred_root_distribution).detach().item()
                        print(inferred_root_distribution)
                        print("starting idx:", starting_idx)
                        thresh = 0.001
                        draw_weighted_graph(inferred_rate_matrix_np,
                                            inferred_output_path,
                                            thresh, node_labels,
                                            node_colors,
                                            state_text = f"log lik: {inferred_log_lik:3f}",
                                            totipotent_state=starting_idx,
                                            self_edges=inferred_growth_rates_np,
                                            state2potency=state2potency)

                        transition_rates_err, label_perm = isomorphic_error(inferred_rate_matrix_np, ground_truth_np)
                        # Use label permutation for error and then invert for type-specfic rates
                        inverted_perm = np.argsort(label_perm)
                        assert all(np.arange(len(inverted_perm))[label_perm][inverted_perm] == np.arange(len(inverted_perm)))
                        per_growth_rate_error = np.abs(inferred_growth_rates_np[label_perm]-ground_truth_growth_rates_np)[inverted_perm]

                        terminal_idxs = set()
                        terminal_growth_rate_error = 0
                        for terminal in terminal_labels:
                            idx = state2idx[terminal]
                            terminal_growth_rate_error += per_growth_rate_error[idx]
                            terminal_idxs.add(idx)
                        progenitor_growth_rate_error = 0
                        for i in range(len(ground_truth_np)):
                            if i not in terminal_idxs:
                                progenitor_growth_rate_error += per_growth_rate_error[i]


                        error_data["num_trees"].append(num_trees)
                        error_data["trial"].append(trial)
                        error_data["subsample_factor"].append(factor)
                        error_data["growth_rates_error"].append(sum(per_growth_rate_error))
                        error_data["progenitor_growth_rate_error"].append(progenitor_growth_rate_error)
                        error_data["terminal_growth_rate_error"].append(terminal_growth_rate_error)
                        error_data["transition_rates_error"].append(transition_rates_err)
                        error_data["total_cell_count"].append(total_cell_count)
                        error_data["is_numerical_likelihood"].append(is_numerical_likelihood)
        
        # Save error info to analyze
        error_data_path = f"{experiment_dir}/error_data.pkl"
        with open(error_data_path, "wb") as fp:
            pickle.dump(error_data, fp)
    
    error_data_path = f"{experiment_dir}/error_data.pkl"
    with open(error_data_path, "rb") as fp:
        error_data = pickle.load(fp)
    
    figure_dir = f"{experiment_dir}/figures"
    os.makedirs(figure_dir, exist_ok=True)

    error_keys = ["terminal_growth_rate_error", "progenitor_growth_rate_error", "transition_rates_error"]

    for error_key in error_keys:
        x_key = "subsample_factor"
        y_key = error_key
        hue_key = "num_trees" #"total_cell_count"
        shape_key="is_numerical_likelihood"

        df = pd.DataFrame(error_data)
        error_key = error_key
        df = df.dropna(subset=[x_key, hue_key, y_key])

        # norm = LogNorm(vmin=df[hue_key].min(), vmax=df[hue_key].max())
        # ax = sns.scatterplot(data=df, x=x_key, y=y_key, hue=hue_key, palette="viridis", s=50,
        #                         legend=False, alpha=0.5)
        # norm = plt.Normalize(df[hue_key].min(), df[hue_key].max())
        # sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm); sm.set_array([])
        # plt.colorbar(sm, ax=ax, label=hue_key.replace("_", " ").title())
        ax = sns.scatterplot(data=df, x=x_key, y=y_key, hue=hue_key, palette="Dark2", s=50,
                             legend=True, alpha=0.5, style=shape_key)

        ax.set(xlabel=x_key.replace("_", " ").title(), ylabel=y_key.replace("_", " ").title())
        ax.set_xscale("log", base=2)
        # ax.set_yscale("log", base=2)
        plt.savefig(f"{figure_dir}/{error_key}_vs_subsampling_scatter.pdf", dpi=400)
        plt.clf()

    for error_key in error_keys:
        df = pd.DataFrame(error_data)

        # Use the correct column name from your dict: error_key
        df = df.dropna(subset=["subsample_factor", error_key, "is_numerical_likelihood", "num_trees"])

        # Build a combined hue: (Analytic | Numerical) x trees=K
        df["model"] = np.where(df["is_numerical_likelihood"], "Numerical", "Analytic")
        df["model_trees"] = df["model"] + " | trees=" + df["num_trees"].astype(str)

        # Consistent ordering
        x_order = sorted(df["subsample_factor"].unique())
        tree_levels = sorted(df["num_trees"].unique())
        hue_order = [f"Analytic | trees={t}" for t in tree_levels] + \
                    [f"Numerical | trees={t}" for t in tree_levels]

        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=df,
            x="subsample_factor",
            y=error_key,
            hue="model_trees",
            order=x_order,
            hue_order=[h for h in hue_order if h in df["model_trees"].unique()],
            showfliers=False,          # optional: hides far outliers for cleaner whiskers
            width=0.8
        )
        sns.despine()
        plt.xlabel("Subsampling factor")
        plt.ylabel(f"{error_key.replace('_', ' ').title()}")
        plt.yscale("log")
        plt.legend(title="Model x Trees", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"{figure_dir}/{error_key}_vs_subsampling_barplot.pdf", dpi=400)
        plt.clf()




@cli.command()
def draw_graph():
    """
    Example usage:
        python scripts/evaluate_results.py draw-graph
    """
    working_dir = f"{base_dir}/experiments"

    # NOTE: To plot TLS graphs
    # input_dir = f"{working_dir}/TLS/results_vanilla_likelihood/reg=1/select_potencies"

    tree_path=f"{working_dir}/TLSC_constrained_transitions/processed_data/trees.pkl"
    # input_dir = f"{working_dir}/TLSC_constrained_transitions/results/reg=0/select_potencies"
    input_dir = f"{working_dir}/TLS_constrained_transitions/results/reg=10000/select_potencies"

    terminal_path = f"{input_dir}/../../../terminal_labels.txt"
    observed_potencies_path = f"{input_dir}/../../../observed_potencies.txt"

    terminal_labels = get_terminal_labels(terminal_path)
    observed_potencies = get_observed_potencies(observed_potencies_path)
    figure_dir = f"{input_dir}/figures"
    os.makedirs(figure_dir, exist_ok=True)
    draw_graph_normal(input_dir, figure_dir, terminal_labels, tree_path=tree_path,
                      observed_potencies=observed_potencies, thresh=1e-4)

@cli.command()
def draw_sse_graph():
    """
    Example usage:
        python scripts/evaluate_results.py draw-sse-graph
    """
    working_dir = f"{base_dir}/experiments"

    # NOTE: For plotting SSE
    input_dir = f"{working_dir}/TLS/results_SSE_unconstrained"
    terminal_labels = get_terminal_labels(f"{input_dir}/../terminal_labels.txt")
    observed_potencies = get_observed_potencies(f"{input_dir}/../observed_potencies_unconstrained.txt")
    thresh = 1e-4

    figure_dir = f"{input_dir}/figures"

    # Get inferred parameters
    model_dict_path = f"{input_dir}/model_dict.pkl"
    with open(model_dict_path, "rb") as fp:
        model_dict = pickle.load(fp)
        inferred_rate_matrix = model_dict["rate_matrix"]
        inferred_root_distribution = model_dict["root_distribution"]
        inferred_growth_rates = model_dict["growth_rates"]
        idx2state = model_dict["idx2state"]
        state2idx = {state: idx for idx, state in idx2state.items()}
        terminal_idxs = [state2idx[state] for state in terminal_labels]

    inferred_rate_matrix_np = inferred_rate_matrix.detach().numpy()
    inferred_growth_rates_np = inferred_growth_rates.detach().numpy()

    node_labels = {i: str(idx2state[i]) for i in idx2state.keys()}
    node_colors = {str(label): color_list[i] for i, label in enumerate(terminal_labels)}
    for idx in idx2state.keys():
        if idx2state[idx] not in observed_potencies:
            label = ""
            node_colors[label] = "#FFFFFF"   # White
            node_labels[idx] = label
        else:
            node_labels[idx] = str(idx2state[idx])

    inferred_rate_matrix_np[inferred_rate_matrix_np < thresh] = 0
    inferred_rate_matrix_np -= np.diag(np.sum(inferred_rate_matrix_np, axis=1))
    idx2potency_ = get_idx2potency(inferred_rate_matrix_np, eps=1e-4)
    idx2potency = {}
    # Remove non-terminals from potencies
    for idx in range(len(inferred_rate_matrix)):
        potency_ = idx2potency_[idx]
        potency = [s for s in potency_ if idx2state[s] in terminal_labels]
        print(idx, potency)
        if len(potency) == 0:
            print("\tSkipping")
            continue
        potency.sort()
        idx2potency[idx] = tuple(potency)

    os.makedirs(figure_dir, exist_ok=True)
    inferred_output_path = f"{figure_dir}/inferred_graph.pdf"
    starting_idx = torch.argmax(inferred_root_distribution).detach().item()
    print("Starting idx:", starting_idx, idx2state[idx])
    draw_weighted_graph(inferred_rate_matrix_np,
                        inferred_output_path,
                        thresh,
                        node_labels,
                        node_colors,
                        totipotent_state=starting_idx,
                        self_edges=inferred_growth_rates_np,
                        state2potency=idx2potency,
                        terminal_idxs=terminal_idxs)


def draw_graph_normal(input_dir,
                      figure_dir,
                      terminal_labels,
                      num_observed_states=None, # TODO: Deprecate this
                      observed_potencies=None,
                      tree_path=None,
                      thresh=1e-4):
    if observed_potencies is None:
        observed_potencies = get_observed_potencies(f"{input_dir}/observed_potencies.txt", is_int_state=False)

    # Get inferred parameters
    model_dict_path = f"{input_dir}/model_dict.pkl"
    with open(model_dict_path, "rb") as fp:
        model_dict = pickle.load(fp)
        inferred_rate_matrix = model_dict["rate_matrix"]
        inferred_root_distribution = model_dict["root_distribution"]
        inferred_pi_params = torch.log(inferred_root_distribution * 1e20)
        inferred_growth_rates = model_dict["growth_rates"]
        idx2state = model_dict["idx2state"]
        state2idx = {state: idx for idx, state in idx2state.items()}
        terminal_idxs = [state2idx[state] for state in terminal_labels]

    if tree_path is not None:
        with open(tree_path, "rb") as fp:
            trees = pickle.load(fp)
            prepped_trees = [_prep_log_tree(tree, len(inferred_rate_matrix), state2idx) for tree in trees]
            inferred_log_lik = log_vec_likelihood(prepped_trees,
                                                inferred_rate_matrix,
                                                inferred_pi_params,
                                                growth_rates=inferred_growth_rates,
                                                state2idx=state2idx).item()
        text = f"Log likelihood: {inferred_log_lik:2g}"
    else:
        text = None


    inferred_rate_matrix_np = inferred_rate_matrix.detach().numpy()
    inferred_growth_rates_np = inferred_growth_rates.detach().numpy()

    node_labels = {i: str(idx2state[i]) for i in idx2state.keys()}
    node_colors = {str(label): color_list[i] for i, label in enumerate(terminal_labels)}
    for idx in idx2state.keys():
        if idx2state[idx] not in observed_potencies:
            label = ""
            node_colors[label] = "#FFFFFF"   # White
            node_labels[idx] = label
        else:
            node_labels[idx] = str(idx2state[idx])

    idx2potency_ = get_idx2potency(inferred_rate_matrix_np)
    idx2potency = {}
    # Remove non-terminals from potencies
    for idx in range(len(inferred_rate_matrix)):
        potency_ = idx2potency_[idx]
        potency = [s for s in potency_ if idx2state[s] in terminal_labels]
        print(idx, potency)
        if len(potency) == 0:
            print("\tSkipping")
            continue
        potency.sort()
        idx2potency[idx] = tuple(potency)

    os.makedirs(figure_dir, exist_ok=True)
    inferred_output_path = f"{figure_dir}/inferred_graph.pdf"
    starting_idx = torch.argmax(inferred_root_distribution).detach().item()
    print("Starting idx:", starting_idx, idx2state[idx])
    draw_weighted_graph(inferred_rate_matrix_np,
                        inferred_output_path,
                        thresh, node_labels,
                        node_colors,
                        totipotent_state=starting_idx,
                        self_edges=inferred_growth_rates_np,
                        state_text=text,
                        state2potency=idx2potency,
                        terminal_idxs=terminal_idxs)

def get_inference_error(input_dir, ground_truth_np, ground_truth_growth_rates_np, labels_are_ints=True, draw_graph=False):
    terminal_labels = get_terminal_labels(f"{input_dir}/terminal_labels.txt", is_int_state=labels_are_ints)
    observed_potencies = get_observed_potencies(f"{input_dir}/observed_potencies.txt", is_int_state=labels_are_ints)

    model_dict_path = f"{input_dir}/model_dict.pkl"
    with open(model_dict_path, "rb") as fp:
        model_dict = pickle.load(fp)
        inferred_rate_matrix_np = model_dict["rate_matrix"].detach().numpy()
        inferred_root_distribution = model_dict["root_distribution"]
        inferred_pi_params = torch.log(inferred_root_distribution * 1e20).detach().numpy()
        inferred_growth_rates_np = model_dict["growth_rates"].detach().numpy()

    if draw_graph:
        draw_graph_normal(input_dir, f"{input_dir}/figures", terminal_labels,
                          num_observed_states=len(observed_potencies))
    
    transition_rates_err, label_perm = isomorphic_error(inferred_rate_matrix_np, ground_truth_np)
    # Use label permutation for error and then invert for type-specfic rates
    inverted_perm = np.argsort(label_perm)
    assert all(np.arange(len(inverted_perm))[label_perm][inverted_perm] == np.arange(len(inverted_perm)))
    growth_rates_err = np.sum(np.abs(inferred_growth_rates_np[label_perm]-ground_truth_growth_rates_np)[inverted_perm])
    return transition_rates_err.item(), growth_rates_err.item()

def get_exact_inference_error(input_dir, ground_truth_np, ground_truth_growth_rates_np, draw_graph=False, labels_are_ints=True):
    terminal_labels = get_terminal_labels(f"{input_dir}/terminal_labels.txt", is_int_state=labels_are_ints)
    observed_potencies = get_observed_potencies(f"{input_dir}/observed_potencies.txt", is_int_state=labels_are_ints)

    # Find run with the lowest loss (will be index of potency sets for constrained)
    idx2loss = {}
    potency_folder_idxs = list_folder_names(input_dir)
    if "figures" in potency_folder_idxs:
        potency_folder_idxs.remove("figures")
    for idx in potency_folder_idxs:
        loss_path = f"{input_dir}/{idx}/loss.txt"
        if not os.path.isfile(loss_path):
            print(f"Skipping: {loss_path}")
            continue
        with open(loss_path, "r") as fp:
            loss = float(fp.readline())
        idx2loss[idx] = loss
    sorted_potencies = sorted(idx2loss.items(), key=lambda item: item[1])
    best_idx = sorted_potencies[0][0]
    best_potency_results_dir = f"{input_dir}/{best_idx}"
    model_dict_path = f"{best_potency_results_dir}/model_dict.pkl"

    if draw_graph:
        draw_graph_normal(model_dict_path, f"{input_dir}/figures", terminal_labels,
                          num_observed_states=len(observed_potencies))

    with open(model_dict_path, "rb") as fp:
        model_dict = pickle.load(fp)
        inferred_rate_matrix_np = model_dict["rate_matrix"].detach().numpy()
        inferred_root_distribution = model_dict["root_distribution"]
        inferred_pi_params = torch.log(inferred_root_distribution * 1e20).detach().numpy()
        inferred_growth_rates_np = model_dict["growth_rates"].detach().numpy()

    transition_rates_err, label_perm = isomorphic_error(inferred_rate_matrix_np, ground_truth_np)
    # Use label permutation for error and then invert for type-specfic rates
    inverted_perm = np.argsort(label_perm)
    assert all(np.arange(len(inverted_perm))[label_perm][inverted_perm] == np.arange(len(inverted_perm)))
    growth_rates_err = np.sum(np.abs(inferred_growth_rates_np[label_perm]-ground_truth_growth_rates_np)[inverted_perm])

    return transition_rates_err.item(), growth_rates_err.item()


def get_relaxed_inference_error(input_dir, ground_truth_np, ground_truth_growth_rates_np,
                                ground_truth_pi_params_np, trees, labels_are_ints=True,
                                draw_graph=False, draw_select_potencies=False):
    terminal_labels = get_terminal_labels(f"{input_dir}/terminal_labels.txt", is_int_state=labels_are_ints)
    observed_potencies = get_observed_potencies(f"{input_dir}/observed_potencies.txt", is_int_state=labels_are_ints)
    num_observed_states = len(observed_potencies)
    thresh = 1e-3

    # Find run that realizes the true number of states with the lowest loss
    reg2loss = {}
    regs = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1 , 3, 10]
    used_regs = []
    losses = []
    for reg in regs:
        loss_path = f"{input_dir}/reg={reg}/loss.txt"
        if not os.path.isfile(loss_path):
            print(f"Skipping: {loss_path}")
            continue
        with open(loss_path, "r") as fp:
            loss = float(fp.readline())
        model_dict_path = f"{input_dir}/reg={reg}/model_dict.pkl"
        with open(model_dict_path, "rb") as fp:
            model_dict = pickle.load(fp)
            inferred_rate_matrix_np = model_dict["rate_matrix"].detach().numpy()
            starting_state = torch.argmax(model_dict["root_distribution"]).detach().item()
        reachable_idxs = get_reachable_idxs(inferred_rate_matrix_np, starting_state, threshold=thresh)
        reg2loss[reg] = (loss, len(reachable_idxs))

        ###############
        if draw_graph:
            results_dir = f"{input_dir}/reg={reg}"
            figure_dir = f"{results_dir}/figures"
            os.makedirs(figure_dir, exist_ok=True)
            model_dict_path = f"{results_dir}/model_dict.pkl"
            loss_path = f"{results_dir}/state_dict.pth"

            if not os.path.isfile(model_dict_path) or not os.path.isfile(loss_path):
                print("Skipping:", model_dict_path)
                continue
            with open(model_dict_path, "rb") as fp:
                model_dict = pickle.load(fp)
                inferred_rate_matrix = model_dict["rate_matrix"]
                inferred_root_distribution = model_dict["root_distribution"]
                inferred_pi_params = torch.log(inferred_root_distribution * 1e20)
                inferred_growth_rates = model_dict["growth_rates"]
                idx2state = model_dict["idx2state"]
                state2idx = {state: idx for idx, state in idx2state.items()}
            prepped_trees = [_prep_log_tree(tree, len(inferred_rate_matrix), state2idx) for tree in trees]
            inferred_log_lik = log_vec_likelihood(prepped_trees,
                                                inferred_rate_matrix,
                                                inferred_pi_params,
                                                growth_rates=inferred_growth_rates,
                                                state2idx=state2idx).item()

            inferred_rate_matrix = inferred_rate_matrix.detach().numpy()
            num_hidden_states = len(inferred_rate_matrix)-num_observed_states
            num_terminal_states = len(terminal_labels)
            
            node_labels = {i: str(idx2state[i]) for i in idx2state.keys()}
            node_colors = {str(label): color_list[i] for i, label in enumerate(terminal_labels)}
            for idx in idx2state.keys():
                if idx2state[idx] not in observed_potencies:
                    label = ""
                    node_colors[label] = "#FFFFFF"   # White
                    node_labels[idx] = label
                else:
                    node_labels[idx] = str(idx2state[idx])
            state2potency = get_idx2potency(inferred_rate_matrix, tree_length=10)

            inferred_output_path = f"{figure_dir}/inferred_graph.pdf"
            starting_idx = torch.argmax(inferred_root_distribution).detach().item()
            draw_weighted_graph(inferred_rate_matrix,
                                inferred_output_path,
                                thresh, node_labels,
                                node_colors,
                                # state_text = f"log lik: {inferred_log_lik:3f}",
                                totipotent_state=starting_idx,
                                self_edges=inferred_growth_rates,
                                state2potency=state2potency)
                            
            losses.append(-inferred_log_lik)
            used_regs.append(reg)

            if draw_select_potencies:
                if not os.path.isfile(f"{results_dir}/select_potencies/model_dict.pkl"):
                    print("Skipping:", model_dict_path)
                    continue
                draw_graph_normal(f"{results_dir}/select_potencies",
                                  f"{results_dir}/select_potencies/figures",
                                  terminal_labels=terminal_labels,
                                  observed_potencies=observed_potencies,
                                  num_observed_states=len(observed_potencies))

        ###############
    if len(used_regs) == 0:
        return None, None

    if len(used_regs) >= 2 and draw_graph:
        figure_outdir = f"{input_dir}/figures"
        os.makedirs(figure_outdir, exist_ok=True)
        # Plot likelihood vs regularization strength
        x_vals = used_regs
        plt.plot(x_vals, losses, marker='o', linewidth=4, markersize=12)
        plt.xlabel("Regularization")
        plt.ylabel("Loss (negative log likelihood)")
        plt.xscale("log")
        plt.savefig(f"{figure_outdir}/loss_vs_regularization.pdf", dpi=400)
        plt.clf()

    num_states_gt = len(ground_truth_growth_rates_np)
    reg2loss_trimmed = {}
    pad_with_zeros = False
    for reg, (loss, num_nodes) in reg2loss.items():
        if num_nodes <= num_states_gt:
            reg2loss_trimmed[reg] = (loss, num_nodes)
    pad_with_zeros = all([num_nodes < num_states_gt for (_, num_nodes) in reg2loss_trimmed.values()])

    sorted_regs = sorted(reg2loss_trimmed.items(), key=lambda item: item[1][0])
    best_reg = sorted_regs[0][0]
    print("Best reg:", best_reg, f"...{input_dir[-40:]}")
    best_reg_results_dir = f"{input_dir}/reg={best_reg}"
    model_dict_path = f"{best_reg_results_dir}/model_dict.pkl"

    # NOTE: Doing Kneedle separately
    # y = losses
    # x = [math.log(reg) for reg in used_regs]
    # kneedle = KneeLocator(x, y, S=1.0, curve="convex", direction="increasing")
    # if kneedle.knee is not None:
    #     knee_x = round(math.exp(kneedle.knee), 4)
    #     knee_y = kneedle.knee_y
    #     num_reachable = reg2loss[knee_x][1]
    #     print(f"knee: ({knee_x}, {knee_y}) \t num_reachable: {num_reachable}")
    # else:
    #     knee_x = "Error"
    #     knee_y = "Error"
    #     num_reachable = "Error"

    # with open(f"{input_dir}/knee.txt", "w") as fp:
    #     fp.write(f"knee_x\t{knee_x}\n")                 # x-value of highest curvature
    #     fp.write(f"knee_y\t{knee_y}\n")                 # y-value of highest curvature
    #     fp.write(f"num_reachable\t{num_reachable}\n")   # num nodes connected to starting state for knee_x
    #     fp.write(f"best_reg\t{best_reg}\n")             # Smallest reg that achieves the right number of states

    with open(model_dict_path, "rb") as fp:
        model_dict = pickle.load(fp)
        inferred_rate_matrix_np = model_dict["rate_matrix"].detach().numpy()
        inferred_root_distribution = model_dict["root_distribution"]
        starting_state = torch.argmax(model_dict["root_distribution"]).detach().item()
        inferred_growth_rates_np = model_dict["growth_rates"].detach().numpy()
    reachable_idxs = np.array(get_reachable_idxs(inferred_rate_matrix_np, starting_state, threshold=thresh))

    # Subset the inferred_rate_matrix and growth rates
    inferred_rate_matrix_np = inferred_rate_matrix_np[reachable_idxs, :][:, reachable_idxs]
    inferred_growth_rates_np = inferred_growth_rates_np[reachable_idxs]
    k = 0
    if pad_with_zeros:
        k = reg2loss_trimmed[best_reg][1]
        inferred_rate_matrix_np = np.pad(inferred_rate_matrix_np, ((0, num_states_gt-k),(0, num_states_gt-k)),
                constant_values=(0, 0))
        inferred_growth_rates_np = np.pad(inferred_growth_rates_np, (0, num_states_gt-k),
                constant_values=(0,))
    print("\tpad_with_zeros = ", pad_with_zeros, num_states_gt-k)

    transition_rates_err, label_perm = isomorphic_error(inferred_rate_matrix_np, ground_truth_np)
    # Use label permutation for error and then invert for type-specfic rates
    inverted_perm = np.argsort(label_perm)
    assert all(np.arange(len(inverted_perm))[label_perm][inverted_perm] == np.arange(len(inverted_perm)))
    growth_rates_err = np.sum(np.abs(inferred_growth_rates_np[label_perm]-ground_truth_growth_rates_np)[inverted_perm])

    return transition_rates_err.item(), growth_rates_err.item()
        


@cli.command()
def evaluate_sample_efficiency_knees():
    """
    Usage example:
        python scripts/evaluate_results.py evaluate-sample-efficiency-knees
    """
    process_time = 2.35
    rate_matrix = 12
    num_states_ground_truth = 9
    thresh = 1e-3
    method = "relaxed"
    working_dir = base_dir
    experiment_name = "sample_efficiency_experiment"
    experiment_dir = f"{working_dir}/results/{experiment_name}/rate_matrix_{rate_matrix}"

    trials = [0, 1, 2, 3, 4]
    tree_counts = [8, 16, 32, 64, 128, 256, 512]

    count2correct = {count: 0 for count in tree_counts}

    for num_trees in tree_counts:
        for trial in trials:
            print(num_trees, trial)
            trees_path = f"{working_dir}/simulated_data/branching_process_experiment" + \
                            f"/{rate_matrix}/trees_{num_trees}/time_{process_time}/trial_{trial}/trees.pkl"
            input_dir = f"{experiment_dir}/{method}/trees_{num_trees}/time_{process_time}/trial_{trial}"

            with open(trees_path, "rb") as fp:
                trees = pickle.load(fp)

            regs = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1 , 3, 10]
            # regs = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1 , 3, 10]
            used_regs = []
            neg_llh = []
            num_reachable_states = []
            reg2num_states = {}
            for reg in regs:
                if reg == 0:
                    loss_path = f"{input_dir}/reg={reg}/loss.txt"
                    model_dict_path = f"{input_dir}/reg={reg}/model_dict.pkl"
                else:
                    loss_path = f"{input_dir}/reg={reg}/select_potencies/loss.txt"
                    model_dict_path = f"{input_dir}/reg={reg}/select_potencies/model_dict.pkl"
                if not os.path.isfile(loss_path):
                    print(f"Skipping: {loss_path}")
                    continue
                with open(model_dict_path, "rb") as fp:
                    model_dict = pickle.load(fp)
                    inferred_rate_matrix = model_dict["rate_matrix"]
                    inferred_rate_matrix_np = inferred_rate_matrix.detach().numpy()
                    inferred_root_distribution = model_dict["root_distribution"]
                    starting_state = torch.argmax(inferred_root_distribution).item()
                    inferred_pi_params = torch.log(inferred_root_distribution * 1e20)
                    inferred_growth_rates = model_dict["growth_rates"]
                    idx2state = model_dict["idx2state"]
                    state2idx = {state: idx for idx, state in idx2state.items()}
                prepped_trees = [_prep_log_tree(tree, len(inferred_rate_matrix), state2idx) for tree in trees]
                inferred_log_lik = log_vec_likelihood(prepped_trees,
                                                    inferred_rate_matrix,
                                                    inferred_pi_params,
                                                    growth_rates=inferred_growth_rates,
                                                    state2idx=state2idx).item()

                reachable_idxs = get_reachable_idxs(inferred_rate_matrix_np, starting_state, threshold=thresh)
                neg_llh.append(-inferred_log_lik)
                num_reachable_states.append(len(reachable_idxs))
                used_regs.append(reg)
                reg2num_states[reg] = len(reachable_idxs)

            # Set y-vals to be the minimum among all the same x-vals
            y_ = neg_llh # [math.log(y_val) for y_val in neg_llh]
            x_ = num_reachable_states
            state2liks = {x_val: [] for x_val in x_}
            for x_val, y_val in zip(x_, y_):
                state2liks[x_val].append(y_val)
            y = []
            x = list(set(x_))
            x.sort()
            for x_val in x:
                y.append(min(state2liks[x_val]))
            print("num states:  ", x)
            print("likelihoods: ", y)
            kneedle = KneeLocator(x, y, S=0.5, curve="convex", direction="decreasing")

            # x = [math.log(reg) for reg in used_regs]
            # y = neg_llh
            # kneedle = KneeLocator(x, y, S=0.5, curve="convex", direction="increasing")
            # x = [round(math.exp(x_val), 5) for x_val in x]

            if kneedle.knee is not None:
                knee_x = kneedle.knee
                num_states_inferred = knee_x
                # knee_x = round(math.exp(kneedle.knee), 5)
                # num_states_inferred = reg2num_states[knee_x]
                knee_y = kneedle.knee_y
                print("Knee:", knee_x, knee_y)
                count2correct[num_trees] += int(num_states_inferred == num_states_ground_truth)
            else:
                knee_x = "Error"
                knee_y = "Error"

            with open(f"{input_dir}/num_states_knee.txt", "w") as fp:
                fp.write(f"knee_x\t{knee_x}\n")                 # x-value of highest curvature
                fp.write(f"knee_y\t{knee_y}\n")                 # y-value of highest curvature

            plt.plot(x, y, marker='o', linewidth=4, markersize=15)
            plt.ylabel("Negative Log Likelihood")
            if kneedle.knee is not None:
                plt.plot([knee_x], [knee_y], marker='*', color ='red', markersize=25)
            plt.xlabel("Number of states")
            plt.savefig(f"{input_dir}/figures/loss_vs_num_states.pdf")
            # plt.xlabel("Regularization strength")
            # plt.xscale('log')
            plt.savefig(f"{input_dir}/figures/loss_vs_regularization.pdf")
            plt.clf()
        
        print(count2correct)

    
@cli.command()
def evaluate_sample_efficiency_experiment():
    """
    Usage example:
        python scripts/evaluate_results.py evaluate-sample-efficiency-experiment
    """

    remake_error_data = True
    draw_graph = True
    labels_are_ints = True
    # process_time = 1.8  # 1.75
    # rate_matrix = 13    # 2
    process_time = 2.35
    rate_matrix = 12

    working_dir = base_dir
    experiment_name = "sample_efficiency_experiment"
    experiment_dir = f"{working_dir}/results/{experiment_name}/rate_matrix_{rate_matrix}"

    if remake_error_data:

        ground_truth_model_info_path = f"{working_dir}/scripts/branching_process_experiment/model_params/rate_matrix_{rate_matrix}.json"
        with open(ground_truth_model_info_path, "r") as fp:
            info_dict = json.load(fp)
            ground_truth_np = np.array(info_dict["rate_matrix"])
            ground_truth_growth_rates_np = np.array(info_dict["growth_rates"])
            ground_truth_root_distribution_np = np.array(info_dict["init_distribution"])
            ground_truth_pi_params = np.log((ground_truth_root_distribution_np + 1e-20) * 1e20)

        methods = [
            "relaxed",
            "constrained_observed",
            "unconstrained",
            # "exact",
        ]

        trials = [0, 1, 2, 3, 4]
        tree_counts = [32, 64, 128, 256, 512, 8, 16] #, 1024]

        error_dict = {
            "transition_rate_error" : [],
            "growth_rate_error": [],
            "method": [],
            "num_trees": [],
            "num_leaves": [],
            "trial": [],
            "num_obs_states": []
        }

        for method in methods:
            print(method)
            for num_trees in tree_counts:
                for trial in trials:
                    print(method, num_trees, trial)
                    trees_path = f"{working_dir}/simulated_data/branching_process_experiment" + \
                                 f"/{rate_matrix}/trees_{num_trees}/time_{process_time}/trial_{trial}/trees.pkl"
                    trial_dir = f"{experiment_dir}/{method}/trees_{num_trees}/time_{process_time}/trial_{trial}"

                    with open(trees_path, "rb") as fp:
                        trees = pickle.load(fp)

                    states = set()
                    num_leaves = 0
                    for tree in trees:
                        tree_leaves = tree.get_leaves()
                        num_leaves += len(tree_leaves)
                        for leaf in tree_leaves:
                            states.add(leaf.state)
                    num_obs_states = len(states)

                    if method == "exact":
                        transition_rate_error, growth_rate_error = get_exact_inference_error(
                                                                                trial_dir,
                                                                                ground_truth_np,
                                                                                ground_truth_growth_rates_np,
                                                                                draw_graph=draw_graph)
                    elif method == "relaxed":
                        transition_rate_error, growth_rate_error = get_relaxed_inference_error(
                                                                                trial_dir,
                                                                                ground_truth_np,
                                                                                ground_truth_growth_rates_np,
                                                                                ground_truth_pi_params,
                                                                                trees,
                                                                                draw_graph=draw_graph,
                                                                                draw_select_potencies=True)
                    elif method == "constrained_observed" or method == "unconstrained":
                        transition_rate_error, growth_rate_error = get_inference_error(
                                                                                trial_dir,
                                                                                ground_truth_np,
                                                                                ground_truth_growth_rates_np,
                                                                                draw_graph=draw_graph)
                    
                    if transition_rate_error is None:
                        continue

                    error_dict["transition_rate_error"].append(transition_rate_error)
                    error_dict["growth_rate_error"].append(growth_rate_error)
                    error_dict["method"].append(method)
                    error_dict["num_trees"].append(num_trees)
                    error_dict["num_leaves"].append(num_leaves)
                    error_dict["trial"].append(trial)
                    error_dict["num_obs_states"].append(num_obs_states)
    
    
        # TODO: uncomment this
        # Save error info to analyze
        error_dict_path = f"{experiment_dir}/error_dict.pkl"
        with open(error_dict_path, "wb") as fp:
            pickle.dump(error_dict, fp)
    
    error_dict_path = f"{experiment_dir}/error_dict.pkl"
    with open(error_dict_path, "rb") as fp:
        error_dict = pickle.load(fp)

    figure_dir = f"{experiment_dir}/figures"
    os.makedirs(figure_dir, exist_ok=True)

    methods2name = {
            "constrained_observed": "Constrained Observed",
            "unconstrained": "SSE",
            "exact": "Exact",
            "relaxed": "Ours"
        }

    method_list = error_dict["method"]
    for i, method in enumerate(method_list):
        method_list[i] = methods2name[method]

    error_keys = ["transition_rate_error", "growth_rate_error"]

    def fmt_mu(v):
        return "μ=-" if pd.isna(v) else f"μ={v:.1f}"
    
    for error_key in error_keys:
        x_key = "num_trees"
        y_key = error_key
        hue_key = "method"
        states_key = "num_obs_states"

        df = pd.DataFrame(error_dict)
        # df = df[df["method"] != "unconstrained"]
        df = df[df["num_trees"] != 8]
        df = df[df["method"] != "Constrained Observed"]
        df = df.dropna(subset=[x_key, hue_key, y_key, states_key]).copy()

        # Order x categories numerically
        order = sorted(pd.unique(df[x_key]))

        plt.figure(figsize=(8,5))
        ax = sns.boxplot(
            data=df,
            x=x_key, y=y_key, hue=hue_key,
            order=order, palette="Dark2", showfliers=False
        )
        ax.set(
            xlabel="Num Trees",
            ylabel=y_key.replace("_", " ").title(),
        )
        # Mean observed states per num_trees (across trials/methods in df)
        means = df.groupby(x_key)[states_key].mean().reindex(order)
        # Two-line x tick labels: "<num_trees>\nμ=<mean states>"
        x_positions = np.arange(len(order))
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{o}\n{fmt_mu(m)}" for o, m in zip(order, means.values)])
        ax.tick_params(axis="x", labelsize=9, pad=6)  # pad gives a little room for the second line
        plt.tight_layout()
        plt.savefig(f"{figure_dir}/{error_key}_vs_num_trees_boxplot.pdf") #, dpi=400)
        plt.clf()

    for y_key in error_keys:
        # Filter once for plotting y; don't require states here
        df = pd.DataFrame(error_dict)
        df_y = df.dropna(subset=[x_key, hue_key, y_key]).copy()
        # X order (numeric, stable)
        order = sorted(pd.unique(df_y[x_key]))
        if len(order) == 0:
            continue
        # Mean observed progenitors per num_trees (use rows that have states)
        mu_states = (df.dropna(subset=[x_key, states_key])
                    .groupby(x_key)[states_key]
                    .mean()
                    .reindex(order))
        # Pre-aggregate mean error per (method, num_trees)
        means_df = (df_y.groupby([hue_key, x_key])[y_key]
                        .mean()
                        .reset_index())
        # Method order and colors
        method_order = sorted(means_df[hue_key].unique())
        palette = sns.color_palette("Dark2", n_colors=len(method_order))
        plt.figure(figsize=(8, 5))
        ax = plt.gca()
        for color, method in zip(palette, method_order):
            sub = (means_df[means_df[hue_key] == method]
                        .set_index(x_key)
                        .reindex(order))  # ensures consistent x across methods
            ax.plot(order, sub[y_key].values, marker="o", linewidth=2, label=method, color=color)
        ax.set_xlabel("Num Trees")
        ax.set_ylabel(y_key.replace("_", " ").title())
        ax.set_xscale("log", base=10)  # uncomment if you want log scale

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, title="Method", frameon=False, loc="best")

        plt.tight_layout()
        plt.savefig(f"{figure_dir}/{y_key}_vs_num_trees_lineplot.pdf") #, dpi=400)
        plt.clf()
    
    for error_key in error_keys:
        x_key = "num_leaves"
        y_key = error_key
        hue_key = "method"
        df = pd.DataFrame(error_dict)
        # df = df[df["method"] != "unconstrained"]
        df = df.dropna(subset=[x_key, hue_key, y_key])
        ax = sns.scatterplot(data=df, x=x_key, y=y_key, hue=hue_key, palette="Dark2", alpha=0.6)
        ax.set(xlabel=x_key.replace("_", " ").title(), ylabel=y_key.replace("_", " ").title())
        # ax.set_yscale("log", base=10)
        ax.set_xscale("log")
        plt.tight_layout()
        plt.savefig(f"{figure_dir}/{error_key}_vs_num_leaves_scatter.pdf") #, dpi=400)
        plt.clf()


@cli.command()
def draw_true_graph():
    """
    python scripts/evaluate_results.py draw-true-graph
    """
    rate_matrix_num = 12
    terminal_idxs = [4, 5, 6, 7, 8]

    # rate_matrix_num = 13
    # terminal_idxs = [0, 1, 2, 3]

    idx2color = {idx: color_list[i] for i, idx in enumerate(terminal_idxs)}
    idx2state = {i: str(i) for i in terminal_idxs}
    state2idx = {state: idx for idx, state in idx2state.items()}
    working_dir = f"{base_dir}/scripts/model_params"
    model_dict_path = f"{working_dir}/rate_matrix_{rate_matrix_num}.json"
    with open(model_dict_path, "r") as fp:
        model_dict = json.load(fp)
        rate_matrix = np.array(model_dict["rate_matrix"])
        root_distribution = np.array(model_dict["init_distribution"])
        growth_rates = np.array(model_dict["growth_rates"])

    num_hidden_states = len(rate_matrix)-len(terminal_idxs)

    node_labels = {i: str(idx2state[i]) for i in idx2state.keys()}
    node_colors = {idx2state[idx]: idx2color[idx] for idx in terminal_idxs}
    for idx in range(len(rate_matrix)):
        if idx not in terminal_idxs:
            label = ""
            node_colors[label] = "#FFFFFF"   # White
            node_labels[idx] = label
    state2potency = get_idx2potency(rate_matrix, tree_length=10)

    inferred_output_path = f"{working_dir}/rate_matrix_{rate_matrix_num}.pdf"
    starting_idx = np.argmax(root_distribution).item()
    thresh = 0.0
    draw_weighted_graph(rate_matrix,
                        inferred_output_path,
                        thresh,
                        node_labels,
                        node_colors,
                        totipotent_state=starting_idx,
                        self_edges=growth_rates,
                        state2potency=state2potency,
                        no_node_labels=True)
    
@cli.command()
def get_experiment_potencies():
    """
    Usage example:
        python scripts/evaluate_results.py get-experiment-potencies
    """

    process_time = 2.35
    rate_matrix = 12
    true_num_states = 9
    experiment_name = "sample_efficiency_experiment"

    working_dir = base_dir
    experiment_dir = f"{working_dir}/results/{experiment_name}/rate_matrix_{rate_matrix}"

    methods = [
        # "constrained_observed",
        # "unconstrained",
        "relaxed"
    ]

    trials = [0, 1, 2, 3, 4]
    tree_counts = [8, 16, 32, 64, 128, 256, 512] #, 1024]

    for method in methods:
        for num_trees in tree_counts:
            for trial in trials:
                print(method, num_trees, trial)
                trial_dir = f"{experiment_dir}/{method}/trees_{num_trees}/time_{process_time}/trial_{trial}"

                if method == "relaxed":
                    regs = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
                    file_output = "ERROR"
                    for reg in regs:
                        model_dict_path = f"{trial_dir}/reg={reg}/model_dict.pkl"
                        with open(model_dict_path, "rb") as fp:
                            model_dict = pickle.load(fp)
                            inferred_rate_matrix_np = model_dict["rate_matrix"].detach().numpy()
                            idx2state = model_dict["idx2state"]
                            starting_state = torch.argmax(model_dict["root_distribution"]).detach().item()
                        reachable_idxs = get_reachable_idxs(inferred_rate_matrix_np, starting_state, threshold=0.00001)
                        idx2potency = get_idx2potency(inferred_rate_matrix_np, tree_length=10)

                        reg_file_output = ""
                        for i in reachable_idxs:
                            reg_file_output += f"{[idx2state[idx] for idx in idx2potency[i]]}"
                            reg_file_output += "\n"
                        with open(f"{trial_dir}/reg={reg}/inferred_potencies.txt", "w") as fp:
                            fp.write(reg_file_output)
                        
                        # take the lowest regularization s.t. we infer leq the true number of states
                        if file_output == "ERROR" and len(reachable_idxs) <= true_num_states:
                            print("\treg:", reg)
                            file_output = reg_file_output
                else:
                    model_dict_path = f"{trial_dir}/model_dict.pkl"
                    with open(model_dict_path, "rb") as fp:
                        model_dict = pickle.load(fp)
                        inferred_rate_matrix_np = model_dict["rate_matrix"].detach().numpy()
                        idx2state = model_dict["idx2state"]
                    idx2potency = get_idx2potency(inferred_rate_matrix_np, tree_length=10)
                    file_output = ""
                    for i in range(len(idx2potency)):
                        file_output += f"{[idx2state[idx] for idx in idx2potency[i]]}"
                        file_output += "\n"
                    
                print(file_output)
                with open(f"{trial_dir}/inferred_potencies.txt", "w") as fp:
                    fp.write(file_output)


@cli.command()
def evaluate_experiment():
    """
    Usage example:
        python scripts/evaluate_results.py evaluate-experiment
    """

    working_dir = base_dir
    # experiment_name = "TLSC"
    experiment_name = "TLS"
    # experiment_name = "test_rates=10_num_trees=128_time=1.75_trial=0"
    # experiment_name = "fly"
    experiment_dir = f"{working_dir}/experiments/{experiment_name}"
    labels_are_ints = not experiment_name.startswith("TLS") and experiment_name != "fly"
    subsampling_rate = 1.0
    
    regs = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    
    trees_path = f"{experiment_dir}/processed_data/trees.pkl"
    tree_length = 1.0
    with open(trees_path, "rb") as fp:
        trees = pickle.load(fp)
        total_leaves = 0
        for tree in trees:
            tree_length = max(tree_length, tree.get_farthest_leaf()[1])
            actual_num_leaves = len(tree.get_leaves())
            total_leaves += actual_num_leaves
        cell_type_counts = Counter()
        for tree in trees:
            counter = Counter()
            for leaf in tree.get_leaves():
                counter[leaf.state] += 1
                cell_type_counts[leaf.state] += 1
        num_observed_states = len(cell_type_counts)
    terminal_labels = get_terminal_labels(f"{experiment_dir}/terminal_labels.txt", is_int_state=labels_are_ints)
    observed_potencies = get_observed_potencies(f"{experiment_dir}/observed_potencies.txt", is_int_state=labels_are_ints)

    losses = []
    used_regs = []
    num_reachable_states = []
    for regularization in regs:
        # results_dir = f"{experiment_dir}/results_vanilla_likelihood/reg={regularization}"
        results_dir = f"{experiment_dir}/results_vanilla_likelihood/reg={regularization}/select_potencies"
        figure_dir = f"{results_dir}/figures"
        os.makedirs(figure_dir, exist_ok=True)

        model_dict_path = f"{results_dir}/model_dict.pkl"
        if not os.path.isfile(model_dict_path):
            print("Skipping", model_dict_path)
            continue
        # state_dict_path = f"{results_dir}/state_dict.pth"
        # error_log_path = f"{experiment_dir}/results/reg={regularization}/inference.err"
        # if not os.path.isfile(model_dict_path) or \
        #     not os.path.isfile(state_dict_path) or \
        #     not os.path.isfile(error_log_path):
        #     print("Skipping:", model_dict_path)
        #     continue
        # # Skip if the run was cancelled or gradient blew up
        # with open(error_log_path, "r") as fp:
        #     first_line = fp.readline()
        #     if first_line.startswith("slurmstepd: error:") or \
        #     first_line.startswith("Traceback (most recent call last):"):
        #         print("[ERROR: run cancelled or exploding grad] Skipping:", model_dict_path)
        #         continue

        with open(model_dict_path, "rb") as fp:
            model_dict = pickle.load(fp)
            inferred_rate_matrix = model_dict["rate_matrix"]
            inferred_root_distribution = model_dict["root_distribution"]
            inferred_pi_params = torch.log(inferred_root_distribution * 1e20)
            inferred_growth_rates = model_dict["growth_rates"]
            idx2state = model_dict["idx2state"]
            state2idx = {state: idx for idx, state in idx2state.items()}
        prepped_trees = [_prep_log_tree(tree, len(inferred_rate_matrix), state2idx) for tree in trees]
        inferred_log_lik = log_vec_likelihood(prepped_trees,
                                            inferred_rate_matrix,
                                            inferred_pi_params,
                                            growth_rates=inferred_growth_rates,
                                            state2idx=state2idx,
                                            rho=subsampling_rate).item()
        inferred_rate_matrix = inferred_rate_matrix.detach().numpy()

        num_hidden_states = len(inferred_rate_matrix)-num_observed_states
        print(f"num_hidden: {num_hidden_states} \t infer: {inferred_log_lik}")
    
        node_labels = {i: str(idx2state[i]) for i in idx2state.keys()}
        if experiment_name.startswith("TLS"):
            node_colors = {
                "NeuralTube": color_list[0],
                "Somite": color_list[1],
                "Endoderm": color_list[2],
                "PCGLC": color_list[3],
                "Endothelial": color_list[4]
            }
        else:
            node_colors = {label: color_list[state2idx[label]] for label in terminal_labels}
        
        for idx in idx2state.keys():
            if idx2state[idx] not in observed_potencies:
                label = ""
                node_colors[label] = "#FFFFFF"   # White
                node_labels[idx] = label
            else:
                node_labels[idx] = idx2state[idx]
        
        state2potency = get_idx2potency(inferred_rate_matrix, tree_length=tree_length)


        inferred_output_path = f"{figure_dir}/inferred_graph.pdf"
        starting_idx = torch.argmax(inferred_root_distribution).detach().item()
        # print(inferred_root_distribution)
        # print("starting idx:", starting_idx)
        thresh = 0.001  # TODO: Make this adaptive to the scale of the rate matrix
        draw_weighted_graph(inferred_rate_matrix,
                            inferred_output_path,
                            thresh, node_labels,
                            node_colors,
                            state_text = f"log lik: {inferred_log_lik:3f}",
                            totipotent_state=starting_idx,
                            self_edges=inferred_growth_rates,
                            state2potency=state2potency)
                        
        num_reachable = len(get_reachable_idxs(inferred_rate_matrix, starting_idx, thresh))
        num_reachable_states.append(num_reachable)
        losses.append(-inferred_log_lik)
        used_regs.append(regularization)

    figure_outdir = f"{experiment_dir}/results_vanilla_likelihood/figures"
    os.makedirs(figure_outdir, exist_ok=True)

    # Set y-vals to be the minimum among all the same x-vals
    y_ = losses # [math.log(y_val) for y_val in neg_llh]
    x_ = num_reachable_states
    state2liks = {x_val: [] for x_val in x_}
    for x_val, y_val in zip(x_, y_):
        state2liks[x_val].append(y_val)
    y = []
    x = list(set(x_))
    x.sort()
    for x_val in x:
        y.append(min(state2liks[x_val]))
    print("num states:  ", x)
    print("likelihoods: ", y)
    kneedle = KneeLocator(x, y, S=0.5, curve="convex", direction="decreasing")

    # x = [math.log(reg) for reg in used_regs]
    # y = neg_llh
    # kneedle = KneeLocator(x, y, S=0.5, curve="convex", direction="increasing")
    # x = [round(math.exp(x_val), 5) for x_val in x]

    if kneedle.knee is not None:
        knee_x = kneedle.knee
        num_states_inferred = knee_x
        # knee_x = round(math.exp(kneedle.knee), 5)
        # num_states_inferred = reg2num_states[knee_x]
        knee_y = kneedle.knee_y
        print("Knee:", knee_x, knee_y)
    else:
        knee_x = "Error"
        knee_y = "Error"

    with open(f"{experiment_dir}/results_vanilla_likelihood/num_states_knee.txt", "w") as fp:
        fp.write(f"knee_x\t{knee_x}\n")                 # x-value of highest curvature
        fp.write(f"knee_y\t{knee_y}\n")                 # y-value of highest curvature
        fp.write(f"num_states\t{num_reachable_states}\n")
        fp.write(f"used_regs\t{used_regs}\n")
        fp.write(f"loss_vals_regs\t{losses}\n")

    plt.plot(x, y, marker='o', linewidth=4, markersize=15)
    plt.ylabel("Negative Log Likelihood")
    if kneedle.knee is not None:
        plt.plot([knee_x], [knee_y], marker='*', color ='red', markersize=25)
    plt.xlabel("Number of states")
    plt.savefig(f"{figure_outdir}/loss_vs_num_states.pdf")
    # plt.xlabel("Regularization strength")
    # plt.xscale('log')
    # plt.savefig(f"{figure_outdir}/loss_vs_regularization.pdf")
    plt.clf()

    # if len(used_regs) >= 2:
    #     # Plot likelihood vs regularization strength
    #     x_vals = used_regs
    #     plt.plot(x_vals, losses, marker='o', linewidth=4, markersize=12)
    #     plt.xlabel("Regularization")
    #     plt.ylabel("Loss (negative log likelihood)")
    #     plt.xscale("log")
    #     plt.savefig(f"{figure_outdir}/loss_vs_regularization.pdf", dpi=400)
    #     plt.clf()

    #     # Plot likelihood vs number of reachable states
    #     x_vals = num_reachable_states
    #     plt.plot(x_vals, losses, marker='o', linewidth=4, markersize=12)
    #     plt.xticks(list(range(min(x_vals), max(x_vals)+1)))
    #     plt.xlabel("Number of states")
    #     plt.ylabel("Loss (negative log likelihood)")
    #     plt.savefig(f"{figure_outdir}/loss_vs_num_progenitors.pdf", dpi=400)
    #     plt.clf()


@cli.command()
def plot_expected_population_composition():
    """
    Usage example:
        python scripts/evaluate_results.py plot-expected-population-composition
    """

    final_time = 120 # 2.35

    experiment_type = "TLSC"
    reg=0.3
    experiment_dir = f"{base_dir}/experiments/{experiment_type}"
    trial_dir = f"{experiment_dir}/results_vanilla_likelihood/reg={reg}/select_potencies"
    terminal2color = {
        'NeuralTube': '#1b9e77',
        'Somite': '#d95f02',
        'Endoderm': '#7570b3',
        'Endothelial': '#66a61e',
        'PCGLC': '#e7298a'
    }
    NMP_color = '#e6ab02'
    totipotent_color = '#666666'
    E_NT_S_color = '#decbe4'
    abrev = {
        'NeuralTube': "NT",
        'Somite': "S",
        'Endoderm': "E",
        'Endothelial': "T",
        'PCGLC': "P"
    }
    other_colors = {
        '#fbb4ae',
        '#b3cde3',
        '#ccebc5',
        '#a6761d'
    }
    terminals = get_terminal_labels(f"{experiment_dir}/terminal_labels.txt")
    observed_potencies = get_observed_potencies(f"{experiment_dir}/observed_potencies.txt")

    trees_path = f"{experiment_dir}/processed_data/trees.pkl"
    tree_length = 1.0
    with open(trees_path, "rb") as fp:
        trees = pickle.load(fp)
        total_leaves = 0
        for tree in trees:
            tree_length = max(tree_length, tree.get_farthest_leaf()[1])
            actual_num_leaves = len(tree.get_leaves())
            total_leaves += actual_num_leaves
        cell_type_counts = Counter()
        for tree in trees:
            counter = Counter()
            for leaf in tree.get_leaves():
                counter[leaf.state] += 1
                cell_type_counts[leaf.state] += 1

    model_dict_path = f"{trial_dir}/model_dict.pkl"
    with open(model_dict_path, "rb") as fp:
        model_dict = pickle.load(fp)
        inferred_rate_matrix = model_dict["rate_matrix"].detach().numpy()
        inferred_growth_rates = model_dict["growth_rates"].detach().numpy()
        idx2state = model_dict["idx2state"]
        starting_idx = torch.argmax(model_dict["root_distribution"]).detach().item()
    reachable_idxs = get_reachable_idxs(inferred_rate_matrix, starting_idx, threshold=0.00001)
    old2new_idx = {prev_idx: i for i, prev_idx in enumerate(reachable_idxs)}
    newidx2state = {old2new_idx[idx]: idx2state[idx] for idx in reachable_idxs}
    state2newidx = {state: newidx for newidx, state in newidx2state.items()}

    # TODO: Order the indexing by potency (0 is least potent, n is most potent)

    true_final = [cell_type_counts[newidx2state[i]] if newidx2state[i] in observed_potencies else 0 for i in range(len(reachable_idxs))]

    lam = inferred_growth_rates[reachable_idxs]
    Q = inferred_rate_matrix[reachable_idxs, :][:, reachable_idxs]
    initial_idx = old2new_idx[starting_idx]

    M = Q.T + np.diag(lam)

    times = np.linspace(0, final_time, 100)

    proportions = np.zeros((len(times), len(lam)))
    for idx, t in enumerate(times):
        m_t = expm(M * t / final_time)[:, initial_idx]
        proportions[idx] = m_t / m_t.sum()

    idx2label ={}
    idx2potency = get_idx2potency(Q, tree_length=100)
    for idx, potency in idx2potency.items():
        new_potency = [idx2state[s] for s in potency if idx2state[s] in terminal2color.keys()]
        print(f"new potency: {new_potency}")
        if len(new_potency) == 1:
            idx2label[idx] = new_potency[0]
        elif len(new_potency) == len(terminals):
            idx2label[idx] = "Totipotent"
        elif tuple(new_potency) == ('NeuralTube', 'Somite'):
            idx2label[idx] = "NMP"
        else:
            abreviated_potency = [abrev[s] for s in new_potency]
            abreviated_potency.sort()
            idx2label[idx] = "(" + ", ".join(abreviated_potency) + ")"
    
    color_list = []
    for idx in range(len(newidx2state)):
        state = newidx2state[idx]
        if state in terminal2color:
            color_list.append(terminal2color[state])
        elif idx2label[idx] == "Totipotent":
            color_list.append(totipotent_color)
        elif idx2label[idx] == "NMP":
            color_list.append(NMP_color)
        elif idx2label[idx] == "(E, NT, S)":
            color_list.append(E_NT_S_color)
        else:
            color_list.append(other_colors.pop())

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, ax = plt.subplots(dpi=400)

    # your stackplot
    ax.stackplot(times, proportions.T, colors=color_list)
    ax.set_xlabel('Time')
    ax.set_ylabel('Expected Proportion')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, max(times))
    ax.legend([f'{idx2label[i]}' for i in range(len(lam))], loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(left=False, bottom=False)

    # --- final proportion bars: axis that shares y with the main one ---
    exp_final = proportions[-1].astype(float); exp_final /= exp_final.sum()
    true_final = np.asarray(true_final, float); true_final /= true_final.sum()

    divider = make_axes_locatable(ax)
    axbars  = divider.append_axes("right", size="20%", pad=0.08, sharey=ax)

    # draw stacked bars
    w = 0.45
    x_exp, x_true = 0.0, 0.7

    bottom = 0.0
    for c, v in zip(color_list, exp_final):
        axbars.bar(x_exp, v, bottom=bottom, width=w, color=c, linewidth=0)
        bottom += v

    bottom = 0.0
    for c, v in zip(color_list, true_final):
        axbars.bar(x_true, v, bottom=bottom, width=w, color=c,
                edgecolor='k', linewidth=0.6, hatch='///', alpha=0.65)
        bottom += v

    # cosmetics for the right bar axis
    axbars.set_xlim(-0.5, 1.25)
    axbars.set_ylim(0, 1)                      # still shares y with ax
    axbars.set_xticks([x_exp, x_true])
    axbars.set_xticklabels(["Expected", "True"],
                        rotation=30, ha='right', rotation_mode='anchor')

    # remove the y-axis on the right bar plot
    axbars.set_yticks([])
    axbars.tick_params(axis='y', left=False, labelleft=False)
    for s in ("top", "right", "left"):
        axbars.spines[s].set_visible(False)

    plt.savefig(f"{trial_dir}/expected_pop.pdf", dpi=400, bbox_inches="tight", transparent=True)
    plt.clf()






@cli.command()
@click.option('-i', '--results-dir', required=True, type=click.Path(exists=True),
              help='Root results directory containing reg=X subdirectories.')
@click.option('--thresh', default=1e-3, show_default=True,
              help='Edge weight threshold; edges below this are hidden.')
@click.option('--use-select-potencies/--no-select-potencies', default=True,
              show_default=True,
              help='If set, plot from select_potencies/ when available.')
def plot_differentiation_maps(results_dir, thresh, use_select_potencies):
    """Plot differentiation maps for all reg=X results in a directory.

    Treats each inferred Q matrix as a weighted directed graph with
    self-edge weights equal to the inferred growth rates.

    Usage example:
        BASE_DIR=/Users/william_hs/Desktop/Projects/troupe;
        python scripts/evaluate_results.py plot-differentiation-maps \
            -i $BASE_DIR/example/results
    """
    reg_dirs = sorted(
        [d for d in os.listdir(results_dir)
         if os.path.isdir(os.path.join(results_dir, d)) and d.startswith('reg=')],
        key=lambda s: float(s.split('=')[1])
    )
    if not reg_dirs:
        click.echo(f"No reg=X directories found in {results_dir}")
        return

    for reg_dir_name in reg_dirs:
        reg_path = os.path.join(results_dir, reg_dir_name)

        # Choose select_potencies if available and requested
        sp_path = os.path.join(reg_path, 'select_potencies')
        if use_select_potencies and os.path.isdir(sp_path) \
                and os.path.isfile(os.path.join(sp_path, 'model_dict.pkl')):
            model_dir = sp_path
            suffix = 'select_potencies'
        elif os.path.isfile(os.path.join(reg_path, 'model_dict.pkl')):
            model_dir = reg_path
            suffix = 'full'
        else:
            click.echo(f"  Skipping {reg_dir_name}: no model_dict.pkl found")
            continue

        model_dict_path = os.path.join(model_dir, 'model_dict.pkl')
        with open(model_dict_path, 'rb') as fp:
            model_dict = pickle.load(fp)

        rate_matrix = model_dict['rate_matrix'].detach().numpy()
        root_distribution = model_dict['root_distribution']
        growth_rates = model_dict['growth_rates'].detach().numpy()
        idx2state = model_dict['idx2state']
        idx2potency = model_dict.get('idx2potency', None)

        starting_idx = torch.argmax(root_distribution).detach().item()

        # Determine terminal states from idx2potency (self-only potency)
        terminal_idxs = []
        if idx2potency is not None:
            for idx, potency in idx2potency.items():
                state = idx2state[idx]
                if potency == (state,) or potency == (idx,):
                    terminal_idxs.append(idx)
        terminal_idxs.sort()

        # Build node labels and colors
        node_labels = {i: str(idx2state[i]) for i in idx2state}
        node_colors = {}
        terminal_color_idx = 0
        for idx in sorted(idx2state.keys()):
            state = str(idx2state[idx])
            if idx in terminal_idxs:
                node_colors[state] = color_list[terminal_color_idx % len(color_list)]
                terminal_color_idx += 1
            else:
                node_colors[state] = '#FFFFFF'

        # Build state2potency for wedge coloring
        state2potency = get_idx2potency(rate_matrix)
        # Filter potencies to terminal states only
        state2potency_filtered = {}
        for idx, potency in state2potency.items():
            filtered = [s for s in potency if s in terminal_idxs]
            if filtered:
                filtered.sort()
                state2potency_filtered[idx] = tuple(filtered)

        # Output
        figure_dir = os.path.join(model_dir, 'figures')
        os.makedirs(figure_dir, exist_ok=True)
        outfile = os.path.join(figure_dir, 'differentiation_map.pdf')

        draw_weighted_graph(
            rate_matrix,
            outfile,
            thresh,
            node_labels,
            node_colors,
            totipotent_state=starting_idx,
            self_edges=growth_rates,
            state2potency=state2potency_filtered,
            terminal_idxs=terminal_idxs if terminal_idxs else None,
        )
        click.echo(f"  {reg_dir_name}/{suffix} -> {outfile}")

    click.echo("Done.")


if __name__ == '__main__':
    cli()