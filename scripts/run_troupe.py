#!/usr/bin/env python
"""Unified TROUPE inference pipeline.

Runs the full two-phase inference pipeline:
  Phase 1: Overparameterized MLE with L1 regularization (per reg value)
  Phase 2: Potency extraction + debiased MLE (per reg value)
  Model Selection: Knee-finding on neg-llh vs number of states

Example usage:
    python scripts/run_troupe.py \
        -i /n/fs/ragr-research/users/wh8114/projects/cell-diff-via-ml/simulated_data/branching_process_experiment/18/trees_64/time_8.0/trial_2/trees.pkl \
        -o /n/fs/ragr-research/users/wh8114/projects/cell-diff-via-ml/simulated_data/branching_process_experiment/tmp/results/18/trees_64/time_8.0/trial_2/trees.pkl \
        --regularizations 0.001 0.01 0.1 1.0 10 100
"""

import argparse
import copy
import logging
import os
import pickle
import shutil
import sys
import warnings
from collections import Counter

import torch
from ete3 import Tree, TreeNode
from kneed import KneeLocator

from likelihood import log_vec_likelihood, _prep_log_tree
from optimizer import compute_mle
from utils import (
    binarize_tree,
    get_terminal_labels,
    get_observed_potencies,
    get_idx2potency,
    get_reachable_idxs,
)

sys.setrecursionlimit(5000)
torch.set_default_dtype(torch.float64)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Load trees
# ---------------------------------------------------------------------------

def load_trees(input_path, newick_format=1):
    """Load trees from a .pkl or .nwk file and binarize them.

    Args:
        input_path: Path to pickled tree list (.pkl) or newick file (.nwk).
        newick_format: ete3 format code for newick parsing.

    Returns:
        List of binarized ete3 TreeNode objects with .state on each leaf.
    """
    if input_path.endswith(".pkl"):
        with open(input_path, "rb") as fp:
            trees = pickle.load(fp)
    elif input_path.endswith(".nwk"):
        trees = []
        with open(input_path, "r") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                t = Tree(line, format=newick_format)
                for leaf in t.get_leaves():
                    leaf.add_feature("state", leaf.name)
                trees.append(t)
    else:
        raise ValueError(f"Unsupported input format: {input_path}. Use .pkl or .nwk")

    trees = [binarize_tree(t) for t in trees]
    return trees


# ---------------------------------------------------------------------------
# 2. Detect terminal labels
# ---------------------------------------------------------------------------

def detect_labels(trees):
    """Scan all leaf states to determine terminal labels and type.

    Args:
        trees: List of ete3 trees with .state on leaves.

    Returns:
        (terminal_labels, is_int_state) where terminal_labels is a sorted
        list of unique leaf states.
    """
    states = set()
    for tree in trees:
        for leaf in tree.get_leaves():
            states.add(leaf.state)
    first_leaf = list(trees[0].get_leaves())[0]
    is_int_state = isinstance(first_leaf.state, int)
    terminal_labels = sorted(states)
    return terminal_labels, is_int_state


# ---------------------------------------------------------------------------
# 3. Auto-generate observed potencies
# ---------------------------------------------------------------------------

def auto_observed_potencies(terminal_labels):
    """Create default singleton potency for each terminal label.

    Args:
        terminal_labels: List of terminal state labels.

    Returns:
        Dict mapping each state to a 1-tuple of itself.
    """
    return {state: (state,) for state in terminal_labels}


# ---------------------------------------------------------------------------
# 4. Generate potency sets
# ---------------------------------------------------------------------------

def generate_potency_sets(trees, terminal_labels, observed_potencies, max_hidden):
    """Generate candidate potency sets from tree clades.

    Computes induced clades via postorder traversal, then adds complementary
    potencies (remove one element, add one element). Ensures all observed
    potencies are present and caps unobserved potencies at max_hidden.

    Args:
        trees: List of ete3 trees.
        terminal_labels: List of terminal state labels.
        observed_potencies: Dict mapping state -> potency tuple.
        max_hidden: Maximum number of unobserved (hidden) potency sets.

    Returns:
        List of potency tuples, sorted by size descending.
    """
    terminal_set = set(terminal_labels)

    # Compute induced clades from trees
    induced_potencies = set()
    for tree in trees:
        for node in tree.traverse("postorder"):
            if node.is_leaf():
                if node.state in terminal_set:
                    clade = frozenset([node.state])
                else:
                    clade = frozenset()
                node.add_feature("clade", clade)
            else:
                clade = frozenset()
                for child in node.get_children():
                    clade = clade | child.clade
                node.add_feature("clade", clade)

            if len(clade) > 0:
                induced_potencies.add(tuple(sorted(clade)))

    logger.info("Trees have %d induced clades", len(induced_potencies))

    # Complementary potencies: remove one element, add one element
    potencies_to_add = set()
    for potency in induced_potencies:
        if len(potency) > 1:
            for state in potency:
                sub = list(potency)
                sub.remove(state)
                potencies_to_add.add(tuple(sub))
        for state in terminal_labels:
            if state not in potency:
                sup = sorted(list(potency) + [state])
                potencies_to_add.add(tuple(sup))

    all_potencies = induced_potencies | potencies_to_add
    potency_list = list(all_potencies)

    logger.info("Found %d total potencies (induced + complementary)", len(potency_list))

    # Ensure all observed potencies are present
    for potency in observed_potencies.values():
        if potency not in potency_list:
            potency_list.append(potency)
            logger.warning("Observed potency %s not in induced set, adding it", potency)

    # Cap unobserved potencies at max_hidden
    observed_potency_set = set(observed_potencies.values())
    observed_pots = [p for p in potency_list if p in observed_potency_set]
    unobserved_pots = [p for p in potency_list if p not in observed_potency_set]
    unobserved_pots.sort(key=len, reverse=True)
    if len(unobserved_pots) > max_hidden:
        logger.info("Capping unobserved potencies from %d to %d", len(unobserved_pots), max_hidden)
        unobserved_pots = unobserved_pots[:max_hidden]

    result = observed_pots + unobserved_pots
    result.sort(key=len, reverse=True)
    return result


# ---------------------------------------------------------------------------
# 5. Build model info
# ---------------------------------------------------------------------------

def build_model_info(states, terminal_labels, observed_potencies, potency_sets):
    """Construct model_info dict and state2idx mapping for Phase 1.

    Observed states are indexed first (sorted), then hidden states are
    named U{idx}. Builds idx2potency by assigning observed potencies to
    their states and remaining potencies to hidden states.

    Args:
        states: Set of observed leaf states.
        terminal_labels: List of terminal state labels.
        observed_potencies: Dict mapping state -> potency tuple.
        potency_sets: List of all potency tuples (observed + hidden).

    Returns:
        (model_info, state2idx) tuple.
    """
    num_obs = len(states)
    num_hidden = len(potency_sets) - len(observed_potencies)
    num_states = num_obs + num_hidden

    # Observed states first (sorted), then hidden
    state_list = sorted(states)
    state2idx = {state: i for i, state in enumerate(state_list)}

    observed_idxs = set(state2idx.values())
    all_idxs = set(range(num_states))
    unobserved_idxs = sorted(all_idxs - observed_idxs)
    for idx in unobserved_idxs:
        state2idx[f"U{idx}"] = idx
    idx2state = {i: state for state, i in state2idx.items()}

    # Build idx2potency
    idx2potency = {}
    unobserved_potencies = list(potency_sets)
    for potency in observed_potencies.values():
        i = unobserved_potencies.index(potency)
        unobserved_potencies.pop(i)

    for idx, state in idx2state.items():
        if state in observed_potencies:
            idx2potency[idx] = observed_potencies[state]
        else:
            idx2potency[idx] = unobserved_potencies.pop(-1)

    # Determine start state (unique most-potent state)
    potency_lengths = [len(p) for p in idx2potency.values()]
    max_len = max(potency_lengths)
    start_state = None
    if potency_lengths.count(max_len) == 1:
        for idx, potency in idx2potency.items():
            if len(potency) == max_len:
                start_state = idx
                break

    model_info = {
        "idx2potency": idx2potency,
        "idx2state": idx2state,
        "start_state": start_state,
        "terminal_states": [state2idx[s] for s in terminal_labels],
        "optimize_growth": True,
    }

    logger.info("idx2state: %s", idx2state)
    logger.info("Start state: %s", start_state)

    return model_info, state2idx


# ---------------------------------------------------------------------------
# 6. Phase 1: Overparameterized MLE
# ---------------------------------------------------------------------------

def run_phase1(trees, model_info, state2idx, num_obs, num_hidden, reg, output_dir, device):
    """Run Phase 1 (overparameterized) MLE for a single regularization value.

    Args:
        trees: Original trees (will be deep-copied and relabeled).
        model_info: Model configuration dict.
        state2idx: Mapping from state labels to integer indices.
        num_obs: Number of observed states.
        num_hidden: Number of hidden states.
        reg: L1 regularization strength.
        output_dir: Directory for this run's outputs.
        device: Torch device.

    Returns:
        output_dir on success, None on failure.
    """
    trees_copy = copy.deepcopy(trees)
    for tree in trees_copy:
        for leaf in tree.get_leaves():
            leaf.state = state2idx[leaf.state]

    os.makedirs(output_dir, exist_ok=True)
    try:
        llh, loss = compute_mle(
            trees_copy,
            (num_obs, num_hidden),
            device,
            output_dir,
            l1_regularization_strength=reg,
            model_type="PureBirthLikelihoodModel",
            model_info=model_info,
        )
        with open(f"{output_dir}/loss.txt", "w") as fp:
            fp.write(f"{loss}")
        return output_dir
    except Exception as e:
        logger.error("Phase 1 failed for reg=%s: %s", reg, e)
        return None


# ---------------------------------------------------------------------------
# 7. Phase 2: Potency extraction + debiased MLE
# ---------------------------------------------------------------------------

def run_phase2(phase1_dir, trees, terminal_labels, observed_potencies,
               is_int_state, debiasing_l1, threshold, device):
    """Extract potencies from Phase 1 model and run debiased Phase 2 MLE.

    Loads the Phase 1 model, finds reachable states, computes potencies,
    builds a reduced state space, and re-runs MLE with warm-start
    initialization.

    Args:
        phase1_dir: Directory containing Phase 1 model_dict.pkl.
        trees: Original trees (will be deep-copied and relabeled).
        terminal_labels: List of terminal state labels.
        observed_potencies: Dict mapping state -> potency tuple.
        is_int_state: Whether states are integers.
        debiasing_l1: L1 regularization for Phase 2.
        threshold: Reachability threshold for get_reachable_idxs.
        device: Torch device.

    Returns:
        Phase 2 output directory on success, None on failure.
    """
    model_dict_path = f"{phase1_dir}/model_dict.pkl"
    if not os.path.isfile(model_dict_path):
        logger.error("Phase 1 model_dict not found: %s", model_dict_path)
        return None

    try:
        with open(model_dict_path, "rb") as fp:
            model_dict = pickle.load(fp)

        inferred_rate_matrix = model_dict["rate_matrix"]
        inferred_growth_rates = model_dict["growth_rates"]
        inferred_rate_matrix_np = inferred_rate_matrix.detach().numpy()
        idx2state = model_dict["idx2state"]
        starting_idx = torch.argmax(model_dict["root_distribution"]).detach().item()

        # Find reachable states
        reachable_idxs = get_reachable_idxs(
            inferred_rate_matrix_np, starting_idx, threshold=threshold
        )
        logger.info("Phase 2: %d reachable states from starting idx %d",
                     len(reachable_idxs), starting_idx)

        # Remap indices
        terminal_set = set(terminal_labels)
        old2newidx = {old_idx: i for i, old_idx in enumerate(reachable_idxs)}
        newidx2state = {old2newidx[idx]: idx2state[idx] for idx in reachable_idxs}
        state2newidx = {state: newidx for newidx, state in newidx2state.items()}

        # Compute potencies for reachable states (filtered to terminals only)
        idx2potency_ = get_idx2potency(inferred_rate_matrix_np, tree_length=10)
        idx2potency = {}
        for idx in reachable_idxs:
            potency_ = idx2potency_[idx]
            potency = sorted(
                [idx2state[s] for s in potency_ if idx2state[s] in terminal_set]
            )
            if len(potency) == 0:
                continue
            idx2potency[old2newidx[idx]] = tuple(potency)

        # Ensure all observed potencies are present
        for state, state_potency in observed_potencies.items():
            if state not in state2newidx:
                logger.warning("Observed state %s not reachable, skipping", state)
                continue
            if state2newidx[state] not in idx2potency:
                potency = tuple(sorted(state_potency))
                idx2potency[state2newidx[state]] = potency

        # Subset Q and growth rates to reachable states
        lam = inferred_growth_rates[reachable_idxs]
        Q = inferred_rate_matrix[reachable_idxs, :][:, reachable_idxs]
        initial_idx = old2newidx[starting_idx]

        # Invert softplus for warm-start initialization
        Q = Q.fill_diagonal_(0).clamp_min(1e-20)
        Q_params_init = torch.log(torch.exp(Q) - 1)
        Q_params_init = Q_params_init.fill_diagonal_(0)
        growth_params_init = torch.log(torch.exp(lam.clamp_min(1e-2)) - 1)

        # Count observed vs hidden in new state space
        leaf_states = set()
        for tree in trees:
            for leaf in tree.get_leaves():
                leaf_states.add(leaf.state)
        num_obs_new = sum(1 for s in newidx2state.values() if s in leaf_states)
        num_hidden_new = len(newidx2state) - num_obs_new

        # Build Phase 2 model_info
        phase2_model_info = {
            "Q_params_init": Q_params_init,
            "growth_params_init": growth_params_init,
            "start_state": initial_idx,
            "idx2state": newidx2state,
            "idx2potency": idx2potency,
            "terminal_states": [
                state2newidx[s] for s in terminal_labels if s in state2newidx
            ],
            "optimize_growth": True,
        }

        # Relabel trees with new state2idx
        trees_copy = copy.deepcopy(trees)
        for tree in trees_copy:
            for leaf in tree.get_leaves():
                leaf.state = state2newidx[leaf.state]

        output_dir = f"{phase1_dir}/select_potencies"
        os.makedirs(output_dir, exist_ok=True)

        llh, loss = compute_mle(
            trees_copy,
            (num_obs_new, num_hidden_new),
            device,
            output_dir,
            l1_regularization_strength=debiasing_l1,
            model_type="PureBirthLikelihoodModel",
            model_info=phase2_model_info,
        )
        with open(f"{output_dir}/loss.txt", "w") as fp:
            fp.write(f"{loss}")

        return output_dir

    except Exception as e:
        logger.error("Phase 2 failed for %s: %s", phase1_dir, e)
        return None


# ---------------------------------------------------------------------------
# 8. Model selection via knee finding
# ---------------------------------------------------------------------------

def run_model_selection(output_dir, trees, reg_values, threshold, knee_sensitivity):
    """Select the best model across regularization values using knee detection.

    For each reg, loads the Phase 2 model (or Phase 1 for reg=0), computes
    the negative log-likelihood, and counts reachable states. Deduplicates
    by state count (keeping min neg-llh), then runs KneeLocator.

    Args:
        output_dir: Base output directory containing reg=X subdirs.
        trees: Original trees for likelihood evaluation.
        reg_values: List of regularization values tested.
        threshold: Reachability threshold.
        knee_sensitivity: KneeLocator S parameter.

    Returns:
        Dict with keys: knee_num_states, knee_loss, best_reg,
        best_model_dir, all_results.

    Raises:
        RuntimeError: If all regularization runs failed.
    """
    results = []

    for reg in reg_values:
        if reg > 0:
            model_dir = f"{output_dir}/reg={reg}/select_potencies"
        else:
            model_dir = f"{output_dir}/reg={reg}"

        model_dict_path = f"{model_dir}/model_dict.pkl"
        if not os.path.isfile(model_dict_path):
            logger.warning("Skipping reg=%s: model_dict not found at %s", reg, model_dict_path)
            continue

        try:
            with open(model_dict_path, "rb") as fp:
                model_dict = pickle.load(fp)

            rate_matrix = model_dict["rate_matrix"]
            root_distribution = model_dict["root_distribution"]
            pi_params = torch.log(root_distribution * 1e20)
            growth_rates = model_dict["growth_rates"]
            idx2state_model = model_dict["idx2state"]
            state2idx_model = {s: i for i, s in idx2state_model.items()}

            num_model_states = len(rate_matrix)
            prepped = [
                _prep_log_tree(t, num_model_states, state2idx_model)
                for t in trees
            ]
            log_lik = log_vec_likelihood(
                prepped,
                rate_matrix,
                pi_params,
                growth_rates=growth_rates,
                state2idx=state2idx_model,
            ).item()

            rate_matrix_np = rate_matrix.detach().numpy()
            starting_idx = torch.argmax(root_distribution).detach().item()
            reachable = get_reachable_idxs(rate_matrix_np, starting_idx, threshold)

            results.append({
                "reg": reg,
                "num_states": len(reachable),
                "neg_llh": -log_lik,
                "model_dir": model_dir,
            })
            logger.info("reg=%s: %d reachable states, neg-llh=%.4f",
                        reg, len(reachable), -log_lik)

        except Exception as e:
            logger.warning("Model selection failed for reg=%s: %s", reg, e)

    if not results:
        raise RuntimeError(
            "All regularization runs failed. No models available for selection."
        )

    # Single result: return it directly
    if len(results) == 1:
        r = results[0]
        return {
            "knee_num_states": r["num_states"],
            "knee_loss": r["neg_llh"],
            "best_reg": r["reg"],
            "best_model_dir": r["model_dir"],
            "all_results": results,
        }

    # Deduplicate: for each distinct state count, keep minimum neg-llh
    state2best = {}
    for r in results:
        ns = r["num_states"]
        if ns not in state2best or r["neg_llh"] < state2best[ns]["neg_llh"]:
            state2best[ns] = r

    x = sorted(state2best.keys())
    y = [state2best[ns]["neg_llh"] for ns in x]

    logger.info("Model selection: num_states=%s, neg_llh=%s", x, y)

    # Run knee detection
    knee_result = None
    if len(x) >= 2:
        kneedle = KneeLocator(
            x, y, S=knee_sensitivity, curve="convex", direction="decreasing"
        )
        if kneedle.knee is not None:
            knee_ns = kneedle.knee
            knee_result = state2best[knee_ns]
            logger.info("Knee found at %d states (neg-llh=%.4f)",
                        knee_ns, knee_result["neg_llh"])

    if knee_result is None:
        logger.warning("No knee found. Falling back to model with lowest neg-llh.")
        knee_result = min(results, key=lambda r: r["neg_llh"])

    return {
        "knee_num_states": knee_result["num_states"],
        "knee_loss": knee_result["neg_llh"],
        "best_reg": knee_result["reg"],
        "best_model_dir": knee_result["model_dir"],
        "all_results": results,
    }


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the full TROUPE inference pipeline."
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to .pkl (pickled trees) or .nwk (newick, one tree per line)",
    )
    parser.add_argument(
        "-o", "--output_dir", required=True,
        help="Base output directory",
    )
    parser.add_argument(
        "--terminal_labels", type=str, default=None,
        help="Path to terminal labels file. Default: auto-detect from leaf states",
    )
    parser.add_argument(
        "--observed_potencies", type=str, default=None,
        help="Path to observed potencies file. Default: each terminal maps to (itself,)",
    )
    parser.add_argument(
        "--regularizations", type=float, nargs="+",
        default=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30],
        help="List of L1 regularization values",
    )
    parser.add_argument(
        "--max_hidden_states", type=int, default=1000,
        help="Max hidden states for potency generation",
    )
    parser.add_argument(
        "--debiasing_l1", type=float, default=0.0001,
        help="L1 regularization for Phase 2 (debiasing)",
    )
    parser.add_argument(
        "--reachability_threshold", type=float, default=0.00001,
        help="Threshold for get_reachable_idxs",
    )
    parser.add_argument(
        "--knee_sensitivity", type=float, default=0.5,
        help="KneeLocator S parameter",
    )
    parser.add_argument(
        "--newick_format", type=int, default=1,
        help="ete3 format for newick parsing",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device (cpu or cuda)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # --- Load trees ---
    logger.info("Loading trees from %s", args.input)
    trees = load_trees(args.input, newick_format=args.newick_format)
    logger.info("Loaded %d trees", len(trees))

    # --- Detect or load terminal labels ---
    detected_labels, is_int_state = detect_labels(trees)

    if args.terminal_labels:
        terminal_labels = get_terminal_labels(args.terminal_labels, is_int_state)
    else:
        terminal_labels = detected_labels
    logger.info("Terminal labels (%d): %s", len(terminal_labels), terminal_labels)

    # --- Load or auto-generate observed potencies ---
    if args.observed_potencies:
        observed_potencies = get_observed_potencies(
            args.observed_potencies, is_int_state
        )
    else:
        observed_potencies = auto_observed_potencies(terminal_labels)
    logger.info("Observed potencies (%d): %s",
                len(observed_potencies), observed_potencies)

    # --- Collect observed states from trees ---
    states = set()
    for tree in trees:
        for leaf in tree.get_leaves():
            states.add(leaf.state)

    # --- Generate potency sets ---
    logger.info("Generating potency sets...")
    potency_sets = generate_potency_sets(
        trees, terminal_labels, observed_potencies, args.max_hidden_states
    )
    num_hidden = len(potency_sets) - len(observed_potencies)
    num_obs = len(states)
    logger.info("Potency sets: %d total (%d observed, %d hidden)",
                len(potency_sets), len(observed_potencies), num_hidden)

    # --- Build model info ---
    model_info, state2idx = build_model_info(
        states, terminal_labels, observed_potencies, potency_sets
    )

    # --- Run inference for each regularization value ---
    os.makedirs(args.output_dir, exist_ok=True)

    for reg in args.regularizations:
        reg_dir = f"{args.output_dir}/reg={reg}"
        logger.info("=" * 60)
        logger.info("Phase 1: reg=%s", reg)
        logger.info("=" * 60)
        phase1_ok = run_phase1(
            trees, model_info, state2idx, num_obs, num_hidden,
            reg, reg_dir, device,
        )

        if phase1_ok and reg > 0:
            logger.info("=" * 60)
            logger.info("Phase 2: reg=%s", reg)
            logger.info("=" * 60)
            run_phase2(
                reg_dir, trees, terminal_labels, observed_potencies,
                is_int_state, args.debiasing_l1,
                args.reachability_threshold, device,
            )

    # --- Model selection ---
    logger.info("=" * 60)
    logger.info("Model selection")
    logger.info("=" * 60)
    selection = run_model_selection(
        args.output_dir, trees, args.regularizations,
        args.reachability_threshold, args.knee_sensitivity,
    )

    # --- Copy best model ---
    best_src = f"{selection['best_model_dir']}/model_dict.pkl"
    best_dst = f"{args.output_dir}/best_model_dict.pkl"
    shutil.copy2(best_src, best_dst)
    logger.info("Best model copied to %s", best_dst)

    # --- Write summary ---
    summary_path = f"{args.output_dir}/troupe_summary.txt"
    with open(summary_path, "w") as fp:
        fp.write(f"best_reg\t{selection['best_reg']}\n")
        fp.write(f"best_model_dir\t{selection['best_model_dir']}\n")
        fp.write(f"knee_num_states\t{selection['knee_num_states']}\n")
        fp.write(f"knee_loss\t{selection['knee_loss']}\n")
        fp.write(
            f"num_regularizations_tested\t{len(selection['all_results'])}\n"
        )
        fp.write("\nAll results:\n")
        fp.write("reg\tnum_states\tneg_llh\n")
        for r in selection["all_results"]:
            fp.write(f"{r['reg']}\t{r['num_states']}\t{r['neg_llh']}\n")

    logger.info("Summary written to %s", summary_path)
    logger.info(
        "Done! Best model: reg=%s, %d states, neg-llh=%.4f",
        selection["best_reg"],
        selection["knee_num_states"],
        selection["knee_loss"],
    )


if __name__ == "__main__":
    main()
