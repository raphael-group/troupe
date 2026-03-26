#!/usr/bin/env python
"""Unified ClaSSE-TROUPE inference pipeline.

Runs the full two-phase inference pipeline using the ClaSSE (Cladogenetic
State-change Speciation and Extinction) likelihood model.  In this model
cells do not change type during their lifetime; type changes occur only at
division, where each daughter draws its type from row i of a row-stochastic
birth kernel B.

  Phase 1: Overparameterized MLE with L1 regularization (per reg value)
  Phase 2: Potency extraction + debiased MLE (per reg value)
  Model Selection: Knee-finding on neg-llh vs number of states

Example usage:
    python scripts/run_classe_troupe.py \
        -i /Users/william_hs/Desktop/Projects/troupe/experiments/subsampled_leaves_4_terminals/trees_64/time_5.0/sample_0.1/trial_0/trees.pkl \
        -o tmp/classe_results_4_terminals \
        --regularizations 0.01 0.1 1.0 \
        --sampling_probability 0.1
"""

import argparse
import copy
import logging
import os
import pickle
import shutil
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from ete3 import Tree, TreeNode
from kneed import KneeLocator

from classe_model import ClaSSELikelihoodModel
from optimizer import constant_rate_mle
from utils import (
    binarize_tree,
    get_terminal_labels,
    get_observed_potencies,
    get_reachable_idxs,
)

sys.setrecursionlimit(5000)

dtype = torch.float64
torch.set_default_dtype(dtype)
EPS = 1e-30

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_softplus_inverse(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, min=EPS)
    return torch.log(torch.expm1(x))


def _uniform_kernel_logits(idx2potency, n_states, device):
    """Return 2-D logit matrix giving uniform B over allowed (i,j) entries."""
    logits = torch.full((n_states, n_states), -1e30, device=device, dtype=dtype)
    if idx2potency is None:
        logits[:] = 0.0
        return logits
    for i, potency in idx2potency.items():
        allowed = [
            j for j, child_potency in idx2potency.items()
            if all(t in potency for t in child_potency)
        ]
        logits[i, allowed] = 0.0
    return logits


def _get_idx2potency_classe(B_np, eps=1e-4, n_steps=100):
    """Infer potency sets from a row-stochastic birth kernel via matrix power.

    Computes B^n_steps to find the long-run reachability.

    Args:
        B_np: (K, K) numpy array — row-stochastic birth kernel.
        eps: Minimum probability to consider a state reachable.
        n_steps: Number of generations to simulate (power of B).

    Returns:
        Dict mapping each state index to a sorted tuple of reachable indices.
    """
    B_power = np.linalg.matrix_power(B_np, n_steps)
    n = len(B_power)
    potency_map = {}
    for i in range(n):
        reachable = [j for j in range(n) if B_power[i, j] > eps]
        potency_map[i] = tuple(sorted(reachable))
    return potency_map


def _compute_classe_mle(
    trees_labeled,
    n_states,
    device,
    output_dir,
    model_info,
    sampling_prob,
    l1_reg=0.0,
    do_logging=True,
    num_iter=50,
    log_iter=1,
):
    """Fit a ClaSSELikelihoodModel via LBFGS and save results.

    Args:
        trees_labeled: Trees with integer leaf states.
        n_states: Total number of states.
        device: Torch device.
        output_dir: Directory to save model_dict.pkl and state_dict.pth.
        model_info: Dict with idx2potency, idx2state, start_state,
            optimize_growth, and optional B_params_init / growth_params_init.
        sampling_prob: Leaf sampling probability eta in (0, 1].
        l1_reg: L1 regularization strength on off-diagonal B entries.
        do_logging: Whether to log progress every log_iter steps.
        num_iter: Maximum number of LBFGS outer iterations.
        log_iter: Logging interval (iterations).

    Returns:
        (llh, final_loss) tuple.
    """
    idx2potency = model_info["idx2potency"]
    idx2state = model_info["idx2state"]
    start_state = model_info.get("start_state")
    optimize_growth = model_info.get("optimize_growth", True)
    integration_max_step = model_info.get("integration_max_step", 0.05)
    ode_method = model_info.get("ode_method", "Dopri5")
    ode_atol = model_info.get("ode_atol", 1e-8)
    ode_rtol = model_info.get("ode_rtol", 1e-6)
    backend = model_info.get("backend", "fundamental")

    # Initial birth-kernel logits
    if "B_params_init" in model_info:
        bk_params_init = model_info["B_params_init"].to(device=device, dtype=dtype)
    else:
        bk_params_init = _uniform_kernel_logits(idx2potency, n_states, device)

    # Initial growth-rate params
    if "growth_params_init" in model_info:
        growth_params_init = model_info["growth_params_init"].to(device=device, dtype=dtype)
    else:
        lam0 = constant_rate_mle(trees_labeled)
        growth_params_init = _safe_softplus_inverse(
            torch.ones(n_states, device=device, dtype=dtype) * lam0
        )

    pi_params_init = torch.zeros(n_states, device=device, dtype=dtype)

    llh = ClaSSELikelihoodModel(
        trees_labeled,
        n_states,
        bk_params_init,
        pi_params_init,
        growth_params_init,
        optimize_growth=optimize_growth,
        idx2potency=idx2potency,
        device=device,
        idx2state=idx2state,
        start_state=start_state,
        sampling_prob=float(sampling_prob),
        integration_max_step=integration_max_step,
        ode_method=ode_method,
        ode_atol=ode_atol,
        ode_rtol=ode_rtol,
        backend=backend,
    )

    tree_idxs = list(range(len(trees_labeled)))
    changeable_params = [p for p in llh.parameters(recurse=True) if p.requires_grad]
    optimizer = optim.LBFGS(
        changeable_params,
        lr=0.1,
        max_iter=1,
        max_eval=2,
        line_search_fn="strong_wolfe",
    )

    rel_loss_thresh = 1e-5 / len(trees_labeled)
    losses = []
    closure_state = {"last_loss": None}

    def _loss():
        total = sum(-llh(j) for j in tree_idxs) / len(tree_idxs)
        if l1_reg > 0:
            B = llh.get_daughter_kernel()
            mask = ~torch.eye(n_states, dtype=torch.bool, device=device)
            total = total + l1_reg * torch.sum(torch.abs(B[mask]))
        return total

    def closure():
        optimizer.zero_grad()
        llh.precompute_ode()
        objective = _loss()
        if not torch.isfinite(objective):
            llh.clear_ode_cache()
            raise RuntimeError(f"Non-finite loss: {objective.item()}")
        objective.backward()
        closure_state["last_loss"] = float(objective.detach().item())
        llh.clear_ode_cache()
        return objective

    start = time.time()
    for i in range(num_iter):
        optimizer.step(closure)
        if closure_state["last_loss"] is None:
            raise RuntimeError("LBFGS step did not evaluate the objective")
        loss_value = closure_state["last_loss"]

        with torch.no_grad():
            losses.append(loss_value)
            B = llh.get_daughter_kernel()
            pi = llh.get_root_distribution()
            if torch.isnan(B).any() or torch.isnan(pi).any():
                raise ValueError("NaN in parameters or loss")

        if losses[-1] <= min(losses) and len(losses) >= 2:
            with torch.no_grad():
                os.makedirs(output_dir, exist_ok=True)
                torch.save(llh.state_dict(), f"{output_dir}/state_dict.pth")
                _save_classe_model_dict(llh, model_info, sampling_prob, output_dir)

        if do_logging and i % log_iter == 0:
            logger.info(
                "Iter %d | loss=%.6f | B diag=%s | lam=%s",
                i, loss_value,
                B.diag().detach().tolist(),
                llh.get_growth_rates().detach().tolist(),
            )
            elapsed = time.time() - start
            logger.info("  %.4f s/iter", elapsed / min(i + 1, log_iter))
            start = time.time()

        if len(losses) > 2:
            rel = abs(losses[-1] - losses[-2]) / (abs(losses[-2]) + EPS)
            if rel <= rel_loss_thresh:
                logger.info("Converged at iteration %d (rel_loss=%.2e)", i, rel)
                break

    # Always save final model
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        torch.save(llh.state_dict(), f"{output_dir}/state_dict.pth")
        _save_classe_model_dict(llh, model_info, sampling_prob, output_dir)

    with open(f"{output_dir}/loss.txt", "w") as fp:
        fp.write(f"{losses[-1]}")

    return llh, losses[-1]


def _save_classe_model_dict(llh, model_info, sampling_prob, output_dir):
    """Serialize model parameters and metadata to model_dict.pkl."""
    B = llh.get_daughter_kernel()
    model_dict = {
        "rate_matrix": B,           # alias for compatibility with reachability utils
        "daughter_kernel": B,
        "growth_rates": llh.get_growth_rates(),
        "root_distribution": llh.get_root_distribution(),
        "sampling_probability": llh.get_sampling_probability(),
        "sampling_prob_float": float(sampling_prob),
        "idx2state": model_info["idx2state"],
        "idx2potency": model_info["idx2potency"],
        "n_states": llh.num_states,
        "start_state": model_info.get("start_state"),
        "backend": model_info.get("backend", "fundamental"),
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/model_dict.pkl", "wb") as fp:
        pickle.dump(model_dict, fp)
    logger.info("Saved model_dict to %s/model_dict.pkl", output_dir)


# ---------------------------------------------------------------------------
# 1. Load trees
# ---------------------------------------------------------------------------

def load_trees(input_path, newick_format=1):
    """Load trees from a .pkl or .nwk file and binarize them."""
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
    """Scan all leaf states to determine terminal labels and type."""
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
    """Create default singleton potency for each terminal label."""
    return {state: (state,) for state in terminal_labels}


# ---------------------------------------------------------------------------
# 4. Generate potency sets
# ---------------------------------------------------------------------------

def generate_potency_sets(trees, terminal_labels, observed_potencies, max_hidden):
    """Generate candidate potency sets from tree clades."""
    terminal_set = set(terminal_labels)

    induced_potencies = set()
    for tree in trees:
        for node in tree.traverse("postorder"):
            if node.is_leaf():
                clade = frozenset([node.state]) if node.state in terminal_set else frozenset()
                node.add_feature("clade", clade)
            else:
                clade = frozenset()
                for child in node.get_children():
                    clade = clade | child.clade
                node.add_feature("clade", clade)
            if len(clade) > 0:
                induced_potencies.add(tuple(sorted(clade)))

    logger.info("Trees have %d induced clades", len(induced_potencies))

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

    for potency in observed_potencies.values():
        if potency not in potency_list:
            potency_list.append(potency)

    observed_potency_set = set(observed_potencies.values())
    observed_pots = [p for p in potency_list if p in observed_potency_set]
    unobserved_pots = [p for p in potency_list if p not in observed_potency_set]
    unobserved_pots.sort(key=len, reverse=True)
    if len(unobserved_pots) > max_hidden:
        unobserved_pots = unobserved_pots[:max_hidden]

    result = observed_pots + unobserved_pots
    result.sort(key=len, reverse=True)
    return result


# ---------------------------------------------------------------------------
# 5. Build model info
# ---------------------------------------------------------------------------

def build_model_info(states, terminal_labels, observed_potencies, potency_sets):
    """Construct model_info dict and state2idx mapping for Phase 1."""
    num_obs = len(states)
    num_hidden = len(potency_sets) - len(observed_potencies)

    state_list = sorted(states)
    state2idx = {state: i for i, state in enumerate(state_list)}

    observed_idxs = set(state2idx.values())
    all_idxs = set(range(num_obs + num_hidden))
    for idx in sorted(all_idxs - observed_idxs):
        state2idx[f"U{idx}"] = idx
    idx2state = {i: state for state, i in state2idx.items()}

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
# 6. Phase 1: Overparameterized ClaSSE MLE
# ---------------------------------------------------------------------------

def run_phase1(trees, model_info, state2idx, num_obs, num_hidden,
               reg, sampling_prob, output_dir, device):
    """Run Phase 1 (overparameterized) ClaSSE MLE for one regularization value.

    Args:
        trees: Original trees (will be deep-copied and relabeled).
        model_info: Model configuration dict.
        state2idx: Mapping from state labels to integer indices.
        num_obs: Number of observed states.
        num_hidden: Number of hidden states.
        reg: L1 regularization strength.
        sampling_prob: Leaf sampling probability eta.
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
        n_states = num_obs + num_hidden
        _compute_classe_mle(
            trees_copy, n_states, device, output_dir,
            model_info, sampling_prob, l1_reg=reg,
        )
        return output_dir
    except Exception as e:
        logger.error("Phase 1 failed for reg=%s: %s", reg, e, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# 7. Phase 2: Potency extraction + debiased ClaSSE MLE
# ---------------------------------------------------------------------------

def run_phase2(phase1_dir, trees, terminal_labels, observed_potencies,
               is_int_state, sampling_prob, debiasing_l1, threshold, device, backend):
    """Extract potencies from Phase 1 model and run debiased Phase 2 ClaSSE MLE.

    Loads the Phase 1 model, finds reachable states from B, computes potencies
    via matrix power, builds a reduced state space, and re-runs MLE with
    warm-start initialization.

    Args:
        phase1_dir: Directory containing Phase 1 model_dict.pkl.
        trees: Original trees (will be deep-copied and relabeled).
        terminal_labels: List of terminal state labels.
        observed_potencies: Dict mapping state -> potency tuple.
        is_int_state: Whether states are integers.
        sampling_prob: Leaf sampling probability eta.
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

        B = model_dict["daughter_kernel"]
        inferred_growth_rates = model_dict["growth_rates"]
        B_np = B.detach().numpy()
        idx2state = model_dict["idx2state"]
        starting_idx = torch.argmax(model_dict["root_distribution"]).item()

        # Find reachable states from B
        reachable_idxs = get_reachable_idxs(B_np, starting_idx, threshold=threshold)
        logger.info("Phase 2: %d reachable states from starting idx %d",
                    len(reachable_idxs), starting_idx)

        # Infer observed-terminal potencies via matrix power on B.
        terminal_set = set(terminal_labels)
        idx2potency_ = _get_idx2potency_classe(B_np, eps=1e-4, n_steps=100)
        old_idx2potency = {}
        for idx in reachable_idxs:
            potency_ = idx2potency_[idx]
            potency = sorted(
                [idx2state[s] for s in potency_ if idx2state[s] in terminal_set]
            )
            old_idx2potency[idx] = tuple(potency)

        # Remove states that cannot reach any observed terminal.
        kept_old_idxs = [idx for idx in reachable_idxs if len(old_idx2potency[idx]) > 0]
        dropped_old_idxs = sorted(set(reachable_idxs) - set(kept_old_idxs))
        if dropped_old_idxs:
            logger.info(
                "Phase 2: dropping %d reachable states with empty observed potency: %s",
                len(dropped_old_idxs),
                {idx: idx2state[idx] for idx in dropped_old_idxs},
            )

        if not kept_old_idxs:
            raise RuntimeError("Phase 2 eliminated every reachable state; no observed-terminal potencies remain.")
        if starting_idx not in kept_old_idxs:
            raise RuntimeError(
                f"Starting state {starting_idx} ({idx2state[starting_idx]}) cannot reach any observed terminal state."
            )

        # Merge states with identical observed-terminal potency.
        potency_to_old_idxs = {}
        for idx in kept_old_idxs:
            potency_to_old_idxs.setdefault(old_idx2potency[idx], []).append(idx)

        ordered_potencies = []
        for state in terminal_labels:
            singleton = (state,)
            if singleton in potency_to_old_idxs:
                ordered_potencies.append(singleton)
        ordered_potencies.extend(
            sorted(
                [potency for potency in potency_to_old_idxs if potency not in ordered_potencies],
                key=lambda potency: (len(potency), potency),
            )
        )

        merged_old_groups = [potency_to_old_idxs[potency] for potency in ordered_potencies]
        old2newidx = {
            old_idx: new_idx
            for new_idx, old_group in enumerate(merged_old_groups)
            for old_idx in old_group
        }
        initial_idx = old2newidx[starting_idx]

        idx2potency = {new_idx: potency for new_idx, potency in enumerate(ordered_potencies)}
        newidx2state = {}
        for new_idx, potency in idx2potency.items():
            if len(potency) == 1 and potency[0] in terminal_set:
                newidx2state[new_idx] = potency[0]
            else:
                newidx2state[new_idx] = f"U{new_idx}"
        state2newidx = {state: new_idx for new_idx, state in newidx2state.items()}

        n_states_new = len(idx2potency)
        logger.info(
            "Phase 2: merged %d kept states into %d potency classes",
            len(kept_old_idxs),
            n_states_new,
        )

        # Warm-start the reduced model by aggregating mass between merged states.
        B_reduced = torch.zeros((n_states_new, n_states_new), dtype=dtype, device=device)
        lam_reduced = torch.zeros(n_states_new, dtype=dtype, device=device)
        for new_idx, old_group in enumerate(merged_old_groups):
            lam_reduced[new_idx] = inferred_growth_rates[old_group].mean()

            row = torch.zeros(n_states_new, dtype=dtype, device=device)
            for old_idx in old_group:
                for target_new_idx, target_group in enumerate(merged_old_groups):
                    row[target_new_idx] += B[old_idx, target_group].sum()
            row /= len(old_group)

            row_sum = row.sum()
            if row_sum <= EPS:
                row[new_idx] = 1.0
                row_sum = 1.0
            B_reduced[new_idx] = row / row_sum

        B_params_init = torch.log(B_reduced.clamp_min(1e-20))
        growth_params_init = _safe_softplus_inverse(lam_reduced.clamp_min(1e-2))

        phase2_model_info = {
            "B_params_init": B_params_init,
            "growth_params_init": growth_params_init,
            "start_state": initial_idx,
            "idx2state": newidx2state,
            "idx2potency": idx2potency,
            "terminal_states": [
                state2newidx[s] for s in terminal_labels if s in state2newidx
            ],
            "optimize_growth": True,
            "backend": backend,
        }

        trees_copy = copy.deepcopy(trees)
        for tree in trees_copy:
            for leaf in tree.get_leaves():
                if leaf.state not in state2newidx:
                    raise RuntimeError(
                        f"Observed leaf state {leaf.state} is absent from the reduced Phase 2 state space."
                    )
                leaf.state = state2newidx[leaf.state]

        output_dir = f"{phase1_dir}/select_potencies"
        os.makedirs(output_dir, exist_ok=True)

        _compute_classe_mle(
            trees_copy, n_states_new, device, output_dir,
            phase2_model_info, sampling_prob, l1_reg=debiasing_l1,
        )

        return output_dir

    except Exception as e:
        logger.error("Phase 2 failed for %s: %s", phase1_dir, e, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# 8. Model selection via knee finding
# ---------------------------------------------------------------------------

def run_model_selection(output_dir, trees, reg_values, threshold,
                        knee_sensitivity, device):
    """Select the best model across regularization values using knee detection.

    For each reg, loads the Phase 2 model (or Phase 1 for reg=0), reconstructs
    a ClaSSELikelihoodModel from the saved state_dict, evaluates the total
    log-likelihood, and counts reachable states from B.

    Args:
        output_dir: Base output directory containing reg=X subdirs.
        trees: Original trees for likelihood evaluation.
        reg_values: List of regularization values tested.
        threshold: Reachability threshold.
        knee_sensitivity: KneeLocator S parameter.
        device: Torch device.

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
        state_dict_path = f"{model_dir}/state_dict.pth"
        if not os.path.isfile(model_dict_path) or not os.path.isfile(state_dict_path):
            logger.warning("Skipping reg=%s: model files not found in %s", reg, model_dir)
            continue

        try:
            with open(model_dict_path, "rb") as fp:
                model_dict = pickle.load(fp)

            B = model_dict["daughter_kernel"]
            B_np = B.detach().numpy()
            idx2state_model = model_dict["idx2state"]
            idx2potency_model = model_dict["idx2potency"]
            n_states = model_dict["n_states"]
            sampling_prob = model_dict["sampling_prob_float"]
            start_state = model_dict.get("start_state")
            backend = model_dict.get("backend", "fundamental")
            state2idx_model = {s: i for i, s in idx2state_model.items()}

            root_distribution = model_dict["root_distribution"]
            starting_idx = torch.argmax(root_distribution).item()
            reachable = get_reachable_idxs(B_np, starting_idx, threshold)

            # Reconstruct model from state_dict for likelihood evaluation
            trees_copy = copy.deepcopy(trees)
            for tree in trees_copy:
                for leaf in tree.get_leaves():
                    if leaf.state not in state2idx_model:
                        continue
                    leaf.state = state2idx_model[leaf.state]

            dummy_bk = torch.zeros(n_states, n_states, dtype=dtype)
            dummy_pi = torch.zeros(n_states, dtype=dtype)
            dummy_lam = torch.zeros(n_states, dtype=dtype)

            eval_model = ClaSSELikelihoodModel(
                trees_copy, n_states,
                dummy_bk, dummy_pi, dummy_lam,
                idx2potency=idx2potency_model,
                device=device,
                idx2state=idx2state_model,
                start_state=start_state,
                sampling_prob=float(sampling_prob),
                backend=backend,
            )
            eval_model.load_state_dict(
                torch.load(state_dict_path, map_location=device)
            )
            eval_model.eval()

            with torch.no_grad():
                eval_model.precompute_ode()
                log_lik = sum(
                    eval_model(i).item() for i in range(len(trees_copy))
                )
                eval_model.clear_ode_cache()

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
        description="Run the full ClaSSE-TROUPE inference pipeline."
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
        "--sampling_probability", type=float, default=1.0,
        help="Leaf sampling probability eta in (0, 1]. Default: 1.0 (all leaves observed)",
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
        help="List of L1 regularization values for Phase 1",
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
    parser.add_argument(
        "--backend", type=str, default="fundamental",
        choices=["fundamental", "vector_transport"],
        help="ClaSSE likelihood backend to use",
    )
    parser.add_argument(
        "--skip_phase_1", action="store_true",
        help="Whether to skip the potency inference part",
    )
    args = parser.parse_args()

    if not (0 < args.sampling_probability <= 1.0):
        parser.error("--sampling_probability must be in (0, 1]")

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
    logger.info(
        "Potency sets: %d total (%d observed, %d hidden)",
        len(potency_sets), len(observed_potencies), num_hidden,
    )

    # --- Build model info ---
    model_info, state2idx = build_model_info(
        states, terminal_labels, observed_potencies, potency_sets
    )
    model_info["backend"] = args.backend

    # --- Run inference for each regularization value ---
    os.makedirs(args.output_dir, exist_ok=True)

    for reg in args.regularizations:
        reg_dir = f"{args.output_dir}/reg={reg}"
        if not args.skip_phase_1:
            logger.info("=" * 60)
            logger.info("Phase 1: reg=%s", reg)
            logger.info("=" * 60)
            phase1_ok = run_phase1(
                trees, model_info, state2idx, num_obs, num_hidden,
                reg, args.sampling_probability, reg_dir, device,
            )

        if (args.skip_phase_1 or phase1_ok) and reg > 0:
            logger.info("=" * 60)
            logger.info("Phase 2: reg=%s", reg)
            logger.info("=" * 60)
            run_phase2(
                reg_dir, trees, terminal_labels, observed_potencies,
                is_int_state, args.sampling_probability,
                args.debiasing_l1, args.reachability_threshold, device, args.backend,
            )

    # --- Model selection ---
    logger.info("=" * 60)
    logger.info("Model selection")
    logger.info("=" * 60)
    selection = run_model_selection(
        args.output_dir, trees, args.regularizations,
        args.reachability_threshold, args.knee_sensitivity, device,
    )

    # --- Copy best model ---
    best_src = f"{selection['best_model_dir']}/model_dict.pkl"
    best_dst = f"{args.output_dir}/best_model_dict.pkl"
    shutil.copy2(best_src, best_dst)
    logger.info("Best model copied to %s", best_dst)

    # --- Write summary ---
    summary_path = f"{args.output_dir}/classe_troupe_summary.txt"
    with open(summary_path, "w") as fp:
        fp.write(f"best_reg\t{selection['best_reg']}\n")
        fp.write(f"best_model_dir\t{selection['best_model_dir']}\n")
        fp.write(f"knee_num_states\t{selection['knee_num_states']}\n")
        fp.write(f"knee_loss\t{selection['knee_loss']}\n")
        fp.write(f"sampling_probability\t{args.sampling_probability}\n")
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
