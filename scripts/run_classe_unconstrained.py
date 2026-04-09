#!/usr/bin/env python
"""Unconstrained ClaSSE MLE (baseline for TROUPE comparison).

Fits a ClaSSE birth-kernel model with a fixed state space and NO potency
constraints.  Every state is free to give birth to daughters of any type.
This is the baseline against which TROUPE's potency constraints are evaluated.

The state space consists of the observed terminal types found in the data plus
an optional number of hidden (unobserved) states specified by --num_hidden.
A single MLE optimisation is run (no regularisation sweep, no Phase 1/2 split).

Usage:
    python scripts/run_classe_unconstrained.py \
        -i /n/fs/ragr-research/users/wh8114/projects/troupe/experiments/subsampled_leaves_4_terminals/trees_32/time_5.0/sample_0.4/trial_0/trees.pkl \
        -o tmp/unconstrained \
        --num_hidden 4 \
        --sampling_probability 0.4
"""

import argparse
import copy
import logging
import os
import pickle
import sys
import time

import torch
import torch.optim as optim
from ete3 import Tree

from classe_model import ClaSSELikelihoodModel
from optimizer import constant_rate_mle
from utils import binarize_tree

sys.setrecursionlimit(5000)

dtype = torch.float64
torch.set_default_dtype(dtype)
EPS = 1e-30

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_softplus_inverse(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, min=EPS)
    return torch.log(torch.expm1(x))


def load_trees(input_path: str, newick_format: int = 1):
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
    return [binarize_tree(t) for t in trees]


def detect_labels(trees):
    """Return sorted list of terminal labels found at tree leaves."""
    states = set()
    for tree in trees:
        for leaf in tree.get_leaves():
            states.add(leaf.state)
    return sorted(states)


def build_model_info(terminal_labels, num_hidden: int, backend: str):
    """Build state-space description for the unconstrained model.

    Observed terminal states are indexed 0 … n_obs-1; hidden states follow.
    All states are assigned potency = full set of terminal labels so that
    downstream utilities reading idx2potency are not confused.  The
    DaughterKernelBuilder receives idx2potency=None (mask = all ones,
    unconstrained), which is equivalent.
    """
    n_obs = len(terminal_labels)
    n_states = n_obs + num_hidden

    state2idx = {state: i for i, state in enumerate(sorted(terminal_labels))}
    idx2state = {i: state for state, i in state2idx.items()}
    for h in range(num_hidden):
        idx = n_obs + h
        idx2state[idx] = f"U{idx}"
        state2idx[f"U{idx}"] = idx

    full_potency = tuple(sorted(terminal_labels))
    idx2potency = {i: full_potency for i in range(n_states)}

    # Root is the first hidden state when hidden states exist so that the model
    # has an unobserved ancestor (matching the TROUPE convention).  With no
    # hidden states the root distribution is left free (optimised).
    start_state = n_obs if num_hidden > 0 else None

    model_info = {
        "idx2state":   idx2state,
        "idx2potency": idx2potency,
        "start_state": start_state,
        "backend":     backend,
    }
    return model_info, state2idx


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_model_dict(llh, model_info, sampling_prob: float, n_states: int,
                     output_dir: str):
    B = llh.get_daughter_kernel()
    model_dict = {
        "rate_matrix":          B,           # alias for compatibility
        "daughter_kernel":      B,
        "growth_rates":         llh.get_growth_rates(),
        "root_distribution":    llh.get_root_distribution(),
        "sampling_probability": llh.get_sampling_probability(),
        "sampling_prob_float":  float(sampling_prob),
        "idx2state":            model_info["idx2state"],
        "idx2potency":          model_info["idx2potency"],
        "n_states":             n_states,
        "start_state":          model_info.get("start_state"),
        "backend":              model_info.get("backend", "fundamental"),
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/model_dict.pkl", "wb") as fp:
        pickle.dump(model_dict, fp)
    logger.info("Saved model_dict to %s/model_dict.pkl", output_dir)


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------

def run_mle(
    trees_labeled,
    n_states: int,
    device,
    output_dir: str,
    model_info: dict,
    sampling_prob: float,
    l1_reg: float = 0.0,
    num_iter: int = 100,
    log_iter: int = 1,
):
    """Fit unconstrained ClaSSE via LBFGS; checkpoint on improvement.

    Returns (model, neg_log_likelihood) where neg_log_likelihood is the pure
    total negative log-likelihood (without regularisation) at the best checkpoint.
    """
    idx2state  = model_info["idx2state"]
    start_state = model_info.get("start_state")
    backend    = model_info.get("backend", "fundamental")

    lam0 = constant_rate_mle(trees_labeled)
    bk_params_init = torch.zeros(n_states, n_states, device=device, dtype=dtype)
    growth_params_init = _safe_softplus_inverse(
        torch.ones(n_states, device=device, dtype=dtype) * lam0
    )
    pi_params_init = torch.zeros(n_states, device=device, dtype=dtype)

    # idx2potency=None → DaughterKernelBuilder uses mask of all ones (unconstrained).
    llh = ClaSSELikelihoodModel(
        trees_labeled, n_states,
        bk_params_init, pi_params_init, growth_params_init,
        optimize_growth=True,
        idx2potency=None,
        device=device,
        idx2state=idx2state,
        start_state=start_state,
        sampling_prob=float(sampling_prob),
        backend=backend,
    )

    tree_idxs = list(range(len(trees_labeled)))
    optimizer = optim.LBFGS(
        [p for p in llh.parameters() if p.requires_grad],
        lr=1.0, max_iter=1, max_eval=20, line_search_fn="strong_wolfe",
    )

    rel_loss_thresh = 1e-5 / len(trees_labeled)
    losses = []
    closure_state = {"last_loss": None}

    def _objective():
        total = sum(-llh(j) for j in tree_idxs) / len(tree_idxs)
        if l1_reg > 0:
            B = llh.get_daughter_kernel()
            offdiag_mask = ~torch.eye(n_states, dtype=torch.bool, device=device)
            total = total + l1_reg * B[offdiag_mask].abs().sum()
        return total

    def closure():
        optimizer.zero_grad()
        llh.precompute_ode()
        obj = _objective()
        if not torch.isfinite(obj):
            llh.clear_ode_cache()
            raise RuntimeError(f"Non-finite loss: {obj.item()}")
        obj.backward()
        closure_state["last_loss"] = float(obj.detach())
        llh.clear_ode_cache()
        return obj

    start = time.time()
    for i in range(num_iter):
        optimizer.step(closure)
        if closure_state["last_loss"] is None:
            raise RuntimeError("LBFGS step did not evaluate the closure")
        loss_value = closure_state["last_loss"]
        losses.append(loss_value)

        with torch.no_grad():
            if torch.isnan(llh.get_daughter_kernel()).any():
                raise ValueError("NaN detected in daughter kernel")

        if losses[-1] <= min(losses):
            os.makedirs(output_dir, exist_ok=True)
            torch.save(llh.state_dict(), f"{output_dir}/state_dict.pth")
            _save_model_dict(llh, model_info, sampling_prob, n_states, output_dir)

        if i % log_iter == 0:
            logger.info(
                "Iter %d | loss=%.6f | B diag=%s | lam=%s",
                i, loss_value,
                llh.get_daughter_kernel().diag().detach().tolist(),
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

    # Restore best checkpoint
    best_state_path = f"{output_dir}/state_dict.pth"
    if os.path.isfile(best_state_path):
        llh.load_state_dict(torch.load(best_state_path, map_location=device))
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        torch.save(llh.state_dict(), best_state_path)
        _save_model_dict(llh, model_info, sampling_prob, n_states, output_dir)

    # Evaluate pure (unregularised) log-likelihood at the best checkpoint.
    with torch.no_grad():
        llh.precompute_ode()
        log_lik = sum(llh(j).item() for j in tree_idxs)
        llh.clear_ode_cache()

    neg_llh = -log_lik
    with open(f"{output_dir}/loss.txt", "w") as fp:
        fp.write(f"{neg_llh:.6f}")

    return llh, neg_llh


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unconstrained ClaSSE MLE (baseline for TROUPE comparison)."
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to .pkl (pickled list of ete3 trees) or .nwk (one tree per line).",
    )
    parser.add_argument(
        "-o", "--output_dir", required=True,
        help="Directory for output files (model_dict.pkl, state_dict.pth, summary).",
    )
    parser.add_argument(
        "--num_hidden", type=int, default=0,
        help="Number of hidden (unobserved) states added beyond observed terminal types. "
             "Default: 0 (observed types only).",
    )
    parser.add_argument(
        "--sampling_probability", type=float, default=1.0,
        help="Leaf sampling probability eta in (0, 1]. Default: 1.0.",
    )
    parser.add_argument(
        "--l1", type=float, default=0.0,
        help="L1 regularisation on off-diagonal birth-kernel entries. "
             "Default: 0.0 (fully unconstrained).",
    )
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device (cpu or cuda).")
    parser.add_argument(
        "--backend", type=str, default="fundamental",
        choices=["fundamental", "vector_transport"],
        help="ClaSSE likelihood backend. Default: fundamental.",
    )
    parser.add_argument("--newick_format", type=int, default=1,
                        help="ete3 newick format integer. Default: 1.")
    parser.add_argument(
        "--num_iter", type=int, default=100,
        help="Maximum LBFGS outer iterations. Default: 100.",
    )
    args = parser.parse_args()

    if not (0 < args.sampling_probability <= 1.0):
        parser.error("--sampling_probability must be in (0, 1]")

    device = torch.device(args.device)

    logger.info("Loading trees from %s", args.input)
    trees = load_trees(args.input, newick_format=args.newick_format)
    logger.info("Loaded %d trees", len(trees))

    terminal_labels = detect_labels(trees)
    n_obs    = len(terminal_labels)
    n_states = n_obs + args.num_hidden
    logger.info("Terminal labels (%d): %s", n_obs, terminal_labels)
    logger.info(
        "State space: %d total (%d observed + %d hidden)",
        n_states, n_obs, args.num_hidden,
    )

    model_info, state2idx = build_model_info(
        terminal_labels, args.num_hidden, args.backend
    )
    logger.info("idx2state:   %s", model_info["idx2state"])
    logger.info("start_state: %s", model_info["start_state"])

    # Relabel leaves from original state labels to integer indices.
    trees_labeled = copy.deepcopy(trees)
    for tree in trees_labeled:
        for leaf in tree.get_leaves():
            leaf.state = state2idx[leaf.state]

    os.makedirs(args.output_dir, exist_ok=True)
    try:
        _llh, neg_llh = run_mle(
            trees_labeled, n_states, device, args.output_dir, model_info,
            args.sampling_probability, l1_reg=args.l1, num_iter=args.num_iter,
        )
    except Exception as e:
        logger.error("Inference failed: %s", e, exc_info=True)
        sys.exit(1)

    logger.info("Done. neg-llh=%.4f, n_states=%d", neg_llh, n_states)

    summary_path = f"{args.output_dir}/classe_unconstrained_summary.txt"
    with open(summary_path, "w") as fp:
        fp.write(f"n_observed_states\t{n_obs}\n")
        fp.write(f"num_hidden\t{args.num_hidden}\n")
        fp.write(f"n_total_states\t{n_states}\n")
        fp.write(f"neg_llh\t{neg_llh:.6f}\n")
        fp.write(f"sampling_probability\t{args.sampling_probability}\n")
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
