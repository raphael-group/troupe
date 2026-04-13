#!/usr/bin/env python
"""Unconstrained ClaSSE MLE (baseline for TROUPE comparison).

Fits a ClaSSE birth-kernel model with a fixed state space and NO potency
constraints.  Every state is free to give birth to daughters of any type.
This is the baseline against which TROUPE's potency constraints are evaluated.

The state space consists of the observed terminal types found in the data plus
an optional number of hidden (unobserved) states specified by --num_hidden.
A single MLE optimization is run (no regularization sweep, no Phase 1/2 split).

NOTE: This assumes that all observed states are terminal states.

Usage:
    python scripts/run_classe_unconstrained.py \
        -i /Users/william_hs/Desktop/Projects/troupe/experiments/subsampled_leaves_4_terminals/trees_64/time_5.0/sample_0.1/trial_0/trees.pkl \
        -o tmp/unconstrained \
        --num_hidden 4 \
        --sampling_probability 0.1 \
        --num_restarts 5 \
        --num_iter 200
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


def _make_bk_params_init(
    n_states: int,
    observed_idxs: set,
    device,
    rng=None,
    start_state=None,
) -> torch.Tensor:
    """Return initial birth-kernel logits as an (n_states, n_states) tensor.

    Observed state rows use a fixed 0.75 self-replication bias.

    The start_state row (the root whose pi is a fixed point mass) is always
    initialised to uniform — i.e. zeros, so softmax gives 1/n_states.  This
    avoids any prior bias about which terminal type the root preferentially
    produces before gradient information is available.

    Other hidden state rows are either:
    - deterministic (rng=None): 0.25 on diagonal, rest uniform.
    - random (rng provided): each row concentrates ~75% mass on a distinct
      random subset of terminal states, breaking the symmetry that locks all
      hidden states into identical rows.

    Args:
        n_states:      Total number of states (observed + hidden).
        observed_idxs: Set of integer indices for observed (terminal) states.
        device:        Torch device.
        rng:           numpy.random.Generator for random restarts, or None for
                       the deterministic default.
        start_state:   Integer index of the root state (pi fixed here), or None.
    """
    obs_sorted = sorted(observed_idxs)
    idxs = list(range(n_states))
    n_obs = len(obs_sorted)
    bk = torch.empty(n_states, n_states, device=device, dtype=dtype)

    for i in range(n_states):
        if i == start_state:
            # Uniform row: softmax(zeros) = 1/n_states for all columns.
            bk[i] = torch.zeros(n_states, device=device, dtype=dtype)
        elif i in observed_idxs:
            offdiag = 0.1 / max(n_states - 1, 1)
            row = torch.full((n_states,), offdiag, device=device, dtype=dtype)
            row[i] = 0.9
            bk[i] = torch.log(row.clamp(min=EPS))
        elif rng is None:
            # Deterministic default for non-root hidden states.
            offdiag = 0.9 / max(n_states - 1, 1)
            row = torch.full((n_states,), offdiag, device=device, dtype=dtype)
            row[i] = 0.1
            bk[i] = torch.log(row.clamp(min=EPS))
        else:
            # Random subset assignment: hidden state i specialises toward a
            # random subset of k terminals, with ~0.9 mass inside the subset
            # and ~0.1 spread over the remaining n_states - k entries.
            allowable_idxs = [idx for idx in idxs if idx in observed_idxs or idx >= i]
            k = int(rng.integers(1, len(allowable_idxs)))
            subset = set(rng.choice(allowable_idxs, size=k, replace=False).tolist())
            n_other = max(n_states - k, 1)
            row = torch.full((n_states,), 0.1 / n_other, device=device, dtype=dtype)
            for j in subset:
                row[j] = 0.9 / k
            bk[i] = torch.log(row.clamp(min=EPS))

    return bk


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
    """Return sorted list of observed labels found at tree leaves."""
    states = set()
    for tree in trees:
        for leaf in tree.get_leaves():
            states.add(leaf.state)
    return sorted(states)


def build_model_info(terminal_labels, num_hidden: int, backend: str):
    """Build state-space description for the unconstrained model.

    Observed terminal states are indexed 0 ... n_obs-1; hidden states follow.
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
    # hidden states the root distribution is left free (optimized).
    start_state = n_obs if num_hidden > 0 else None

    model_info = {
        "idx2state":   idx2state,
        "idx2potency": idx2potency,
        "start_state": start_state,
        "backend":     backend,
        "observed_idxs": {state2idx[state] for state in terminal_labels} 
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
# Optimization
# ---------------------------------------------------------------------------

def _run_one_restart(
    trees_labeled,
    n_states: int,
    device,
    restart_dir: str,
    model_info: dict,
    sampling_prob: float,
    bk_params_init: torch.Tensor,
    l1_reg: float,
    num_iter: int,
    log_iter: int,
) -> tuple:
    """Run LBFGS from a given birth-kernel initialisation.

    Returns (llh_model, neg_log_likelihood).
    """
    idx2state   = model_info["idx2state"]
    start_state = model_info.get("start_state")
    backend     = model_info.get("backend", "fundamental")

    lam0 = constant_rate_mle(trees_labeled)
    growth_params_init = _safe_softplus_inverse(
        torch.ones(n_states, device=device, dtype=dtype) * lam0
    )
    pi_params_init = torch.zeros(n_states, device=device, dtype=dtype)

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
        if llh.pi_params.requires_grad:
            with torch.no_grad():
                llh.pi_params.clamp_(-500.0, 500.0)
        # NOTE: Let's not do this because it can hide other issues...
        # Clamp pi_params before every forward pass (including inside LBFGS
        # line-search evaluations).  When start_state=None the root logits are
        # free; with a flat likelihood in that direction LBFGS can step them
        # past 700, causing exp() to overflow to +inf.
        # if llh.pi_params.requires_grad:
        #     with torch.no_grad():
        #         llh.pi_params.clamp_(-500.0, 500.0)
        llh.precompute_ode()
        obj = _objective()
        if not torch.isfinite(obj):
            llh.clear_ode_cache()
            raise RuntimeError(f"Non-finite loss: {obj.item()}")
        obj.backward()
        closure_state["last_loss"] = float(obj.detach())
        llh.clear_ode_cache()
        return obj

    t0 = time.time()
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
            os.makedirs(restart_dir, exist_ok=True)
            torch.save(llh.state_dict(), f"{restart_dir}/state_dict.pth")
            _save_model_dict(llh, model_info, sampling_prob, n_states, restart_dir)

        if i % log_iter == 0:
            logger.info(
                "Iter %d | loss=%.6f | B diag=%s | lam=%s",
                i, loss_value,
                llh.get_daughter_kernel().diag().detach().tolist(),
                llh.get_growth_rates().detach().tolist(),
            )
            elapsed = time.time() - t0
            logger.info("  %.4f s/iter", elapsed / min(i + 1, log_iter))
            t0 = time.time()

        if len(losses) > 2:
            rel = abs(losses[-1] - losses[-2]) / (abs(losses[-2]) + EPS)
            if rel <= rel_loss_thresh:
                logger.info("Converged at iteration %d (rel_loss=%.2e)", i, rel)
                break

    # Restore best checkpoint for this restart.
    best_path = f"{restart_dir}/state_dict.pth"
    if os.path.isfile(best_path):
        llh.load_state_dict(torch.load(best_path, map_location=device))
    with torch.no_grad():
        torch.save(llh.state_dict(), best_path)
        _save_model_dict(llh, model_info, sampling_prob, n_states, restart_dir)

    with torch.no_grad():
        llh.precompute_ode()
        log_lik = sum(llh(j).item() for j in tree_idxs)
        llh.clear_ode_cache()

    return llh, -log_lik


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
    num_restarts: int = 1,
    seed: int = 0,
):
    """Fit unconstrained ClaSSE via LBFGS with optional random restarts.

    Restart 0 uses a deterministic initialisation:
      - Observed states: 0.75 self-replication, 0.25 uniform over the rest.
      - Hidden states:   0.25 self-replication, 0.75 uniform over the rest.

    Restarts 1 ... num_restarts-1 use random initialisations designed to break
    the symmetry that causes hidden states to collapse to identical rows:
      - Observed states: same deterministic 0.75 self-replication.
      - Hidden states: each row concentrates ~0.75 mass on a distinct random
        subset of terminal states, seeding the optimizer at different vertices
        of the probability simplex.

    The best solution across all restarts (lowest neg-llh) is written to
    output_dir and returned.

    Returns (best_model, neg_log_likelihood).
    """
    observed_idxs = model_info["observed_idxs"]
    start_state   = model_info.get("start_state")

    best_llh  = None
    best_neg_llh = float("inf")

    import numpy as np
    rng_master = np.random.default_rng(seed)

    for r in range(num_restarts):
        restart_dir = os.path.join(output_dir, f"restart_{r}")
        rng = None if r == 0 else np.random.default_rng(rng_master.integers(2**31))

        logger.info("=== Restart %d / %d ===", r, num_restarts - 1)
        bk_params_init = _make_bk_params_init(n_states, observed_idxs, device, rng,
                                               start_state=start_state)

        try:
            llh, neg_llh = _run_one_restart(
                trees_labeled, n_states, device, restart_dir, model_info,
                sampling_prob, bk_params_init, l1_reg, num_iter, log_iter,
            )
        except Exception as e:
            logger.warning("Restart %d failed: %s", r, e)
            continue

        logger.info("Restart %d neg-llh=%.6f", r, neg_llh)

        if neg_llh < best_neg_llh:
            best_neg_llh = neg_llh
            # Free the previous best model before storing the new one so that
            # at most one completed model is alive between restarts.
            best_llh = None
            best_llh = llh
            # Copy best restart's artifacts to the top-level output_dir.
            os.makedirs(output_dir, exist_ok=True)
            import shutil
            for fname in ("state_dict.pth", "model_dict.pkl"):
                src = os.path.join(restart_dir, fname)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(output_dir, fname))
        else:
            # Not the best - discard immediately so it doesn't persist until
            # the next iteration overwrites `llh`.
            del llh

    if best_llh is None:
        raise RuntimeError("All restarts failed.")

    with open(f"{output_dir}/loss.txt", "w") as fp:
        fp.write(f"{best_neg_llh:.6f}")

    logger.info("Best neg-llh across %d restart(s): %.6f", num_restarts, best_neg_llh)
    return best_llh, best_neg_llh


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unconstrained ClaSSE MLE (baseline for TROUPE comparison). Assumes all observed labels are terminal"
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
        help="L1 regularization on off-diagonal birth-kernel entries. "
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
        help="Maximum LBFGS outer iterations per restart. Default: 100.",
    )
    parser.add_argument(
        "--num_restarts", type=int, default=1,
        help="Number of random restarts. Restart 0 is deterministic; subsequent "
             "restarts use random hidden-state initialisations to escape local minima. "
             "Default: 1 (no restarts).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Master random seed for restart initialisations. Default: 0.",
    )
    args = parser.parse_args()

    if not (0 < args.sampling_probability <= 1.0):
        parser.error("--sampling_probability must be in (0, 1]")

    device = torch.device(args.device)

    logger.info("Loading trees from %s", args.input)
    trees = load_trees(args.input, newick_format=args.newick_format)
    logger.info("Loaded %d trees", len(trees))
    
    # NOTE: All observed labels are treated as terminals
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
            num_restarts=args.num_restarts, seed=args.seed,
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
