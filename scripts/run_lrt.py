#!/usr/bin/env python
"""Parametric bootstrap likelihood ratio test for ClaSSE birth kernel constraints.

Tests whether specific birth kernel entries are statistically significant.
Because the MLE may lie on a boundary, the classical chi-squared asymptotic
is invalid; the null distribution is calibrated via parametric bootstrap.

Example usage (ratio constraint):
    python scripts/run_lrt.py \\
        -i experiments/c_elegans/packer/processed_data/10/0.5/trees.pkl \\
        --model lrt_results/model_dict.pkl \\
        --constraint "B[NMP,NeuralTube] >= 2.0 * B[NMP,Somite]" \\
        --B 99 \\
        --output lrt_results/nmp_test

Example usage (lower-bound constraint):
    python scripts/run_lrt.py \\
        -i trees.pkl \\
        --model model_dict.pkl \\
        --constraint "B[NMP,NeuralTube] >= 0.3" \\
        --B 99 \\
        --output lrt_results/nmp_lb_test
"""

import argparse
import logging
import os
import pickle
import re
import sys
import warnings

import torch
from ete3 import Tree, TreeNode

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from constraints import KernelConstraint, constraint_from_names
from bootstrap import estimate_simulation_time
from lrt import parametric_bootstrap_lrt
from utils import binarize_tree

sys.setrecursionlimit(5000)
torch.set_default_dtype(torch.float64)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constraint string parser
# ---------------------------------------------------------------------------

def parse_constraint_str(constraint_str: str, state2idx: dict) -> KernelConstraint:
    """Parse a human-readable constraint string into a KernelConstraint.

    Supported formats (state names may contain letters, digits, underscores,
    hyphens, and dots — any non-whitespace non-bracket characters):
      Ratio:       "B[A,B] >= k * B[C,D]"
      Lower bound: "B[A,B] >= c"

    Args:
        constraint_str: Raw constraint string from --constraint flag.
        state2idx: Mapping from state label to integer index.

    Returns:
        KernelConstraint object.

    Raises:
        ValueError: If the string cannot be parsed or references unknown states.
    """
    s = constraint_str.strip()
    STATE_RE = r"([^\[\],\s]+)"

    # Try ratio form: B[A,B] >= k * B[C,D]
    ratio_pat = (
        r"B\[" + STATE_RE + r"," + STATE_RE + r"\]"
        r"\s*>=\s*([\d.eE+\-]+)\s*\*\s*"
        r"B\[" + STATE_RE + r"," + STATE_RE + r"\]"
    )
    m = re.fullmatch(ratio_pat, s)
    if m:
        from_state, to_state, k_str, from_state2, to_state2 = m.groups()
        _check_states(state2idx, [from_state, to_state, from_state2, to_state2], constraint_str)
        return constraint_from_names(
            state2idx, from_state, to_state,
            from_state2=from_state2, to_state2=to_state2,
            k=float(k_str), label=constraint_str,
        )

    # Try lower-bound form: B[A,B] >= c
    lb_pat = r"B\[" + STATE_RE + r"," + STATE_RE + r"\]\s*>=\s*([\d.eE+\-]+)"
    m = re.fullmatch(lb_pat, s)
    if m:
        from_state, to_state, c_str = m.groups()
        _check_states(state2idx, [from_state, to_state], constraint_str)
        return constraint_from_names(
            state2idx, from_state, to_state,
            min_val=float(c_str), label=constraint_str,
        )

    raise ValueError(
        f"Cannot parse constraint: {constraint_str!r}\n"
        "Expected format:\n"
        "  Ratio:       B[A,B] >= k * B[C,D]\n"
        "  Lower bound: B[A,B] >= c"
    )


def _check_states(state2idx, names, constraint_str):
    missing = [n for n in names if n not in state2idx]
    if missing:
        raise ValueError(
            f"Unknown state(s) in constraint {constraint_str!r}: {missing}\n"
            f"Available states: {sorted(state2idx)}"
        )


# ---------------------------------------------------------------------------
# Tree loading
# ---------------------------------------------------------------------------

def load_trees(input_path: str, state2idx: dict = None, newick_format: int = 1) -> list:
    """Load and binarize trees; relabel leaf states to integers using state2idx."""
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

    if state2idx is not None:
        # Relabel leaf states from string names to integer indices.
        first_leaf = trees[0].get_leaves()[0]
        already_int = isinstance(first_leaf.state, int) or (
            isinstance(first_leaf.state, str) and first_leaf.state.lstrip("-").isdigit()
        )
        if not already_int:
            for tree in trees:
                for leaf in tree.get_leaves():
                    if leaf.state not in state2idx:
                        raise ValueError(
                            f"Leaf state {leaf.state!r} not in model state2idx. "
                            f"Available: {sorted(state2idx)}"
                        )
                    leaf.state = state2idx[leaf.state]

    return trees


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parametric bootstrap LRT for ClaSSE birth kernel constraints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Path to trees (.pkl or .nwk).")
    parser.add_argument("--model", required=True,
                        help="Path to model_dict.pkl from a fitted ClaSSE-TROUPE run.")
    parser.add_argument("--constraint", action="append", dest="constraints",
                        metavar="EXPR", required=True,
                        help=("Constraint expression, e.g. "
                              "'B[NMP,NeuralTube] >= 2.0 * B[NMP,Somite]'. "
                              "Can be repeated for multiple simultaneous constraints."))
    parser.add_argument("-o", "--output", default="lrt_results",
                        help="Output directory (default: lrt_results).")
    parser.add_argument("--B", type=int, default=99,
                        help="Number of bootstrap replicates (default: 99).")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level (default: 0.05).")
    parser.add_argument("--T", type=float, default=None,
                        help="Simulation time horizon. If omitted, estimated from trees.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for bootstrap simulation (default: 0).")
    parser.add_argument("--sampling_prob", type=float, default=None,
                        help="Leaf sampling probability. Overrides value in model_dict.")
    parser.add_argument("--num_iter_uncon", type=int, default=100,
                        help="LBFGS iterations for unconstrained fit (default: 100).")
    parser.add_argument("--num_iter_con", type=int, default=50,
                        help="LBFGS iterations per penalty phase (default: 50).")
    parser.add_argument("--penalty_schedule", type=float, nargs="+",
                        default=[1.0, 10.0, 100.0, 1000.0],
                        help="Mu values for quadratic penalty schedule (default: 1 10 100 1000).")
    parser.add_argument("--constraint_tol", type=float, default=1e-5,
                        help="Tolerance for constraint satisfaction (default: 1e-5).")
    parser.add_argument("--device", default="cpu",
                        help="Torch device: 'cpu' or 'cuda' (default: cpu).")
    parser.add_argument("--verbose", action="store_true",
                        help="Log optimization progress for each replicate.")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # Load model dict.
    # ------------------------------------------------------------------
    logger.info("Loading model from %s", args.model)
    with open(args.model, "rb") as fp:
        model_dict = pickle.load(fp)

    idx2state = model_dict["idx2state"]
    idx2potency = model_dict["idx2potency"]
    state2idx = {state: idx for idx, state in idx2state.items()}
    n_states = model_dict.get("n_states", len(idx2state))
    start_state = model_dict.get("start_state")

    sampling_prob = args.sampling_prob
    if sampling_prob is None:
        sampling_prob = float(model_dict.get("sampling_prob_float",
                                             model_dict.get("sampling_probability", 1.0)))
    logger.info("Sampling probability: %.4f", sampling_prob)

    # Build model_info from model_dict.
    model_info = {
        "idx2potency": idx2potency,
        "idx2state": idx2state,
        "start_state": start_state,
        "backend": model_dict.get("backend", "fundamental"),
    }
    # Warm-start unconstrained fits from the loaded model's parameters.
    if "daughter_kernel" in model_dict:
        B_loaded = model_dict["daughter_kernel"]
        if hasattr(B_loaded, "detach"):
            import torch.nn.functional as F
            # Convert kernel back to logits (inverse softmax isn't unique, but
            # log is a reasonable proxy for warm-starting).
            logits = torch.log(B_loaded.clamp(min=1e-10))
            model_info["B_params_init"] = logits
    if "growth_rates" in model_dict:
        lam = model_dict["growth_rates"]
        if hasattr(lam, "detach"):
            import torch.nn.functional as F
            model_info["growth_params_init"] = torch.log(torch.expm1(lam.clamp(min=1e-10)))

    # ------------------------------------------------------------------
    # Parse constraints.
    # ------------------------------------------------------------------
    constraints = []
    for cstr in args.constraints:
        try:
            c = parse_constraint_str(cstr, state2idx)
        except ValueError as e:
            logger.error("Failed to parse constraint %r:\n%s", cstr, e)
            sys.exit(1)
        logger.info("Constraint: %s", c.label)
        constraints.append(c)

    # ------------------------------------------------------------------
    # Load trees.
    # ------------------------------------------------------------------
    logger.info("Loading trees from %s", args.input)
    trees = load_trees(args.input, state2idx=state2idx)
    logger.info("Loaded %d trees.", len(trees))

    # ------------------------------------------------------------------
    # Run LRT.
    # ------------------------------------------------------------------
    results = parametric_bootstrap_lrt(
        observed_trees=trees,
        null_model_dict=model_dict,
        n_states=n_states,
        model_info=model_info,
        sampling_prob=sampling_prob,
        constraints=constraints,
        device=device,
        T=args.T,
        B=args.B,
        alpha=args.alpha,
        seed=args.seed,
        output_dir=args.output,
        num_iter_uncon=args.num_iter_uncon,
        num_iter_con=args.num_iter_con,
        penalty_weight_schedule=tuple(args.penalty_schedule),
        constraint_tol=args.constraint_tol,
        do_logging=args.verbose,
    )

    # ------------------------------------------------------------------
    # Print summary.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  LIKELIHOOD RATIO TEST RESULT")
    print("=" * 60)
    for c in constraints:
        print(f"  H0: {c.label}")
    print(f"\n  Observed LRT statistic:  {results['lrt_stat_observed']:.4f}")
    print(f"  Log-lik unconstrained:   {results['log_lik_unconstrained']:.4f}")
    print(f"  Log-lik null:            {results['log_lik_null']:.4f}")
    print(f"\n  Bootstrap replicates:    {results['B']}")
    print(f"  p-value:                 {results['p_value']:.4f}")
    print(f"  Critical value (alpha={args.alpha}): {results['critical_value']:.4f}")
    print(f"  Decision:                {'REJECT H0' if results['reject'] else 'FAIL TO REJECT H0'}")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
