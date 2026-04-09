#!/usr/bin/env python3
"""
Compute the min cut (max flow) from a start state to each observed state
in a weighted directed graph defined by an adjacency matrix.

By the max-flow / min-cut theorem, the min cut equals the maximum probability
mass that can flow from the start state to each target.  A value of 0 means
the target is structurally unreachable — no path exists regardless of threshold.

Usage examples
--------------
# From a raw .npy adjacency matrix
python scripts/min_cut_diagnostics.py \
    --matrix B.npy --start 7 --observed 0 1 2 3 4 5 6

# Directly from a ClaSSE model_dict.pkl (state labels resolved automatically)
python scripts/min_cut_diagnostics.py \
    --model_dict tmp/c_elegans_subsample/reg=0.01/model_dict.pkl

# Override which states to treat as observed
python scripts/min_cut_diagnostics.py \
    --model_dict tmp/c_elegans_subsample/reg=0.01/model_dict.pkl \
    --observed 0 1 2 3 4 5 6
"""

import argparse
import pickle
import sys

import numpy as np
import networkx as nx


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def min_cuts_from_source(
    adj: np.ndarray,
    start: int,
    targets: list[int],
) -> dict[int, float]:
    """Return the min-cut value from `start` to each state in `targets`.

    Args:
        adj: (N, N) adjacency / capacity matrix.  adj[i, j] is the edge
             capacity from state i to state j; zero entries are absent edges.
        start: Source node index.
        targets: List of sink node indices.

    Returns:
        Dict mapping each target index to its min-cut value.
        0.0 means the target is completely unreachable from `start`.
    """
    n = adj.shape[0]
    if adj.shape != (n, n):
        raise ValueError(f"adj must be square, got shape {adj.shape}")

    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                G.add_edge(i, j, capacity=float(adj[i, j]))

    results = {}
    for t in targets:
        if t == start:
            results[t] = float("inf")
        elif not nx.has_path(G, start, t):
            results[t] = 0.0
        else:
            cut_value, _ = nx.minimum_cut(G, start, t)
            results[t] = float(cut_value)

    return results


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_matrix(path: str):
    """Load an adjacency matrix from .npy or .pkl.

    For a .pkl that contains a ClaSSE model_dict, returns
    (adj_numpy, idx2state, start_state, terminal_idxs).
    Otherwise returns (adj_numpy, None, None, None).
    """
    if path.endswith(".npy"):
        return np.load(path).astype(float), None, None, None

    if not path.endswith(".pkl"):
        raise ValueError(f"Unsupported file format: {path}. Use .npy or .pkl.")

    with open(path, "rb") as f:
        obj = pickle.load(f)

    # Raw numpy array
    if isinstance(obj, np.ndarray):
        return obj.astype(float), None, None, None

    # Torch tensor
    if hasattr(obj, "numpy"):
        return obj.detach().cpu().numpy().astype(float), None, None, None

    # ClaSSE model_dict
    if isinstance(obj, dict) and "daughter_kernel" in obj:
        B = obj["daughter_kernel"]
        adj = B.detach().cpu().numpy() if hasattr(B, "numpy") else np.array(B)
        idx2state = obj.get("idx2state")
        idx2potency = obj.get("idx2potency", {})
        # Infer start state from root_distribution
        rd = obj.get("root_distribution")
        if rd is not None:
            rd_np = rd.detach().cpu().numpy() if hasattr(rd, "numpy") else np.array(rd)
            start = int(np.argmax(rd_np))
        else:
            start = obj.get("start_state")
        # Terminal states = singleton potencies
        terminal_idxs = sorted(
            idx for idx, potency in idx2potency.items()
            if len(potency) == 1 and potency[0] == idx2state.get(idx)
        ) if idx2potency and idx2state else None
        return adj.astype(float), idx2state, start, terminal_idxs

    raise ValueError(f"Cannot interpret .pkl contents as a matrix: {type(obj)}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(
    cuts: "dict[int, float]",
    start: int,
    idx2state: "dict | None",
):
    label = (lambda i: idx2state[i]) if idx2state else (lambda i: str(i))

    print(f"\nMin cuts from '{label(start)}'")
    print(f"  (0.0 = structurally unreachable; *** flagged)\n")
    col = 30
    print(f"  {'Target':<{col}} {'Min cut':>10}")
    print("  " + "-" * (col + 12))
    for t, v in sorted(cuts.items(), key=lambda x: x[1]):
        flag = "  ***" if v == 0.0 else ""
        val_str = "inf" if v == float("inf") else f"{v:.6f}"
        print(f"  {label(t):<{col}} {val_str:>10}{flag}")
    print()

    unreachable = [t for t, v in cuts.items() if v == 0.0]
    if unreachable:
        names = [label(t) for t in unreachable]
        print(f"  WARNING: {len(unreachable)} target(s) are unreachable: {names}")
    else:
        print("  All targets are reachable.")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--model_dict",
        metavar="PATH",
        help="ClaSSE model_dict.pkl (start state and terminal indices resolved automatically).",
    )
    src.add_argument(
        "--matrix",
        metavar="PATH",
        help="Adjacency matrix (.npy or .pkl containing a bare numpy array / tensor).",
    )
    p.add_argument(
        "--start", type=int, default=None,
        help="Start state index (required when --matrix is used; overrides model_dict value).",
    )
    p.add_argument(
        "--observed", type=int, nargs="+", default=None,
        help="Target state indices (overrides the terminal states inferred from model_dict).",
    )
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    path = args.model_dict or args.matrix
    adj, idx2state, inferred_start, inferred_terminals = load_matrix(path)

    start = args.start if args.start is not None else inferred_start
    if start is None:
        sys.exit("--start is required when not loading from a model_dict.")

    targets = args.observed if args.observed is not None else inferred_terminals
    if targets is None:
        sys.exit("--observed is required when not loading from a model_dict.")

    cuts = min_cuts_from_source(adj, start, targets)
    print_results(cuts, start, idx2state)


if __name__ == "__main__":
    main()
