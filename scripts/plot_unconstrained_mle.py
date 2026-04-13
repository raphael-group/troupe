#!/usr/bin/env python
"""Plot the unconstrained ClaSSE MLE birth kernel as a directed graph.

The birth kernel B[i,j] gives the probability that a cell of type i gives
birth to a daughter of type j.  This script:

  1. Loads model_dict.pkl from an unconstrained MLE output directory.
  2. Thresholds the off-diagonal entries of B.
  3. Converts the thresholded graph to a DAG by greedily removing the
     minimum-weight edge from each cycle (same approach as the C. elegans
     build_diff_map.py).  The DAG is used only for node positioning, forcing
     a clean hierarchical 'dot' layout.
  4. Draws the full significant graph: DAG edges as solid arrows, removed
     back-edges as dashed arrows in grey.  Growth rates are shown as
     self-loops.

Usage:
    python scripts/plot_unconstrained_mle.py \
        -i tmp/unconstrained/model_dict.pkl \
        -o tmp/unconstrained/figures/birth_kernel.pdf \
        --threshold 0.02
"""

import argparse
import os
import pickle
import sys
from collections import defaultdict, deque

import networkx as nx
import numpy as np
import torch
from graphviz import Digraph

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

TERMINAL_COLORS = [
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
HIDDEN_COLOR = "#DDDDDD"
ROOT_COLOR   = "#888888"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _html_pill(text):
    return (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLPADDING="1" CELLSPACING="0" '
        f'BGCOLOR="white"><TR><TD>{text}</TD></TR></TABLE>>'
    )


def _infer_fmt(path):
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    return ext if ext in {"png", "pdf", "svg", "jpg"} else "pdf"


# ---------------------------------------------------------------------------
# DAG conversion
# ---------------------------------------------------------------------------

def make_dag(B_thresh):
    """Greedily remove min-weight edges from cycles until the graph is a DAG.

    Parameters
    ----------
    B_thresh : (n, n) ndarray
        Thresholded off-diagonal birth kernel (diagonal must be zero).

    Returns
    -------
    dag : (n, n) ndarray — copy of B_thresh with back-edges zeroed.
    back_edges : list of (i, j, weight) for removed back-edges.
    """
    dag = B_thresh.copy()
    G = nx.DiGraph()
    n = dag.shape[0]
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if dag[i, j] > 0:
                G.add_edge(i, j, weight=dag[i, j])

    back_edges = []
    while not nx.is_directed_acyclic_graph(G):
        try:
            cycle = nx.find_cycle(G)
        except nx.NetworkXNoCycle:
            break

        # Remove the lightest edge; break ties by preferring the edge whose
        # reverse has higher weight (clearer back-edge).
        u, v = min(cycle, key=lambda e: (dag[e[0], e[1]], -dag[e[1], e[0]]))
        back_edges.append((u, v, float(dag[u, v])))
        dag[u, v] = 0.0
        G.remove_edge(u, v)

    return dag, back_edges


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _dag_levels(nodes, dag_adj):
    """Kahn's algorithm with longest-path leveling. Returns node->level dict."""
    n = len(nodes)
    adj   = defaultdict(list)
    indeg = {u: 0 for u in nodes}
    for i in nodes:
        for j in nodes:
            if i != j and dag_adj[i, j] > 0:
                adj[i].append(j)
                indeg[j] += 1

    q = deque([u for u in nodes if indeg[u] == 0])
    level = {u: 0 for u in q}
    while q:
        u = q.popleft()
        for v in adj[u]:
            level[v] = max(level.get(v, 0), level[u] + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    for u in nodes:
        level.setdefault(u, 0)
    return level


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_birth_kernel(
    B_off,
    growth_rates,
    dag_adj,
    back_edges,
    idx2state,
    start_state,
    terminal_idxs,
    outfile,
    state_text=None,
):
    n = B_off.shape[0]
    nodes = list(range(n))
    terminal_set = set(terminal_idxs)

    # Node colors
    term_colors = {}
    for k, idx in enumerate(sorted(terminal_set)):
        term_colors[idx] = TERMINAL_COLORS[k % len(TERMINAL_COLORS)]

    def node_color(i):
        if i in terminal_set:
            return term_colors[i]
        if i == start_state:
            return ROOT_COLOR
        return HIDDEN_COLOR

    # Collect edges
    dag_edges  = [(i, j, float(dag_adj[i, j]))
                  for i in nodes for j in nodes
                  if dag_adj[i, j] > 0 and i != j]
    # back_edges already filtered to significant (above threshold) by caller

    all_weights = [w for _, _, w in dag_edges] + [w for _, _, w in back_edges]
    max_w = max(all_weights, default=1.0)

    def edge_attrs(w, dashed=False):
        pen = max(7.0 * (w / max_w), 0.8)
        arr = max(0.3, 0.1 * pen)
        attrs = {
            "penwidth":    f"{pen:.3g}",
            "arrowsize":   f"{arr:.3g}",
            "xlabel":      _html_pill(f"{w:.2g}"),
            "labelfontsize": "8",
        }
        if dashed:
            attrs["style"] = "dashed"
            attrs["color"] = "#999999"
        return attrs

    # DAG-based levels for rank= subgraphs
    levels = _dag_levels(nodes, dag_adj)
    if terminal_idxs:
        max_level = max(levels.values())
        max_level = max(max_level, max(levels.get(t, 0) for t in terminal_idxs))
        for t in terminal_idxs:
            levels[t] = max_level

    fmt = _infer_fmt(outfile)
    g = Digraph("G", format=fmt, engine="dot")
    g.attr(rankdir="TB", splines="spline", concentrate="false")
    if fmt == "png":
        g.attr(dpi="400")
    if state_text:
        g.attr(label=state_text, labelloc="t", labeljust="l")
    g.attr("node", shape="circle", style="filled", fontname="Helvetica-Bold",
           fontsize="10", penwidth="1", width="0.85", height="0.85", fixedsize="true")
    g.attr("edge", fontname="Helvetica", fontsize="8", arrowhead="normal")

    # Nodes
    for i in sorted(nodes):
        g.node(str(i), label=str(idx2state[i]), fillcolor=node_color(i))

    # DAG edges (solid)
    for u, v, w in dag_edges:
        g.edge(str(u), str(v), **edge_attrs(w, dashed=False))

    # Back-edges (dashed grey) — shown but don't influence layout
    for u, v, w in back_edges:
        g.edge(str(u), str(v), **edge_attrs(w, dashed=True))

    # Self-loops for growth rates
    max_lam = max(float(growth_rates[i]) for i in nodes)
    for i in nodes:
        lam = float(growth_rates[i])
        pen = max(7.0 * (lam / max_lam), 0.8)
        arr = max(0.3, 0.1 * pen)
        g.edge(str(i), str(i),
               penwidth=f"{pen:.3g}", arrowsize=f"{arr:.3g}",
               xlabel=_html_pill(f"λ={lam:.2g}"), labelfontsize="8")

    # Rank subgraphs: root pinned to top, terminals pinned to bottom,
    # intermediate levels get rank='same' for horizontal alignment.
    level2nodes = defaultdict(list)
    for u in nodes:
        level2nodes[levels[u]].append(u)

    for lvl in sorted(level2nodes):
        lvl_nodes = sorted(level2nodes[lvl])
        is_top    = start_state is not None and all(u == start_state for u in lvl_nodes)
        is_bottom = terminal_idxs and all(u in terminal_set for u in lvl_nodes)

        with g.subgraph() as s:
            s.attr(rank="min" if is_top else ("max" if is_bottom else "same"))
            for u in lvl_nodes:
                s.node(str(u))

    base = os.path.splitext(outfile)[0]
    g.render(base, cleanup=True)
    print(f"Saved to {outfile}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot unconstrained ClaSSE MLE birth kernel as a directed graph."
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Path to model_dict.pkl (or directory containing it).")
    parser.add_argument("-o", "--output", default=None,
                        help="Output file path (.pdf, .png, .svg). "
                             "Default: figures/birth_kernel.pdf next to model_dict.pkl.")
    parser.add_argument("--threshold", type=float, default=0.02,
                        help="Minimum birth-kernel probability to show an edge. "
                             "Default: 0.02.")
    args = parser.parse_args()

    # Resolve input path
    model_path = args.input
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, "model_dict.pkl")
    if not os.path.isfile(model_path):
        print(f"ERROR: model_dict.pkl not found at {model_path}")
        sys.exit(1)

    with open(model_path, "rb") as fp:
        model_dict = pickle.load(fp)

    # Output path
    if args.output is None:
        fig_dir = os.path.join(os.path.dirname(model_path), "figures")
        os.makedirs(fig_dir, exist_ok=True)
        outfile = os.path.join(fig_dir, "birth_kernel.pdf")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        outfile = args.output

    # Extract model components
    B = model_dict["daughter_kernel"].detach().numpy()
    growth_rates = model_dict["growth_rates"].detach().numpy()
    idx2state  = model_dict["idx2state"]
    start_state = model_dict.get("start_state")
    neg_llh = float(model_dict.get("neg_llh", float("nan")))
    if np.isnan(neg_llh):
        loss_path = os.path.join(os.path.dirname(model_path), "loss.txt")
        if os.path.isfile(loss_path):
            with open(loss_path) as fp:
                neg_llh = float(fp.read().strip())

    n = B.shape[0]

    # Identify terminal (observed) states.
    # Convention: start_state == n_obs; observed indices are 0 … start_state-1.
    # Fall back to checking for hidden-state naming pattern "U<int>".
    if start_state is not None:
        terminal_idxs = list(range(start_state))
    else:
        terminal_idxs = [
            i for i, s in idx2state.items()
            if not (isinstance(s, str) and s.startswith("U") and s[1:].isdigit())
        ]
    terminal_idxs.sort()

    print(f"States ({n} total): {idx2state}")
    print(f"Terminal idxs: {terminal_idxs}")
    print(f"Root (start_state): {start_state}")
    print(f"Threshold: {args.threshold}")

    # Build off-diagonal birth kernel, apply threshold
    B_off = B.copy()
    np.fill_diagonal(B_off, 0.0)
    B_thresh = np.where(B_off >= args.threshold, B_off, 0.0)

    n_sig = int((B_thresh > 0).sum())
    print(f"Significant edges (>= {args.threshold}): {n_sig}")

    # Convert thresholded graph to DAG
    dag_adj, back_edges = make_dag(B_thresh)

    if back_edges:
        print(f"Removed {len(back_edges)} back-edge(s) for DAG layout:")
        for u, v, w in back_edges:
            print(f"  {idx2state[u]} -> {idx2state[v]}  (w={w:.3f})")
    else:
        print("Graph is already a DAG — no edges removed.")

    state_text = f"neg-llh={neg_llh:.2f}  threshold={args.threshold}"

    draw_birth_kernel(
        B_off=B_thresh,
        growth_rates=growth_rates,
        dag_adj=dag_adj,
        back_edges=back_edges,
        idx2state=idx2state,
        start_state=start_state,
        terminal_idxs=terminal_idxs,
        outfile=outfile,
        state_text=state_text,
    )


if __name__ == "__main__":
    main()
