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
     back-edges as dashed arrows in grey.  Self-renewal probability (the
     diagonal of the birth kernel B, i.e. the probability that a division
     produces two daughters of the parent's own type) is shown as
     self-loops.

Usage:
    python scripts/plot_unconstrained_mle.py \
        -i /Users/william_hs/Desktop/Projects/troupe/results/TLS_no_endothelial_new_classe/sample_0.05/model_dict.pkl
        -o /Users/william_hs/Desktop/Projects/troupe/results/TLS_no_endothelial_new_classe/sample_0.05/figures/birth_kernel.pdf \
        --threshold 0.01
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

TLS_COLORS = {
    'NeuralTube':  '#1b9e77',
    'Somite':      '#d95f02',
    'Endoderm':    '#7570b3',
    'PCGLC':       '#e7298a',
    'Endothelial': '#66a61e',
    'NMP':         '#ff7f00',
}
FALLBACK_COLORS = [
    '#1b9e77', '#d95f02', '#7570b3', '#e7298a',
    '#66a61e', '#e6ab02', '#a6761d', '#f781bf', '#999999',
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

def _reachable_terminals(start, B_off, terminal_set, nodes):
    """BFS over the thresholded birth-kernel graph; return sorted reachable terminal idxs."""
    adj = {i: [j for j in nodes if B_off[i, j] > 0] for i in nodes}
    visited = set()
    queue = deque([start])
    while queue:
        u = queue.popleft()
        if u in visited:
            continue
        visited.add(u)
        for v in adj[u]:
            if v not in visited:
                queue.append(v)
    return sorted(t for t in terminal_set if t in visited)


def draw_birth_kernel(
    B_off,
    self_renewal_prob,
    dag_adj,
    back_edges,
    idx2state,
    start_state,
    terminal_idxs,
    outfile,
    state_text=None,
    forced_levels=None,
):
    n = B_off.shape[0]
    nodes = list(range(n))
    terminal_set = set(terminal_idxs)

    # Node colors: use TLS palette by name, fall back to generic list by rank.
    term_colors = {}
    for k, idx in enumerate(sorted(terminal_set)):
        name = str(idx2state[idx])
        term_colors[idx] = TLS_COLORS.get(name, FALLBACK_COLORS[k % len(FALLBACK_COLORS)])

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

    def edge_attrs(w):
        pen = max(7.0 * (w / max_w), 0.8)
        arr = max(0.3, 0.1 * pen)
        return {
            "penwidth":      f"{pen:.3g}",
            "arrowsize":     f"{arr:.3g}",
            "xlabel":        _html_pill(f"{w:.2g}"),
            "labelfontsize": "8",
        }

    # Level assignment: forced_levels (name→level dict) overrides DAG computation.
    if forced_levels is not None:
        name2idx = {str(idx2state[i]): i for i in nodes}
        levels = {}
        max_forced = max(forced_levels.values(), default=0)
        for name, lvl in forced_levels.items():
            if name in name2idx:
                levels[name2idx[name]] = lvl
        bottom = max_forced + 1
        for i in nodes:
            if i not in levels:
                # Terminals always at the bottom row; unassigned non-terminals
                # stay above it at the last explicitly forced level.
                levels[i] = bottom if i in terminal_set else max_forced
    else:
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

    # All nodes get a potency pie: wedge colors for each terminal reachable from
    # that state in the thresholded birth-kernel graph.  A terminal state always
    # includes itself; if it reaches other terminals those appear too.
    for i in sorted(nodes):
        label = str(idx2state[i])
        pot = _reachable_terminals(i, B_off, terminal_set, nodes)
        if pot:
            wedge_colors = [term_colors[t] for t in pot]
            g.node(str(i), label=label, style="wedged,filled",
                   fillcolor=":".join(wedge_colors))
        else:
            g.node(str(i), label=label, fillcolor=node_color(i))

    # All edges drawn solid with uniform styling (DAG edges + back-edges alike)
    for u, v, w in dag_edges + list(back_edges):
        g.edge(str(u), str(v), **edge_attrs(w))

    # Self-loops for self-renewal probability (same styling as other edges)
    max_srp = max(float(self_renewal_prob[i]) for i in nodes)
    for i in nodes:
        srp = float(self_renewal_prob[i])
        pen = max(7.0 * (srp / max_srp), 0.8)
        arr = max(0.3, 0.1 * pen)
        g.edge(str(i), str(i),
               penwidth=f"{pen:.3g}", arrowsize=f"{arr:.3g}",
               xlabel=_html_pill(f"{srp:.2g}"),
               labelfontsize="8")

    # Rank subgraphs: min level pinned to top, max level pinned to bottom.
    level2nodes = defaultdict(list)
    for u in nodes:
        level2nodes[levels[u]].append(u)

    all_levels = sorted(level2nodes)
    min_lvl, max_lvl = all_levels[0], all_levels[-1]
    for lvl in all_levels:
        with g.subgraph() as s:
            s.attr(rank="min" if lvl == min_lvl else ("max" if lvl == max_lvl else "same"))
            for u in sorted(level2nodes[lvl]):
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
    parser.add_argument("--state_levels", nargs="*", default=None, metavar="NAME=LEVEL",
                        help="Pin states to explicit rank levels (0 = top). "
                             "States not listed default to the bottom level. "
                             "Example: --state_levels U5=0 U6=1 NMP=2")
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
    # Diagonal of the (row-stochastic) birth kernel is the self-renewal
    # probability: P(both daughters are the parent's own type).
    self_renewal_prob = np.diag(B).copy()
    idx2state  = model_dict["idx2state"]
    start_state = model_dict.get("start_state")
    neg_llh = float(model_dict.get("neg_llh", float("nan")))
    if np.isnan(neg_llh):
        loss_path = os.path.join(os.path.dirname(model_path), "loss.txt")
        if os.path.isfile(loss_path):
            with open(loss_path) as fp:
                neg_llh = float(fp.read().strip())

    n = B.shape[0]

    # Identify true terminal states from idx2potency (the potency tuple lists
    # exactly the terminal names specified via --terminal_states).  Fall back to
    # all observed states (range(start_state)) if idx2potency is absent.
    idx2potency = model_dict.get("idx2potency")
    if idx2potency:
        potency_names = set(next(iter(idx2potency.values())))
        terminal_idxs = sorted(i for i, s in idx2state.items()
                               if str(s) in potency_names)
    elif start_state is not None:
        terminal_idxs = list(range(start_state))
    else:
        terminal_idxs = sorted(
            i for i, s in idx2state.items()
            if not (isinstance(s, str) and s.startswith("U") and s[1:].isdigit())
        )

    print(f"States ({n} total): {idx2state}")
    print(f"Terminal idxs: {terminal_idxs}")
    print(f"Root (start_state): {start_state}")
    print(f"Threshold: {args.threshold}")

    # Build off-diagonal birth kernel, apply threshold
    B_off = B.copy()
    np.fill_diagonal(B_off, 0.0)
    B_thresh = np.where(B_off >= args.threshold, B_off, 0.0)

    # Prevent disconnected nodes: for any non-root node with no incoming edges
    # after thresholding, restore its strongest incoming edge.
    for j in range(n):
        if j == start_state:
            continue
        if B_thresh[:, j].sum() == 0:
            i_best = int(np.argmax(B_off[:, j]))
            if B_off[i_best, j] > 0:
                B_thresh[i_best, j] = B_off[i_best, j]
                print(f"Restored weakest-incoming edge {idx2state[i_best]} -> "
                      f"{idx2state[j]} (w={B_off[i_best, j]:.4f}) to prevent disconnection")

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

    forced_levels = None
    if args.state_levels:
        forced_levels = {}
        for item in args.state_levels:
            name, lvl = item.split("=", 1)
            forced_levels[name.strip()] = int(lvl.strip())
        print(f"Forced levels: {forced_levels}")

    state_text = f"neg-llh={neg_llh:.2f}  threshold={args.threshold}"

    draw_birth_kernel(
        B_off=B_thresh,
        self_renewal_prob=self_renewal_prob,
        dag_adj=dag_adj,
        back_edges=back_edges,
        idx2state=idx2state,
        start_state=start_state,
        terminal_idxs=terminal_idxs,
        outfile=outfile,
        state_text=state_text,
        forced_levels=forced_levels,
    )


if __name__ == "__main__":
    main()
