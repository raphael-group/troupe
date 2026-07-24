import os
import numpy as np
from collections import defaultdict, deque
from graphviz import Digraph
import matplotlib.pyplot as plt


def _augment_edges_for_required_nodes(adj_matrix, edges, required_nodes, active_nodes=None):
    """Ensure required nodes are present and have at least one incoming edge.

    This is intended for visualization only. It preserves rare observed states
    that would otherwise disappear under a global edge threshold.
    """
    if required_nodes is None:
        return edges, set(active_nodes) if active_nodes is not None else set()

    nodes = set(active_nodes) if active_nodes is not None else set()
    required_nodes = set(required_nodes)
    nodes |= required_nodes

    augmented_edges = list(edges)
    existing_incoming = {v for u, v, _ in augmented_edges if u != v}
    n = adj_matrix.shape[0]

    for target in sorted(required_nodes):
        if target in existing_incoming:
            continue

        candidate_sources = [
            u for u in nodes
            if 0 <= u < n and u != target and float(adj_matrix[u, target]) > 0.0
        ]
        if not candidate_sources:
            candidate_sources = [
                u for u in range(n)
                if u != target and float(adj_matrix[u, target]) > 0.0
            ]
        if not candidate_sources:
            continue

        best_source = max(candidate_sources, key=lambda u: float(adj_matrix[u, target]))
        augmented_edges.append((best_source, target, float(adj_matrix[best_source, target])))
        nodes.add(best_source)
        existing_incoming.add(target)

    return augmented_edges, nodes


def compute_active_graph(adj_matrix, threshold=0, totipotent_state=None,
                          terminal_idxs=None, self_edges=None):
    """Compute the pruned node/edge set used for plotting.

    Mirrors the reachability pruning performed by ``draw_weighted_graph`` so
    that other plots (e.g., growth-rate bar charts) can be restricted to
    exactly the same set of active states.

    Inputs
        adj_matrix:        Numpy adjacency matrix.
        threshold:          Set all edges below this value to 0
        totipotent_state:   Starting state
        terminal_idxs:      Row idxs of terminal states
        self_edges:         Optional self-edge weights (e.g., growth rates)

    Returns:
        (nodes, edges) where nodes is a sorted list of active state indices
        and edges is a list of (u, v, w) tuples that survived thresholding
        and pruning.
    """
    n = adj_matrix.shape[0]

    edges = []  # (u, v, w)
    for i in range(n):
        for j in range(n):
            w = float(adj_matrix[i, j])
            if w > threshold:
                edges.append((i, j, w))
    if self_edges is not None:
        for i in range(n):
            w = float(self_edges[i])
            edges.append((i, i, w))

    nodes_in_edges = {u for u, v, _ in edges} | {v for u, v, _ in edges}
    all_nodes = set(range(n))
    required_nodes = set(terminal_idxs) if terminal_idxs is not None else set()

    if totipotent_state is not None:
        adj = defaultdict(list)
        for u, v, w in edges:
            adj[u].append(v)
        reachable = set()
        q = deque([totipotent_state])
        reachable.add(totipotent_state)
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in reachable:
                    reachable.add(v); q.append(v)
        edges = [(u, v, w) for (u, v, w) in edges if u in reachable and v in reachable]
        edges, required_augmented_nodes = _augment_edges_for_required_nodes(
            adj_matrix, edges, required_nodes, active_nodes=reachable
        )
        nodes = reachable | required_augmented_nodes
    else:
        seed_nodes = nodes_in_edges if nodes_in_edges else all_nodes
        edges, required_augmented_nodes = _augment_edges_for_required_nodes(
            adj_matrix, edges, required_nodes, active_nodes=seed_nodes
        )
        nodes = seed_nodes | required_augmented_nodes

    return sorted(nodes), edges


def draw_weighted_graph(adj_matrix,
                        outfile,
                        threshold=0,
                        node_labels=None,       # int -> str dict
                        node_colors=None,       # str -> hex dict (keys should be node labels)
                        state_text=None,
                        totipotent_state=None,  # starting_state
                        state2potency=None,     # int -> Iterable[int] (terminals)
                        self_edges=None,
                        no_node_labels=False,
                        terminal_idxs=None,
                        scale_by_transitions=False):
    """
    Inputs
        adj_matrix:         Numpy adjacency matrix.
        outfile:            Path to save the plot to.
        threshold:          Set all edges below this value to 0
        node_labels:        Maps columns of the adjacency matrix to their labels
        node_colors:        Maps labels to their colors
        state_text:         Optional text to include in the figure E.g., likelihood of the matrix
        totipotent_state:   Starting state
        state2potency:      Row index to the terminal states (idxs) that it can transition into
        self_edges:         Optional plotting of self-edges (e.g., growth rates)
        no_node_labels:     If True, don't include node labels
        terminal_idxs:      Row idxs of terminal states
    """
    n = adj_matrix.shape[0]
    if node_labels is None:
        node_labels = {i: str(i) for i in range(n)}

    def color_for(i):
        if node_colors is None:
            return 'lightblue'
        lab = node_labels[i]
        if isinstance(node_colors, dict):
            return node_colors.get(lab, 'lightblue')
        try:
            return node_colors[i]
        except Exception:
            return 'lightblue'

    # Collect edges above threshold and prune to the active node set
    nodes_list, edges = compute_active_graph(
        adj_matrix, threshold, totipotent_state, terminal_idxs, self_edges
    )
    nodes = set(nodes_list)

    # Degrees (ignoring self-loops for terminal detection)
    outdeg = defaultdict(int)
    for u, v, w in edges:
        if u != v:
            outdeg[u] += 1
    if terminal_idxs is None:
        terminals = sorted([i for i in nodes if outdeg[i] == 0])
    else:
        terminals = terminal_idxs

    max_w = max([w for _, _, w in edges], default=1.0)
    max_transition = max([w for u, v, w in edges if u != v], default=1.0)
    max_growth = max([w for u, v, w in edges if u == v], default=1.0)

    # DAG detection
    dag_levels = _dag_levels(nodes, edges)  # returns dict node->level or None if cyclic
    is_dag = dag_levels is not None                     # NEW
    use_circle = not is_dag and (totipotent_state is not None)  # NEW

    # Engine selection + (for non-circle DAGs) level tweaking
    if is_dag and not use_circle:
        # Force all terminals to global bottom (layered DOT layout)
        max_level = max(dag_levels.values()) if dag_levels else 0
        if terminals:
            max_level = max(max_level, max(dag_levels.get(t, 0) for t in terminals))
            for t in terminals:
                dag_levels[t] = max_level
        use_engine = 'dot'
    else:
        # Either non-DAG, or DAG where we want a circular layout (use_circle = True)
        if use_circle:
            use_engine = 'twopi'  # NEW: radial / circular around root
        else:
            # Fallback: previous behavior
            use_engine = 'dot' if totipotent_state is not None else 'sfdp'

    g = Digraph('G', format=_infer_fmt(outfile), engine=use_engine)
    g.attr(rankdir='TB', splines='spline', concentrate='false')
    if outfile.endswith('.png'):
        g.attr(dpi='400')
    if state_text:
        g.attr(label=state_text, labelloc='t', labeljust='l')
    g.attr('node', shape='circle', style='filled', fontname='Helvetica-Bold',
           fontsize='10', penwidth='1', width='0.85', height='0.85', fixedsize='true')
    g.attr('edge', fontname='Helvetica', fontsize='8', arrowhead='normal')

    # If circular DAG layout requested, set root at the starting state
    if use_circle:  # NEW
        g.attr(root=str(totipotent_state))
        # --- spacing tweaks ---
        # minimum separation between nodes (in inches)
        # g.attr(nodesep='0.9')       # default is 0.25

        # vertical separation between ranks (for dot-style layouts)
        g.attr(ranksep='1.3')       # default is 0.5

        # radial layout: spread circles apart
        g.attr(mindist='2.0')   # minimum distance between nodes on same circle
        # g.attr(sep='+10')       # extra separation between clusters (in points)

        # ----------------------

    # Nodes (handle wedged potency fills)
    for i in sorted(nodes):
        if no_node_labels:
            label = ""
        else:
            label = node_labels.get(i, str(i))
        if state2potency is not None and i in state2potency:
            pot = [t for t in state2potency[i] if t in nodes]
            if pot:
                wedge_colors = [color_for(t) for t in pot]
                g.node(str(i), label=label, style='wedged,filled', fillcolor=':'.join(wedge_colors))
                continue
        g.node(str(i), label=label, fillcolor=color_for(i))

    # Edges with scaled widths/labels
    for u, v, w in edges:
        if scale_by_transitions:
            if u == v:
                max_w = max_growth
            else:
                max_w = max_transition
        pen = max(7.0 * (w / max_w), 1.0)
        arr = max(0.3, 0.1 * pen)
        attrs = {'penwidth': f'{pen:.3g}', 'arrowsize': f'{arr:.3g}'}
        if w >= 0.1 * max_w or not use_circle:  # NOTE: Uncomment if you only want large edge labels
            attrs['xlabel'] = _html_pill(f'{w:.2g}')
            attrs['labelfontsize'] = '8'
        g.edge(str(u), str(v), **attrs)

    # Apply ranks if DAG and we're *not* using the circular layout
    if dag_levels is not None and nodes and not use_circle:  # NEW condition
        # Group by level
        level2nodes = defaultdict(list)
        for u in nodes:
            level2nodes[dag_levels[u]].append(u)
        for lvl in sorted(level2nodes.keys()):
            with g.subgraph() as s:
                s.attr(rank='same')
                for u in sorted(level2nodes[lvl]):
                    s.node(str(u))

    # Fallback layering if not DAG: terminals still at bottom; BFS layering if totipotent_state
    elif dag_levels is None and terminals:  # NEW: make sure this only triggers when not DAG
        with g.subgraph() as s:
            s.attr(rank='max')
            for t in terminals:
                s.node(str(t))
        if totipotent_state is not None:
            dists = _bfs_distances(nodes, edges, start=totipotent_state)
            layers = defaultdict(list)
            for i, d in dists.items():
                layers[d].append(i)
            for d in sorted(layers.keys()):
                with g.subgraph() as s:
                    s.attr(rank='same')
                    for i in sorted(layers[d]):
                        s.node(str(i))

    # Render
    base, ext = os.path.splitext(outfile)
    out_base = base if ext else outfile
    g.render(out_base, cleanup=True)


def _infer_fmt(path):
    ext = os.path.splitext(path)[1].lower().lstrip('.')
    return ext if ext in {'png', 'pdf', 'svg', 'jpg', 'jpeg'} else 'png'

def _html_pill(text):
    return f'<<TABLE BORDER="0" CELLBORDER="0" CELLPADDING="1" CELLSPACING="0" BGCOLOR="white"><TR><TD>{text}</TD></TR></TABLE>>'


def _bfs_distances(nodes, edges, start):
    adj = defaultdict(list)
    for u, v, _ in edges:
        adj[u].append(v)
    dist = {start: 0}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v in nodes and v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    for i in nodes:
        dist.setdefault(i, 10**9)  # should not happen if pruned; keeps shape
    return dist


def _dag_levels(nodes, edges):
    """
    Return node->level (0 for sources, strictly increasing along edges) if acyclic,
    else return None. Self-loops are ignored for the acyclicity test and levels.
    """
    nodes = set(nodes)
    adj = defaultdict(list)
    indeg = defaultdict(int)
    for u, v, _ in edges:
        if u == v:
            continue  # ignore self-loops for DAG test/levels
        if u in nodes and v in nodes:
            adj[u].append(v)
            indeg[v] += 1
            indeg.setdefault(u, indeg.get(u, 0))

    # Include isolated nodes as sources
    for u in nodes:
        indeg.setdefault(u, 0)

    # Kahn's algorithm with longest-path style leveling
    from collections import deque
    q = deque([u for u in nodes if indeg[u] == 0])
    if not q and nodes:
        return None  # no sources -> has cycles (ignoring self-loops)

    level = {u: 0 for u in q}  # sources at level 0
    seen = 0
    while q:
        u = q.popleft()
        seen += 1
        for v in adj[u]:
            # ensure v's level >= u+1
            level[v] = max(level.get(v, 0), level[u] + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if seen < len(nodes):
        return None  # cycle detected (ignoring self-loops)

    # Assign level 0 to any isolated nodes not visited (safety)
    for u in nodes:
        level.setdefault(u, 0)
    return level


def plot_growth_rate_bar_chart(growth_rates,
                                outfile,
                                nodes,
                                node_labels=None,
                                node_colors=None,
                                state2potency=None,
                                bar_color='#4c72b0',
                                label_states=False):
    """Bar plot of per-state growth rates, x-axis annotated with potency pies.

    Each bar's height is the inferred growth rate for that state. Below each
    bar, a small pie chart is drawn whose wedges indicate the state's
    potency (terminal fates it can differentiate into), colored the same way
    as the corresponding nodes in ``draw_weighted_graph``. Each pie is drawn
    as its own tiny vector Axes (not a rasterized image), so it stays crisp
    at any zoom level or output resolution.

    Inputs
        growth_rates:  Array-like of growth rates, indexed by state idx.
        outfile:       Path to save the plot to.
        nodes:         Iterable of state idxs to include (left to right).
        node_labels:   idx -> str, used to look up colors/labels.
        node_colors:   label -> hex color dict (terminal-state colors).
        state2potency: idx -> tuple of terminal idxs, used to build pie wedges.
                       States absent from this mapping (or with no filtered
                       potency) are drawn as solid circles using their own color.
        bar_color:     Fill color for the bars.
        label_states:  If True, also print the state label under each pie.
    """
    nodes = list(nodes)
    if not nodes:
        return
    if node_labels is None:
        node_labels = {i: str(i) for i in nodes}

    def color_for(i):
        if node_colors is None:
            return 'lightblue'
        lab = node_labels.get(i, str(i))
        if isinstance(node_colors, dict):
            return node_colors.get(lab, 'lightblue')
        try:
            return node_colors[i]
        except Exception:
            return 'lightblue'

    heights = [float(growth_rates[i]) for i in nodes]

    fig_width = max(4.0, 1.0 * len(nodes))
    fig_height = 4.5
    figsize = (fig_width, fig_height)
    left_margin, right_margin = 0.15, 0.97
    top_margin = 0.95
    bottom_margin = 0.34 if label_states else 0.26
    fig, ax = plt.subplots(figsize=figsize)
    # Some callers (e.g. evaluate_results.py) set rcParams['figure.autolayout']
    # globally, which would silently re-run tight_layout at save time and
    # shift the manually-positioned pie axes below out from under their bars.
    fig.set_layout_engine('none')
    fig.subplots_adjust(left=left_margin, right=right_margin,
                         top=top_margin, bottom=bottom_margin)
    x = np.arange(len(nodes))
    bars = ax.bar(x, heights, color=bar_color, width=0.6, zorder=3,
                   edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Growth rate')
    xlim = (-0.6, len(nodes) - 0.4)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, max(heights) * 1.12 if max(heights) > 0 else 1.0)
    ax.set_xticks([])
    ax.spines[['top', 'right']].set_visible(False)
    ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=8)

    # Each pie is its own small vector Axes, positioned in figure-fraction
    # coordinates directly beneath its bar. This (rather than rasterizing an
    # icon image) keeps the wedges crisp at any zoom or output resolution.
    ax_pos = ax.get_position()
    inches_per_unit = ax_pos.width * fig_width / (xlim[1] - xlim[0])
    diameter_in = 0.65 * inches_per_unit
    w_frac = diameter_in / fig_width
    h_frac = diameter_in / fig_height
    gap_frac = 0.05 / fig_height  # small gap between axes bottom and pie top

    for xi, i in zip(x, nodes):
        pot = state2potency.get(i) if state2potency else None
        wedge_colors = [color_for(t) for t in pot] if pot else [color_for(i)]

        axes_frac_x = (xi - xlim[0]) / (xlim[1] - xlim[0])
        fig_x = ax_pos.x0 + axes_frac_x * ax_pos.width
        pie_y0 = ax_pos.y0 - gap_frac - h_frac
        pie_ax = fig.add_axes([fig_x - w_frac / 2, pie_y0, w_frac, h_frac])
        pie_ax.set_xticks([])
        pie_ax.set_yticks([])
        for spine in pie_ax.spines.values():
            spine.set_visible(False)
        pie_ax.patch.set_alpha(0.0)
        sizes = [1.0 / len(wedge_colors)] * len(wedge_colors)
        pie_ax.pie(sizes, colors=wedge_colors,
                    wedgeprops=dict(edgecolor='black', linewidth=0.6))
        pie_ax.set_aspect('equal')

        if label_states:
            pie_ax.text(0.5, -0.18, node_labels.get(i, str(i)),
                        transform=pie_ax.transAxes, ha='center', va='top',
                        fontsize=7)

    fig.savefig(outfile)
    plt.close(fig)
