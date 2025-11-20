import os
import numpy as np
from collections import defaultdict, deque
from graphviz import Digraph

import os
import numpy as np
from collections import defaultdict, deque
from graphviz import Digraph

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
                        terminal_idxs=None):
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

    # Collect edges above threshold
    edges = []  # (u, v, w)
    for i in range(n):
        for j in range(n):
            w = float(adj_matrix[i, j])
            if w > threshold:
                edges.append((i, j, w))
    if self_edges is not None:
        for i in range(n):
            w = float(self_edges[i])
            # if w > threshold:
            edges.append((i, i, w))

    nodes_in_edges = {u for u, v, _ in edges} | {v for u, v, _ in edges}
    all_nodes = set(range(n))

    # Optional pruning from starting state
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
        nodes = reachable
    else:
        nodes = nodes_in_edges if nodes_in_edges else all_nodes

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
