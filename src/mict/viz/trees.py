"""viz/trees.py — Chow-Liu tree plotting (ported from msc_chow_liu.draw_hierarchical_tree).

Separated from the engine so headless analysis never imports matplotlib. The default
node coloring uses ``mict.rois.region_color``; pass ``color_fn`` to override.
"""

from __future__ import annotations

import networkx as nx

from ..rois import region_color


def _hierarchy_pos(G: "nx.DiGraph", root: int, width: float = 2.0, vert_gap: float = 0.5) -> dict:
    """Top-down hierarchical layout for a BFS-rooted directed tree (no graphviz)."""
    def _count(node):
        children = list(G.successors(node))
        return 1 if not children else sum(_count(c) for c in children)

    def _recurse(node, left, right, y, pos):
        pos[node] = ((left + right) / 2, y)
        children = list(G.successors(node))
        if children:
            total = sum(_count(c) for c in children)
            dx = (right - left) / total
            x = left
            for child in children:
                sz = _count(child)
                _recurse(child, x, x + dx * sz, y - vert_gap, pos)
                x += dx * sz
        return pos

    return _recurse(root, 0, width, 0, {})


def draw_hierarchical_tree(
    tree: "nx.Graph",
    node_labels: list[str],
    root: int | None = None,
    title: str = "",
    ax=None,
    color_fn=None,
    label_fontsize: int = 7,
):
    """Draw a CL tree with a top-down hierarchical layout.

    ``color_fn(label) -> hex`` overrides the default region coloring. Root defaults
    to the highest-degree node for a balanced layout.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 8))
    if tree.number_of_nodes() == 0:
        ax.axis("off")
        return ax

    # Lay out each connected component (the consensus "tree" can be a forest) and
    # offset components horizontally so all nodes get a position.
    pos = {}
    x_off = 0.0
    for comp in nx.connected_components(tree):
        sub = tree.subgraph(comp)
        croot = root if (root is not None and root in comp) else max(sub.degree, key=lambda x: x[1])[0]
        cpos = _hierarchy_pos(nx.bfs_tree(sub, source=croot), croot)
        xs = [p[0] for p in cpos.values()]
        span = (max(xs) - min(xs)) if len(xs) > 1 else 1.0
        for nidx, (px, py) in cpos.items():
            pos[nidx] = (px + x_off, py)
        x_off += span + 0.5

    cfn = color_fn or region_color
    colors = [cfn(node_labels[n]) for n in tree.nodes()]

    weights = [tree[u][v].get("weight", 1.0) for u, v in tree.edges()]
    max_w = max(weights) if weights else 1.0
    widths = [1.0 + 3.0 * (w / max_w) for w in weights]

    nx.draw_networkx_edges(tree, pos, width=widths, alpha=0.6, edge_color="#555555", ax=ax)
    nx.draw_networkx_nodes(tree, pos, node_color=colors, node_size=140, ax=ax)

    y_vals = [y for _, y in pos.values()]
    offset = (max(y_vals) - min(y_vals)) * 0.04 if len(y_vals) > 1 else 0.04
    for n in tree.nodes():
        x, y = pos[n]
        ax.text(x, y + offset, node_labels[n], fontsize=label_fontsize,
                ha="left", va="bottom", rotation=45, rotation_mode="anchor", clip_on=False)

    ax.set_title(title, fontsize=10, pad=4)
    ax.axis("off")
    return ax


# ───────────────────────── publication figures (method-agnostic) ──────────────
# These work for the CL tree or any FC-based sparse network (fc_mst, pcorr_mst, …):
# pass that method's per-subject trees + the edge-frequency matrix.

# Consensus-frequency edge tiers (matches the original gcss fig6 styling).
def _edge_style(freq: float) -> tuple[float, str, float]:
    if freq >= 0.70:
        return 2.6, "solid", 0.95     # canonical (≥70% of subjects)
    if freq >= 0.40:
        return 1.5, "solid", 0.65     # intermediate (40–69%)
    return 0.8, "dashed", 0.45        # idiosyncratic (<40%)


def _component_pos(tree: "nx.Graph", vert_gap: float = 0.5) -> dict:
    """Forest-aware hierarchical positions for all nodes of ``tree``."""
    pos, x_off = {}, 0.0
    for comp in nx.connected_components(tree):
        sub = tree.subgraph(comp)
        croot = max(sub.degree, key=lambda x: x[1])[0]
        cpos = _hierarchy_pos(nx.bfs_tree(sub, source=croot), croot, vert_gap=vert_gap)
        xs = [p[0] for p in cpos.values()]
        span = (max(xs) - min(xs)) if len(xs) > 1 else 1.0
        for nidx, (px, py) in cpos.items():
            pos[nidx] = (px + x_off, py)
        x_off += span + 0.5
    return pos


def _draw_styled_tree(tree, roi_keys, freq, ax, hub_degree=3, title="",
                      show_labels=False, label_fontsize=6):
    """One tree with frequency-tiered edges, hub-node outlines, region colours.

    ``show_labels=False`` (default, for the subject grid) relies on the shared colour
    legend — matching the original fig6 declutter; set True for a single large panel.
    """
    if tree.number_of_nodes() == 0:
        ax.axis("off"); return
    pos = _component_pos(tree)
    deg = dict(tree.degree())
    for u, v in tree.edges():
        lw, ls, al = _edge_style(float(freq[u, v]) if freq is not None else 1.0)
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color="#444444", lw=lw, ls=ls, alpha=al, zorder=1)
    for n in tree.nodes():
        is_hub = deg.get(n, 0) >= hub_degree
        ax.scatter(*pos[n], s=180 if is_hub else 110, color=region_color(roi_keys[n]),
                   edgecolors="black" if is_hub else "none",
                   linewidths=1.6 if is_hub else 0, zorder=2)
    if show_labels:
        ys = [p[1] for p in pos.values()]
        off = (max(ys) - min(ys)) * 0.05 if len(ys) > 1 else 0.05
        for n in tree.nodes():
            ax.text(pos[n][0], pos[n][1] + off, roi_keys[n].replace("_", " "),
                    fontsize=label_fontsize, ha="left", va="bottom",
                    rotation=40, rotation_mode="anchor", clip_on=False)
    ax.set_title(title, fontsize=9, pad=3)
    ax.axis("off")


def _roi_legend(ax, roi_keys):
    """Region-colour + edge-style legend (shared across subject panels)."""
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    from ..rois import roi_regions, REGION_COLORS
    regions = {roi_regions().get(k) for k in roi_keys}
    handles = [mpatches.Patch(color=c, label=r) for r, c in REGION_COLORS.items() if r in regions]
    handles += [
        mlines.Line2D([], [], color="#444", lw=2.6, label="canonical ≥70%"),
        mlines.Line2D([], [], color="#444", lw=1.5, label="intermediate 40–69%"),
        mlines.Line2D([], [], color="#444", lw=0.8, ls="dashed", label="idiosyncratic <40%"),
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor="#999",
                      markeredgecolor="black", markersize=10, label="hub (degree ≥3)"),
    ]
    ax.legend(handles=handles, loc="center", ncol=3, frameon=False, fontsize=7)
    ax.axis("off")


def subject_tree_grid(trees: dict, roi_keys, freq, ncols: int = 5,
                      title: str = "", subtitles: dict | None = None, hub_degree: int = 3):
    """F1 — per-subject sparse-network grid (CL or FC method). Returns the figure.

    ``trees``  : {subject: nx.Graph}; ``freq``: (N,N) edge-frequency across these subjects;
    ``subtitles``: optional {subject: str} (e.g. mean MI) appended to each panel title.
    """
    import math
    import matplotlib.pyplot as plt

    subs = list(trees)
    n = len(subs)
    nrows = math.ceil(n / ncols) + 1  # extra row for the legend
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows))
    axes = axes.reshape(nrows, ncols)
    for ax in axes.ravel():
        ax.axis("off")
    for k, sub in enumerate(subs):
        ax = axes[k // ncols, k % ncols]
        t = f"{sub}" + (f"\n{subtitles[sub]}" if subtitles and sub in subtitles else "")
        _draw_styled_tree(trees[sub], roi_keys, freq, ax, hub_degree=hub_degree, title=t)
    _roi_legend(axes[-1, 0], roi_keys)  # legend in the spare bottom row
    if title:
        fig.suptitle(title, fontsize=13, y=0.99)
    fig.tight_layout()
    return fig


def labeled_tree_figure(tree, roi_keys, ax=None, edge_freq=None, n_subjects=None,
                        edge_label="weight", title="", hub_degree=3, node_scale=1.0,
                        label_fontsize=9):
    """A single readable, labeled tree (fig5 style): hierarchical layout, large region-
    coloured nodes, ROI names in white boxes, edges labelled by weight or ``k/N`` frequency.

    edge_label: 'weight' (MI / |r|, 2 dp), 'freq' (``k/N`` — needs edge_freq + n_subjects),
                or 'none'. Returns the figure (creates one if ``ax`` is None).
    """
    import matplotlib.pyplot as plt

    if tree.number_of_nodes() == 0:
        if ax is None:
            fig, ax = plt.subplots(figsize=(13, 9))
            ax.axis("off")
            return fig
        ax.axis("off"); return None

    # Generous vertical spacing so node labels (drawn below each node) and edge labels
    # don't collide — important for deep/linear trees (e.g. cast disuse chains).
    pos = _component_pos(tree, vert_gap=1.3)
    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
    xspan = (max(xs) - min(xs)) or 1.0
    yspan = (max(ys) - min(ys)) or 1.0

    fig = None
    if ax is None:  # size the canvas to the tree's shape so deep chains get vertical room
        w_in = max(9, min(22, 2.2 * xspan + 4))
        h_in = max(6, min(22, 1.7 * yspan + 3))
        fig, ax = plt.subplots(figsize=(w_in, h_in))

    deg = dict(tree.degree())
    ws = [tree[u][v].get("weight", 1.0) for u, v in tree.edges()]
    mx = max(ws) if ws else 1.0

    for u, v in tree.edges():
        w = tree[u][v].get("weight", 1.0)
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color="#777", lw=1.5 + 5.5 * (w / mx), alpha=0.7, zorder=1,
                solid_capstyle="round")
        if edge_label != "none":
            if edge_label == "freq" and edge_freq is not None and n_subjects:
                txt = f"{int(round(float(edge_freq[u, v]) * n_subjects))}/{n_subjects}"
            else:
                txt = f"{w:.2f}"
            # place the label 40% from the parent (upper) node → clear of the child's
            # label box (drawn below the child)
            hy, ly = max(pos[u][1], pos[v][1]), min(pos[u][1], pos[v][1])
            hx = pos[u][0] if pos[u][1] >= pos[v][1] else pos[v][0]
            lx = pos[v][0] if pos[u][1] >= pos[v][1] else pos[u][0]
            ax.text(hx + 0.4 * (lx - hx), hy + 0.4 * (ly - hy), txt,
                    fontsize=label_fontsize - 2, ha="center", va="center", zorder=3,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
    for nd in tree.nodes():
        hub = deg.get(nd, 0) >= hub_degree
        ax.scatter(*pos[nd], s=(900 + 500 * deg.get(nd, 0)) * node_scale,
                   color=region_color(roi_keys[nd]), edgecolors="black",
                   linewidths=2.0 if hub else 1.0, zorder=2)
    off = 0.30  # fixed fraction of the vert_gap=1.3 level spacing → label just below each node
    for nd in tree.nodes():
        ax.text(pos[nd][0], pos[nd][1] - off, roi_keys[nd].replace("_", " "),
                fontsize=label_fontsize, ha="center", va="top", zorder=4, weight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7", alpha=0.9))
    ax.margins(0.16)
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    return fig


def _wrap_label(label: str) -> str:
    toks = label.split("_")
    return (" ".join(toks[:-1]) + "\n" + toks[-1]) if len(toks) >= 3 else " ".join(toks)


def _relax_overlaps(pos: dict, min_dist: float = 0.22, iters: int = 120) -> dict:
    """Normalize a layout to a unit box, then push apart any nodes closer than ``min_dist``
    (so equal-size ellipses never overlap). Returns positions in the relaxed space."""
    import numpy as np
    nodes = list(pos)
    P = np.array([pos[k] for k in nodes], dtype=float)
    span = (P.max(0) - P.min(0))
    P = (P - P.min(0)) / (span.max() or 1.0)
    for _ in range(iters):
        moved = False
        for i in range(len(P)):
            for j in range(i + 1, len(P)):
                d = P[j] - P[i]
                dist = float(np.hypot(*d))
                if dist < min_dist:
                    if dist < 1e-6:
                        d = np.array([0.01 * (i - j + 0.5), 0.01]); dist = float(np.hypot(*d))
                    u = d / (dist + 1e-9)
                    shift = u * (min_dist - dist) / 2
                    P[i] -= shift; P[j] += shift; moved = True
        if not moved:
            break
    return {k: (float(P[i, 0]), float(P[i, 1])) for i, k in enumerate(nodes)}


def aesthetic_tree_figure(tree, roi_keys, edge_freq=None, n_subjects=None, ax=None,
                          title="", palette=None, edge_value="MI"):
    """Aesthetic per-subject CL tree (fig10 style): each ROI a distinctly-coloured ellipse
    with its name inside, organic (spring) layout, edge width ∝ weight with the value
    labelled, and edge colour by group-frequency tier (canonical ≥70% / intermediate /
    idiosyncratic). Returns the figure (or draws into ``ax``).
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse

    from ..rois import roi_palette

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 11))
        fig.patch.set_facecolor("#f5f5f5")
    ax.set_facecolor("#f5f5f5")
    if tree.number_of_nodes() == 0:
        ax.axis("off"); return fig

    palette = palette or roi_palette()
    n = tree.number_of_nodes()
    # Fruchterman-Reingold with strong repulsion → even spread (kamada packs hubs too tight
    # for large labelled ellipses); kamada layout as the deterministic init.
    pos = nx.spring_layout(tree, k=2.6 / (n ** 0.5), iterations=400, seed=1,
                           pos=nx.kamada_kawai_layout(tree))
    pos = _relax_overlaps(pos, min_dist=0.30)   # guarantee spacing > ellipse size
    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
    ew, eh = 0.20, 0.115                          # absolute (relaxed) units; < min_dist

    ws = [tree[u][v].get("weight", 1.0) for u, v in tree.edges()]
    mx = max(ws) if ws else 1.0
    for u, v in tree.edges():
        w = tree[u][v].get("weight", 1.0)
        f = float(edge_freq[u, v]) if edge_freq is not None else 1.0
        col = "#222222" if f >= 0.7 else "#888888" if f >= 0.4 else "#c0c0c0"
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=col,
                lw=1.0 + 5.5 * (w / mx), alpha=0.9, zorder=1, solid_capstyle="round")
        ax.text((pos[u][0]+pos[v][0])/2, (pos[u][1]+pos[v][1])/2, f"{edge_value}={w:.2f}",
                fontsize=7, ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    for nd in tree.nodes():
        c = palette.get(roi_keys[nd], "#888888")
        ax.add_patch(Ellipse(pos[nd], ew, eh, facecolor=c, edgecolor="white", lw=1.8, zorder=2))
        ax.text(pos[nd][0], pos[nd][1], _wrap_label(roi_keys[nd]), color="white",
                ha="center", va="center", fontsize=7.5, weight="bold", zorder=3)

    ax.set_xlim(min(xs) - ew, max(xs) + ew)
    ax.set_ylim(min(ys) - eh, max(ys) + eh)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=12, weight="bold")
    if edge_freq is not None and n_subjects:
        handles = [Line2D([], [], color="#222222", lw=3.2, label=f"canonical (≥{int(0.7*n_subjects)}/{n_subjects})"),
                   Line2D([], [], color="#888888", lw=2.2, label="intermediate"),
                   Line2D([], [], color="#c0c0c0", lw=1.4, label="idiosyncratic")]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, 1.02),
                  fontsize=7, frameon=True, framealpha=0.9,
                  title="group CL edge", title_fontsize=7)
    return fig


def condition_grid(cells: dict, roi_keys, row_labels, col_labels,
                   highlight: tuple | None = None, title: str = "", hub_degree: int = 3):
    """Grid of sparse networks: ``cells[(r, c)] = nx.Graph`` (rows × cols).

    Edge width ∝ edge weight; nodes region-coloured; hubs (deg ≥ hub_degree) outlined;
    an optional ``highlight=(roiA, roiB)`` edge (e.g. the casted-limb bilateral edge) is
    drawn in red with its weight labelled — for the cast pre→cast→post disuse story.
    Returns the figure.
    """
    import matplotlib.pyplot as plt

    nr, nc = len(row_labels), len(col_labels)
    hi = None
    if highlight and highlight[0] in roi_keys and highlight[1] in roi_keys:
        hi = frozenset((roi_keys.index(highlight[0]), roi_keys.index(highlight[1])))
    fig, axes = plt.subplots(nr, nc, figsize=(3.4 * nc, 3.2 * nr), squeeze=False)
    for r in range(nr):
        for c in range(nc):
            ax = axes[r][c]; ax.axis("off")
            g = cells.get((r, c))
            if g is None or g.number_of_nodes() == 0:
                continue
            pos = _component_pos(g)
            ws = [g[u][v].get("weight", 1.0) for u, v in g.edges()]
            mx = max(ws) if ws else 1.0
            deg = dict(g.degree())
            for u, v in g.edges():
                is_hi = hi is not None and frozenset((u, v)) == hi
                w = g[u][v].get("weight", 1.0)
                ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                        color="#d62728" if is_hi else "#555", lw=1.0 + 4.0 * (w / mx),
                        alpha=0.95 if is_hi else 0.6, zorder=3 if is_hi else 1)
                if is_hi:
                    ax.text((pos[u][0]+pos[v][0])/2, (pos[u][1]+pos[v][1])/2, f"{w:.2f}",
                            fontsize=8, color="#d62728", weight="bold", ha="center", zorder=4)
            for nd in g.nodes():
                is_hub = deg.get(nd, 0) >= hub_degree
                ax.scatter(*pos[nd], s=170 if is_hub else 100, color=region_color(roi_keys[nd]),
                           edgecolors="black" if is_hub else "none",
                           linewidths=1.5 if is_hub else 0, zorder=2)
            if r == 0:
                ax.set_title(col_labels[c], fontsize=12, pad=6)
            if c == 0:
                ax.text(-0.08, 0.5, row_labels[r], transform=ax.transAxes, fontsize=12,
                        rotation=90, va="center", ha="right", weight="bold")
    if title:
        fig.suptitle(title, fontsize=13, y=1.0)
    fig.tight_layout()
    return fig


def consensus_tree_figure(freq, roi_keys, n_subjects: int, layout: dict | None = None,
                          title: str = "", min_freq: float = 0.1):
    """F2 — consensus sparse network: MST of the edge-frequency, edge width ∝ frequency,
    each edge labelled ``k/N``, node size ∝ degree, region colours. Returns the figure.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    freq = np.asarray(freq)
    n = freq.shape[0]
    # consensus backbone = MST of the frequency graph (keeps it tree-sparse + connected)
    Gfull = nx.from_numpy_array(np.where(freq >= min_freq, freq, 0.0))
    consensus = (nx.maximum_spanning_tree(Gfull, weight="weight")
                 if Gfull.number_of_edges() else nx.Graph())
    consensus.add_nodes_from(range(n))

    if layout is None:
        layout = (nx.kamada_kawai_layout(Gfull) if Gfull.number_of_edges()
                  else nx.circular_layout(consensus))

    fig, ax = plt.subplots(figsize=(11, 9))
    deg = dict(consensus.degree())
    for u, v in consensus.edges():
        f = float(freq[u, v])
        lw, _, _ = _edge_style(f)
        x = [layout[u][0], layout[v][0]]; y = [layout[u][1], layout[v][1]]
        ax.plot(x, y, color="#666", lw=1.0 + 6.0 * f, alpha=0.3 + 0.6 * f, zorder=1,
                solid_capstyle="round")
        ax.text((x[0]+x[1])/2, (y[0]+y[1])/2, f"{int(round(f*n_subjects))}/{n_subjects}",
                fontsize=7, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7), zorder=3)
    ys = [p[1] for p in layout.values()]
    yspan = (max(ys) - min(ys)) if len(ys) > 1 else 1.0
    for nd in consensus.nodes():
        ax.scatter(*layout[nd], s=360 + 280 * deg.get(nd, 0),
                   color=region_color(roi_keys[nd]), edgecolors="black", linewidths=1.2, zorder=2)
        ax.text(layout[nd][0], layout[nd][1] - 0.045 * yspan, roi_keys[nd].replace("_", " "),
                fontsize=8.5, ha="center", va="top", zorder=4, weight="bold",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="0.7", alpha=0.85))
    ax.margins(0.12)
    ax.set_title(title or f"Consensus network (N={n_subjects})", fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    return fig
