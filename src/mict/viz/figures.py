"""viz/figures.py — participant-specific + consensus motor-network figures.

`motor_report` emits the two headline figures (subject grid F1, consensus F2) for any
method — the CL tree or an FC-based sparse network (fc_mst, pcorr_mst, glasso, fc_topn) —
so CL and FC are compared apples-to-apples with the *same* figure style and sparsity.
"""

from __future__ import annotations

import json

import networkx as nx
import numpy as np

from ..fc_sparse import METHOD_LABELS, edge_frequency_matrix, sparse_network
from ..paths import dataset as dcfg
from ..paths import results_path


def load_subject_timeseries(dataset: str, space: str, subjects: list[str],
                            session: str = "all_sessions") -> dict:
    """Return ``{subject: (ts (T,N), roi_keys)}`` for subjects with a timeseries on disk."""
    out = {}
    for sub in subjects:
        d = results_path(dataset, space, "timeseries", sub, session=session, create=False)
        ts_path = d / "timeseries.npy"
        if not ts_path.exists():
            continue
        keys = json.load(open(d / "roi_keys.json"))["roi_keys"]
        out[sub] = (np.load(ts_path), keys)
    return out


def _graph(res: dict, n: int) -> nx.Graph:
    if res["tree"] is not None:
        return res["tree"]
    g = nx.from_numpy_array(res["adjacency"])          # glasso / top-N: weighted graph
    g.remove_edges_from([(u, v) for u, v, d in g.edges(data=True) if d["weight"] == 0])
    return g


def network_figures(dataset: str, space: str, method: str, subjects: list[str] | None = None,
                    session: str = "all_sessions", num_bins: int = 100, out_dir=None):
    """Build per-subject sparse networks by ``method`` and save F1 (grid) + F2 (consensus)."""
    import matplotlib
    matplotlib.use("Agg")
    from .trees import consensus_tree_figure, subject_tree_grid

    subjects = subjects or dcfg(dataset)["subjects"]
    ts_by_sub = load_subject_timeseries(dataset, space, subjects, session)
    if not ts_by_sub:
        print(f"  [{method}] no timeseries found for {dataset}/{session}")
        return
    keys = next(iter(ts_by_sub.values()))[1]
    n = len(keys)

    trees, edge_sets, subtitles = {}, {}, {}
    for sub, (ts, _k) in ts_by_sub.items():
        res = sparse_network(ts, method, num_bins=num_bins)
        trees[sub] = _graph(res, n)
        edge_sets[sub] = res["edges"]
        adj = np.asarray(res["adjacency"])
        nz = adj[np.triu_indices(n, 1)]
        nz = nz[nz > 0]
        subtitles[sub] = f"mean {res['weight']}={nz.mean():.2f}" if nz.size else ""

    freq = edge_frequency_matrix(edge_sets, n)
    nsub = len(trees)
    out_dir = out_dir or results_path(dataset, space, f"figures/{method}", create=True)

    label = METHOD_LABELS.get(method, method)
    f1 = subject_tree_grid(trees, keys, freq, subtitles=subtitles,
                           title=f"{dataset} — per-subject {label} networks (N={nsub})")
    f1.savefig(out_dir / "F1_subject_grid.pdf", bbox_inches="tight")
    f1.savefig(out_dir / "F1_subject_grid.png", dpi=150, bbox_inches="tight")

    f2 = consensus_tree_figure(freq, keys, nsub,
                               title=f"{dataset} — consensus {label} network (N={nsub})")
    f2.savefig(out_dir / "F2_consensus.pdf", bbox_inches="tight")
    f2.savefig(out_dir / "F2_consensus.png", dpi=150, bbox_inches="tight")

    import matplotlib.pyplot as plt
    plt.close("all")
    print(f"  [{method}] F1+F2 saved → {out_dir}  ({nsub} subjects, {n} ROIs)")
    return out_dir


def labeled_subjects_individual(dataset: str, space: str, method: str, subjects=None,
                                session: str = "all_sessions", num_bins: int = 100,
                                edge_label: str = "weight"):
    """One **full-size labeled tree per subject** (most readable for individual inspection).

    For per-condition data (e.g. cast pre/cast/post) pass ``session`` — it's appended to the
    filename. Saves to ``figures/<method>/subjects_labeled/<subject>[_<session>].{pdf,png}``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .trees import labeled_tree_figure

    subjects = subjects or dcfg(dataset)["subjects"]
    ts_by_sub = load_subject_timeseries(dataset, space, subjects, session)
    if not ts_by_sub:
        print(f"  [{method}] no timeseries for {dataset}/{session}"); return
    out_dir = results_path(dataset, space, f"figures/{method}/subjects_labeled", create=True)
    tag = "" if session == "all_sessions" else f"_{session}"
    label = METHOD_LABELS.get(method, method)
    for sub, (ts, keys) in ts_by_sub.items():
        res = sparse_network(ts, method, num_bins=num_bins)
        g = _graph(res, len(keys))
        fig = labeled_tree_figure(g, keys, edge_label=edge_label,
                                  title=f"{dataset} {sub}{tag.replace('_',' ')} — {label}")
        fig.savefig(out_dir / f"{sub}{tag}.pdf", bbox_inches="tight")
        fig.savefig(out_dir / f"{sub}{tag}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  [{method}{tag}] {len(ts_by_sub)} individual labeled trees → {out_dir}")


def aesthetic_subjects(dataset: str, space: str, method: str = "cl", subjects=None,
                       session: str = "all_sessions", num_bins: int = 100, out_dir=None):
    """One **aesthetic per-subject tree** (fig10 style) per subject — distinctly-coloured
    labelled ellipses, edge width ∝ weight, edges coloured by group-frequency tier.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .trees import aesthetic_tree_figure

    subjects = subjects or dcfg(dataset)["subjects"]
    ts_by_sub = load_subject_timeseries(dataset, space, subjects, session)
    if not ts_by_sub:
        print(f"  [{method}] no timeseries for {dataset}/{session}"); return
    keys = next(iter(ts_by_sub.values()))[1]
    n = len(keys)

    trees, edge_sets = {}, {}
    for sub, (ts, _k) in ts_by_sub.items():
        res = sparse_network(ts, method, num_bins=num_bins)
        trees[sub] = _graph(res, n)
        edge_sets[sub] = res["edges"]
    freq = edge_frequency_matrix(edge_sets, n)
    nsub = len(trees)
    ev = {"cl": "MI"}.get(method, "w")
    out_dir = out_dir or results_path(dataset, space, f"figures/{method}/aesthetic", create=True)
    tag = "" if session == "all_sessions" else f"_{session}"
    for sub, tree in trees.items():
        fig = aesthetic_tree_figure(tree, keys, edge_freq=freq, n_subjects=nsub,
                                    title=f"{sub}{tag.replace('_',' ')} — motor Chow-Liu tree", edge_value=ev)
        fig.savefig(out_dir / f"{sub}{tag}.pdf", bbox_inches="tight", facecolor="#f5f5f5")
        fig.savefig(out_dir / f"{sub}{tag}.png", dpi=150, bbox_inches="tight", facecolor="#f5f5f5")
        plt.close(fig)
    print(f"  [{method}{tag}] {nsub} aesthetic subject trees → {out_dir}")
    return out_dir


def motor_report(dataset: str, space: str = "surface",
                 methods=("cl", "fc_mst", "pcorr_mst"),
                 subjects=None, session: str = "all_sessions", num_bins: int = 100):
    """Generate F1+F2 for each method — the apples-to-apples figure set."""
    for m in methods:
        network_figures(dataset, space, m, subjects=subjects, session=session, num_bins=num_bins)


def labeled_report(dataset: str, space: str, method: str, subjects=None,
                   session: str = "all_sessions", num_bins: int = 100, out_dir=None):
    """Readable, **labeled** figures (fig5 style): labeled consensus tree + labeled
    per-subject grid — node names shown, edges labelled by k/N (consensus) or weight."""
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .trees import labeled_tree_figure

    subjects = subjects or dcfg(dataset)["subjects"]
    ts_by_sub = load_subject_timeseries(dataset, space, subjects, session)
    if not ts_by_sub:
        print(f"  [{method}] no timeseries for {dataset}/{session}"); return
    keys = next(iter(ts_by_sub.values()))[1]
    n = len(keys)

    trees, edge_sets = {}, {}
    for sub, (ts, _k) in ts_by_sub.items():
        res = sparse_network(ts, method, num_bins=num_bins)
        trees[sub] = _graph(res, n)
        edge_sets[sub] = res["edges"]
    freq = edge_frequency_matrix(edge_sets, n)
    nsub = len(trees)
    out_dir = out_dir or results_path(dataset, space, f"figures/{method}", create=True)
    label = METHOD_LABELS.get(method, method)

    # labeled consensus (MST of the edge-frequency graph), edges labelled k/N
    G = nx.from_numpy_array(np.where(freq >= 0.1, freq, 0.0))
    cons = nx.maximum_spanning_tree(G, weight="weight") if G.number_of_edges() else nx.Graph()
    fig = labeled_tree_figure(cons, keys, edge_freq=freq, n_subjects=nsub, edge_label="freq",
                              title=f"{dataset} — consensus {label} network (N={nsub})")
    fig.savefig(out_dir / "F2_consensus_labeled.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "F2_consensus_labeled.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # labeled per-subject grid (node names shown; edges by weight, not labelled to reduce clutter)
    ncols = 5
    subs = list(trees)
    nrows = math.ceil(len(subs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.3 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, sub in enumerate(subs):
        labeled_tree_figure(trees[sub], keys, ax=axes[k // ncols][k % ncols],
                            edge_label="none", title=sub, label_fontsize=7, node_scale=0.45)
    fig.suptitle(f"{dataset} — per-subject {label} networks (labeled)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "F1_subject_grid_labeled.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "F1_subject_grid_labeled.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{method}] labeled consensus + subject grid → {out_dir}")
