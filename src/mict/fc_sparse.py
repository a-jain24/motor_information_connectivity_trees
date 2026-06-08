"""fc_sparse.py — FC-based sparse-network estimators, for apples-to-apples vs the CL tree.

The Chow-Liu tree is the maximum spanning tree (MST) of pairwise mutual information —
a sparse, direct-dependency graph with exactly N-1 edges. To compare what FC gives us
at the *same* sparsity and with the *same* graph machinery, this module builds analogous
sparse graphs from linear FC:

  fc_mst     : MST of |Pearson r|              (Tewarie et al. 2015 — unbiased MST of FC)
  pcorr_mst  : MST of |partial correlation|    (direct connectivity; Smith et al. 2011)
  glasso     : sparse precision (graphical LASSO; Friedman 2008, Varoquaux & Craddock 2013)
  fc_topn    : density-matched top-(N-1) |Pearson r| edges

Theory hook (why this is exactly apples-to-apples): under Gaussianity
``MI(i,j) = -1/2 log(1 - r_ij^2)`` is monotonic in ``|r_ij|``, so the **MST of MI (CL tree)
equals the MST of |r| (fc_mst)**. Any CL≠fc_mst difference isolates the *non-Gaussian /
nonlinear* dependence MI captures and correlation misses. pcorr/glasso add the *direct vs
indirect* axis (removing transitive paths the way the tree assumption does, but linearly).

All estimators take an ROI timeseries ``ts`` of shape ``(T, N)`` and return an ``(N, N)``
symmetric adjacency with a zero diagonal; ``sparse_network`` wraps them into a uniform
result (adjacency + networkx tree + undirected edge set) for the figure/comparison code.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from .chow_liu import chow_liu_tree, cl_adjacency, edge_set, mi_to_graph
from .connectivity import compute_fc, compute_mi, top_n_fc_edges

METHODS = ["cl", "fc_mst", "pcorr_mst", "glasso", "fc_topn"]

METHOD_LABELS = {
    "cl": "Chow-Liu (MI MST)",
    "fc_mst": "FC MST (|r|)",
    "pcorr_mst": "Partial-corr MST",
    "glasso": "Graphical LASSO",
    "fc_topn": "FC top-(N-1)",
}


# ───────────────────────── estimators ─────────────────────────

def partial_correlation(ts: np.ndarray, shrink: bool = True) -> np.ndarray:
    """Partial correlation matrix from the (regularized) precision matrix.

    pc_ij = -prec_ij / sqrt(prec_ii prec_jj). Ledoit-Wolf shrinkage by default for a
    well-conditioned inverse with modest N (Smith et al. 2011 recommend regularization).
    """
    ts = np.asarray(ts, dtype=float)
    if shrink:
        from sklearn.covariance import LedoitWolf
        cov = LedoitWolf().fit(ts).covariance_
    else:
        cov = np.cov(ts, rowvar=False)
    prec = np.linalg.pinv(cov)
    d = np.sqrt(np.clip(np.diag(prec), 1e-12, None))
    pc = -prec / np.outer(d, d)
    np.fill_diagonal(pc, 0.0)
    return pc


def graphical_lasso(ts: np.ndarray) -> np.ndarray:
    """Sparse partial-correlation matrix via graphical LASSO (CV-tuned, robust fallback)."""
    from sklearn.covariance import GraphicalLassoCV, GraphicalLasso

    ts = np.asarray(ts, dtype=float)
    prec = None
    try:
        prec = GraphicalLassoCV().fit(ts).precision_
    except Exception:
        for alpha in (0.01, 0.05, 0.1, 0.2):
            try:
                prec = GraphicalLasso(alpha=alpha, max_iter=200).fit(ts).precision_
                break
            except Exception:
                continue
    if prec is None:  # last resort: shrunk partial correlation (dense)
        return partial_correlation(ts)
    d = np.sqrt(np.clip(np.diag(prec), 1e-12, None))
    pc = -prec / np.outer(d, d)
    np.fill_diagonal(pc, 0.0)
    return pc


def mst_from_weights(weights: np.ndarray) -> nx.Graph:
    """Maximum spanning tree on |weights| — the FC analog of the CL tree (N-1 edges)."""
    w = np.abs(np.asarray(weights, dtype=float))
    np.fill_diagonal(w, 0.0)
    G = mi_to_graph(w)  # reuse: builds a weighted graph on positive off-diagonals
    return nx.maximum_spanning_tree(G, algorithm="kruskal", weight="weight")


def _tree_adjacency(tree: nx.Graph, n: int) -> np.ndarray:
    adj = np.zeros((n, n), dtype=float)
    for u, v, dat in tree.edges(data=True):
        adj[u, v] = adj[v, u] = dat.get("weight", 1.0)
    return adj


# ───────────────────────── unified interface ─────────────────────────

def sparse_network(ts: np.ndarray, method: str, num_bins: int = 100) -> dict:
    """Build a sparse network from ``ts`` (T, N) by ``method`` (see METHODS).

    Returns ``{adjacency (N,N), tree (nx.Graph|None), edges (set of frozensets), weight}``.
    The MST methods (cl/fc_mst/pcorr_mst) give exactly N-1 edges = the CL sparsity.
    """
    ts = np.asarray(ts)
    n = ts.shape[1]

    if method == "cl":
        mi = compute_mi(ts, num_bins=num_bins)
        tree = chow_liu_tree(mi)
        return {"adjacency": cl_adjacency(mi), "tree": tree,
                "edges": edge_set(tree), "weight": "MI"}

    if method == "fc_mst":
        fc = compute_fc(ts)
        tree = mst_from_weights(fc)
        return {"adjacency": _tree_adjacency(tree, n), "tree": tree,
                "edges": edge_set(tree), "weight": "|r|"}

    if method == "pcorr_mst":
        pc = partial_correlation(ts)
        tree = mst_from_weights(pc)
        return {"adjacency": _tree_adjacency(tree, n), "tree": tree,
                "edges": edge_set(tree), "weight": "|partial r|"}

    if method == "glasso":
        pc = graphical_lasso(ts)
        adj = np.abs(pc)
        es = {frozenset((i, j)) for i in range(n) for j in range(i + 1, n) if adj[i, j] > 0}
        return {"adjacency": adj, "tree": None, "edges": es, "weight": "glasso |partial r|"}

    if method == "fc_topn":
        fc = compute_fc(ts)
        es = top_n_fc_edges(fc, n - 1)
        adj = np.zeros((n, n))
        for e in es:
            i, j = tuple(e)
            adj[i, j] = adj[j, i] = abs(fc[i, j])
        return {"adjacency": adj, "tree": None, "edges": es, "weight": "|r| (top N-1)"}

    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")


def edge_frequency_matrix(edge_sets: dict | list, n: int) -> np.ndarray:
    """Fraction of subjects in which each edge appears → (N, N) symmetric frequency."""
    sets = list(edge_sets.values()) if isinstance(edge_sets, dict) else list(edge_sets)
    freq = np.zeros((n, n))
    for es in sets:
        for e in es:
            i, j = tuple(e)
            freq[i, j] += 1
            freq[j, i] += 1
    if sets:
        freq /= len(sets)
    return freq
