"""chow_liu.py — Chow-Liu tree construction (space-agnostic).

A Chow-Liu tree is the maximum spanning tree of the complete weighted graph whose
edge weights are pairwise mutual information — the best tree-structured approximation
(in KL divergence) to the joint distribution (Chow & Liu, 1968).

Ported from ``msc_chow_liu.py``; plotting lives in ``mict.viz.trees``.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def mi_to_graph(mi_matrix: np.ndarray) -> nx.Graph:
    """Convert an (N, N) MI matrix to a fully connected weighted graph (w > 0 edges)."""
    mi = np.asarray(mi_matrix)
    n = mi.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            w = float(mi[i, j])
            if np.isfinite(w) and w > 0:
                G.add_edge(i, j, weight=w)
    return G


def chow_liu_tree(mi_matrix: np.ndarray) -> nx.Graph:
    """Return the Chow-Liu tree (maximum spanning tree) of the MI matrix.

    Uses Kruskal on the MI-weighted graph. For a fully positive MI matrix the tree
    has exactly N-1 edges and is connected; if the MI graph is disconnected
    (isolated zero-MI nodes), the result is a maximum spanning forest.
    """
    G = mi_to_graph(mi_matrix)
    return nx.maximum_spanning_tree(G, algorithm="kruskal", weight="weight")


def cl_adjacency(mi_matrix: np.ndarray) -> np.ndarray:
    """Return the (N, N) CL-tree adjacency with MI weights on tree edges, else 0."""
    mi = np.asarray(mi_matrix)
    n = mi.shape[0]
    tree = chow_liu_tree(mi)
    adj = np.zeros((n, n), dtype=float)
    for u, v, d in tree.edges(data=True):
        adj[u, v] = adj[v, u] = d["weight"]
    return adj


def edge_set(tree: nx.Graph) -> frozenset:
    """Undirected edge set as a frozenset of frozensets — for Jaccard comparisons."""
    return frozenset(frozenset(e) for e in tree.edges())


def consensus_vote_matrix(trees: dict | list, n: int) -> np.ndarray:
    """Fraction of trees containing each edge → (N, N) symmetric vote matrix."""
    tree_iter = trees.values() if isinstance(trees, dict) else trees
    tree_list = list(tree_iter)
    votes = np.zeros((n, n), dtype=float)
    for t in tree_list:
        adj = nx.to_numpy_array(t, nodelist=range(n), weight=None)
        votes += (adj > 0).astype(float)
    if tree_list:
        votes /= len(tree_list)
    return votes


def consensus_tree(trees: dict | list, n: int, threshold: float = 0.5) -> tuple[nx.Graph, np.ndarray]:
    """Build a consensus CL tree: MST of the edge-vote matrix above ``threshold``.

    Returns ``(consensus_tree, vote_matrix)``. The tree is empty if no edge passes.
    """
    votes = consensus_vote_matrix(trees, n)
    thresholded = np.where(votes >= threshold, votes, 0.0)
    G = nx.from_numpy_array(thresholded)
    if G.number_of_edges() == 0:
        return nx.Graph(), votes
    return nx.maximum_spanning_tree(G, algorithm="kruskal", weight="weight"), votes
