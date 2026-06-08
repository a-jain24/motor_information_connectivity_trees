"""Unit tests for Chow-Liu tree construction (mict.chow_liu)."""

import networkx as nx
import numpy as np

from mict.chow_liu import (
    mi_to_graph, chow_liu_tree, cl_adjacency, edge_set,
    consensus_vote_matrix, consensus_tree,
)
from mict.mutual_info import pairwise_mi


def _random_mi(n, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.random((n, n))
    m = (m + m.T) / 2
    np.fill_diagonal(m, 0.0)
    return m


def test_tree_has_n_minus_one_edges():
    tree = chow_liu_tree(_random_mi(11))
    assert tree.number_of_edges() == 10


def test_tree_is_connected():
    assert nx.is_connected(chow_liu_tree(_random_mi(11)))


def test_tree_includes_max_weight_edge():
    mi = np.array([[0, 0.9, 0.2, 0.1],
                   [0.9, 0, 0.3, 0.5],
                   [0.2, 0.3, 0, 0.7],
                   [0.1, 0.5, 0.7, 0]], dtype=float)
    assert chow_liu_tree(mi).has_edge(0, 1)


def test_cl_adjacency_matches_tree():
    mi = _random_mi(8, seed=5)
    adj = cl_adjacency(mi)
    tree = chow_liu_tree(mi)
    np.testing.assert_allclose(adj, adj.T)
    assert int((adj > 0).sum() / 2) == tree.number_of_edges()
    for u, v in tree.edges():
        assert adj[u, v] > 0


def test_cl_recovers_known_chain(known_chain_ts):
    """A-B-C-D-E chain: the two end nodes (0, 4) must have degree 1."""
    mi = pairwise_mi(known_chain_ts, num_bins=50)
    tree = chow_liu_tree(mi)
    deg = dict(tree.degree())
    assert deg[0] == 1 and deg[4] == 1, f"end-node degrees: {deg[0]}, {deg[4]}"
    # adjacency edges of the AR chain should be present
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        assert tree.has_edge(a, b), f"missing chain edge ({a},{b})"


def test_consensus_recovers_shared_edges():
    n = 6
    base = _random_mi(n, seed=1)
    # Force a dominant edge (0,1) in every subject; jitter the rest
    trees = {}
    for s in range(10):
        m = base + 0.01 * np.random.default_rng(s).random((n, n))
        m = (m + m.T) / 2
        m[0, 1] = m[1, 0] = 100.0
        np.fill_diagonal(m, 0.0)
        trees[f"S{s}"] = chow_liu_tree(m)
    votes = consensus_vote_matrix(trees, n)
    assert votes[0, 1] == 1.0  # edge (0,1) in all subjects
    ctree, _ = consensus_tree(trees, n, threshold=0.5)
    assert ctree.has_edge(0, 1)


def test_edge_set_jaccard_self_is_one():
    tree = chow_liu_tree(_random_mi(7, seed=2))
    es = edge_set(tree)
    jaccard = len(es & es) / len(es | es)
    assert jaccard == 1.0
