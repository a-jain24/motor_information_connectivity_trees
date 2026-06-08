"""stats.py — edge-set similarity and the CL-vs-FC conservation test (space-agnostic).

Implements the primary hypothesis test (cl_vs_fc_comparison_plan.md / Q1): CL-tree
edge sets are more conserved across individuals than the top-N FC edges.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def pairwise_jaccard(edge_sets: dict | list) -> list[float]:
    """Jaccard for every pair of subjects' edge sets → list over the C(n,2) pairs."""
    vals = list(edge_sets.values()) if isinstance(edge_sets, dict) else list(edge_sets)
    return [jaccard(a, b) for a, b in combinations(vals, 2)]


def conservation_test(cl_edge_sets: dict, fc_edge_sets: dict) -> dict:
    """Paired comparison of CL vs FC inter-subject Jaccard across all subject pairs.

    Returns mean/SD for each and a paired t-test (CL − FC).
    """
    from scipy.stats import ttest_rel

    subs = list(cl_edge_sets.keys())
    pairs = list(combinations(subs, 2))
    j_cl = [jaccard(cl_edge_sets[a], cl_edge_sets[b]) for a, b in pairs]
    j_fc = [jaccard(fc_edge_sets[a], fc_edge_sets[b]) for a, b in pairs]
    t, p = ttest_rel(j_cl, j_fc)
    return {
        "n_pairs": len(pairs),
        "cl_jaccard_mean": float(np.mean(j_cl)), "cl_jaccard_sd": float(np.std(j_cl)),
        "fc_jaccard_mean": float(np.mean(j_fc)), "fc_jaccard_sd": float(np.std(j_fc)),
        "delta": float(np.mean(j_cl) - np.mean(j_fc)),
        "t": float(t), "p": float(p),
        "j_cl": j_cl, "j_fc": j_fc,
    }
