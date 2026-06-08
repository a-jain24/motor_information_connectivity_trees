"""connectivity.py — FC, MI, and CL from an ROI-timeseries matrix (space-agnostic).

This is the shared downstream of the ``(T, n_ROI)`` pipeline boundary: every dataset
and image space funnels into these three measures so CL trees and FC matrices are
directly comparable. Folds the old ``msc_covariance.py`` (FC) together with
``mutual_info`` (MI) and ``chow_liu`` (CL).
"""

from __future__ import annotations

import numpy as np

from .chow_liu import cl_adjacency, chow_liu_tree
from .mutual_info import pairwise_mi


def compute_fc(ts: np.ndarray, zero_diagonal: bool = True) -> np.ndarray:
    """Pearson correlation FC matrix for ``ts`` (T, N) → (N, N)."""
    ts = np.asarray(ts)
    if ts.ndim != 2:
        raise ValueError(f"ts must be 2-D (T, N); got shape {ts.shape}")
    fc = np.corrcoef(ts, rowvar=False)
    fc = np.atleast_2d(fc)
    if zero_diagonal:
        np.fill_diagonal(fc, 0.0)
    return fc


def compute_mi(ts: np.ndarray, num_bins: int = 100, use_torch: bool = False) -> np.ndarray:
    """Pairwise MI matrix for ``ts`` (T, N) → (N, N). Thin wrapper over mutual_info."""
    return pairwise_mi(ts, num_bins=num_bins, use_torch=use_torch)


def compute_connectivity(
    ts: np.ndarray,
    num_bins: int = 100,
    use_torch: bool = False,
) -> dict:
    """Compute FC, MI, and the CL tree (+ adjacency) from one ROI-timeseries matrix.

    Returns a dict with keys ``fc``, ``mi``, ``cl_adj`` (all (N, N) arrays) and
    ``cl_tree`` (an ``nx.Graph``).
    """
    ts = np.asarray(ts)
    fc = compute_fc(ts)
    mi = compute_mi(ts, num_bins=num_bins, use_torch=use_torch)
    return {
        "fc": fc,
        "mi": mi,
        "cl_adj": cl_adjacency(mi),
        "cl_tree": chow_liu_tree(mi),
    }


def top_n_fc_edges(fc: np.ndarray, n: int) -> set:
    """Return the top-``n`` FC edges by |r| as a set of frozensets — for FC-vs-CL Jaccard."""
    fc = np.asarray(fc)
    N = fc.shape[0]
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    pairs.sort(key=lambda p: abs(fc[p[0], p[1]]), reverse=True)
    return {frozenset(p) for p in pairs[:n]}
