"""mutual_info.py — the single pairwise mutual-information estimator.

Histogram (binned) MI estimator, the same approach used throughout the original
``fmri_connectivity_trees`` pipeline, consolidated here into one implementation
(the old repo had it in ~4 places with subtly different zero-handling).

Input convention: a timeseries matrix ``ts`` of shape ``(T, N)`` — T timepoints,
N ROIs — matching how ROI timeseries are naturally laid out. Returns the
symmetric ``(N, N)`` MI matrix with a zero diagonal.

The numpy path is the default and is exercised by the test suite (no GPU needed).
For large N a torch path is available via ``use_torch=True`` if torch is installed;
for the motor ROI counts here (≤ ~18) numpy is already instant.

Discretization is per-column (each ROI binned over its own range), the standard
histogram-MI convention. Bias note: the binned estimator is positively biased at
small T (~K²/T; Paninski 2003) — negligible for concatenated all-session data,
non-trivial for single short runs.
"""

from __future__ import annotations

import numpy as np


def discretize(ts: np.ndarray, num_bins: int = 100) -> np.ndarray:
    """Discretize each column of ``ts`` (T, N) into ``num_bins`` bins → int (T, N).

    Bin edges are ``linspace(col.min(), col.max(), num_bins + 1)[1:-1]`` and values
    are assigned with ``np.digitize`` → integers in ``[0, num_bins - 1]``. A constant
    column maps to all zeros.
    """
    ts = np.asarray(ts, dtype=np.float64)
    if ts.ndim != 2:
        raise ValueError(f"ts must be 2-D (T, N); got shape {ts.shape}")
    T, N = ts.shape
    disc = np.zeros((T, N), dtype=np.int64)
    for j in range(N):
        col = ts[:, j]
        lo, hi = col.min(), col.max()
        if hi <= lo:
            continue  # constant column → bin 0
        edges = np.linspace(lo, hi, num_bins + 1)[1:-1]
        disc[:, j] = np.digitize(col, edges)
    return disc


def _mi_from_joint(joint: np.ndarray) -> float:
    """MI (nats) from an unnormalized joint count matrix (K, K)."""
    total = joint.sum()
    if total <= 0:
        return 0.0
    p = joint / total
    pa = p.sum(axis=1)
    pb = p.sum(axis=0)
    nz = p > 0
    pm = np.outer(pa, pb)
    mi = float(np.sum(p[nz] * np.log(p[nz] / pm[nz])))
    return max(mi, 0.0)  # clamp tiny negatives from floating error


def pairwise_mi(ts: np.ndarray, num_bins: int = 100, use_torch: bool = False) -> np.ndarray:
    """Pairwise mutual-information matrix for ``ts`` (T, N) → (N, N), zero diagonal.

    Parameters
    ----------
    ts : (T, N) array — T timepoints, N ROIs.
    num_bins : histogram bins per ROI (default 100, matching the MSC pipeline).
    use_torch : if True and torch is importable, use the GPU/torch path.
    """
    ts = np.asarray(ts)
    if ts.ndim != 2:
        raise ValueError(f"ts must be 2-D (T, N); got shape {ts.shape}")
    if use_torch:
        try:
            return _pairwise_mi_torch(ts, num_bins)
        except ImportError:
            pass  # fall through to numpy
    return _pairwise_mi_numpy(ts, num_bins)


def _pairwise_mi_numpy(ts: np.ndarray, num_bins: int) -> np.ndarray:
    K = num_bins
    disc = discretize(ts, num_bins)          # (T, N) int
    T, N = disc.shape
    mi = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        ai = disc[:, i]
        for j in range(i + 1, N):
            joint = np.bincount(K * ai + disc[:, j], minlength=K * K).reshape(K, K)
            m = _mi_from_joint(joint.astype(np.float64))
            mi[i, j] = mi[j, i] = m
    return mi


def _pairwise_mi_torch(ts: np.ndarray, num_bins: int) -> np.ndarray:
    import torch

    K = num_bins
    dev = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    x = torch.as_tensor(np.asarray(ts, dtype=np.float32), device=dev)  # (T, N)
    T, N = x.shape
    # per-column discretization
    disc = torch.empty((T, N), dtype=torch.long, device=dev)
    for j in range(N):
        col = x[:, j]
        lo, hi = col.min(), col.max()
        if hi <= lo:
            disc[:, j] = 0
        else:
            edges = torch.linspace(lo.item(), hi.item(), K + 1, device=dev)[1:-1]
            disc[:, j] = torch.bucketize(col, edges)
    mi = torch.zeros((N, N), device=dev)
    for i in range(N):
        ai = disc[:, i]
        for j in range(i + 1, N):
            joint = torch.bincount(K * ai + disc[:, j], minlength=K * K).reshape(K, K).float()
            tot = joint.sum()
            if tot <= 0:
                continue
            p = joint / tot
            pa = p.sum(1)
            pb = p.sum(0)
            pm = torch.outer(pa, pb)
            mask = p > 0
            m = torch.sum(p[mask] * torch.log(p[mask] / pm[mask])).item()
            mi[i, j] = mi[j, i] = max(m, 0.0)
    return mi.cpu().numpy()


def entropy(ts_col: np.ndarray, num_bins: int = 100) -> float:
    """Shannon entropy (nats) of a single discretized timeseries column."""
    disc = discretize(np.asarray(ts_col).reshape(-1, 1), num_bins)[:, 0]
    counts = np.bincount(disc, minlength=num_bins).astype(np.float64)
    p = counts / counts.sum()
    nz = p > 0
    return float(-np.sum(p[nz] * np.log(p[nz])))
