"""conditional.py — discrete conditional mutual information (the MI-native directness test).

CMI(A;B|C) = Σ p(a,b,c) · log[ p(a,b,c)·p(c) / (p(a,c)·p(b,c)) ].

For an *indirect* pair routed through C, conditioning on C explains the dependence away
(CMI → 0); for a *direct* edge it survives. Used by direct_interpretable_topology.md A2.

Estimating a 3-way joint needs coarse bins (default K=8) — at K=100 the K³ histogram is
hopelessly undersampled at fMRI sample sizes.
"""

from __future__ import annotations

import numpy as np

from .mutual_info import discretize


def _disc(x: np.ndarray, num_bins: int) -> np.ndarray:
    return discretize(np.asarray(x).reshape(-1, 1), num_bins)[:, 0]


def mutual_info_pair(a: np.ndarray, b: np.ndarray, num_bins: int = 8) -> float:
    """MI(A;B) at the same coarse binning used for CMI (for a fair MI vs CMI comparison)."""
    da, db = _disc(a, num_bins), _disc(b, num_bins)
    K = num_bins
    joint = np.bincount(K * da + db, minlength=K * K).reshape(K, K).astype(float)
    p = joint / joint.sum()
    pa, pb = p.sum(1), p.sum(0)
    nz = p > 0
    pm = np.outer(pa, pb)
    return max(float(np.sum(p[nz] * np.log(p[nz] / pm[nz]))), 0.0)


def conditional_mutual_info(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                            num_bins: int = 8) -> float:
    """CMI(A;B|C) in nats, via the 3-way binned joint (coarse K for estimability)."""
    K = num_bins
    da, db, dc = _disc(a, K), _disc(b, K), _disc(c, K)
    idx = (da * K + db) * K + dc
    joint = np.bincount(idx, minlength=K ** 3).reshape(K, K, K).astype(float)
    joint /= joint.sum()
    pc = joint.sum(axis=(0, 1))           # (K,)        P(c)
    pac = joint.sum(axis=1)               # (Ka, Kc)    P(a,c)
    pbc = joint.sum(axis=0)               # (Kb, Kc)    P(b,c)
    nz = joint > 0
    num = joint * pc[None, None, :]
    den = pac[:, None, :] * pbc[None, :, :]
    val = np.sum(joint[nz] * np.log(num[nz] / den[nz]))
    return max(float(val), 0.0)


def explained_away(a, b, c, num_bins: int = 8) -> float:
    """Fraction of MI(A;B) removed by conditioning on C: 1 − CMI(A;B|C)/MI(A;B).

    ≈1 ⇒ the A–B dependence is indirect (routed through C); ≈0 ⇒ direct.
    """
    mi = mutual_info_pair(a, b, num_bins)
    if mi <= 1e-9:
        return 0.0
    return 1.0 - conditional_mutual_info(a, b, c, num_bins) / mi
