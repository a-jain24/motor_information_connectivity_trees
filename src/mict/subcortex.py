"""subcortex.py — motor-thalamus ROI definition (motor_system_plan.md §D).

FC-seed method (inter-effector study): seed from the cortical motor ROIs, then per
hemisphere keep the thalamic CIFTI voxels whose FC to the seed exceeds a per-subject
percentile threshold. Operates entirely on the CIFTI subcortical grayordinates
(THALAMUS_LEFT / THALAMUS_RIGHT) — no separate atlas resampling needed; Morel labeling
(VLa/VLpd/VLpv) is an optional downstream refinement.
"""

from __future__ import annotations

import numpy as np

from . import cifti as _cifti


def define_thalamus_rois(rest_data: np.ndarray, cif, seed_cols: np.ndarray,
                         pct: float = 95.0) -> dict[str, np.ndarray]:
    """Return ``{L_Thal, R_Thal}`` grayordinate-column masks.

    rest_data : (T, G) resting-state grayordinate timeseries.
    cif       : a ``Cifti`` whose models include THALAMUS_LEFT/RIGHT (same G as rest_data).
    seed_cols : cortical motor ROI columns (e.g. all M1 effector vertices) → mean = seed.
    pct       : per-subject percentile; voxels with FC ≥ this within a hemisphere are kept.
    """
    seed = rest_data[:, np.asarray(seed_cols, dtype=int)].mean(axis=1)
    out: dict[str, np.ndarray] = {}
    for label, struct in [("L_Thal", _cifti.THALAMUS_LEFT), ("R_Thal", _cifti.THALAMUS_RIGHT)]:
        cols = cif.model(struct).columns
        x = rest_data[:, cols]                       # (T, n_vox)
        fc = _corr_to_seed(seed, x)
        thr = np.nanpercentile(fc, pct)
        sel = cols[fc >= thr]
        out[label] = np.asarray(sel, dtype=np.int32)
    return out


def _corr_to_seed(seed: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Pearson r between a 1-D seed and each column of x (vectorized)."""
    s = seed - seed.mean()
    xc = x - x.mean(axis=0)
    num = xc.T @ s
    den = np.sqrt((xc ** 2).sum(axis=0) * (s ** 2).sum()) + 1e-12
    return num / den
