"""preprocessing.py — the audited nuisance-confound model (default).

Encodes the April-2026 preprocessing audit (documented in the old
``canonical_circuits/motor_cortex/analysis_plan.md``) as the single default:

  - ALL cosine DCT regressors present (dynamic; ~27/session → ~0.0075 Hz HPF),
    not just cosine00–03 (which gave a far-too-low ~0.001 Hz cutoff).
  - CSF + white matter.
  - 6 motion parameters + their 6 first derivatives.
  - ALL frame-censoring regressors: motion_outlier* and non_steady_state_outlier*.
  - TR = 2.2 s for MSC/cast1/PS1 (cast2/cast3 = 1.1 s — pass explicitly).

Task localization additionally uses ODD runs only (the `allruns` GLM inverts M1
effector z-maps; odd-run averaging is correct). Even runs are reserved for
split-half test-retest.
"""

from __future__ import annotations

import numpy as np

# Odd runs define fROIs (Bug-1 fix); even runs are held out for test-retest.
ODD_RUNS = [f"run-{i:02d}" for i in range(1, 21, 2)]   # run-01,03,...,19
EVEN_RUNS = [f"run-{i:02d}" for i in range(2, 21, 2)]  # run-02,04,...,20

_BASE_CONFOUNDS = [
    "csf", "white_matter",
    "trans_x", "trans_y", "trans_z",
    "rot_x", "rot_y", "rot_z",
    "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
    "rot_x_derivative1", "rot_y_derivative1", "rot_z_derivative1",
]


def select_confound_columns(columns) -> list:
    """Return the audited confound column names present in an fMRIPrep confounds TSV.

    Dynamically picks up every cosine* (full HPF basis) and every motion_outlier* /
    non_steady_state_outlier* (frame censoring), plus the fixed base set above.
    """
    cols = list(columns)
    cosine = [c for c in cols if c.startswith("cosine")]
    censor = [c for c in cols if c.startswith("motion_outlier")
              or c.startswith("non_steady_state")]
    base = [c for c in _BASE_CONFOUNDS if c in cols]
    return cosine + base + censor


def build_confound_matrix(confounds_df) -> np.ndarray:
    """Select the audited columns from an fMRIPrep confounds DataFrame → (T, K) array.

    NaNs (e.g. the first derivative/outlier rows) are zero-filled, as in the audit.
    """
    cols = select_confound_columns(confounds_df.columns)
    return confounds_df[cols].fillna(0.0).to_numpy(dtype=float)


def clean_timeseries(
    ts: np.ndarray,
    confounds: np.ndarray | None = None,
    t_r: float = 2.2,
    detrend: bool = True,
    standardize: str = "zscore_sample",
) -> np.ndarray:
    """Detrend + confound-regress + standardize an ROI timeseries (T, N) via nilearn.

    Thin wrapper over ``nilearn.signal.clean`` with the audited defaults
    (``t_r=2.2``; override for cast2/cast3 at TR 1.1 s).
    """
    from nilearn import signal

    return signal.clean(
        np.asarray(ts, dtype=float),
        confounds=confounds,
        detrend=detrend,
        standardize=standardize,
        t_r=t_r,
    )
