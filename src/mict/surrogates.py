"""surrogates.py — linear surrogates that preserve correlations but destroy nonlinearity.

Used by the nonlinear-sensitivity test (nonlinear_sensitivity_plan.md A1): if the observed
CL↔FC-MST disagreement exceeds what these linear surrogates produce, the disagreement
reflects genuine non-Gaussian / nonlinear dependence rather than finite-sample noise.

  gaussian_surrogate         : multivariate Gaussian with the empirical covariance.
  phase_randomized_surrogate : multivariate Fourier phase randomization (Prichard & Theiler
                               1994) — same random phase per frequency across ROIs, so the
                               cross-spectrum (hence linear FC + autocorrelation) is preserved
                               while nonlinear phase structure is destroyed.
"""

from __future__ import annotations

import numpy as np


def gaussian_surrogate(ts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(T, N) → Gaussian surrogate with the same N×N covariance (no higher-order structure)."""
    ts = np.asarray(ts, dtype=float)
    T, N = ts.shape
    cov = np.cov(ts, rowvar=False)
    mean = ts.mean(axis=0)
    return rng.multivariate_normal(mean, cov, size=T)


def phase_randomized_surrogate(ts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(T, N) → multivariate phase-randomized surrogate preserving auto/cross-spectra.

    One random phase per non-DC/non-Nyquist frequency, shared across all ROIs (so inter-ROI
    phase *differences* — the linear cross-correlation — are preserved); higher-order phase
    coupling is randomized.
    """
    ts = np.asarray(ts, dtype=float)
    T, N = ts.shape
    F = np.fft.rfft(ts, axis=0)          # (nf, N)
    nf = F.shape[0]
    phases = rng.uniform(0, 2 * np.pi, size=nf)
    phases[0] = 0.0                       # keep DC
    if T % 2 == 0:
        phases[-1] = 0.0                 # Nyquist must stay real
    Fs = np.abs(F) * np.exp(1j * (np.angle(F) + phases[:, None]))
    return np.fft.irfft(Fs, n=T, axis=0)


SURROGATES = {"gaussian": gaussian_surrogate, "phase": phase_randomized_surrogate}
