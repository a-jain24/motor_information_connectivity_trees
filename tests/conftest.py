"""Shared fixtures: synthetic timeseries with known dependency structure."""

import numpy as np
import pytest


@pytest.fixture
def independent_ts():
    """Two independent Gaussian timeseries: MI should be near zero."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((4000, 2))


@pytest.fixture
def dependent_ts():
    """A signal and its copy: MI should be large (≈ entropy of the signal)."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(4000)
    return np.column_stack([x, x])


@pytest.fixture
def known_chain_ts():
    """Timeseries from a known 5-node linear chain A-B-C-D-E (AR coupling).

    Adjacent nodes are strongly coupled, distant nodes only indirectly — the CL
    tree should recover the chain (end nodes degree 1).
    """
    rng = np.random.default_rng(0)
    T, N = 8000, 5
    noise = 0.2
    ts = np.zeros((T, N))
    ts[:, 0] = rng.standard_normal(T)
    for i in range(1, N):
        ts[:, i] = 0.8 * ts[:, i - 1] + noise * rng.standard_normal(T)
    return ts


@pytest.fixture
def roi_names_18():
    return [
        "L_M1_hand", "R_M1_hand", "L_M1_foot", "R_M1_foot", "L_M1_face", "R_M1_face",
        "SMA", "L_PMd", "R_PMd", "L_PMv", "R_PMv",
        "L_Thal", "R_Thal",
        "L_Cereb_hand", "R_Cereb_hand", "L_Cereb_foot", "R_Cereb_foot",
    ]
