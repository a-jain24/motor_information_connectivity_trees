"""Unit tests for the MI estimator (mict.mutual_info)."""

import numpy as np

from mict.mutual_info import pairwise_mi, discretize, entropy


def test_mi_independent_is_near_zero(independent_ts):
    # The histogram estimator is positively biased ~K^2/T (Paninski 2003), so the
    # bin/sample ratio must be sane for the absolute value to approach 0. At K=10,
    # T=4000 the bias is ~0.01 nats; at K=50 it would be ~0.2 (pure bias, not signal).
    mi = pairwise_mi(independent_ts, num_bins=10)
    assert mi[0, 1] < 0.05, f"expected MI ≈ 0 for independent TS, got {mi[0, 1]:.4f}"


def test_mi_dependent_is_large(dependent_ts):
    mi = pairwise_mi(dependent_ts, num_bins=50)
    assert mi[0, 1] > 1.0, f"expected MI >> 0 for identical TS, got {mi[0, 1]:.4f}"


def test_mi_independent_much_less_than_dependent(independent_ts, dependent_ts):
    """The bias-robust property the pipeline actually relies on (relative ranking)."""
    mi_indep = pairwise_mi(independent_ts, num_bins=50)[0, 1]
    mi_dep = pairwise_mi(dependent_ts, num_bins=50)[0, 1]
    assert mi_dep > 5 * mi_indep, f"dependent {mi_dep:.3f} not >> independent {mi_indep:.3f}"


def test_mi_matrix_symmetric_zero_diag(independent_ts):
    mi = pairwise_mi(independent_ts, num_bins=50)
    np.testing.assert_allclose(mi, mi.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(mi), 0.0, atol=1e-12)


def test_mi_nonnegative():
    rng = np.random.default_rng(3)
    ts = rng.standard_normal((2000, 8))
    mi = pairwise_mi(ts, num_bins=100)
    assert np.all(mi >= 0.0), f"negative MI found: {mi.min():.6f}"


def test_mi_monotone_with_coupling():
    """Stronger linear coupling ⇒ higher MI."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal(8000)
    weak = np.column_stack([x, 0.2 * x + rng.standard_normal(8000)])
    strong = np.column_stack([x, 0.9 * x + 0.1 * rng.standard_normal(8000)])
    assert pairwise_mi(strong, 60)[0, 1] > pairwise_mi(weak, 60)[0, 1]


def test_discretize_range_and_constant():
    rng = np.random.default_rng(1)
    ts = rng.standard_normal((500, 3))
    ts[:, 2] = 5.0  # constant column
    d = discretize(ts, num_bins=20)
    assert d.shape == (500, 3)
    assert d.min() >= 0 and d.max() <= 19
    assert np.all(d[:, 2] == 0)  # constant → bin 0


def test_entropy_positive():
    rng = np.random.default_rng(2)
    assert entropy(rng.standard_normal(4000), num_bins=50) > 0.0
