"""Tests for the trustworthiness module.

The implementation in ``trustworthiness`` must reproduce
``sklearn.manifold.trustworthiness`` bit-for-bit (up to ``atol=1e-10``) on
random, tie-free data. The tests also cover the edge cases that the
paper's formula prescribes (identity embedding scores 1.0, a random
permutation of the embedding scores badly for small ``k``) and verify the
input-validation surface.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.manifold import trustworthiness as sk_trustworthiness

from starfold.trustworthiness import (
    trustworthiness,
    trustworthiness_curve,
)

# 5 seeded datasets: vary N and the relative dimensionality of X_high/X_low.
# Each row: (seed, n_samples, n_features_high, n_features_low).
_DATASETS = [
    (0, 100, 8, 2),
    (1, 500, 6, 2),
    (2, 2000, 10, 3),
    (3, 2000, 4, 2),
    (4, 5000, 5, 2),
]

_K_VALUES = (5, 15, 30)


def _make_pair(
    seed: int,
    n: int,
    d_high: int,
    d_low: int,
) -> tuple[np.ndarray, np.ndarray]:
    """A (high, low) pair where the low embedding is a noisy linear projection."""
    rng = np.random.default_rng(seed)
    x_high = rng.normal(size=(n, d_high))
    projection = rng.normal(size=(d_high, d_low))
    x_low = x_high @ projection + 0.1 * rng.normal(size=(n, d_low))
    return x_high, x_low


@pytest.mark.parametrize(("seed", "n", "d_high", "d_low"), _DATASETS)
@pytest.mark.parametrize("k", _K_VALUES)
def test_matches_sklearn(
    seed: int,
    n: int,
    d_high: int,
    d_low: int,
    k: int,
) -> None:
    """Our implementation equals sklearn's to numerical precision."""
    x_high, x_low = _make_pair(seed, n, d_high, d_low)
    ours = trustworthiness(x_high, x_low, k=k)
    theirs = sk_trustworthiness(x_high, x_low, n_neighbors=k)
    assert ours == pytest.approx(theirs, abs=1e-10)


@pytest.mark.parametrize("k", _K_VALUES)
def test_identity_embedding_is_perfect(k: int) -> None:
    """An embedding equal to the input must score exactly 1.0."""
    rng = np.random.default_rng(42)
    x = rng.normal(size=(500, 6))
    assert trustworthiness(x, x, k=k) == 1.0


def test_random_permutation_is_poor() -> None:
    """Shuffling the embedding independently of the input destroys trust.

    For a uniformly random permutation, the expected score is

        1 - (n - 1 - k) * (n - k) / [(n - 1) * (2n - 3k - 1)],

    which for n=500, k=5 is about 0.503. The observed score must land near
    that value, which is dramatically below the identity score of 1.0.
    """
    rng = np.random.default_rng(123)
    n, k = 500, 5
    x_high = rng.normal(size=(n, 6))
    perm = rng.permutation(n)
    t_identity = trustworthiness(x_high, x_high, k=k)
    t_shuffled = trustworthiness(x_high, x_high[perm], k=k)
    expected_random = 1.0 - (n - 1 - k) * (n - k) / ((n - 1) * (2 * n - 3 * k - 1))
    assert t_identity == 1.0
    assert abs(t_shuffled - expected_random) < 0.05
    assert t_shuffled < t_identity - 0.4


def test_score_is_in_unit_interval() -> None:
    rng = np.random.default_rng(7)
    x_high = rng.normal(size=(300, 4))
    x_low = rng.normal(size=(300, 2))  # independent -> bad embedding
    t = trustworthiness(x_high, x_low, k=10)
    assert 0.0 <= t <= 1.0


def test_curve_matches_pointwise() -> None:
    """``trustworthiness_curve`` equals repeated ``trustworthiness`` calls."""
    rng = np.random.default_rng(2024)
    x_high = rng.normal(size=(400, 5))
    projection = rng.normal(size=(5, 2))
    x_low = x_high @ projection + 0.05 * rng.normal(size=(400, 2))
    ks = [5, 10, 20, 40]
    curve = trustworthiness_curve(x_high, x_low, k_values=ks)
    for k in ks:
        assert curve[k] == pytest.approx(trustworthiness(x_high, x_low, k=k), abs=1e-12)


def test_curve_preserves_input_order_and_deduplicates() -> None:
    rng = np.random.default_rng(99)
    x = rng.normal(size=(200, 3))
    curve = trustworthiness_curve(x, x, k_values=[15, 5, 15, 10])
    assert list(curve) == [15, 5, 10]
    for value in curve.values():
        assert value == 1.0


def test_curve_empty_k_values() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 3))
    assert trustworthiness_curve(x, x, k_values=[]) == {}


@pytest.mark.parametrize(
    ("k", "n"),
    [(0, 100), (-1, 100), (50, 100), (60, 100)],
)
def test_rejects_out_of_range_k(k: int, n: int) -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, 3))
    with pytest.raises(ValueError, match="k must satisfy"):
        trustworthiness(x, x, k=k)


def test_rejects_non_integer_k() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 3))
    with pytest.raises(TypeError, match="k must be an integer"):
        trustworthiness(x, x, k=5.5)  # type: ignore[arg-type]


def test_rejects_mismatched_shapes() -> None:
    rng = np.random.default_rng(0)
    x_high = rng.normal(size=(100, 3))
    x_low = rng.normal(size=(99, 2))
    with pytest.raises(ValueError, match="same number of samples"):
        trustworthiness(x_high, x_low, k=5)


def test_rejects_non_2d_inputs() -> None:
    with pytest.raises(ValueError, match="must be 2-D arrays"):
        trustworthiness(np.zeros(10), np.zeros((10, 2)), k=3)
