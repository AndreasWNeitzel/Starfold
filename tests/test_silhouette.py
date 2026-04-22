"""Tests for :mod:`starfold.silhouette`.

The chunked implementation must agree with
:func:`sklearn.metrics.silhouette_score` at ``atol=1e-10`` on small
inputs. Singleton clusters, outliers, non-contiguous labels, and
single-cluster corner cases are also covered.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import silhouette_samples, silhouette_score

from starfold.silhouette import SilhouetteResult, chunked_silhouette


@pytest.fixture
def well_separated_three_clusters() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    parts = [rng.normal(loc, 0.25, size=(40, 2)) for loc in ((-5, 0), (5, 0), (0, 5))]
    X = np.vstack(parts).astype(np.float64)
    labels = np.repeat([0, 1, 2], 40).astype(np.intp)
    return X, labels


@pytest.mark.parametrize("chunk_size", [1, 8, 64, 512])
def test_matches_sklearn_silhouette_score(
    well_separated_three_clusters: tuple[np.ndarray, np.ndarray],
    chunk_size: int,
) -> None:
    X, labels = well_separated_three_clusters
    ours = chunked_silhouette(X, labels, chunk_size=chunk_size)
    sk = float(silhouette_score(X, labels, metric="euclidean"))
    assert ours.overall == pytest.approx(sk, abs=1e-10)


def test_per_sample_matches_sklearn_silhouette_samples(
    well_separated_three_clusters: tuple[np.ndarray, np.ndarray],
) -> None:
    X, labels = well_separated_three_clusters
    ours = chunked_silhouette(X, labels, chunk_size=32)
    sk = silhouette_samples(X, labels, metric="euclidean")
    np.testing.assert_allclose(ours.per_sample, sk, atol=1e-10)


def test_outliers_are_nan_in_per_sample_and_dropped_from_overall() -> None:
    rng = np.random.default_rng(1)
    cluster_a = rng.normal((-5, 0), 0.3, size=(20, 2))
    cluster_b = rng.normal((5, 0), 0.3, size=(20, 2))
    outliers = rng.normal((0, 20), 0.5, size=(5, 2))
    X = np.vstack([cluster_a, cluster_b, outliers]).astype(np.float64)
    labels = np.concatenate(
        [np.zeros(20, dtype=np.intp), np.ones(20, dtype=np.intp), -np.ones(5, dtype=np.intp)]
    )
    ours = chunked_silhouette(X, labels, chunk_size=16)
    assert ours.n_outliers == 5
    # Outlier rows have NaN per-sample silhouettes.
    assert np.all(np.isnan(ours.per_sample[-5:]))
    # Non-outlier rows are finite.
    assert np.all(np.isfinite(ours.per_sample[:40]))
    # Overall score equals the non-outlier sklearn score on the same rows.
    mask = labels >= 0
    sk_subset = float(silhouette_score(X[mask], labels[mask], metric="euclidean"))
    assert ours.overall == pytest.approx(sk_subset, abs=1e-10)


def test_singleton_clusters_contribute_zero_silhouette() -> None:
    # Two dense clusters plus a single-point cluster on its own. By convention
    # the singleton's silhouette is 0 (undefined cohesion). The two dense
    # clusters should otherwise match sklearn's silhouette_samples.
    rng = np.random.default_rng(2)
    X = np.vstack(
        [
            rng.normal((-5, 0), 0.2, size=(15, 2)),
            rng.normal((5, 0), 0.2, size=(15, 2)),
            np.array([[0.0, 10.0]]),
        ]
    ).astype(np.float64)
    labels = np.concatenate(
        [
            np.zeros(15, dtype=np.intp),
            np.ones(15, dtype=np.intp),
            np.array([2], dtype=np.intp),
        ]
    )
    ours = chunked_silhouette(X, labels, chunk_size=8)
    assert ours.per_sample[-1] == pytest.approx(0.0, abs=1e-12)
    assert ours.cluster_sizes[2] == 1
    assert ours.per_cluster[2] == pytest.approx(0.0, abs=1e-12)


def test_raises_when_single_cluster_remains() -> None:
    X = np.random.default_rng(3).normal(size=(40, 2))
    labels = np.zeros(40, dtype=np.intp)
    with pytest.raises(ValueError, match="two distinct"):
        chunked_silhouette(X, labels)


def test_raises_when_all_samples_are_outliers() -> None:
    X = np.random.default_rng(4).normal(size=(10, 2))
    labels = -np.ones(10, dtype=np.intp)
    with pytest.raises(ValueError, match="every sample is an outlier"):
        chunked_silhouette(X, labels)


def test_noncontiguous_labels_are_handled() -> None:
    # Labels 0, 3, 7 -- legal HDBSCAN output after filtering but not a
    # contiguous range. per_cluster must still be retrievable under the
    # original label values.
    rng = np.random.default_rng(5)
    X = np.vstack([rng.normal(c, 0.25, size=(12, 2)) for c in ((-5, 0), (5, 0), (0, 5))]).astype(
        np.float64
    )
    labels = np.repeat([0, 3, 7], 12).astype(np.intp)
    ours = chunked_silhouette(X, labels, chunk_size=8)
    assert ours.per_cluster.shape == (8,)
    assert ours.cluster_sizes[0] == 12
    assert ours.cluster_sizes[3] == 12
    assert ours.cluster_sizes[7] == 12
    # Missing labels are NaN (not scored).
    assert np.all(np.isnan(ours.per_cluster[[1, 2, 4, 5, 6]]))


def test_memory_budget_scales_with_chunk_size_not_quadratic() -> None:
    # Functional, not memory, check: the result must be identical (up to
    # floating-point tolerance) across every chunk size we try, which is
    # what we rely on for streaming large N.
    rng = np.random.default_rng(6)
    X = rng.normal(size=(200, 3)).astype(np.float64)
    labels = np.concatenate(
        [np.zeros(80, dtype=np.intp), np.ones(80, dtype=np.intp), 2 * np.ones(40, dtype=np.intp)]
    )
    ref = chunked_silhouette(X, labels, chunk_size=1)
    for chunk in (8, 32, 200, 1024):
        other = chunked_silhouette(X, labels, chunk_size=chunk)
        # Floating-point reduction order changes with chunk size, so the
        # tolerance follows BLAS-level rounding (not exact equality).
        assert other.overall == pytest.approx(ref.overall, abs=1e-10)
        np.testing.assert_allclose(other.per_sample, ref.per_sample, atol=1e-10)


def test_returns_silhouette_result_with_expected_fields(
    well_separated_three_clusters: tuple[np.ndarray, np.ndarray],
) -> None:
    X, labels = well_separated_three_clusters
    ours = chunked_silhouette(X, labels, chunk_size=32)
    assert isinstance(ours, SilhouetteResult)
    assert ours.per_sample.shape == (X.shape[0],)
    assert ours.per_cluster.shape == (3,)
    assert ours.cluster_sizes.sum() == X.shape[0]
    assert ours.n_outliers == 0


def test_rejects_mismatched_shapes() -> None:
    X = np.random.default_rng(7).normal(size=(30, 2))
    with pytest.raises(ValueError, match="same number of samples"):
        chunked_silhouette(X, np.zeros(29, dtype=np.intp))
    with pytest.raises(ValueError, match="2-D array"):
        chunked_silhouette(X.ravel(), np.zeros(60, dtype=np.intp))
    with pytest.raises(ValueError, match="1-D array"):
        chunked_silhouette(X, np.zeros((30, 2), dtype=np.intp))


def test_rejects_bad_chunk_size(
    well_separated_three_clusters: tuple[np.ndarray, np.ndarray],
) -> None:
    X, labels = well_separated_three_clusters
    with pytest.raises(ValueError, match="chunk_size"):
        chunked_silhouette(X, labels, chunk_size=0)
