"""Unit tests for the subsample-stability diagnostic.

Covers the shape contract of :class:`SubsampleStability`, the ARI and
n_clusters readouts on a clean 3-blob embedding, reproducibility under
a fixed seed, and the boundary checks on the public API.
"""

from __future__ import annotations

import numpy as np
import pytest

from starfold.clustering import run_hdbscan
from starfold.stability import SubsampleStability, compute_subsample_stability


def _reference_fit(
    X: np.ndarray,
    mcs: int = 20,
    ms: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    res = run_hdbscan(X, min_cluster_size=mcs, min_samples=ms, engine="cpu")
    return res.labels, res.cluster_persistence


def test_subsample_stability_shapes(
    gmm_three_clusters_2d: tuple[np.ndarray, np.ndarray],
) -> None:
    X, _ = gmm_three_clusters_2d
    labels, pers = _reference_fit(X)
    out = compute_subsample_stability(
        X,
        labels,
        pers,
        min_cluster_size=20,
        min_samples=5,
        n_subsamples=5,
        subsample_fraction=0.8,
        engine="cpu",
        random_state=0,
    )
    assert isinstance(out, SubsampleStability)
    assert out.n_subsamples == 5
    assert out.subsample_fraction == pytest.approx(0.8)
    assert out.n_clusters.shape == (5,)
    assert out.ari.shape == (5,)
    assert out.persistence_sum.shape == (5,)
    assert out.persistence_per_cluster.shape == (5, pers.shape[0])
    assert out.n_clusters.dtype == np.intp


def test_subsample_stability_is_reproducible(
    gmm_three_clusters_2d: tuple[np.ndarray, np.ndarray],
) -> None:
    X, _ = gmm_three_clusters_2d
    labels, pers = _reference_fit(X)
    kwargs = {
        "min_cluster_size": 20,
        "min_samples": 5,
        "n_subsamples": 4,
        "subsample_fraction": 0.8,
        "engine": "cpu",
        "random_state": 42,
    }
    a = compute_subsample_stability(X, labels, pers, **kwargs)  # type: ignore[arg-type]
    b = compute_subsample_stability(X, labels, pers, **kwargs)  # type: ignore[arg-type]
    np.testing.assert_array_equal(a.n_clusters, b.n_clusters)
    np.testing.assert_allclose(a.ari, b.ari, atol=0.0)
    np.testing.assert_allclose(a.persistence_sum, b.persistence_sum, atol=0.0)
    np.testing.assert_allclose(
        np.nan_to_num(a.persistence_per_cluster, nan=-1.0),
        np.nan_to_num(b.persistence_per_cluster, nan=-1.0),
        atol=0.0,
    )


def test_subsample_stability_recovers_three_clusters_on_clean_data(
    gmm_three_clusters_2d: tuple[np.ndarray, np.ndarray],
) -> None:
    X, _ = gmm_three_clusters_2d
    labels, pers = _reference_fit(X)
    out = compute_subsample_stability(
        X,
        labels,
        pers,
        min_cluster_size=20,
        min_samples=5,
        n_subsamples=8,
        subsample_fraction=0.8,
        engine="cpu",
        random_state=0,
    )
    # Clean, well-separated blobs -- every subsample should recover three
    # clusters with ARI close to 1 vs the reference.
    assert np.all(out.n_clusters == 3)
    finite_ari = out.ari[np.isfinite(out.ari)]
    assert float(finite_ari.min()) > 0.9


def test_subsample_stability_rejects_non_2d_embedding() -> None:
    rng = np.random.default_rng(0)
    X_3d = rng.normal(size=(50, 3))
    labels = np.zeros(50, dtype=np.intp)
    pers = np.array([0.5])
    with pytest.raises(ValueError, match="shape"):
        compute_subsample_stability(
            X_3d, labels, pers,
            min_cluster_size=5, min_samples=1, n_subsamples=2, engine="cpu",
        )


def test_subsample_stability_rejects_label_mismatch() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    labels_wrong = np.zeros(40, dtype=np.intp)
    pers = np.array([0.5])
    with pytest.raises(ValueError, match="one entry per embedding row"):
        compute_subsample_stability(
            X, labels_wrong, pers,
            min_cluster_size=5, min_samples=1, n_subsamples=2, engine="cpu",
        )


def test_subsample_stability_rejects_bad_fraction() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    labels = np.zeros(50, dtype=np.intp)
    pers = np.array([0.5])
    with pytest.raises(ValueError, match="subsample_fraction"):
        compute_subsample_stability(
            X, labels, pers,
            min_cluster_size=5, min_samples=1,
            subsample_fraction=1.5, n_subsamples=2, engine="cpu",
        )
    with pytest.raises(ValueError, match="subsample_fraction"):
        compute_subsample_stability(
            X, labels, pers,
            min_cluster_size=5, min_samples=1,
            subsample_fraction=0.0, n_subsamples=2, engine="cpu",
        )
