"""Tests for :mod:`starfold.uncertainty` and the pipeline integration."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

import starfold as sf
from starfold.uncertainty import (
    UncertaintyPropagation,
    _broadcast_sigma,
    _effective_n_jobs,
    _tally_membership,
    build_replica_augmented_matrix,
    consensus_from_augmented_labels,
    propagate_uncertainty,
)


def _fit_pipeline(
    X: np.ndarray,
    *,
    random_state: int = 0,
    n_trials: int = 4,
) -> sf.PipelineResult:
    pipeline = sf.UnsupervisedPipeline(
        umap_kwargs={"n_epochs": 50, "n_neighbors": 10},
        hdbscan_optuna_trials=n_trials,
        skip_noise_baseline=True,
        engine="cpu",
        random_state=random_state,
    )
    return pipeline.fit(X)


@pytest.fixture(scope="module")
def simple_two_blob_fit() -> tuple[np.ndarray, sf.PipelineResult]:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(loc, 0.3, size=(80, 3)) for loc in (-5.0, 5.0)]).astype(np.float64)
    result = _fit_pipeline(X, random_state=0)
    return X, result


# -----------------------
# _broadcast_sigma
# -----------------------


def test_broadcast_sigma_scalar() -> None:
    arr, shape = _broadcast_sigma(0.3, n_samples=4, n_features=2)
    assert arr.shape == (4, 2)
    assert shape == "scalar"
    assert np.allclose(arr, 0.3)


def test_broadcast_sigma_per_feature() -> None:
    arr, shape = _broadcast_sigma(np.array([0.1, 0.5]), n_samples=3, n_features=2)
    assert arr.shape == (3, 2)
    assert shape == "per_feature"
    assert np.allclose(arr[:, 0], 0.1)
    assert np.allclose(arr[:, 1], 0.5)


def test_broadcast_sigma_per_sample_feature() -> None:
    raw = np.arange(6.0).reshape(3, 2)
    arr, shape = _broadcast_sigma(raw, n_samples=3, n_features=2)
    assert shape == "per_sample_feature"
    assert np.allclose(arr, raw)
    # Ensure it's a copy, not the original
    arr[0, 0] = 999.0
    assert raw[0, 0] == 0.0


def test_broadcast_sigma_rejects_wrong_feature_length() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        _broadcast_sigma(np.array([0.1, 0.2, 0.3]), n_samples=4, n_features=2)


def test_broadcast_sigma_rejects_wrong_2d_shape() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        _broadcast_sigma(np.zeros((3, 3)), n_samples=3, n_features=2)


def test_broadcast_sigma_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _broadcast_sigma(-0.1, n_samples=3, n_features=2)
    with pytest.raises(ValueError, match="non-negative"):
        _broadcast_sigma(np.array([0.1, -0.2]), n_samples=3, n_features=2)


def test_broadcast_sigma_rejects_nonfinite() -> None:
    with pytest.raises(ValueError, match=r"non-negative|finite"):
        _broadcast_sigma(np.array([0.1, np.inf]), n_samples=3, n_features=2)


def test_broadcast_sigma_rejects_high_dim() -> None:
    with pytest.raises(ValueError, match="ndim"):
        _broadcast_sigma(np.zeros((2, 2, 2)), n_samples=2, n_features=2)


# -----------------------
# _tally_membership
# -----------------------


def test_tally_membership_counts_fractions() -> None:
    # 4 draws, 2 samples, 2 clusters
    #   sample 0: labels 0,0,1,-1
    #   sample 1: labels 1,1,1,1
    draws = np.array(
        [
            [0, 1],
            [0, 1],
            [1, 1],
            [-1, 1],
        ],
        dtype=np.intp,
    )
    membership = _tally_membership(draws, n_clusters=2, n_samples=2)
    assert membership.shape == (2, 3)
    # sample 0: 2 in cluster 0, 1 in cluster 1, 1 outlier
    assert np.allclose(membership[0], [0.5, 0.25, 0.25])
    # sample 1: all in cluster 1
    assert np.allclose(membership[1], [0.0, 1.0, 0.0])


def test_tally_membership_handles_all_outliers() -> None:
    draws = np.full((3, 2), -1, dtype=np.intp)
    membership = _tally_membership(draws, n_clusters=1, n_samples=2)
    # All draws landed in the outlier column (index n_clusters = 1)
    assert np.allclose(membership[:, 0], 0.0)
    assert np.allclose(membership[:, 1], 1.0)


def test_tally_membership_rows_sum_to_one() -> None:
    rng = np.random.default_rng(0)
    draws = rng.integers(-1, 3, size=(50, 10), dtype=np.intp)
    membership = _tally_membership(draws, n_clusters=3, n_samples=10)
    assert np.allclose(membership.sum(axis=1), 1.0)


# -----------------------
# _effective_n_jobs
# -----------------------


def test_effective_n_jobs_clamps_to_draws() -> None:
    assert _effective_n_jobs(n_jobs=10, n_draws=3) <= 3
    assert _effective_n_jobs(n_jobs=10, n_draws=1) == 1


def test_effective_n_jobs_negative_one_picks_cpu_count() -> None:
    # Pick at least 1, at most n_draws
    n = _effective_n_jobs(n_jobs=-1, n_draws=4)
    assert 1 <= n <= 4


# -----------------------
# propagate_uncertainty (CPU path)
# -----------------------


def test_propagate_uncertainty_returns_expected_shapes(
    simple_two_blob_fit: tuple[np.ndarray, sf.PipelineResult],
) -> None:
    X, result = simple_two_blob_fit
    prop = result.propagate_uncertainty(X, sigma=0.1, n_draws=5, random_state=0)
    assert isinstance(prop, UncertaintyPropagation)
    assert prop.membership.shape == (X.shape[0], result.n_clusters + 1)
    assert prop.consensus_label.shape == (X.shape[0],)
    assert prop.instability.shape == (X.shape[0],)
    assert prop.n_draws == 5
    assert prop.n_samples == X.shape[0]
    assert prop.n_clusters == result.n_clusters


def test_propagate_uncertainty_memberships_are_probabilities(
    simple_two_blob_fit: tuple[np.ndarray, sf.PipelineResult],
) -> None:
    X, result = simple_two_blob_fit
    prop = result.propagate_uncertainty(X, sigma=0.1, n_draws=8, random_state=0)
    assert np.all(prop.membership >= 0.0)
    assert np.all(prop.membership <= 1.0)
    assert np.allclose(prop.membership.sum(axis=1), 1.0)
    assert np.all(prop.instability >= 0.0)
    assert np.all(prop.instability <= 1.0)


def test_propagate_uncertainty_zero_sigma_is_stable(
    simple_two_blob_fit: tuple[np.ndarray, sf.PipelineResult],
) -> None:
    X, result = simple_two_blob_fit
    prop = result.propagate_uncertainty(X, sigma=0.0, n_draws=3, random_state=0)
    # With sigma = 0 every draw is identical -> every sample goes to one
    # single column with membership 1.0 -> instability 0.
    assert np.allclose(prop.instability, 0.0)


def test_propagate_uncertainty_larger_sigma_raises_instability(
    simple_two_blob_fit: tuple[np.ndarray, sf.PipelineResult],
) -> None:
    X, result = simple_two_blob_fit
    small = result.propagate_uncertainty(X, sigma=0.05, n_draws=8, random_state=0)
    big = result.propagate_uncertainty(X, sigma=2.0, n_draws=8, random_state=0)
    assert big.instability.mean() >= small.instability.mean()


def test_propagate_uncertainty_reproducible(
    simple_two_blob_fit: tuple[np.ndarray, sf.PipelineResult],
) -> None:
    X, result = simple_two_blob_fit
    a = result.propagate_uncertainty(X, sigma=0.2, n_draws=6, random_state=42)
    b = result.propagate_uncertainty(X, sigma=0.2, n_draws=6, random_state=42)
    assert np.array_equal(a.membership, b.membership)
    assert np.array_equal(a.consensus_label, b.consensus_label)


def test_propagate_uncertainty_per_feature_sigma(
    simple_two_blob_fit: tuple[np.ndarray, sf.PipelineResult],
) -> None:
    X, result = simple_two_blob_fit
    sigma = np.array([0.1, 0.1, 0.1])
    prop = result.propagate_uncertainty(X, sigma=sigma, n_draws=4, random_state=0)
    assert prop.sigma_shape == "per_feature"


def test_propagate_uncertainty_per_sample_per_feature_sigma(
    simple_two_blob_fit: tuple[np.ndarray, sf.PipelineResult],
) -> None:
    X, result = simple_two_blob_fit
    sigma = np.full_like(X, 0.1, dtype=np.float64)
    prop = result.propagate_uncertainty(X, sigma=sigma, n_draws=4, random_state=0)
    assert prop.sigma_shape == "per_sample_feature"


def test_propagate_uncertainty_parallel_matches_sequential(
    simple_two_blob_fit: tuple[np.ndarray, sf.PipelineResult],
) -> None:
    X, result = simple_two_blob_fit
    seq = result.propagate_uncertainty(X, sigma=0.2, n_draws=6, random_state=0, n_jobs=1)
    par = result.propagate_uncertainty(X, sigma=0.2, n_draws=6, random_state=0, n_jobs=2)
    assert np.array_equal(seq.membership, par.membership)


def test_propagate_uncertainty_rejects_n_draws_zero(
    simple_two_blob_fit: tuple[np.ndarray, sf.PipelineResult],
) -> None:
    X, result = simple_two_blob_fit
    with pytest.raises(ValueError, match="n_draws"):
        result.propagate_uncertainty(X, sigma=0.1, n_draws=0)


def test_propagate_uncertainty_rejects_shape_mismatch(
    simple_two_blob_fit: tuple[np.ndarray, sf.PipelineResult],
) -> None:
    _, result = simple_two_blob_fit
    # X with wrong ndim
    with pytest.raises(ValueError, match="2-D"):
        propagate_uncertainty(
            np.zeros(5, dtype=np.float64),
            sigma=0.1,
            scaler=result.scaler,
            umap_model=result.umap_model,
            hdbscan_model=result.search.model,
            n_clusters=result.n_clusters,
        )


def test_propagate_uncertainty_pipeline_result_preserves_umap_model(
    simple_two_blob_fit: tuple[np.ndarray, sf.PipelineResult],
) -> None:
    _, result = simple_two_blob_fit
    assert result.umap_model is not None


def test_propagate_uncertainty_summary_is_string(
    simple_two_blob_fit: tuple[np.ndarray, sf.PipelineResult],
) -> None:
    X, result = simple_two_blob_fit
    prop = result.propagate_uncertainty(X, sigma=0.1, n_draws=3, random_state=0)
    text = prop.summary()
    assert "uncertainty propagation" in text
    assert "mean instability" in text


# -----------------------
# cuml-path NotImplementedError
# -----------------------


class _FakeCumlUMAP:
    """Stand-in for cuml.manifold.UMAP to exercise the isinstance guard."""


def test_propagate_uncertainty_rejects_non_umap_model() -> None:
    # The CPU-only guard rejects anything that isn't an umap.UMAP instance,
    # which is the mechanism used for cuml rejection.
    scaler = StandardScaler().fit(np.zeros((5, 2)))
    with pytest.raises(NotImplementedError, match="CPU-only"):
        propagate_uncertainty(
            np.zeros((5, 2)),
            sigma=0.1,
            scaler=scaler,
            umap_model=_FakeCumlUMAP(),
            hdbscan_model=None,
            n_clusters=1,
        )


def test_propagate_uncertainty_on_result_without_umap_model_raises() -> None:
    # Synthesise a PipelineResult with umap_model=None; the method must
    # refuse to propagate rather than silently fall back.
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(loc, 0.3, size=(40, 3)) for loc in (-5.0, 5.0)]).astype(np.float64)
    real = _fit_pipeline(X, random_state=0)
    fake = sf.PipelineResult(
        embedding=real.embedding,
        labels=real.labels,
        probabilities=real.probabilities,
        persistence=real.persistence,
        significant=real.significant,
        trustworthiness=real.trustworthiness,
        continuity=real.continuity,
        n_clusters=real.n_clusters,
        best_params=real.best_params,
        search=real.search,
        noise_baseline=real.noise_baseline,
        credibility=real.credibility,
        hierarchy=real.hierarchy,
        scaler=real.scaler,
        umap_model=None,
        config=real.config,
    )
    with pytest.raises(NotImplementedError, match="UMAP model"):
        fake.propagate_uncertainty(X, sigma=0.1, n_draws=3)


# -----------------------
# confident_labels
# -----------------------


def test_confident_labels_returns_consensus_when_above_threshold() -> None:
    membership = np.array(
        [
            [0.9, 0.05, 0.05],  # confident cluster 0
            [0.55, 0.4, 0.05],  # not confident at 0.8
            [0.0, 0.0, 1.0],  # outlier column
        ],
        dtype=np.float64,
    )
    consensus = np.array([0, 0, -1], dtype=np.intp)
    instab = 1.0 - membership.max(axis=1)
    prop = UncertaintyPropagation(
        membership=membership,
        consensus_label=consensus,
        instability=instab,
        n_draws=20,
        sigma_shape="scalar",
    )
    out = prop.confident_labels(threshold=0.8)
    assert out.dtype == np.intp
    np.testing.assert_array_equal(out, [0, -1, -1])


def test_confident_labels_threshold_validation() -> None:
    prop = UncertaintyPropagation(
        membership=np.zeros((0, 2), dtype=np.float64),
        consensus_label=np.empty(0, dtype=np.intp),
        instability=np.empty(0, dtype=np.float64),
        n_draws=0,
        sigma_shape="scalar",
    )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        prop.confident_labels(threshold=-0.1)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        prop.confident_labels(threshold=1.5)


def test_confident_labels_empty_result() -> None:
    prop = UncertaintyPropagation(
        membership=np.zeros((0, 2), dtype=np.float64),
        consensus_label=np.empty(0, dtype=np.intp),
        instability=np.empty(0, dtype=np.float64),
        n_draws=0,
        sigma_shape="scalar",
    )
    out = prop.confident_labels(threshold=0.5)
    assert out.shape == (0,)
    assert out.dtype == np.intp


# -----------------------
# build_replica_augmented_matrix / consensus_from_augmented_labels
# -----------------------


def test_build_replica_augmented_matrix_shapes_and_groups() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 3))
    x_aug, groups = build_replica_augmented_matrix(
        X,
        0.2,
        n_replicas=4,
        random_state=7,
    )
    assert x_aug.shape == (10 * 5, 3)
    assert groups.shape == (50,)
    # Clean block: first n_samples rows are the originals exactly.
    np.testing.assert_array_equal(x_aug[:10], X.astype(np.float64))
    # Every original sample is referenced (1 + n_replicas) times.
    counts = np.bincount(groups)
    assert np.all(counts == 5)


def test_build_replica_augmented_matrix_n_replicas_zero_is_identity() -> None:
    X = np.arange(20, dtype=np.float64).reshape(10, 2)
    x_aug, groups = build_replica_augmented_matrix(X, 0.5, n_replicas=0)
    np.testing.assert_array_equal(x_aug, X)
    np.testing.assert_array_equal(groups, np.arange(10))


def test_build_replica_augmented_matrix_rejects_negative_n_replicas() -> None:
    with pytest.raises(ValueError, match="n_replicas"):
        build_replica_augmented_matrix(np.zeros((5, 2)), 0.1, n_replicas=-1)


def test_build_replica_is_reproducible_under_seed() -> None:
    X = np.random.default_rng(42).normal(size=(8, 2))
    a, _ = build_replica_augmented_matrix(X, 0.3, n_replicas=3, random_state=99)
    b, _ = build_replica_augmented_matrix(X, 0.3, n_replicas=3, random_state=99)
    np.testing.assert_array_equal(a, b)


def test_consensus_from_augmented_labels_tallies_correctly() -> None:
    # Three original samples, two replicas plus clean -> 9 rows.
    group_ids = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.intp)
    labels = np.array([0, 1, -1, 0, 1, 1, 0, 0, 1], dtype=np.intp)
    prop = consensus_from_augmented_labels(labels, group_ids, n_clusters=2)
    # sample 0: cluster 0 three times out of three -> membership [1, 0, 0]
    np.testing.assert_allclose(prop.membership[0], [1.0, 0.0, 0.0])
    # sample 1: cluster 1 twice, cluster 0 once -> [1/3, 2/3, 0]
    np.testing.assert_allclose(prop.membership[1], [1 / 3, 2 / 3, 0.0])
    # sample 2: -1 once (outlier col), cluster 1 twice -> [0, 2/3, 1/3]
    np.testing.assert_allclose(prop.membership[2], [0.0, 2 / 3, 1 / 3])
    np.testing.assert_array_equal(prop.consensus_label, [0, 1, 1])
    assert prop.n_draws == 3


def test_consensus_rejects_uneven_group_counts() -> None:
    labels = np.array([0, 1, 0, 1], dtype=np.intp)
    groups = np.array([0, 0, 0, 1], dtype=np.intp)  # sample 0 has 3 rows, sample 1 has 1
    with pytest.raises(ValueError, match="same number of times"):
        consensus_from_augmented_labels(labels, groups, n_clusters=2)


def test_consensus_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        consensus_from_augmented_labels(
            np.array([0, 1], dtype=np.intp),
            np.array([0, 1, 2], dtype=np.intp),
            n_clusters=2,
        )


# -----------------------
# fit_with_uncertainty
# -----------------------


def test_fit_with_uncertainty_returns_augmented_fit() -> None:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(loc, 0.3, size=(40, 3)) for loc in (-5.0, 5.0)]).astype(np.float64)
    pipeline = sf.UnsupervisedPipeline(
        umap_kwargs={"n_epochs": 50, "n_neighbors": 10},
        hdbscan_optuna_trials=4,
        skip_noise_baseline=True,
        engine="cpu",
        random_state=0,
    )
    fit = pipeline.fit_with_uncertainty(X, sigma=0.1, n_replicas=2)
    # Augmented matrix is n_samples * (1 + n_replicas).
    assert fit.augmented_result.embedding.shape == (80 * 3, 2)
    assert fit.propagation.membership.shape[0] == 80
    assert fit.propagation.membership.shape[1] == fit.augmented_result.n_clusters + 1
    # Membership rows must sum to 1 (per-sample total fraction).
    np.testing.assert_allclose(fit.propagation.membership.sum(axis=1), 1.0, atol=1e-12)
    # Consensus labels length matches original n_samples.
    assert fit.consensus_label.shape == (80,)
    assert fit.instability.shape == (80,)
    assert fit.n_replicas == 2
    assert fit.sigma_summary[0] == pytest.approx(0.1)


def test_fit_with_uncertainty_zero_replicas_matches_clean_fit() -> None:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(loc, 0.3, size=(40, 3)) for loc in (-5.0, 5.0)]).astype(np.float64)
    pipeline = sf.UnsupervisedPipeline(
        umap_kwargs={"n_epochs": 50, "n_neighbors": 10},
        hdbscan_optuna_trials=4,
        skip_noise_baseline=True,
        engine="cpu",
        random_state=0,
    )
    fit = pipeline.fit_with_uncertainty(X, sigma=0.0, n_replicas=0)
    clean = pipeline.fit(X)
    np.testing.assert_array_equal(fit.augmented_result.labels, clean.labels)
    np.testing.assert_array_equal(fit.consensus_label, clean.labels)
    # With n_replicas=0 every "draw" is a single clean copy, so
    # instability must be exactly 0.
    np.testing.assert_array_equal(fit.instability, np.zeros(80))


def test_fit_with_uncertainty_rejects_negative_replicas() -> None:
    pipeline = sf.UnsupervisedPipeline(
        umap_kwargs={"n_epochs": 50, "n_neighbors": 10},
        hdbscan_optuna_trials=4,
        skip_noise_baseline=True,
        engine="cpu",
        random_state=0,
    )
    X = np.random.default_rng(0).normal(size=(40, 3))
    with pytest.raises(ValueError, match="n_replicas"):
        pipeline.fit_with_uncertainty(X, sigma=0.1, n_replicas=-1)


def test_fit_with_uncertainty_summary_mentions_n_replicas() -> None:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(loc, 0.3, size=(30, 3)) for loc in (-5.0, 5.0)]).astype(np.float64)
    pipeline = sf.UnsupervisedPipeline(
        umap_kwargs={"n_epochs": 50, "n_neighbors": 10},
        hdbscan_optuna_trials=4,
        skip_noise_baseline=True,
        engine="cpu",
        random_state=0,
    )
    fit = pipeline.fit_with_uncertainty(X, sigma=0.05, n_replicas=2)
    text = fit.summary()
    assert "uncertainty-aware fit" in text
    assert "n_replicas=2" in text


# -----------------------
# refit_subcluster
# -----------------------


def test_refit_subcluster_returns_new_pipeline_result() -> None:
    rng = np.random.default_rng(0)
    # Three well-separated blobs in 3-D so the outer fit is stable.
    X = np.vstack([rng.normal(loc, 0.3, size=(40, 3)) for loc in (-8.0, 0.0, 8.0)]).astype(
        np.float64
    )
    pipeline = sf.UnsupervisedPipeline(
        umap_kwargs={"n_epochs": 50, "n_neighbors": 10},
        hdbscan_optuna_trials=4,
        skip_noise_baseline=True,
        engine="cpu",
        random_state=0,
    )
    result = pipeline.fit(X)
    assert result.n_clusters >= 1
    target = 0
    sub = result.refit_subcluster(X, target)
    # The sub-fit should only see the subset.
    n_subset = int((result.labels == target).sum())
    assert sub.labels.shape == (n_subset,)


def test_refit_subcluster_rejects_negative_cluster() -> None:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(loc, 0.3, size=(40, 3)) for loc in (-8.0, 8.0)]).astype(np.float64)
    pipeline = sf.UnsupervisedPipeline(
        umap_kwargs={"n_epochs": 50, "n_neighbors": 10},
        hdbscan_optuna_trials=4,
        skip_noise_baseline=True,
        engine="cpu",
        random_state=0,
    )
    result = pipeline.fit(X)
    with pytest.raises(ValueError, match="non-negative"):
        result.refit_subcluster(X, -1)


def test_refit_subcluster_rejects_out_of_range_cluster() -> None:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(loc, 0.3, size=(40, 3)) for loc in (-8.0, 8.0)]).astype(np.float64)
    pipeline = sf.UnsupervisedPipeline(
        umap_kwargs={"n_epochs": 50, "n_neighbors": 10},
        hdbscan_optuna_trials=4,
        skip_noise_baseline=True,
        engine="cpu",
        random_state=0,
    )
    result = pipeline.fit(X)
    with pytest.raises(ValueError, match="out of range"):
        result.refit_subcluster(X, result.n_clusters + 5)


def test_refit_subcluster_rejects_shape_mismatch() -> None:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(loc, 0.3, size=(40, 3)) for loc in (-8.0, 8.0)]).astype(np.float64)
    pipeline = sf.UnsupervisedPipeline(
        umap_kwargs={"n_epochs": 50, "n_neighbors": 10},
        hdbscan_optuna_trials=4,
        skip_noise_baseline=True,
        engine="cpu",
        random_state=0,
    )
    result = pipeline.fit(X)
    with pytest.raises(ValueError, match="n_samples"):
        result.refit_subcluster(X[:10], 0)
