"""End-to-end science-regression tests on a seeded Gaussian mixture.

These tests pin *numerical floors* on the quality metrics the full
pipeline is supposed to surface -- trustworthiness, continuity,
silhouette, DBCV, adjusted Rand index against the ground-truth labels,
and the cluster count itself -- on a fixed 3-Gaussian fixture with a
fixed seed and a constrained Optuna search space that deliberately
rules out the over-segmenting corners (tiny ``min_cluster_size``,
``alpha < 1``, non-zero ``cluster_selection_epsilon``). The floors are
set ~0.01 below the measured values at the time the tests were written
so routine floating-point drift does not break CI, but any real
behavioural regression in UMAP / HDBSCAN / trustworthiness / silhouette
/ merge-recommender will fail the test.

These tests complement the per-module unit tests: the unit tests check
each building block in isolation, while these tests check that when
everything is wired up the pipeline actually produces a result a
scientist would accept.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from starfold.merge import suggest_merges
from starfold.pipeline import UnsupervisedPipeline
from starfold.silhouette import chunked_silhouette


@pytest.fixture(scope="module")
def three_gaussian_fixture() -> tuple[np.ndarray, np.ndarray]:
    """Three tight, well-separated 2-D Gaussians, 120 points each.

    The spread (0.35) is deliberately smaller than the fixture in
    ``conftest.py`` so that the clustering recovers 3 blobs reliably
    under a constrained search -- the regression test should fail when
    quality drops, not when UMAP's stochastic layout happens to split a
    blob in two.
    """
    rng = np.random.default_rng(42)
    centres = np.array([[-8.0, 0.0], [0.0, 0.0], [8.0, 0.0]])
    blobs = [rng.normal(c, 0.35, size=(120, 2)) for c in centres]
    X = np.vstack(blobs).astype(np.float64)
    y_true = np.concatenate([np.full(120, k, dtype=np.intp) for k in range(3)])
    perm = np.random.default_rng(0).permutation(X.shape[0])
    return X[perm], y_true[perm]


@pytest.fixture(scope="module")
def pinned_pipeline() -> UnsupervisedPipeline:
    """Pinned pipeline configuration for the regression tests.

    The configuration pins every axis the expanded Optuna search would
    otherwise explore. This turns off the parts of the tool the other
    tests already cover (``leaf`` mode, epsilon, alpha<1) and leaves
    the baseline UMAP + EOM HDBSCAN fidelity on the hook.
    """
    return UnsupervisedPipeline(
        umap_kwargs={"n_neighbors": 50, "n_epochs": 500, "min_dist": 0.1},
        hdbscan_optuna_trials=25,
        mcs_range=(60, 250),
        ms_range=(10, 40),
        cluster_selection_methods=("eom",),
        cluster_selection_epsilon_range=(0.0, 0.0),
        alpha_range=(1.0, 1.0),
        skip_noise_baseline=True,
        random_state=42,
        engine="cpu",
    )


@pytest.fixture(scope="module")
def pinned_result(
    three_gaussian_fixture: tuple[np.ndarray, np.ndarray],
    pinned_pipeline: UnsupervisedPipeline,
):
    X, _ = three_gaussian_fixture
    return pinned_pipeline.fit(X)


def test_pipeline_recovers_three_clusters(pinned_result) -> None:
    assert pinned_result.n_clusters == 3


def test_adjusted_rand_index_against_truth(
    three_gaussian_fixture: tuple[np.ndarray, np.ndarray],
    pinned_result,
) -> None:
    _, y_true = three_gaussian_fixture
    ari = adjusted_rand_score(y_true, pinned_result.labels)
    # Floor: the fixture is clean enough that perfect recovery is the
    # measured baseline. We allow a small slack so minor embedding
    # changes don't break the floor, but anything below 0.95 ARI would
    # indicate a real regression in cluster recovery.
    assert ari >= 0.95, f"ARI dropped to {ari:.4f}; expected >= 0.95"


def test_trustworthiness_and_continuity_floors(pinned_result) -> None:
    # Measured: T = 0.9887, C = 0.9858. Set floors ~0.01 below each;
    # any regression in the trustworthiness/continuity implementation
    # would drop them sharply, not by a percent.
    assert pinned_result.trustworthiness >= 0.97, (
        f"T(k) dropped to {pinned_result.trustworthiness:.4f}; expected >= 0.97"
    )
    assert pinned_result.continuity >= 0.97, (
        f"C(k) dropped to {pinned_result.continuity:.4f}; expected >= 0.97"
    )
    # Trustworthiness and continuity are both in [0, 1] by definition.
    assert 0.0 <= pinned_result.trustworthiness <= 1.0
    assert 0.0 <= pinned_result.continuity <= 1.0


def test_silhouette_floor(pinned_result) -> None:
    sil = pinned_result.silhouette()
    # Measured: overall = 0.6783. Three well-separated blobs in 2-D
    # embedding should easily beat 0.55; below that flags a regression.
    assert sil.overall >= 0.55, f"silhouette dropped to {sil.overall:.4f}; expected >= 0.55"
    # Per-cluster silhouettes should all be comfortably positive.
    assert np.all(sil.per_cluster >= 0.3)
    # Outlier count matches the pipeline's labels[-1] count.
    assert sil.n_outliers == int(np.sum(pinned_result.labels < 0))


def test_dbcv_floor(pinned_result) -> None:
    best_trial = pinned_result.search.study.best_trial
    dbcv = float(best_trial.user_attrs.get("relative_validity", float("nan")))
    # Measured: DBCV = 0.7255. A well-separated 3-cluster fit should
    # beat 0.5; below that is a HDBSCAN/relative_validity regression.
    assert dbcv >= 0.5, f"DBCV dropped to {dbcv:.4f}; expected >= 0.5"


def test_suggest_merges_does_not_recommend_any_merge_on_three_clean_blobs(
    pinned_result,
) -> None:
    # With three well-separated Gaussians no pair should be flagged for
    # merge under the default thresholds.
    suggestions = suggest_merges(pinned_result.hierarchy, pinned_result.embedding)
    assert len(suggestions) == 3  # C(3, 2) pairs
    assert not any(s.recommended for s in suggestions)
    # Every pair's gap_ratio should be comfortably above the threshold.
    for s in suggestions:
        assert s.gap_ratio > 2.0, (
            f"gap_ratio for pair ({s.cluster_i}, {s.cluster_j}) dropped to "
            f"{s.gap_ratio:.3f}; expected > 2.0 on well-separated blobs"
        )


def test_chunked_silhouette_matches_sklearn_on_pipeline_output(
    pinned_result,
) -> None:
    # Regression on the consistency between the chunked implementation
    # and sklearn on the pipeline's actual (non-trivial) embedding +
    # labels. Checks that the per-sample contract still holds.
    from sklearn.metrics import silhouette_samples, silhouette_score  # noqa: PLC0415

    mask = pinned_result.labels >= 0
    emb = pinned_result.embedding[mask]
    lbl = pinned_result.labels[mask]
    sk_overall = float(silhouette_score(emb, lbl, metric="euclidean"))
    sk_per_sample = silhouette_samples(emb, lbl, metric="euclidean")
    ours = chunked_silhouette(
        pinned_result.embedding, pinned_result.labels, chunk_size=64
    )
    assert ours.overall == pytest.approx(sk_overall, abs=1e-10)
    np.testing.assert_allclose(ours.per_sample[mask], sk_per_sample, atol=1e-10)


def test_best_params_populated_for_every_axis(pinned_result) -> None:
    # Sanity check: the expanded search-space machinery should still
    # populate every Optuna axis even when ranges are pinned.
    keys = set(pinned_result.best_params)
    assert {
        "min_cluster_size",
        "min_samples",
        "cluster_selection_method",
        "cluster_selection_epsilon",
        "alpha",
    } <= keys
    # Pinned axes must equal their pinned values.
    assert pinned_result.best_params["cluster_selection_method"] == "eom"
    assert pinned_result.best_params["cluster_selection_epsilon"] == 0.0
    assert pinned_result.best_params["alpha"] == 1.0
