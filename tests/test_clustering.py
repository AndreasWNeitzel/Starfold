"""Tests for the clustering wrappers.

The wrappers are exercised on seeded Gaussian-mixture data where the
ground-truth labels are known. The tests verify:

* HDBSCAN recovers the expected number of clusters and achieves
  ARI > 0.9 on a well-separated 3-blob fixture,
* Optuna search is reproducible under a fixed ``random_state``,
* the ``min_cluster_size`` upper bound is auto-capped on small samples,
* input validation rejects malformed calls.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from starfold.clustering import (
    HDBSCANResult,
    OptunaSearchResult,
    _effective_mcs_bounds,
    run_hdbscan,
    search_hdbscan,
)

# --------------------------------------------------------------- run_hdbscan


def test_run_hdbscan_recovers_three_clusters(
    gmm_three_clusters_2d: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y_true = gmm_three_clusters_2d
    result = run_hdbscan(X, min_cluster_size=20, engine="cpu")
    assert isinstance(result, HDBSCANResult)
    assert result.n_clusters == 3
    assert result.cluster_persistence.shape == (3,)
    assert adjusted_rand_score(y_true, result.labels) > 0.9


def test_run_hdbscan_probabilities_in_unit_interval(
    gmm_three_clusters_2d: tuple[np.ndarray, np.ndarray],
) -> None:
    X, _ = gmm_three_clusters_2d
    result = run_hdbscan(X, min_cluster_size=20, engine="cpu")
    assert result.probabilities.shape == (X.shape[0],)
    assert float(result.probabilities.min()) >= 0.0
    assert float(result.probabilities.max()) <= 1.0


def test_run_hdbscan_is_deterministic(
    gmm_three_clusters_2d: tuple[np.ndarray, np.ndarray],
) -> None:
    X, _ = gmm_three_clusters_2d
    a = run_hdbscan(X, min_cluster_size=20, min_samples=5, engine="cpu")
    b = run_hdbscan(X, min_cluster_size=20, min_samples=5, engine="cpu")
    np.testing.assert_array_equal(a.labels, b.labels)
    np.testing.assert_allclose(a.cluster_persistence, b.cluster_persistence, atol=0.0)


def test_run_hdbscan_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        run_hdbscan(np.zeros(10), min_cluster_size=5)


def test_run_hdbscan_rejects_too_small_mcs() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    with pytest.raises(ValueError, match="min_cluster_size"):
        run_hdbscan(X, min_cluster_size=1)


def test_run_hdbscan_rejects_invalid_min_samples() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    with pytest.raises(ValueError, match="min_samples"):
        run_hdbscan(X, min_cluster_size=5, min_samples=0)


def test_run_hdbscan_rejects_unknown_engine() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    with pytest.raises(ValueError, match="engine"):
        run_hdbscan(X, min_cluster_size=5, engine="gpu")  # type: ignore[arg-type]


# -------------------------------------------------------------- search_hdbscan


def test_search_hdbscan_finds_three_clusters(
    gmm_three_clusters_2d: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y_true = gmm_three_clusters_2d
    search = search_hdbscan(
        X,
        n_trials=15,
        mcs_range=(5, 50),
        ms_range=(1, 20),
        random_state=0,
        engine="cpu",
    )
    assert isinstance(search, OptunaSearchResult)
    assert {"min_cluster_size", "min_samples"} <= set(search.best_params)
    assert "cluster_selection_method" in search.best_params
    assert "cluster_selection_epsilon" in search.best_params
    assert "alpha" in search.best_params
    assert search.hdbscan_result.n_clusters == 3
    assert adjusted_rand_score(y_true, search.hdbscan_result.labels) > 0.9


def test_search_hdbscan_is_reproducible(
    gmm_three_clusters_2d: tuple[np.ndarray, np.ndarray],
) -> None:
    X, _ = gmm_three_clusters_2d
    a = search_hdbscan(
        X, n_trials=10, mcs_range=(5, 50), ms_range=(1, 20), random_state=42, engine="cpu"
    )
    b = search_hdbscan(
        X, n_trials=10, mcs_range=(5, 50), ms_range=(1, 20), random_state=42, engine="cpu"
    )
    assert a.best_params == b.best_params
    assert a.best_persistence_sum == pytest.approx(b.best_persistence_sum, abs=1e-12)
    np.testing.assert_array_equal(a.hdbscan_result.labels, b.hdbscan_result.labels)


def test_search_hdbscan_different_seeds_visit_different_points(
    gmm_three_clusters_2d: tuple[np.ndarray, np.ndarray],
) -> None:
    X, _ = gmm_three_clusters_2d
    a = search_hdbscan(
        X, n_trials=8, mcs_range=(5, 60), ms_range=(1, 20), random_state=0, engine="cpu"
    )
    b = search_hdbscan(
        X, n_trials=8, mcs_range=(5, 60), ms_range=(1, 20), random_state=1, engine="cpu"
    )
    trials_a = [(t.params["min_cluster_size"], t.params["min_samples"]) for t in a.study.trials]
    trials_b = [(t.params["min_cluster_size"], t.params["min_samples"]) for t in b.study.trials]
    assert trials_a != trials_b


def test_search_hdbscan_rejects_bad_n_trials() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    with pytest.raises(ValueError, match="n_trials"):
        search_hdbscan(X, n_trials=0, engine="cpu")


def test_search_hdbscan_rejects_bad_mcs_range() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    with pytest.raises(ValueError, match="mcs_range"):
        search_hdbscan(X, mcs_range=(1, 10), engine="cpu")
    with pytest.raises(ValueError, match="mcs_range"):
        search_hdbscan(X, mcs_range=(20, 10), engine="cpu")


def test_search_hdbscan_rejects_bad_ms_range() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    with pytest.raises(ValueError, match="ms_range"):
        search_hdbscan(X, ms_range=(0, 10), engine="cpu")
    with pytest.raises(ValueError, match="ms_range"):
        search_hdbscan(X, ms_range=(20, 10), engine="cpu")


def test_search_hdbscan_rejects_unknown_objective() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    with pytest.raises(ValueError, match="objective"):
        search_hdbscan(
            X, n_trials=3, engine="cpu",
            objective="max_persistence",  # type: ignore[arg-type]
        )


def test_search_hdbscan_combined_geom_objective(
    gmm_three_clusters_2d: tuple[np.ndarray, np.ndarray],
) -> None:
    X, _ = gmm_three_clusters_2d
    search = search_hdbscan(
        X,
        n_trials=12,
        mcs_range=(5, 60),
        ms_range=(1, 20),
        random_state=0,
        engine="cpu",
        objective="combined_geom",
    )
    study = search.study
    assert len(study.trials) == 12
    # Each trial records the full per-trial metric panel.
    expected_keys = {
        "relative_validity",
        "persistence_sum",
        "persistence_median",
        "persistence_max",
        "persistence_mean",
        "n_clusters",
        "outlier_fraction",
    }
    for trial in study.trials:
        assert expected_keys.issubset(trial.user_attrs.keys())
    # Best trial value equals sqrt(max(DBCV, 0) * median_persistence).
    best = study.best_trial
    dbcv = best.user_attrs["relative_validity"]
    persistence_median = best.user_attrs["persistence_median"]
    expected_value = float(np.sqrt(max(dbcv, 0.0) * persistence_median))
    assert best.value == pytest.approx(expected_value, abs=1e-12)


# ------------------------------------------------------------ _effective_mcs_bounds


@pytest.mark.parametrize(
    ("n", "user", "expected"),
    [
        (10_000, (5, 500), (5, 500)),
        (3_000, (5, 500), (5, 300)),
        (200, (5, 500), (5, 20)),
        (50, (5, 500), (5, 5)),
        (30, (10, 500), (10, 10)),
    ],
)
def test_effective_mcs_bounds(n: int, user: tuple[int, int], expected: tuple[int, int]) -> None:
    assert _effective_mcs_bounds(n, user) == expected


def test_search_respects_auto_capped_mcs() -> None:
    """On a small sample the search must never sample an MCS above ``n // 10``."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(loc, size=(40, 2)) for loc in (-5, 5)])  # n=80, cap -> 8
    search = search_hdbscan(
        X, n_trials=10, mcs_range=(5, 500), ms_range=(1, 20), random_state=0, engine="cpu"
    )
    mcs_values = [t.params["min_cluster_size"] for t in search.study.trials]
    assert max(mcs_values) <= 8
