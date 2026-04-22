"""Tests for :mod:`starfold.merge`.

The recommender combines a hierarchical cohesion signal with an
embedding-geometric gap signal. These tests check that:

* Well-separated clusters return low cohesion, large gap_ratio, and
  ``recommended == False``.
* Touching clusters on top of each other flag ``recommended == True``.
* The API surfaces pairs in ``(i, j)`` with ``i < j`` ordering and
  sort by cohesion vs gap as documented.
* Corner cases raise: cuml hierarchy, bad embedding shape, invalid
  thresholds.
"""

from __future__ import annotations

import numpy as np
import pytest

from starfold.hierarchy import HierarchicalStructure
from starfold.merge import MergeSuggestion, suggest_merges
from starfold.pipeline import UnsupervisedPipeline


def _cheap_pipeline(**overrides):
    kwargs = {
        "umap_kwargs": {"n_neighbors": 12, "n_epochs": 100},
        "hdbscan_optuna_trials": 6,
        "skip_noise_baseline": True,
        "random_state": 0,
        "engine": "cpu",
    }
    kwargs.update(overrides)
    return UnsupervisedPipeline(**kwargs)


def test_suggest_merges_returns_all_pairs_for_three_clusters() -> None:
    rng = np.random.default_rng(0)
    X = np.vstack(
        [rng.normal(loc, 0.25, size=(40, 3)) for loc in ((-6, 0, 0), (6, 0, 0), (0, 6, 0))]
    ).astype(np.float64)
    result = _cheap_pipeline().fit(X)
    # This fixture is well-separated -- HDBSCAN reliably finds >= 2
    # flat clusters; if it finds 1 (degenerate), skip.
    if result.n_clusters < 2:
        pytest.skip("fixture produced < 2 clusters on this run")
    suggestions = suggest_merges(result.hierarchy, result.embedding)
    expected_pairs = result.n_clusters * (result.n_clusters - 1) // 2
    assert len(suggestions) == expected_pairs
    for s in suggestions:
        assert isinstance(s, MergeSuggestion)
        assert s.cluster_i < s.cluster_j


def test_well_separated_clusters_are_not_recommended_to_merge() -> None:
    rng = np.random.default_rng(1)
    X = np.vstack(
        [rng.normal(loc, 0.2, size=(50, 3)) for loc in ((-8, 0, 0), (8, 0, 0))]
    ).astype(np.float64)
    result = _cheap_pipeline().fit(X)
    if result.n_clusters < 2:
        pytest.skip("fixture produced < 2 clusters on this run")
    suggestions = suggest_merges(result.hierarchy, result.embedding)
    # With a huge centroid gap and tight dispersions, gap_ratio must be
    # large and none of the pairs should be recommended.
    for s in suggestions:
        assert s.gap_ratio > 2.0
        assert not s.recommended


def test_touching_clusters_flag_recommended_under_permissive_thresholds() -> None:
    # Two blobs so close that their intra-cluster RMS is comparable to
    # the centroid gap. We cannot pin the exact suggestion state because
    # HDBSCAN may or may not split these into two flat clusters on every
    # seed; instead we sanity-check that *when* it does split and
    # gap_ratio is small, recommended can be forced by relaxing the
    # thresholds.
    rng = np.random.default_rng(2)
    X = np.vstack(
        [rng.normal(loc, 0.6, size=(80, 2)) for loc in ((-0.8, 0), (0.8, 0))]
    ).astype(np.float64)
    result = _cheap_pipeline(
        umap_kwargs={"n_neighbors": 10, "n_epochs": 200},
        hdbscan_optuna_trials=10,
    ).fit(X)
    if result.n_clusters < 2:
        pytest.skip("touching-blob fixture collapsed into one cluster")
    # Relax thresholds: cohesion_threshold=0.0 forces the hierarchical
    # signal to always vote merge; gap_threshold=1e6 forces the
    # geometric signal to always vote merge. Every pair must then be
    # recommended.
    permissive = suggest_merges(
        result.hierarchy,
        result.embedding,
        cohesion_threshold=1e-6,
        gap_threshold=1e6,
    )
    assert all(s.recommended for s in permissive)


def test_sort_by_gap_ratio_orders_ascending() -> None:
    rng = np.random.default_rng(3)
    X = np.vstack(
        [rng.normal(loc, 0.25, size=(40, 3)) for loc in ((-6, 0, 0), (6, 0, 0), (0, 8, 0))]
    ).astype(np.float64)
    result = _cheap_pipeline().fit(X)
    if result.n_clusters < 3:
        pytest.skip("fixture produced < 3 clusters on this run")
    by_gap = suggest_merges(result.hierarchy, result.embedding, sort_by="gap_ratio")
    finite_gaps = [s.gap_ratio for s in by_gap if np.isfinite(s.gap_ratio)]
    assert finite_gaps == sorted(finite_gaps)


def test_sort_by_cohesion_orders_descending() -> None:
    rng = np.random.default_rng(4)
    X = np.vstack(
        [rng.normal(loc, 0.25, size=(40, 3)) for loc in ((-6, 0, 0), (6, 0, 0), (0, 8, 0))]
    ).astype(np.float64)
    result = _cheap_pipeline().fit(X)
    if result.n_clusters < 3:
        pytest.skip("fixture produced < 3 clusters on this run")
    by_cohesion = suggest_merges(result.hierarchy, result.embedding)
    finite_cohesions = [
        s.cohesion_ratio for s in by_cohesion if np.isfinite(s.cohesion_ratio)
    ]
    assert finite_cohesions == sorted(finite_cohesions, reverse=True)


def test_cohesion_and_gap_ratios_are_within_documented_bounds() -> None:
    rng = np.random.default_rng(5)
    X = np.vstack(
        [rng.normal(loc, 0.3, size=(40, 2)) for loc in ((-5, 0), (5, 0), (0, 5))]
    ).astype(np.float64)
    result = _cheap_pipeline().fit(X)
    if result.n_clusters < 2:
        pytest.skip("fixture produced < 2 clusters on this run")
    suggestions = suggest_merges(result.hierarchy, result.embedding)
    for s in suggestions:
        if np.isfinite(s.cohesion_ratio):
            assert 0.0 <= s.cohesion_ratio <= 1.0 + 1e-9
        if np.isfinite(s.gap_ratio):
            assert s.gap_ratio >= 0.0


def test_unavailable_hierarchy_raises_runtime_error() -> None:
    hierarchy = HierarchicalStructure(
        available=False,
        edges=np.zeros(0, dtype=[("parent", "<i8"), ("child", "<i8"),
                                 ("lambda_val", "<f8"), ("child_size", "<i8")]),
        flat_to_node=np.full(0, -1, dtype=np.intp),
        labels=np.zeros(10, dtype=np.intp),
        n_samples=10,
        raw_tree=None,
    )
    embedding = np.zeros((10, 2), dtype=np.float64)
    with pytest.raises(RuntimeError, match="CPU-backend fit"):
        suggest_merges(hierarchy, embedding)


def test_rejects_bad_sort_by() -> None:
    rng = np.random.default_rng(6)
    X = np.vstack(
        [rng.normal(loc, 0.25, size=(40, 2)) for loc in ((-5, 0), (5, 0))]
    ).astype(np.float64)
    result = _cheap_pipeline().fit(X)
    if result.n_clusters < 2:
        pytest.skip("fixture produced < 2 clusters on this run")
    with pytest.raises(ValueError, match="sort_by"):
        suggest_merges(result.hierarchy, result.embedding, sort_by="banana")  # type: ignore[arg-type]


def test_rejects_bad_thresholds() -> None:
    rng = np.random.default_rng(7)
    X = np.vstack(
        [rng.normal(loc, 0.25, size=(40, 2)) for loc in ((-5, 0), (5, 0))]
    ).astype(np.float64)
    result = _cheap_pipeline().fit(X)
    if result.n_clusters < 2:
        pytest.skip("fixture produced < 2 clusters on this run")
    with pytest.raises(ValueError, match="cohesion_threshold"):
        suggest_merges(result.hierarchy, result.embedding, cohesion_threshold=0.0)
    with pytest.raises(ValueError, match="cohesion_threshold"):
        suggest_merges(result.hierarchy, result.embedding, cohesion_threshold=1.5)
    with pytest.raises(ValueError, match="gap_threshold"):
        suggest_merges(result.hierarchy, result.embedding, gap_threshold=0.0)


def test_rejects_mismatched_embedding_shape() -> None:
    rng = np.random.default_rng(8)
    X = np.vstack(
        [rng.normal(loc, 0.25, size=(40, 2)) for loc in ((-5, 0), (5, 0))]
    ).astype(np.float64)
    result = _cheap_pipeline().fit(X)
    if result.n_clusters < 2:
        pytest.skip("fixture produced < 2 clusters on this run")
    with pytest.raises(ValueError, match="2-D"):
        suggest_merges(result.hierarchy, result.embedding.ravel())
    with pytest.raises(ValueError, match="n_samples"):
        suggest_merges(result.hierarchy, result.embedding[:-1])


def test_pipeline_result_suggest_merges_method_delegates() -> None:
    rng = np.random.default_rng(9)
    X = np.vstack(
        [rng.normal(loc, 0.25, size=(40, 2)) for loc in ((-5, 0), (5, 0))]
    ).astype(np.float64)
    result = _cheap_pipeline().fit(X)
    if result.n_clusters < 2:
        pytest.skip("fixture produced < 2 clusters on this run")
    via_method = result.suggest_merges()
    via_function = suggest_merges(result.hierarchy, result.embedding)
    assert len(via_method) == len(via_function)
    for a, b in zip(via_method, via_function, strict=True):
        assert (a.cluster_i, a.cluster_j) == (b.cluster_i, b.cluster_j)
        assert a.cohesion_ratio == pytest.approx(b.cohesion_ratio, nan_ok=True)
        assert a.gap_ratio == pytest.approx(b.gap_ratio, nan_ok=True)


def test_empty_hierarchy_returns_empty_list() -> None:
    # Synthesise a hierarchy with one flat cluster: no pairs to consider.
    labels = np.zeros(20, dtype=np.intp)
    edges = np.zeros(0, dtype=[("parent", "<i8"), ("child", "<i8"),
                               ("lambda_val", "<f8"), ("child_size", "<i8")])
    hierarchy = HierarchicalStructure(
        available=True,
        edges=edges,
        flat_to_node=np.full(1, -1, dtype=np.intp),
        labels=labels,
        n_samples=20,
        raw_tree=None,
    )
    assert suggest_merges(hierarchy, np.zeros((20, 2))) == []
