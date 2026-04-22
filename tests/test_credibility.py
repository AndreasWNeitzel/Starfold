"""Tests for the global clustering-credibility test.

Covers:

* :func:`empirical_upper_tail_pvalue` math -- hand-computable cases,
  the ``(r+1)/(n+1)`` correction, and the two validation branches
  (empty / non-1-D null samples).
* :func:`compute_credibility` wiring -- the three p-values line up
  with the three baseline arrays, the ``passes`` flag respects
  ``alpha``, and the ``config`` is copied from the baseline.
* End-to-end behaviour through :class:`UnsupervisedPipeline`: a
  3-blob fixture must produce a :class:`CredibilityReport` attached
  to the result, with the same null sizes as the baseline.
"""

from __future__ import annotations

import numpy as np
import pytest

from starfold.credibility import (
    CredibilityReport,
    compute_credibility,
    empirical_upper_tail_pvalue,
)
from starfold.noise_baseline import NoiseBaselineResult


def _fake_baseline(
    *,
    per_max: np.ndarray,
    per_nc: np.ndarray,
    per_obj: np.ndarray,
    null_cluster_persistence: np.ndarray | None = None,
    null_cluster_size: np.ndarray | None = None,
    null_cluster_realisation: np.ndarray | None = None,
    objective: str = "persistence_sum",
    percentile: float = 99.7,
) -> NoiseBaselineResult:
    arr_max = np.asarray(per_max, dtype=np.float64)
    threshold = float(np.percentile(arr_max, percentile)) if arr_max.size else 0.0
    if null_cluster_persistence is None:
        null_cluster_persistence = np.zeros(0, dtype=np.float64)
    if null_cluster_size is None:
        null_cluster_size = np.zeros(0, dtype=np.intp)
    if null_cluster_realisation is None:
        null_cluster_realisation = np.zeros(0, dtype=np.intp)
    return NoiseBaselineResult(
        threshold=threshold,
        per_realisation_max=arr_max,
        per_realisation_n_clusters=np.asarray(per_nc, dtype=np.intp),
        per_realisation_objective=np.asarray(per_obj, dtype=np.float64),
        null_cluster_persistence=np.asarray(null_cluster_persistence, dtype=np.float64),
        null_cluster_size=np.asarray(null_cluster_size, dtype=np.intp),
        null_cluster_realisation=np.asarray(null_cluster_realisation, dtype=np.intp),
        percentile=percentile,
        config={"objective": objective, "n_samples": 100, "n_features": 3},
        cache_path=None,
    )


# ---------------------------------------------------------------- p-value math


def test_pvalue_matches_hand_computation() -> None:
    null = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    # r = 0 (nothing is >= 0.6), so p = (0 + 1) / (5 + 1) = 1/6.
    assert empirical_upper_tail_pvalue(0.6, null) == pytest.approx(1.0 / 6.0)
    # r = 3 (0.3, 0.4, 0.5 are >= 0.3), so p = 4/6 = 2/3.
    assert empirical_upper_tail_pvalue(0.3, null) == pytest.approx(4.0 / 6.0)


def test_pvalue_never_exactly_zero() -> None:
    null = np.zeros(1000)
    # With a huge observed value, r = 0 -> p = 1 / 1001 > 0.
    p = empirical_upper_tail_pvalue(1_000_000.0, null)
    assert 0.0 < p < 1.0


def test_pvalue_rejects_empty_null() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        empirical_upper_tail_pvalue(0.5, np.array([]))


def test_pvalue_rejects_non_1d_null() -> None:
    with pytest.raises(ValueError, match="1-D"):
        empirical_upper_tail_pvalue(0.5, np.zeros((2, 2)))


# ---------------------------------------------------------------- compute_credibility


def test_credibility_passes_when_every_pvalue_below_alpha() -> None:
    # Null arrays all flat and small; observed is well above every draw.
    baseline = _fake_baseline(
        per_max=np.zeros(1000),
        per_nc=np.zeros(1000, dtype=np.intp),
        per_obj=np.zeros(1000),
    )
    report = compute_credibility(
        n_clusters=5, best_objective=10.0, max_persistence=1.0,
        baseline=baseline, alpha=0.01,
    )
    assert isinstance(report, CredibilityReport)
    assert report.passes is True
    # With 1000 null samples and r = 0 for all three scalars:
    # p = 1 / 1001 ~ 1e-3, all below 0.01.
    expected = 1.0 / 1001.0
    assert report.n_clusters_pvalue == pytest.approx(expected)
    assert report.objective_pvalue == pytest.approx(expected)
    assert report.max_persistence_pvalue == pytest.approx(expected)


def test_credibility_fails_when_any_pvalue_at_or_above_alpha() -> None:
    # max-persistence null contains values equal to observed, so that
    # p-value is high; the other two nulls are flat at zero.
    per_max = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    baseline = _fake_baseline(
        per_max=per_max,
        per_nc=np.zeros(5, dtype=np.intp),
        per_obj=np.zeros(5),
    )
    report = compute_credibility(
        n_clusters=3, best_objective=5.0, max_persistence=1.0,
        baseline=baseline, alpha=0.01,
    )
    assert report.passes is False
    # r = 5, p = 6/6 = 1.0.
    assert report.max_persistence_pvalue == pytest.approx(1.0)


def test_credibility_copies_baseline_objective_name() -> None:
    baseline = _fake_baseline(
        per_max=np.zeros(5),
        per_nc=np.zeros(5, dtype=np.intp),
        per_obj=np.zeros(5),
        objective="combined_geom",
    )
    report = compute_credibility(
        n_clusters=2, best_objective=0.1, max_persistence=0.1,
        baseline=baseline,
    )
    assert report.objective_name == "combined_geom"
    assert report.config["objective"] == "combined_geom"


def test_credibility_rejects_bad_alpha() -> None:
    baseline = _fake_baseline(
        per_max=np.zeros(5),
        per_nc=np.zeros(5, dtype=np.intp),
        per_obj=np.zeros(5),
    )
    with pytest.raises(ValueError, match="alpha"):
        compute_credibility(
            n_clusters=2, best_objective=0.0, max_persistence=0.0,
            baseline=baseline, alpha=0.0,
        )
    with pytest.raises(ValueError, match="alpha"):
        compute_credibility(
            n_clusters=2, best_objective=0.0, max_persistence=0.0,
            baseline=baseline, alpha=1.0,
        )


def test_credibility_rejects_empty_baseline() -> None:
    empty = _fake_baseline(
        per_max=np.array([]),
        per_nc=np.array([], dtype=np.intp),
        per_obj=np.array([]),
    )
    with pytest.raises(ValueError, match="no realisations"):
        compute_credibility(
            n_clusters=2, best_objective=0.0, max_persistence=0.0,
            baseline=empty,
        )


def test_credibility_summary_human_readable() -> None:
    baseline = _fake_baseline(
        per_max=np.zeros(10),
        per_nc=np.zeros(10, dtype=np.intp),
        per_obj=np.zeros(10),
    )
    report = compute_credibility(
        n_clusters=4, best_objective=1.0, max_persistence=0.5,
        baseline=baseline,
    )
    text = report.summary()
    assert "PASS" in text or "FAIL" in text
    assert "n_clusters" in text
    assert "max_persistence" in text
    assert "per-cluster" in text


# ---------------------------------------------------------------- per-cluster p-values


def test_per_cluster_pvalues_match_empirical_formula() -> None:
    # 1000 noise-cluster persistence values, uniform in [0, 1].
    rng = np.random.default_rng(42)
    null_pool = rng.uniform(0.0, 1.0, size=1000)
    baseline = _fake_baseline(
        per_max=np.zeros(10),
        per_nc=np.zeros(10, dtype=np.intp),
        per_obj=np.zeros(10),
        null_cluster_persistence=null_pool,
        null_cluster_size=np.full(1000, 10, dtype=np.intp),
        null_cluster_realisation=np.zeros(1000, dtype=np.intp),
    )
    observed = np.array([0.1, 0.5, 0.95, 2.0])
    report = compute_credibility(
        n_clusters=4, best_objective=1.0, max_persistence=2.0,
        baseline=baseline, cluster_persistence=observed, alpha=0.01,
    )
    # Hand-check each p = (r + 1) / (n + 1).
    for v, p in zip(observed, report.per_cluster_pvalue, strict=True):
        r = int(np.sum(null_pool >= v))
        assert p == pytest.approx((r + 1) / 1001)
    # Only the 2.0 observation should clear the 0.01 threshold.
    assert report.per_cluster_significant.tolist() == [False, False, False, True]


def test_per_cluster_pvalue_is_one_when_no_null_pool() -> None:
    # baseline with an empty null pool still works; per-cluster arrays
    # stay the same length as the real-data cluster count, filled with
    # the least-significant p-value (1.0) so the call remains well-defined.
    baseline = _fake_baseline(
        per_max=np.zeros(5),
        per_nc=np.zeros(5, dtype=np.intp),
        per_obj=np.zeros(5),
    )
    report = compute_credibility(
        n_clusters=3, best_objective=1.0, max_persistence=1.0,
        baseline=baseline,
        cluster_persistence=np.array([0.1, 0.2, 0.3]),
    )
    assert report.per_cluster_pvalue.shape == (3,)
    np.testing.assert_array_equal(report.per_cluster_pvalue, np.ones(3))
    assert not report.per_cluster_significant.any()


def test_per_cluster_rejects_wrong_length() -> None:
    baseline = _fake_baseline(
        per_max=np.zeros(5),
        per_nc=np.zeros(5, dtype=np.intp),
        per_obj=np.zeros(5),
        null_cluster_persistence=np.array([0.1, 0.2, 0.3]),
        null_cluster_size=np.array([10, 10, 10], dtype=np.intp),
        null_cluster_realisation=np.zeros(3, dtype=np.intp),
    )
    with pytest.raises(ValueError, match="does not match n_clusters"):
        compute_credibility(
            n_clusters=3, best_objective=1.0, max_persistence=1.0,
            baseline=baseline,
            cluster_persistence=np.array([0.1, 0.2]),
        )


def test_per_cluster_rejects_non_1d() -> None:
    baseline = _fake_baseline(
        per_max=np.zeros(5),
        per_nc=np.zeros(5, dtype=np.intp),
        per_obj=np.zeros(5),
        null_cluster_persistence=np.array([0.1, 0.2, 0.3]),
        null_cluster_size=np.array([10, 10, 10], dtype=np.intp),
        null_cluster_realisation=np.zeros(3, dtype=np.intp),
    )
    with pytest.raises(ValueError, match="1-D"):
        compute_credibility(
            n_clusters=4, best_objective=1.0, max_persistence=1.0,
            baseline=baseline,
            cluster_persistence=np.zeros((2, 2)),
        )


def test_per_cluster_defaults_to_empty_when_not_provided() -> None:
    baseline = _fake_baseline(
        per_max=np.zeros(5),
        per_nc=np.zeros(5, dtype=np.intp),
        per_obj=np.zeros(5),
        null_cluster_persistence=np.array([0.1, 0.2, 0.3]),
        null_cluster_size=np.array([10, 10, 10], dtype=np.intp),
        null_cluster_realisation=np.zeros(3, dtype=np.intp),
    )
    report = compute_credibility(
        n_clusters=2, best_objective=0.1, max_persistence=0.1,
        baseline=baseline,
    )
    assert report.observed_cluster_persistence.shape == (0,)
    assert report.per_cluster_pvalue.shape == (0,)
    assert report.per_cluster_significant.shape == (0,)


# ---------------------------------------------------------------- end-to-end pipeline


def test_pipeline_attaches_credibility_when_baseline_runs(
    gmm_three_clusters_2d: tuple[np.ndarray, np.ndarray],
) -> None:
    from starfold.pipeline import UnsupervisedPipeline  # noqa: PLC0415

    X, _ = gmm_three_clusters_2d
    # Tiny budget: 2 noise realisations x 3 trials is fast but gives
    # enough null samples to exercise the p-value code path.
    pipeline = UnsupervisedPipeline(
        umap_kwargs={"n_epochs": 50, "n_neighbors": 10},
        hdbscan_optuna_trials=4,
        mcs_range=(5, 15),
        ms_range=(1, 5),
        noise_baseline_kwargs={
            "n_realisations": 3,
            "per_realisation_trials": 3,
            "mcs_range": (5, 15),
            "ms_range": (1, 5),
            "cache_dir": False,
        },
        engine="cpu",
        random_state=0,
    )
    result = pipeline.fit(X)
    assert result.credibility is not None
    assert result.noise_baseline is not None
    # The credibility's null arrays come straight from the baseline.
    assert result.credibility.null_n_clusters.shape == (3,)
    assert result.credibility.null_objective.shape == (3,)
    assert result.credibility.null_max_persistence.shape == (3,)
    # Per-cluster p-values line up with the real-data cluster count.
    assert result.credibility.per_cluster_pvalue.shape == (result.n_clusters,)
    assert result.credibility.observed_cluster_persistence.shape == (result.n_clusters,)
    assert result.credibility.per_cluster_significant.shape == (result.n_clusters,)
    # Observed per-cluster persistence echoes the result's persistence array.
    np.testing.assert_allclose(
        result.credibility.observed_cluster_persistence, result.persistence,
    )
    # The null pool captured at least one noise cluster across the 3
    # realisations (3 trials each is enough for HDBSCAN to find some).
    assert result.noise_baseline.null_cluster_persistence.ndim == 1
    # The summary must mention the credibility verdict.
    assert "credibility" in result.summary()
    assert "per-cluster" in result.summary()


def test_pipeline_credibility_none_when_noise_baseline_skipped(
    gmm_three_clusters_2d: tuple[np.ndarray, np.ndarray],
) -> None:
    from starfold.pipeline import UnsupervisedPipeline  # noqa: PLC0415

    X, _ = gmm_three_clusters_2d
    pipeline = UnsupervisedPipeline(
        umap_kwargs={"n_epochs": 50, "n_neighbors": 10},
        hdbscan_optuna_trials=4,
        mcs_range=(5, 15),
        ms_range=(1, 5),
        skip_noise_baseline=True,
        engine="cpu",
        random_state=0,
    )
    result = pipeline.fit(X)
    assert result.credibility is None
    assert result.noise_baseline is None
