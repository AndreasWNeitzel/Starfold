"""Tests for the end-to-end UnsupervisedPipeline.

Covers:

* the pipeline recovers the expected number of clusters on a 3-blob
  GMM with ARI > 0.9,
* reproducibility across identical seeds,
* ``skip_noise_baseline=True`` skips the expensive step cleanly,
* the noise-baseline path populates ``significant``,
* input validation rejects non-2D arrays,
* the summary string mentions the key metrics,
* the ``combined_geom`` objective produces a valid run,
* the tuning and quality dashboards return well-formed figures.
"""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from starfold.pipeline import PipelineResult, UnsupervisedPipeline


def _cheap_pipeline(**overrides: object) -> UnsupervisedPipeline:
    kwargs: dict[str, object] = {
        "umap_kwargs": {"n_epochs": 2000, "n_neighbors": 30},
        "hdbscan_optuna_trials": 20,
        "mcs_range": (30, 100),
        "ms_range": (5, 20),
        "engine": "cpu",
        "skip_noise_baseline": True,
        "random_state": 0,
    }
    kwargs.update(overrides)
    return UnsupervisedPipeline(**kwargs)  # type: ignore[arg-type]


def test_pipeline_recovers_three_clusters(
    gmm_three_clusters_2d: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y_true = gmm_three_clusters_2d
    pipeline = _cheap_pipeline()
    result = pipeline.fit(X)
    assert isinstance(result, PipelineResult)
    assert result.n_clusters == 3
    assert result.embedding.shape == (X.shape[0], 2)
    assert adjusted_rand_score(y_true, result.labels) > 0.9
    assert 0.0 <= result.trustworthiness <= 1.0


def test_pipeline_is_reproducible(
    gmm_three_clusters_2d: tuple[np.ndarray, np.ndarray],
) -> None:
    X, _ = gmm_three_clusters_2d
    a = _cheap_pipeline(random_state=42).fit(X)
    b = _cheap_pipeline(random_state=42).fit(X)
    np.testing.assert_array_equal(a.labels, b.labels)
    np.testing.assert_allclose(a.embedding, b.embedding, atol=0.0)
    assert a.best_params == b.best_params
    assert a.trustworthiness == pytest.approx(b.trustworthiness, abs=1e-12)


def test_pipeline_skips_noise_baseline() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 3))
    result = _cheap_pipeline().fit(X)
    assert result.noise_baseline is None
    assert result.significant is None


def test_pipeline_runs_noise_baseline(tmp_path: object) -> None:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(loc, size=(50, 3)) for loc in (-5, 5)])  # n=100
    pipeline = _cheap_pipeline(
        skip_noise_baseline=False,
        noise_baseline_kwargs={
            "n_realisations": 3,
            "per_realisation_trials": 3,
            "mcs_range": (5, 10),
            "ms_range": (1, 5),
            "cache_dir": str(tmp_path),
        },
    )
    result = pipeline.fit(X)
    assert result.noise_baseline is not None
    assert result.significant is not None
    assert result.significant.shape == (result.n_clusters,)
    assert result.significant.dtype == bool


def test_noise_baseline_umap_kwargs_override(tmp_path: object) -> None:
    # The nested ``umap_kwargs`` in ``noise_baseline_kwargs`` should
    # override the main pipeline's UMAP configuration for noise fits
    # only. The main-fit UMAP runs at n_epochs=2000 (via _cheap_pipeline)
    # but the baseline should record n_epochs=300 in its config.
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(loc, size=(50, 3)) for loc in (-5, 5)])
    pipeline = _cheap_pipeline(
        skip_noise_baseline=False,
        noise_baseline_kwargs={
            "n_realisations": 2,
            "per_realisation_trials": 2,
            "mcs_range": (5, 10),
            "ms_range": (1, 5),
            "cache_dir": str(tmp_path),
            "umap_kwargs": {
                "n_neighbors": 30,
                "min_dist": 0.0,
                "n_epochs": 300,
            },
        },
    )
    result = pipeline.fit(X)
    assert result.noise_baseline is not None
    recorded = result.noise_baseline.config["umap_kwargs"]["n_epochs"]
    assert recorded == 300


def test_pipeline_rejects_non_2d() -> None:
    pipeline = _cheap_pipeline()
    with pytest.raises(ValueError, match="2-D"):
        pipeline.fit(np.zeros(10))


def test_pipeline_summary_mentions_metrics(
    gmm_three_clusters_2d: tuple[np.ndarray, np.ndarray],
) -> None:
    X, _ = gmm_three_clusters_2d
    result = _cheap_pipeline().fit(X)
    text = result.summary()
    assert "n_clusters" in text
    assert "trustworthiness" in text
    assert "persistence" in text


def test_pipeline_scaler_is_fitted(
    gmm_three_clusters_2d: tuple[np.ndarray, np.ndarray],
) -> None:
    X, _ = gmm_three_clusters_2d
    result = _cheap_pipeline().fit(X)
    transformed = result.scaler.transform(X)
    np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(transformed.std(axis=0, ddof=0), 1.0, atol=1e-10)


# --------------------------------------------------------------- combined_geom


def test_pipeline_combined_geom_objective(
    gmm_three_clusters_2d: tuple[np.ndarray, np.ndarray],
) -> None:
    X, _ = gmm_three_clusters_2d
    result = _cheap_pipeline(hdbscan_objective="combined_geom").fit(X)
    assert result.config["hdbscan_objective"] == "combined_geom"
    best = result.search.study.best_trial
    dbcv = best.user_attrs["relative_validity"]
    persistence_median = best.user_attrs["persistence_median"]
    assert np.isfinite(dbcv)
    assert np.isfinite(persistence_median)
    expected = float(np.sqrt(max(dbcv, 0.0) * persistence_median))
    assert best.value == pytest.approx(expected, abs=1e-12)


def test_summary_contains_dbcv_line(
    gmm_three_clusters_2d: tuple[np.ndarray, np.ndarray],
) -> None:
    X, _ = gmm_three_clusters_2d
    result = _cheap_pipeline(hdbscan_objective="combined_geom").fit(X)
    text = result.summary()
    assert "DBCV" in text
    assert "persistence_med" in text
    assert "objective" in text
    assert "combined_geom" in text


# --------------------------------------------------------------- dashboards


def test_plot_tuning_dashboard_returns_eight_panel_figure(
    gmm_three_clusters_2d: tuple[np.ndarray, np.ndarray],
) -> None:
    X, _ = gmm_three_clusters_2d
    result = _cheap_pipeline(hdbscan_objective="combined_geom").fit(X)
    fig = result.plot_tuning_dashboard()
    try:
        # 2x4 subplot grid; colorbars add extra axes, so allow >= 8.
        assert len(fig.axes) >= 8
        titles = [ax.get_title() for ax in fig.axes]
        tags = "".join(titles)
        for marker in ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"):
            assert marker in tags
        assert "starfold HDBSCAN tuning dashboard" in fig.get_suptitle()
    finally:
        plt.close(fig)


def test_plot_tuning_dashboard_respects_figsize(
    gmm_three_clusters_2d: tuple[np.ndarray, np.ndarray],
) -> None:
    X, _ = gmm_three_clusters_2d
    result = _cheap_pipeline().fit(X)
    fig = result.plot_tuning_dashboard(figsize=(16.0, 8.0))
    try:
        w, h = fig.get_size_inches()
        assert float(w) == pytest.approx(16.0)
        assert float(h) == pytest.approx(8.0)
    finally:
        plt.close(fig)


def test_plot_quality_dashboard_returns_six_panel_figure(
    gmm_three_clusters_2d: tuple[np.ndarray, np.ndarray],
) -> None:
    X, _ = gmm_three_clusters_2d
    result = _cheap_pipeline().fit(X)
    fig = result.plot_quality_dashboard(
        X,
        n_subsamples=5,
        subsample_fraction=0.8,
        k_values=(5, 10, 15),
        random_state=0,
    )
    try:
        # Six-panel grid; the membership-confidence panel adds a colorbar.
        assert len(fig.axes) >= 6
        titles = [ax.get_title() for ax in fig.axes]
        tags = "".join(titles)
        for marker in ("(a)", "(b)", "(c)", "(d)", "(e)", "(f)"):
            assert marker in tags
        assert "pipeline-quality dashboard" in fig.get_suptitle()
    finally:
        plt.close(fig)


def test_plot_quality_dashboard_rejects_non_2d(
    gmm_three_clusters_2d: tuple[np.ndarray, np.ndarray],
) -> None:
    X, _ = gmm_three_clusters_2d
    result = _cheap_pipeline().fit(X)
    with pytest.raises(ValueError, match="2-D"):
        result.plot_quality_dashboard(np.zeros(10))


def test_plot_quality_dashboard_accepts_precomputed_stability(
    gmm_three_clusters_2d: tuple[np.ndarray, np.ndarray],
) -> None:
    from starfold.stability import compute_subsample_stability  # noqa: PLC0415

    X, _ = gmm_three_clusters_2d
    result = _cheap_pipeline().fit(X)
    stability = compute_subsample_stability(
        result.embedding,
        result.labels,
        result.persistence,
        min_cluster_size=result.best_params["min_cluster_size"],
        min_samples=result.best_params["min_samples"],
        n_subsamples=4,
        subsample_fraction=0.8,
        engine="cpu",
        random_state=0,
    )
    fig = result.plot_quality_dashboard(
        X,
        stability=stability,
        k_values=(5, 10, 15),
    )
    try:
        assert len(fig.axes) >= 6
        assert stability.n_subsamples == 4
    finally:
        plt.close(fig)
