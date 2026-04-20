"""Tests for the end-to-end UnsupervisedPipeline.

Covers:

* the pipeline recovers the expected number of clusters on a 3-blob
  GMM with ARI > 0.9,
* reproducibility across identical seeds,
* ``skip_noise_baseline=True`` skips the expensive step cleanly,
* the noise-baseline path populates ``significant``,
* input validation rejects non-2D arrays,
* the summary string mentions the key metrics.
"""

from __future__ import annotations

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
