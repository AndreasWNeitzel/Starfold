"""Round-trip tests for PipelineResult.save / load_pipeline_result."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from starfold.io import load_pipeline_result
from starfold.pipeline import UnsupervisedPipeline

if TYPE_CHECKING:
    from pathlib import Path


def _run_pipeline(X: np.ndarray, *, with_baseline: bool, cache_dir: Path) -> object:
    kwargs: dict[str, object] = {
        "umap_kwargs": {"n_epochs": 100, "n_neighbors": 10},
        "hdbscan_optuna_trials": 10,
        "mcs_range": (5, 20),
        "ms_range": (1, 5),
        "engine": "cpu",
        "random_state": 0,
    }
    if with_baseline:
        kwargs["skip_noise_baseline"] = False
        kwargs["noise_baseline_kwargs"] = {
            "n_realisations": 3,
            "per_realisation_trials": 3,
            "mcs_range": (5, 10),
            "ms_range": (1, 5),
            "cache_dir": str(cache_dir),
        }
    else:
        kwargs["skip_noise_baseline"] = True
    return UnsupervisedPipeline(**kwargs).fit(X)  # type: ignore[arg-type]


def test_save_load_round_trip_without_baseline(
    tmp_path: Path,
    gmm_three_clusters_2d: tuple[np.ndarray, np.ndarray],
) -> None:
    X, _ = gmm_three_clusters_2d
    result = _run_pipeline(X, with_baseline=False, cache_dir=tmp_path / "cache")
    directory = result.save(tmp_path / "run")  # type: ignore[attr-defined]

    loaded = load_pipeline_result(directory)
    np.testing.assert_allclose(loaded["embedding"], result.embedding, atol=0.0)  # type: ignore[attr-defined]
    np.testing.assert_array_equal(loaded["labels"], result.labels)  # type: ignore[attr-defined]
    np.testing.assert_allclose(loaded["probabilities"], result.probabilities, atol=0.0)  # type: ignore[attr-defined]
    np.testing.assert_allclose(loaded["persistence"], result.persistence, atol=0.0)  # type: ignore[attr-defined]
    assert loaded["trustworthiness"] == pytest.approx(result.trustworthiness)  # type: ignore[attr-defined]
    assert loaded["best_params"] == result.best_params  # type: ignore[attr-defined]
    assert "significant" not in loaded
    assert "noise_threshold" not in loaded


def test_save_load_round_trip_with_baseline(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(loc, size=(40, 3)) for loc in (-5, 5)])
    result = _run_pipeline(X, with_baseline=True, cache_dir=tmp_path / "cache")
    directory = result.save(tmp_path / "run")  # type: ignore[attr-defined]

    loaded = load_pipeline_result(directory)
    assert "significant" in loaded
    np.testing.assert_array_equal(loaded["significant"], result.significant)  # type: ignore[attr-defined]
    assert loaded["noise_threshold"] == pytest.approx(result.noise_baseline.threshold)  # type: ignore[attr-defined]
    assert loaded["noise_percentile"] == pytest.approx(result.noise_baseline.percentile)  # type: ignore[attr-defined]
    np.testing.assert_allclose(
        loaded["noise_per_realisation_max"],
        result.noise_baseline.per_realisation_max,  # type: ignore[attr-defined]
        atol=0.0,
    )
    np.testing.assert_array_equal(
        loaded["noise_per_realisation_n_clusters"],
        result.noise_baseline.per_realisation_n_clusters,  # type: ignore[attr-defined]
    )
    np.testing.assert_allclose(
        loaded["noise_per_realisation_objective"],
        result.noise_baseline.per_realisation_objective,  # type: ignore[attr-defined]
        atol=0.0,
    )
    np.testing.assert_allclose(
        loaded["noise_null_cluster_persistence"],
        result.noise_baseline.null_cluster_persistence,  # type: ignore[attr-defined]
        atol=0.0,
    )
    np.testing.assert_array_equal(
        loaded["noise_null_cluster_size"],
        result.noise_baseline.null_cluster_size,  # type: ignore[attr-defined]
    )
    np.testing.assert_array_equal(
        loaded["noise_null_cluster_realisation"],
        result.noise_baseline.null_cluster_realisation,  # type: ignore[attr-defined]
    )
    assert "credibility" in loaded
    cred = result.credibility  # type: ignore[attr-defined]
    assert loaded["credibility"]["passes"] == cred.passes
    assert loaded["credibility"]["alpha"] == pytest.approx(cred.alpha)
    assert loaded["credibility"]["objective_name"] == cred.objective_name
    assert loaded["credibility"]["n_clusters_pvalue"] == pytest.approx(cred.n_clusters_pvalue)
    assert loaded["credibility"]["objective_pvalue"] == pytest.approx(cred.objective_pvalue)
    assert loaded["credibility"]["max_persistence_pvalue"] == pytest.approx(
        cred.max_persistence_pvalue
    )
    np.testing.assert_allclose(
        loaded["credibility_observed_cluster_persistence"],
        cred.observed_cluster_persistence,
        atol=0.0,
    )
    np.testing.assert_allclose(
        loaded["credibility_per_cluster_pvalue"],
        cred.per_cluster_pvalue,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        loaded["credibility_per_cluster_significant"],
        cred.per_cluster_significant,
    )
    assert loaded["credibility"]["per_cluster_significant_count"] == int(
        cred.per_cluster_significant.sum()
    )
    assert loaded["credibility"]["per_cluster_total"] == int(cred.per_cluster_significant.size)


def test_scaler_round_trip_preserves_transform(
    tmp_path: Path,
    gmm_three_clusters_2d: tuple[np.ndarray, np.ndarray],
) -> None:
    X, _ = gmm_three_clusters_2d
    result = _run_pipeline(X, with_baseline=False, cache_dir=tmp_path / "cache")
    directory = result.save(tmp_path / "run")  # type: ignore[attr-defined]
    loaded = load_pipeline_result(directory)
    np.testing.assert_allclose(
        loaded["scaler"].transform(X),
        result.scaler.transform(X),  # type: ignore[attr-defined]
        atol=1e-12,
    )
