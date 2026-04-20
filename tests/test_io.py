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
