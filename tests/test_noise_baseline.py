"""Tests for the noise-baseline computation.

The baseline is evaluated on tiny (n_samples, n_features) matrices with
a cheap UMAP configuration so the suite completes in a few seconds.
The tests verify:

* the threshold and per-realisation maxima are shape-correct,
* results are bit-reproducible under a fixed ``random_state``,
* disk caching short-circuits a second call,
* ``force_recompute=True`` re-runs the computation,
* input validation rejects malformed calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from starfold.noise_baseline import (
    NoiseBaselineResult,
    _cache_key,
    _canonical_umap_kwargs,
    compute_noise_baseline,
    default_cache_dir,
)

if TYPE_CHECKING:
    from pathlib import Path

TINY_UMAP = {"n_epochs": 50, "n_neighbors": 10}


def _baseline(**overrides: object) -> NoiseBaselineResult:
    kwargs: dict[str, object] = {
        "n_samples": 150,
        "n_features": 3,
        "umap_kwargs": TINY_UMAP,
        "n_realisations": 3,
        "per_realisation_trials": 3,
        "mcs_range": (5, 15),
        "ms_range": (1, 5),
        "random_state": 0,
        "engine": "cpu",
        "cache_dir": False,
    }
    kwargs.update(overrides)
    return compute_noise_baseline(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------- core behaviour


def test_baseline_shapes_and_types() -> None:
    result = _baseline()
    assert isinstance(result, NoiseBaselineResult)
    assert result.per_realisation_max.shape == (3,)
    assert result.per_realisation_max.dtype == np.float64
    assert result.per_realisation_n_clusters.shape == (3,)
    assert np.issubdtype(result.per_realisation_n_clusters.dtype, np.integer)
    assert result.per_realisation_objective.shape == (3,)
    assert result.per_realisation_objective.dtype == np.float64
    # Per-cluster null pool: three 1-D arrays of equal length, consistent
    # with the per-realisation cluster counts.
    assert result.null_cluster_persistence.ndim == 1
    assert result.null_cluster_size.shape == result.null_cluster_persistence.shape
    assert result.null_cluster_realisation.shape == result.null_cluster_persistence.shape
    assert result.null_cluster_persistence.size == int(result.per_realisation_n_clusters.sum())
    # Sizes are positive integers; realisation indices are in range.
    if result.null_cluster_size.size:
        assert np.all(result.null_cluster_size >= 1)
        assert int(result.null_cluster_realisation.max()) < result.per_realisation_max.size
    assert result.threshold >= 0.0
    assert result.percentile == 99.7
    assert result.objective == "persistence_sum"


def test_baseline_records_objective_name() -> None:
    result = _baseline(objective="combined_geom")
    assert result.objective == "combined_geom"
    assert result.config["objective"] == "combined_geom"


def test_baseline_n_clusters_are_non_negative_integers() -> None:
    result = _baseline()
    assert np.all(result.per_realisation_n_clusters >= 0)


def test_baseline_threshold_equals_percentile_of_maxima() -> None:
    result = _baseline()
    expected = float(np.percentile(result.per_realisation_max, 99.7))
    assert result.threshold == pytest.approx(expected, abs=1e-12)


def test_baseline_is_reproducible() -> None:
    a = _baseline(random_state=42)
    b = _baseline(random_state=42)
    np.testing.assert_allclose(a.per_realisation_max, b.per_realisation_max, atol=0.0)
    assert a.threshold == pytest.approx(b.threshold, abs=0.0)


def test_baseline_parallel_matches_sequential() -> None:
    # n_jobs must not perturb the result: each realisation uses its own
    # deterministic sub-seed (base_seed + k), so the output array is
    # invariant under execution order.
    serial = _baseline(random_state=7, n_realisations=4)
    parallel = _baseline(random_state=7, n_realisations=4, n_jobs=2)
    np.testing.assert_allclose(
        serial.per_realisation_max,
        parallel.per_realisation_max,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        serial.per_realisation_n_clusters,
        parallel.per_realisation_n_clusters,
    )
    np.testing.assert_allclose(
        serial.per_realisation_objective,
        parallel.per_realisation_objective,
        atol=0.0,
    )
    np.testing.assert_allclose(
        serial.null_cluster_persistence,
        parallel.null_cluster_persistence,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        serial.null_cluster_size,
        parallel.null_cluster_size,
    )
    np.testing.assert_array_equal(
        serial.null_cluster_realisation,
        parallel.null_cluster_realisation,
    )


def test_baseline_different_seed_gives_different_values() -> None:
    a = _baseline(random_state=0)
    b = _baseline(random_state=100)
    assert not np.array_equal(a.per_realisation_max, b.per_realisation_max)


# ---------------------------------------------------------------- caching


def test_baseline_cache_round_trip(tmp_path: Path) -> None:
    first = _baseline(cache_dir=tmp_path)
    assert first.cache_path is not None
    assert first.cache_path.exists()

    second = _baseline(cache_dir=tmp_path)
    np.testing.assert_allclose(first.per_realisation_max, second.per_realisation_max, atol=0.0)
    np.testing.assert_allclose(
        first.null_cluster_persistence,
        second.null_cluster_persistence,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        first.null_cluster_size,
        second.null_cluster_size,
    )
    np.testing.assert_array_equal(
        first.null_cluster_realisation,
        second.null_cluster_realisation,
    )
    assert second.cache_path == first.cache_path


def test_baseline_force_recompute_overwrites_cache(tmp_path: Path) -> None:
    original = _baseline(cache_dir=tmp_path)
    assert original.cache_path is not None
    poisoned = original.cache_path.read_bytes()

    corrupted = tmp_path / original.cache_path.name
    corrupted.write_bytes(b"not-an-npz")

    recomputed = _baseline(cache_dir=tmp_path, force_recompute=True)
    assert recomputed.cache_path == original.cache_path
    assert recomputed.cache_path.read_bytes() != b"not-an-npz"
    assert recomputed.cache_path.read_bytes() == poisoned


def test_baseline_cache_disabled_when_random_state_is_none(tmp_path: Path) -> None:
    result = _baseline(random_state=None, cache_dir=tmp_path)
    assert result.cache_path is None
    assert not any(tmp_path.iterdir())


def test_baseline_cache_key_stable() -> None:
    config = {
        "n_samples": 10,
        "n_features": 2,
        "umap_kwargs": _canonical_umap_kwargs({}),
        "n_realisations": 3,
        "per_realisation_trials": 3,
        "percentile": 99.7,
        "mcs_range": [5, 15],
        "ms_range": [1, 5],
        "random_state": 0,
    }
    assert _cache_key(config) == _cache_key(dict(config))
    config2 = dict(config, random_state=1)
    assert _cache_key(config) != _cache_key(config2)


def test_default_cache_dir_is_writable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    path = default_cache_dir()
    assert "starfold" in str(path).lower()


# ---------------------------------------------------------------- validation


def test_rejects_unknown_umap_kwargs() -> None:
    with pytest.raises(ValueError, match="unsupported keys"):
        _baseline(umap_kwargs={"banana": 1})


def test_rejects_bad_percentile() -> None:
    with pytest.raises(ValueError, match="percentile"):
        _baseline(percentile=0.0)
    with pytest.raises(ValueError, match="percentile"):
        _baseline(percentile=101.0)


def test_rejects_bad_counts() -> None:
    with pytest.raises(ValueError, match="n_samples"):
        _baseline(n_samples=1)
    with pytest.raises(ValueError, match="n_features"):
        _baseline(n_features=0)
    with pytest.raises(ValueError, match="n_realisations"):
        _baseline(n_realisations=0)
    with pytest.raises(ValueError, match="per_realisation_trials"):
        _baseline(per_realisation_trials=0)
