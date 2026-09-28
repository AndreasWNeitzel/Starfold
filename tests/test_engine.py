from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from starfold import _engine
from starfold._engine import (
    cuml_status,
    derive_seed,
    library_versions,
    parallel_map,
    release_backend_caches,
    require_resolved,
    resolve_engine,
)

EXPECTED_VERSION_KEYS = {
    "numpy",
    "scipy",
    "scikit-learn",
    "umap-learn",
    "hdbscan",
    "optuna",
    "joblib",
    "cuml",
}


def _square_with_env(value: int) -> tuple[int, str | None]:
    return value * value, os.environ.get("NUMBA_NUM_THREADS")


def test_resolve_cpu_is_identity():
    assert resolve_engine("cpu") == "cpu"


def test_resolve_rejects_unknown_selector():
    with pytest.raises(ValueError, match="engine must be"):
        resolve_engine("gpu")  # type: ignore[arg-type]


def test_require_resolved_accepts_only_concrete_backends():
    assert require_resolved("cpu") == "cpu"
    assert require_resolved("cuml") == "cuml"
    for engine in ("auto", "gpu", "CPU", ""):
        with pytest.raises(ValueError, match="resolve 'auto' first"):
            require_resolved(engine)


def test_resolution_without_cuml(monkeypatch):
    monkeypatch.setattr(_engine, "_cuml_probe", lambda: (False, "ImportError: no cuml"))
    assert cuml_status() == (False, "ImportError: no cuml")
    with pytest.warns(RuntimeWarning, match="running on the CPU because cuML is not usable"):
        assert resolve_engine("auto") == "cpu"
    with pytest.raises(ImportError, match="no cuml"):
        resolve_engine("cuml")


def test_resolution_with_cuml(monkeypatch):
    monkeypatch.setattr(_engine, "_cuml_probe", lambda: (True, None))
    assert cuml_status() == (True, None)
    assert resolve_engine("auto") == "cuml"
    assert resolve_engine("cuml") == "cuml"


def test_release_backend_caches_runs_without_rapids(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setitem(sys.modules, "rmm", None)
    release_backend_caches()


def test_library_versions_keys_and_values():
    versions = library_versions()
    assert set(versions) == EXPECTED_VERSION_KEYS
    assert versions["numpy"] == np.__version__
    assert all(isinstance(value, str) and value for value in versions.values())


def test_derive_seed_matches_seed_sequence():
    expected = int(np.random.SeedSequence([7, 1, 2]).generate_state(1)[0])
    assert derive_seed(7, 1, 2) == expected


def test_derive_seed_range_and_sensitivity():
    seeds = {derive_seed(0, stage, index) for stage in range(4) for index in range(8)}
    assert len(seeds) == 32
    assert all(0 <= seed < 2**32 for seed in seeds)
    assert derive_seed(1, 0, 1) != derive_seed(1, 1, 0)
    assert derive_seed(1) != derive_seed(2)


def test_parallel_map_serial_preserves_order():
    results = parallel_map(_square_with_env, [3, 1, 2], n_jobs=1)
    assert [value for value, _env in results] == [9, 1, 4]


def test_parallel_map_empty():
    assert parallel_map(_square_with_env, [], n_jobs=1) == []
    assert parallel_map(_square_with_env, [], n_jobs=2) == []


def test_parallel_map_workers_are_single_threaded(monkeypatch):
    monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
    results = parallel_map(_square_with_env, [4, 5, 6], n_jobs=2)
    assert [value for value, _env in results] == [16, 25, 36]
    assert all(env == "1" for _value, env in results)
    assert "NUMBA_NUM_THREADS" not in os.environ
