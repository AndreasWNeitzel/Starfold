"""Tests for the :mod:`starfold._engine` backend resolver.

The resolver drives CPU/GPU selection for UMAP, HDBSCAN, and the noise
baseline. Its contract:

* ``"cpu"`` always resolves to ``"cpu"``.
* ``"cuml"`` resolves to ``"cuml"`` when importable and raises otherwise.
* ``"auto"`` prefers ``"cuml"`` when importable and falls back silently.
* An unknown selector raises ``ValueError``.
"""

from __future__ import annotations

import pytest

from starfold._engine import cuml_is_importable, resolve_engine


def test_cpu_always_resolves_to_cpu() -> None:
    assert resolve_engine("cpu") == "cpu"


def test_auto_matches_cuml_availability() -> None:
    expected = "cuml" if cuml_is_importable() else "cpu"
    assert resolve_engine("auto") == expected


def test_cuml_strict_matches_availability() -> None:
    if cuml_is_importable():
        assert resolve_engine("cuml") == "cuml"
    else:
        with pytest.raises(ImportError, match="cuml"):
            resolve_engine("cuml")


def test_invalid_engine_raises() -> None:
    with pytest.raises(ValueError, match="engine"):
        resolve_engine("gpu")  # type: ignore[arg-type]


def test_cuml_is_importable_returns_bool() -> None:
    assert isinstance(cuml_is_importable(), bool)
