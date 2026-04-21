"""Backend-engine resolution shared by :mod:`embedding`, :mod:`clustering`, and friends.

A single :data:`Engine` alias and :func:`resolve_engine` helper keep the
CPU/GPU selection logic consistent across UMAP, HDBSCAN, and the noise
baseline. ``"auto"`` prefers RAPIDS :mod:`cuml` when importable, falling
back silently to the CPU path. ``"cuml"`` is strict and raises when
:mod:`cuml` is missing; ``"cpu"`` is the deterministic, always-available
reference path.
"""

from __future__ import annotations

from typing import Literal

__all__ = ["Engine", "ResolvedEngine", "cuml_is_importable", "resolve_engine"]

Engine = Literal["auto", "cpu", "cuml"]
ResolvedEngine = Literal["cpu", "cuml"]


def cuml_is_importable() -> bool:
    """Return ``True`` iff RAPIDS :mod:`cuml` can be imported in this process.

    The check is cheap and intentionally tolerant: any ``ImportError`` from
    the ``cuml`` package (including transitive failures from a missing
    CUDA driver or mismatched cuDF/cuML versions) is treated as "not
    available" so ``engine="auto"`` degrades gracefully.
    """
    try:
        import cuml  # noqa: PLC0415
    except ImportError:
        return False
    return cuml is not None


def resolve_engine(engine: Engine) -> ResolvedEngine:
    """Resolve an :data:`Engine` selector to a concrete backend name.

    Parameters
    ----------
    engine : {"auto", "cpu", "cuml"}
        User-facing selector. ``"auto"`` returns ``"cuml"`` when it is
        importable and ``"cpu"`` otherwise. ``"cuml"`` is strict.

    Returns
    -------
    {"cpu", "cuml"}
        Concrete backend name.

    Raises
    ------
    ImportError
        If ``engine="cuml"`` and :mod:`cuml` cannot be imported.
    ValueError
        If ``engine`` is not one of the three documented values.
    """
    if engine == "cpu":
        return "cpu"
    if engine == "cuml":
        if not cuml_is_importable():
            msg = "engine='cuml' requires the optional RAPIDS cuml dependency."
            raise ImportError(msg)
        return "cuml"
    if engine == "auto":
        return "cuml" if cuml_is_importable() else "cpu"
    msg = f"engine must be 'auto', 'cpu', or 'cuml' (got {engine!r})."
    raise ValueError(msg)
