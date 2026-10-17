"""Engine resolution, seed derivation, library versions, and bounded process parallelism.

``"auto"`` prefers RAPIDS cuML when it imports with a visible CUDA device and
falls back to the CPU reference path otherwise. ``"cuml"`` is strict and
``"cpu"`` always works. The probe runs once per process: a failing cuML import
is slow and its outcome cannot change within one run.
"""

from __future__ import annotations

import functools
import gc
import warnings
from contextlib import suppress
from importlib.metadata import PackageNotFoundError, distributions, version
from typing import TYPE_CHECKING, Literal, TypeVar, cast

import numpy as np
from joblib import Parallel, delayed, parallel_config

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "Engine",
    "ResolvedEngine",
    "cuml_status",
    "derive_seed",
    "library_versions",
    "parallel_map",
    "release_backend_caches",
    "require_resolved",
    "resolve_engine",
]

Engine = Literal["auto", "cpu", "cuml"]
ResolvedEngine = Literal["cpu", "cuml"]

T = TypeVar("T")
R = TypeVar("R")

_PINNED_DISTRIBUTIONS = (
    "numpy",
    "scipy",
    "scikit-learn",
    "umap-learn",
    "hdbscan",
    "optuna",
    "joblib",
    "pynndescent",
)


def release_backend_caches() -> None:
    """Collect fitted-estimator cycles and release the CuPy and RMM pools.

    The release is best effort because RAPIDS installations differ in their
    allocator. Without RAPIDS, only the collection runs.
    """
    gc.collect()
    with suppress(Exception):
        import cupy as cp  # noqa: PLC0415

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    with suppress(Exception):
        import rmm  # noqa: PLC0415

        release = getattr(rmm.mr.get_current_device_resource(), "release", None)
        if callable(release):
            release()


def _cuda_device_probe() -> tuple[bool, str | None]:
    """Check that at least one CUDA device is visible through the CuPy runtime."""
    try:
        from cupy.cuda import runtime  # noqa: PLC0415

        count = int(runtime.getDeviceCount())
    except Exception as error:  # noqa: BLE001 - CUDA runtimes raise backend-specific types.
        return False, f"{type(error).__name__}: {error}"
    if count < 1:
        return False, "no visible CUDA devices"
    return True, None


@functools.cache
def _cuml_probe() -> tuple[bool, str | None]:
    """Probe cuML importability and CUDA usability once per process."""
    try:
        import cuml  # noqa: PLC0415
    except (AttributeError, ImportError, OSError, RuntimeError) as error:
        return False, f"{type(error).__name__}: {error}"
    if cuml is None:
        return False, "cuml imported as None"
    device_ok, reason = _cuda_device_probe()
    if not device_ok:
        return False, f"CUDA unavailable: {reason}"
    return True, None


def cuml_status() -> tuple[bool, str | None]:
    """Return whether cuML imports with a usable CUDA device, and why not.

    The probe runs once per process and its outcome is cached.

    Returns
    -------
    available : bool
        ``True`` only when the import succeeds and a device is visible.
    reason : str or None
        Failure text of the cached probe, or ``None`` when available.
    """
    return _cuml_probe()


def require_resolved(engine: str) -> ResolvedEngine:
    """Return ``engine`` if it names a concrete backend, else raise.

    Stage functions take a resolved backend. Any other string, including
    ``"auto"``, is rejected so a typo can never select the CPU path silently.

    Raises
    ------
    ValueError
        If ``engine`` is not ``"cpu"`` or ``"cuml"``.
    """
    if engine in {"cpu", "cuml"}:
        return cast("ResolvedEngine", engine)
    msg = f"engine must be 'cpu' or 'cuml' here (got {engine!r}); resolve 'auto' first."
    raise ValueError(msg)


def resolve_engine(engine: Engine) -> ResolvedEngine:
    """Resolve an engine selector to a concrete backend name.

    Parameters
    ----------
    engine : {"auto", "cpu", "cuml"}
        ``"auto"`` returns ``"cuml"`` when importable, else ``"cpu"``.

    Returns
    -------
    {"cpu", "cuml"}
        Concrete backend name.

    Raises
    ------
    ImportError
        If ``engine="cuml"`` and cuML cannot be used.
    ValueError
        If ``engine`` is not one of the three documented values.
    """
    if engine == "cpu":
        return "cpu"
    if engine == "cuml":
        available, reason = _cuml_probe()
        if not available:
            msg = "engine='cuml' requires an operational RAPIDS/cuML CUDA backend."
            raise ImportError(f"{msg} {reason}" if reason else msg)
        return "cuml"
    if engine == "auto":
        available, reason = _cuml_probe()
        if available:
            return "cuml"
        warnings.warn(
            "engine='auto' is running on the CPU because cuML is not usable"
            + (f" ({reason})." if reason else ".")
            + " Pass engine='cuml' to fail instead of falling back.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu"
    msg = f"engine must be 'auto', 'cpu', or 'cuml' (got {engine!r})."
    raise ValueError(msg)


def library_versions() -> dict[str, str]:
    """Return the installed versions of every library that can change a result.

    Returns
    -------
    dict[str, str]
        Keys ``numpy``, ``scipy``, ``scikit-learn``, ``umap-learn``,
        ``hdbscan``, ``optuna``, ``joblib``, ``pynndescent`` and ``cuml``, each mapped to the
        distribution version or ``"absent"``. RAPIDS wheels carry a CUDA
        suffix (``cuml-cu12``, ``cuml-cu13``), so ``cuml`` is matched by prefix.
    """
    versions: dict[str, str] = {}
    for name in _PINNED_DISTRIBUTIONS:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "absent"
    versions["cuml"] = "absent"
    for distribution in distributions():
        name = str(distribution.metadata["Name"] or "").strip().lower()
        if name == "cuml" or name.startswith("cuml-cu"):
            versions["cuml"] = str(distribution.version)
            break
    return versions


def derive_seed(base: int, *ids: int) -> int:
    """Derive a 32-bit seed for one (stage, index, ...) position from a base seed.

    Seeds are spawned through ``numpy.random.SeedSequence``, giving every fit,
    world and null realisation a seed fixed by its identifiers and not by the
    worker that executes it.

    Parameters
    ----------
    base : int
        Non-negative base seed.
    *ids : int
        Stage and index identifiers.

    Returns
    -------
    int
        Seed in ``[0, 2**32)``.
    """
    return int(np.random.SeedSequence([base, *ids]).generate_state(1)[0])


def parallel_map(fn: Callable[[T], R], items: Sequence[T], *, n_jobs: int) -> list[R]:
    """Map ``fn`` over ``items`` serially or in single-threaded loky worker processes.

    Parameters
    ----------
    fn : callable
        Module-level picklable callable.
    items : sequence
        Work items; results keep their order.
    n_jobs : int
        ``1`` runs a plain loop in the calling process. Any other value is
        passed to ``joblib.Parallel`` with the loky backend and
        ``inner_max_num_threads=1``, which caps the BLAS, OpenMP and Numba
        threads of every worker. Throughput thus comes from processes, not
        threads. The calling process is left unchanged.

    Returns
    -------
    list
        One result per item, in order.
    """
    if n_jobs == 1:
        return [fn(item) for item in items]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=n_jobs)(delayed(fn)(item) for item in items)
    return list(results)
