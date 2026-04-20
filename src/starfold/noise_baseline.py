"""Statistical noise baseline for cluster-persistence significance.

Implements the 99.7th-percentile noise baseline described in
Neitzel et al. (2025) §3.3. For a given sample shape ``(n_samples,
n_features)`` and UMAP configuration, the baseline is computed by

1. drawing ``n_realisations`` matrices of independent standard-normal
   noise,
2. running :func:`starfold.embedding.run_umap` on each,
3. tuning HDBSCAN with :func:`starfold.clustering.search_hdbscan` on
   that embedding for ``per_realisation_trials`` trials,
4. recording the *maximum* cluster-persistence score produced by the
   best HDBSCAN on that realisation, and
5. taking the ``percentile``-th percentile across realisations.

A real-data cluster's persistence exceeds this threshold with
probability ``1 - percentile/100`` under the null of "structureless
Gaussian noise embedded with the same UMAP settings", which the paper
uses as a crude but paper-silent-otherwise significance gate.

The procedure is deliberately expensive. Results are cached on disk
under :func:`platformdirs.user_cache_dir` keyed by a hash of the
inputs; repeated calls with identical arguments return immediately.
Pass ``force_recompute=True`` to invalidate the cache for one call.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from platformdirs import user_cache_dir

from starfold.clustering import Engine, search_hdbscan
from starfold.embedding import run_umap

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "NoiseBaselineResult",
    "compute_noise_baseline",
    "default_cache_dir",
]


@dataclass(frozen=True)
class NoiseBaselineResult:
    """Outcome of a noise-baseline computation.

    Parameters
    ----------
    threshold
        The persistence value at the requested percentile across
        noise realisations. Clusters on real data with
        ``cluster_persistence > threshold`` are flagged significant.
    per_realisation_max
        Per-realisation maximum cluster persistence, length
        ``n_realisations``.
    percentile
        The percentile used to collapse ``per_realisation_max`` into
        ``threshold`` (e.g. 99.7 for the 3-sigma gate).
    config
        Frozen record of the inputs that define the cache key. Useful
        when the threshold is attached to a :class:`PipelineResult`
        and needs to be round-tripped.
    cache_path
        On-disk location of the cached result, or ``None`` when
        caching was disabled.
    """

    threshold: float
    per_realisation_max: NDArray[np.floating[Any]]
    percentile: float
    config: dict[str, Any] = field(default_factory=dict)
    cache_path: Path | None = None


def default_cache_dir() -> Path:
    """Return the user-level cache directory used for baselines."""
    return Path(user_cache_dir("starfold"))


def _canonical_umap_kwargs(umap_kwargs: dict[str, Any]) -> dict[str, Any]:
    allowed = {"n_neighbors", "min_dist", "n_epochs", "metric", "n_components"}
    extra = set(umap_kwargs) - allowed
    if extra:
        msg = f"umap_kwargs contains unsupported keys: {sorted(extra)}."
        raise ValueError(msg)
    return {
        "n_neighbors": int(umap_kwargs.get("n_neighbors", 15)),
        "min_dist": float(umap_kwargs.get("min_dist", 0.0)),
        "n_epochs": int(umap_kwargs.get("n_epochs", 10_000)),
        "metric": str(umap_kwargs.get("metric", "euclidean")),
        "n_components": int(umap_kwargs.get("n_components", 2)),
    }


def _cache_key(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _load_cached(path: Path) -> NoiseBaselineResult | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            per_realisation = np.asarray(payload["per_realisation_max"], dtype=np.float64)
            threshold = float(payload["threshold"])
            percentile = float(payload["percentile"])
            config = json.loads(str(payload["config"]))
    except (OSError, KeyError, ValueError):
        return None
    return NoiseBaselineResult(
        threshold=threshold,
        per_realisation_max=per_realisation,
        percentile=percentile,
        config=config,
        cache_path=path,
    )


def _save_cached(path: Path, result: NoiseBaselineResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        per_realisation_max=result.per_realisation_max,
        threshold=np.float64(result.threshold),
        percentile=np.float64(result.percentile),
        config=np.str_(json.dumps(result.config, sort_keys=True)),
    )


def _validate_inputs(
    *,
    n_samples: int,
    n_features: int,
    n_realisations: int,
    per_realisation_trials: int,
    percentile: float,
) -> None:
    if n_samples < 2:
        msg = f"n_samples must be >= 2 (got {n_samples})."
        raise ValueError(msg)
    if n_features < 1:
        msg = f"n_features must be >= 1 (got {n_features})."
        raise ValueError(msg)
    if n_realisations < 1:
        msg = f"n_realisations must be >= 1 (got {n_realisations})."
        raise ValueError(msg)
    if per_realisation_trials < 1:
        msg = f"per_realisation_trials must be >= 1 (got {per_realisation_trials})."
        raise ValueError(msg)
    if not 0.0 < percentile <= 100.0:
        msg = f"percentile must be in (0, 100] (got {percentile})."
        raise ValueError(msg)


def _resolve_cache_path(
    *,
    cache_dir: Path | str | bool | None,
    config: dict[str, Any],
    random_state: int | None,
) -> Path | None:
    if cache_dir is False or random_state is None:
        return None
    base = default_cache_dir() if cache_dir is None or cache_dir is True else Path(cache_dir)
    return base / f"noise_baseline_{_cache_key(config)}.npz"


def _one_realisation(
    *,
    n_samples: int,
    n_features: int,
    umap_kwargs: dict[str, Any],
    per_realisation_trials: int,
    mcs_range: tuple[int, int],
    ms_range: tuple[int, int],
    engine: Engine,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(size=(n_samples, n_features))
    emb = run_umap(noise, random_state=seed, **umap_kwargs)
    search = search_hdbscan(
        emb,
        n_trials=per_realisation_trials,
        mcs_range=mcs_range,
        ms_range=ms_range,
        random_state=seed,
        engine=engine,
    )
    persistence = search.hdbscan_result.cluster_persistence
    return float(persistence.max()) if persistence.size else 0.0


def compute_noise_baseline(
    n_samples: int,
    n_features: int,
    umap_kwargs: dict[str, Any] | None = None,
    *,
    n_realisations: int = 1000,
    per_realisation_trials: int = 20,
    percentile: float = 99.7,
    mcs_range: tuple[int, int] = (5, 500),
    ms_range: tuple[int, int] = (1, 50),
    engine: Engine = "auto",
    random_state: int | None = None,
    cache_dir: Path | str | bool | None = None,
    force_recompute: bool = False,
) -> NoiseBaselineResult:
    """Compute a significance threshold from structureless Gaussian noise.

    For each of ``n_realisations`` independent ``N(0, 1)`` matrices of
    shape ``(n_samples, n_features)``, this function runs UMAP with
    ``umap_kwargs``, tunes HDBSCAN for ``per_realisation_trials`` Optuna
    trials, and records the maximum cluster persistence. The
    ``percentile``-th percentile of those maxima is returned as the
    significance threshold.

    Parameters
    ----------
    n_samples, n_features : int
        Shape of each synthetic noise matrix. Match these to the real
        data whose clusters you plan to evaluate.
    umap_kwargs : dict, optional
        UMAP configuration. Passed through to :func:`run_umap`. Only
        the keys ``n_neighbors``, ``min_dist``, ``n_epochs``,
        ``metric``, ``n_components`` are accepted; unknown keys raise
        :class:`ValueError`. Defaults to the paper's settings.
    n_realisations : int, default 1000
        Number of noise realisations. 1000 matches Neitzel et al. (2025).
    per_realisation_trials : int, default 20
        Optuna trials per realisation. The paper is silent here; 20 is
        sufficient because we only need the per-realisation optimum,
        not a global one. See ``docs/design_decisions.md``.
    percentile : float, default 99.7
        Percentile across per-realisation maxima. 99.7 is the 3-sigma
        gate used in the paper.
    mcs_range, ms_range : tuple of int
        Search ranges forwarded to :func:`search_hdbscan`.
    engine : {"auto", "cpu", "cuml"}, default ``"auto"``
        Backend for HDBSCAN.
    random_state : int or None, default None
        Seed. Realisation ``k`` uses seed ``random_state + k`` when
        ``random_state`` is an integer; with ``None``, results are not
        reproducible and the cache is bypassed.
    cache_dir : path-like or False, optional
        Directory for the ``.npz`` cache file. Defaults to
        :func:`default_cache_dir`. Pass ``cache_dir=False`` to disable
        caching entirely.
    force_recompute : bool, default False
        Recompute even if a cached result exists. The fresh result
        overwrites the cache.

    Returns
    -------
    NoiseBaselineResult
        The threshold, the per-realisation maxima, the percentile, the
        frozen configuration used as a cache key, and the cache path.

    Notes
    -----
    The cache key is a SHA-256 of a canonical JSON of
    ``(n_samples, n_features, umap_kwargs, n_realisations,
    per_realisation_trials, percentile, mcs_range, ms_range,
    random_state)``. Changing any of these invalidates the cache.
    The ``engine`` is not part of the key: CPU and GPU backends must
    produce statistically equivalent thresholds.

    Examples
    --------
    >>> from starfold.noise_baseline import compute_noise_baseline
    >>> baseline = compute_noise_baseline(
    ...     n_samples=200, n_features=3,
    ...     umap_kwargs={"n_epochs": 100},
    ...     n_realisations=3, per_realisation_trials=3,
    ...     random_state=0, cache_dir=False,
    ... )
    >>> baseline.per_realisation_max.shape
    (3,)
    """
    _validate_inputs(
        n_samples=n_samples,
        n_features=n_features,
        n_realisations=n_realisations,
        per_realisation_trials=per_realisation_trials,
        percentile=percentile,
    )

    canonical_umap = _canonical_umap_kwargs(dict(umap_kwargs or {}))
    config = {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "umap_kwargs": canonical_umap,
        "n_realisations": int(n_realisations),
        "per_realisation_trials": int(per_realisation_trials),
        "percentile": float(percentile),
        "mcs_range": list(mcs_range),
        "ms_range": list(ms_range),
        "random_state": None if random_state is None else int(random_state),
    }

    cache_path = _resolve_cache_path(cache_dir=cache_dir, config=config, random_state=random_state)
    if cache_path is not None and not force_recompute:
        cached = _load_cached(cache_path)
        if cached is not None:
            return cached

    base_seed = 0 if random_state is None else int(random_state)
    per_realisation_max = np.empty(n_realisations, dtype=np.float64)
    for k in range(n_realisations):
        per_realisation_max[k] = _one_realisation(
            n_samples=n_samples,
            n_features=n_features,
            umap_kwargs=canonical_umap,
            per_realisation_trials=per_realisation_trials,
            mcs_range=mcs_range,
            ms_range=ms_range,
            engine=engine,
            seed=base_seed + k,
        )

    threshold = float(np.percentile(per_realisation_max, percentile))
    result = NoiseBaselineResult(
        threshold=threshold,
        per_realisation_max=per_realisation_max,
        percentile=float(percentile),
        config=config,
        cache_path=cache_path,
    )
    if cache_path is not None:
        _save_cached(cache_path, result)
    return result
