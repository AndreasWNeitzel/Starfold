"""Statistical noise baseline for cluster-persistence significance.

Implements the 99.7th-percentile noise baseline described in
Neitzel et al. (2025) §3.3. For a given sample shape ``(n_samples,
n_features)`` and UMAP configuration, the baseline is computed by

1. drawing ``n_realisations`` matrices of independent standard-normal
   noise,
2. running :func:`starfold.embedding.run_umap` on each,
3. tuning HDBSCAN with :func:`starfold.clustering.search_hdbscan` on
   that embedding for ``per_realisation_trials`` trials,
4. recording (a) the *maximum* cluster-persistence score, (b) the
   *number* of clusters in the best trial, and (c) the *best-trial
   Optuna objective value* on that realisation, and
5. taking the ``percentile``-th percentile of the per-realisation
   maxima to set the per-cluster significance threshold.

A real-data cluster's persistence exceeds this threshold with
probability ``1 - percentile/100`` under the null of "structureless
Gaussian noise embedded with the same UMAP settings", which the paper
uses as a crude but paper-silent-otherwise significance gate.

The full per-realisation arrays (max persistence, n_clusters, and
best-trial objective) are also retained so downstream code in
:mod:`starfold.credibility` can compute empirical *global* p-values for
a run, which answer the question "HDBSCAN always finds at least a
couple of clusters on noise, so is my real-data run actually
distinguishable from noise?".

The procedure is deliberately expensive. Results are cached on disk
under :func:`platformdirs.user_cache_dir` keyed by a hash of the
inputs; repeated calls with identical arguments return immediately.
Pass ``force_recompute=True`` to invalidate the cache for one call.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from joblib import Parallel, delayed
from platformdirs import user_cache_dir

from starfold.clustering import Engine, TrialObjective, search_hdbscan
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
    per_realisation_n_clusters
        Per-realisation number of HDBSCAN clusters in the best
        Optuna trial, length ``n_realisations``. Used to judge
        whether a real-data run yields an unusually large or small
        cluster count relative to noise.
    per_realisation_objective
        Per-realisation best-trial Optuna objective value (on the
        same ``objective`` scale used by the real-data pipeline),
        length ``n_realisations``. Used by
        :mod:`starfold.credibility` as the primary null distribution
        for the global clustering-credibility p-value.
    null_cluster_persistence
        Flat concatenation of *every* cluster's persistence value
        across every noise realisation, shape
        ``(total_null_clusters,)``. This is the null distribution for
        the per-cluster credibility p-value: "how often does a single
        noise cluster exceed persistence P?". Storing the flat array
        rather than a ragged list keeps Monte Carlo cheap (a single
        sort + searchsorted per observed cluster).
    null_cluster_size
        Matching cluster-size array (number of members per noise
        cluster), same shape as ``null_cluster_persistence``. Kept for
        diagnostic plots; the per-cluster p-value conditions only on
        persistence by default.
    null_cluster_realisation
        Realisation index for each null cluster, same shape as
        ``null_cluster_persistence``. Recovers the ragged per-realisation
        layout when needed.
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
    per_realisation_n_clusters: NDArray[np.integer[Any]]
    per_realisation_objective: NDArray[np.floating[Any]]
    null_cluster_persistence: NDArray[np.floating[Any]]
    null_cluster_size: NDArray[np.integer[Any]]
    null_cluster_realisation: NDArray[np.integer[Any]]
    percentile: float
    config: dict[str, Any] = field(default_factory=dict)
    cache_path: Path | None = None

    @property
    def objective(self) -> str:
        """Name of the Optuna objective recorded per realisation."""
        return str(self.config.get("objective", "persistence_sum"))


def default_cache_dir() -> Path:
    """Return the user-level cache directory used for baselines."""
    return Path(user_cache_dir("starfold"))


def _canonical_umap_kwargs(umap_kwargs: dict[str, Any]) -> dict[str, Any]:
    # ``engine`` is intentionally not part of the cache key: CPU and GPU
    # UMAP produce statistically equivalent noise baselines, so sharing
    # the cache across backends is the right default.
    allowed = {"n_neighbors", "min_dist", "n_epochs", "metric", "n_components"}
    ignored = {"engine"}
    extra = set(umap_kwargs) - allowed - ignored
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
            # Legacy caches (pre-credibility / pre-percluster) are
            # invalidated by demanding every key be present.
            per_n_clusters = np.asarray(
                payload["per_realisation_n_clusters"], dtype=np.intp
            )
            per_objective = np.asarray(
                payload["per_realisation_objective"], dtype=np.float64
            )
            null_pers = np.asarray(
                payload["null_cluster_persistence"], dtype=np.float64
            )
            null_size = np.asarray(payload["null_cluster_size"], dtype=np.intp)
            null_real = np.asarray(payload["null_cluster_realisation"], dtype=np.intp)
            threshold = float(payload["threshold"])
            percentile = float(payload["percentile"])
            config = json.loads(str(payload["config"]))
    except (OSError, KeyError, ValueError):
        return None
    return NoiseBaselineResult(
        threshold=threshold,
        per_realisation_max=per_realisation,
        per_realisation_n_clusters=per_n_clusters,
        per_realisation_objective=per_objective,
        null_cluster_persistence=null_pers,
        null_cluster_size=null_size,
        null_cluster_realisation=null_real,
        percentile=percentile,
        config=config,
        cache_path=path,
    )


def _save_cached(path: Path, result: NoiseBaselineResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        per_realisation_max=result.per_realisation_max,
        per_realisation_n_clusters=result.per_realisation_n_clusters,
        per_realisation_objective=result.per_realisation_objective,
        null_cluster_persistence=result.null_cluster_persistence,
        null_cluster_size=result.null_cluster_size,
        null_cluster_realisation=result.null_cluster_realisation,
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


def _effective_n_jobs(*, n_jobs: int, engine: Engine, n_realisations: int) -> int:
    """Clamp ``n_jobs`` to a sensible value for the current context.

    Parallel GPU work across joblib worker processes risks blowing up
    VRAM (every worker loads a separate cuML context); sequential runs
    are fine under CPU too for tiny sweeps where process-spawn overhead
    dwarfs the work itself. Returns 1 in those cases; otherwise clamps
    ``n_jobs`` to ``min(n_realisations, os.cpu_count())``.
    """
    if engine == "cuml":
        return 1
    if n_realisations < 2:
        return 1
    cpu = os.cpu_count() or 1
    if n_jobs == -1:
        return min(n_realisations, cpu)
    return max(1, min(n_jobs, n_realisations, cpu))


def _one_realisation(
    *,
    n_samples: int,
    n_features: int,
    umap_kwargs: dict[str, Any],
    per_realisation_trials: int,
    mcs_range: tuple[int, int],
    ms_range: tuple[int, int],
    cluster_selection_methods: tuple[str, ...],
    cluster_selection_epsilon_range: tuple[float, float],
    alpha_range: tuple[float, float],
    engine: Engine,
    objective: TrialObjective,
    seed: int,
) -> tuple[float, int, float, NDArray[np.floating[Any]], NDArray[np.integer[Any]]]:
    """Run one noise realisation.

    Returns ``(max_persistence, n_clusters, best_objective,
    per_cluster_persistence, per_cluster_size)`` where the last two
    arrays are length-``n_clusters`` views of the best trial's
    cluster geometry. Thread the real-data pipeline's ``objective``
    through so the null distribution is comparable to the real-data
    optimisation target.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(size=(n_samples, n_features))
    umap_call_kwargs = dict(umap_kwargs)
    umap_call_kwargs.setdefault("engine", engine)
    emb = run_umap(noise, random_state=seed, **umap_call_kwargs)
    search = search_hdbscan(
        emb,
        n_trials=per_realisation_trials,
        mcs_range=mcs_range,
        ms_range=ms_range,
        cluster_selection_methods=cluster_selection_methods,
        cluster_selection_epsilon_range=cluster_selection_epsilon_range,
        alpha_range=alpha_range,
        random_state=seed,
        engine=engine,
        objective=objective,
    )
    persistence = np.asarray(search.hdbscan_result.cluster_persistence, dtype=np.float64)
    labels = np.asarray(search.hdbscan_result.labels, dtype=np.intp)
    max_persistence = float(persistence.max()) if persistence.size else 0.0
    n_clusters = int(search.hdbscan_result.n_clusters)
    # Cluster sizes via bincount on non-negative labels.
    if n_clusters > 0:
        positive = labels[labels >= 0]
        sizes = np.bincount(positive, minlength=n_clusters).astype(np.intp)
    else:
        sizes = np.zeros(0, dtype=np.intp)
    try:
        best_objective = float(search.study.best_value)
    except ValueError:
        best_objective = 0.0
    del noise, emb, search
    gc.collect()
    return max_persistence, n_clusters, best_objective, persistence, sizes


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
    cluster_selection_methods: tuple[str, ...] = ("eom", "leaf"),
    cluster_selection_epsilon_range: tuple[float, float] = (0.0, 0.5),
    alpha_range: tuple[float, float] = (0.7, 1.5),
    engine: Engine = "auto",
    objective: TrialObjective = "persistence_sum",
    random_state: int | None = None,
    cache_dir: Path | str | bool | None = None,
    force_recompute: bool = False,
    n_jobs: int = 1,
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
        sufficient because only the per-realisation optimum is needed,
        not a global one. See ``docs/design_decisions.md``.
    percentile : float, default 99.7
        Percentile across per-realisation maxima. 99.7 is the 3-sigma
        gate used in the paper.
    mcs_range, ms_range : tuple of int
        Search ranges forwarded to :func:`search_hdbscan`.
    cluster_selection_methods : tuple of str, default ``("eom", "leaf")``
        HDBSCAN selection-method axis forwarded to
        :func:`search_hdbscan`. Pass a single-element tuple to pin the
        method. Must match the real-data pipeline so the null and the
        observation share a search space.
    cluster_selection_epsilon_range : tuple of float, default ``(0.0, 0.5)``
        Range for the ``cluster_selection_epsilon`` Optuna axis.
    alpha_range : tuple of float, default ``(0.7, 1.5)``
        Range for the HDBSCAN ``alpha`` Optuna axis (log-uniform).
    engine : {"auto", "cpu", "cuml"}, default ``"auto"``
        Backend for HDBSCAN.
    objective : {"persistence_sum", "combined_geom"}, default ``"persistence_sum"``
        Optuna scalar objective. Must match the objective used by
        the real-data pipeline whose output is being tested, so the
        null ``per_realisation_objective`` distribution is comparable.
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
    n_jobs : int, default 1
        Number of parallel worker processes for the per-realisation
        loop. ``-1`` uses every CPU core. Each realisation is an
        independent UMAP + Optuna + HDBSCAN fit so the loop is
        embarrassingly parallel; the only shared state is the cache
        file, which is written once after the loop completes.
        Parallelism is skipped (``n_jobs=1``) for tiny runs or when a
        GPU engine is selected, because GPU backends serialise poorly
        across joblib workers.

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
    cluster_selection_methods, cluster_selection_epsilon_range,
    alpha_range, objective, random_state)``. Changing any of these
    invalidates the cache.
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
        "cluster_selection_methods": sorted(cluster_selection_methods),
        "cluster_selection_epsilon_range": list(cluster_selection_epsilon_range),
        "alpha_range": list(alpha_range),
        "objective": str(objective),
        "random_state": None if random_state is None else int(random_state),
    }

    cache_path = _resolve_cache_path(cache_dir=cache_dir, config=config, random_state=random_state)
    if cache_path is not None and not force_recompute:
        cached = _load_cached(cache_path)
        if cached is not None:
            return cached

    base_seed = 0 if random_state is None else int(random_state)
    per_realisation_max = np.empty(n_realisations, dtype=np.float64)
    per_realisation_n_clusters = np.empty(n_realisations, dtype=np.intp)
    per_realisation_objective = np.empty(n_realisations, dtype=np.float64)
    effective_jobs = _effective_n_jobs(n_jobs=n_jobs, engine=engine, n_realisations=n_realisations)
    if effective_jobs == 1:
        outputs = [
            _one_realisation(
                n_samples=n_samples,
                n_features=n_features,
                umap_kwargs=canonical_umap,
                per_realisation_trials=per_realisation_trials,
                mcs_range=mcs_range,
                ms_range=ms_range,
                cluster_selection_methods=cluster_selection_methods,
                cluster_selection_epsilon_range=cluster_selection_epsilon_range,
                alpha_range=alpha_range,
                engine=engine,
                objective=objective,
                seed=base_seed + k,
            )
            for k in range(n_realisations)
        ]
    else:
        outputs = list(
            Parallel(n_jobs=effective_jobs, backend="loky", prefer="processes")(
                delayed(_one_realisation)(
                    n_samples=n_samples,
                    n_features=n_features,
                    umap_kwargs=canonical_umap,
                    per_realisation_trials=per_realisation_trials,
                    mcs_range=mcs_range,
                    ms_range=ms_range,
                    cluster_selection_methods=cluster_selection_methods,
                    cluster_selection_epsilon_range=cluster_selection_epsilon_range,
                    alpha_range=alpha_range,
                    engine=engine,
                    objective=objective,
                    seed=base_seed + k,
                )
                for k in range(n_realisations)
            )
        )
    null_cluster_persistence_parts: list[NDArray[np.floating[Any]]] = []
    null_cluster_size_parts: list[NDArray[np.integer[Any]]] = []
    null_cluster_realisation_parts: list[NDArray[np.integer[Any]]] = []
    for k, (max_persistence, n_clusters_k, best_objective, per_pers, per_sizes) in enumerate(
        outputs
    ):
        per_realisation_max[k] = max_persistence
        per_realisation_n_clusters[k] = n_clusters_k
        per_realisation_objective[k] = best_objective
        if per_pers.size:
            null_cluster_persistence_parts.append(per_pers)
            null_cluster_size_parts.append(per_sizes)
            null_cluster_realisation_parts.append(
                np.full(per_pers.size, k, dtype=np.intp)
            )

    null_cluster_persistence = (
        np.concatenate(null_cluster_persistence_parts).astype(np.float64)
        if null_cluster_persistence_parts
        else np.zeros(0, dtype=np.float64)
    )
    null_cluster_size = (
        np.concatenate(null_cluster_size_parts).astype(np.intp)
        if null_cluster_size_parts
        else np.zeros(0, dtype=np.intp)
    )
    null_cluster_realisation = (
        np.concatenate(null_cluster_realisation_parts).astype(np.intp)
        if null_cluster_realisation_parts
        else np.zeros(0, dtype=np.intp)
    )

    threshold = float(np.percentile(per_realisation_max, percentile))
    result = NoiseBaselineResult(
        threshold=threshold,
        per_realisation_max=per_realisation_max,
        per_realisation_n_clusters=per_realisation_n_clusters,
        per_realisation_objective=per_realisation_objective,
        null_cluster_persistence=null_cluster_persistence,
        null_cluster_size=null_cluster_size,
        null_cluster_realisation=null_cluster_realisation,
        percentile=float(percentile),
        config=config,
        cache_path=cache_path,
    )
    if cache_path is not None:
        _save_cached(cache_path, result)
    return result
