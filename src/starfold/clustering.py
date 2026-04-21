"""HDBSCAN clustering with Optuna hyperparameter search.

Two public entry points:

* :func:`run_hdbscan` fits a single HDBSCAN at a fixed
  ``min_cluster_size`` / ``min_samples`` pair and returns labels,
  per-cluster persistence, and sample-level membership probabilities.
* :func:`search_hdbscan` runs a TPE-sampled Optuna study that
  *maximises the sum of cluster persistences* (Neitzel et al. 2025,
  §3.3) and refits HDBSCAN at the best parameters.

By default the CPU :mod:`hdbscan` package is used. When the optional
RAPIDS :mod:`cuml` dependency is importable, ``engine="auto"``
delegates to :class:`cuml.cluster.HDBSCAN`, whose public attributes
(``labels_``, ``cluster_persistence_``, ``probabilities_``) match the
CPU implementation.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import hdbscan as _hdbscan
import numpy as np
import optuna

from starfold._engine import Engine, ResolvedEngine, resolve_engine

TrialObjective = Literal["persistence_sum", "combined_geom"]

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "Engine",
    "HDBSCANResult",
    "OptunaSearchResult",
    "TrialObjective",
    "run_hdbscan",
    "search_hdbscan",
]


@dataclass(frozen=True)
class HDBSCANResult:
    """Outcome of a single HDBSCAN fit.

    Parameters
    ----------
    labels
        Integer cluster assignment per sample. ``-1`` flags outliers.
    cluster_persistence
        Per-cluster persistence score, indexed by non-negative label.
    probabilities
        Strength of cluster membership per sample in ``[0, 1]``.
    n_clusters
        Number of distinct non-negative cluster labels.
    """

    labels: NDArray[np.intp]
    cluster_persistence: NDArray[np.floating[Any]]
    probabilities: NDArray[np.floating[Any]]
    n_clusters: int


@dataclass(frozen=True)
class OptunaSearchResult:
    """Outcome of an Optuna search over HDBSCAN's hyperparameters.

    Parameters
    ----------
    best_params
        The ``min_cluster_size`` / ``min_samples`` pair that maximised
        the sum of cluster-persistence scores.
    best_persistence_sum
        The corresponding objective value.
    study
        The underlying :class:`optuna.Study`, retained for trial-history
        and importance inspection. Every completed trial carries the
        following ``user_attrs``: ``relative_validity`` (DBCV proxy from
        the minimum spanning tree), ``n_clusters``, ``outlier_fraction``,
        and ``persistence_max``.
    hdbscan_result
        :class:`HDBSCANResult` produced by refitting HDBSCAN at
        ``best_params``.
    model
        The fitted ``hdbscan.HDBSCAN`` instance from the refit, or
        ``None`` on the cuml backend (cuml returns a different object
        that does not expose the condensed-tree API). Retained so that
        diagnostics such as the condensed-tree plot can be drawn.
    """

    best_params: dict[str, int]
    best_persistence_sum: float
    study: optuna.Study
    hdbscan_result: HDBSCANResult
    model: _hdbscan.HDBSCAN | None = None


_resolve_engine = resolve_engine  # backward-compatible alias for internal callers


def _as_2d_float(X: ArrayLike) -> NDArray[np.floating[Any]]:
    x = np.asarray(X, dtype=np.float64)
    if x.ndim != 2:
        msg = f"X must be a 2-D array (got shape {x.shape})."
        raise ValueError(msg)
    return x


def _pack(labels: ArrayLike, persistence: ArrayLike, probs: ArrayLike) -> HDBSCANResult:
    labels_arr = np.asarray(labels, dtype=np.intp)
    persistence_arr = np.asarray(persistence, dtype=np.float64)
    probs_arr = np.asarray(probs, dtype=np.float64)
    positive = labels_arr[labels_arr >= 0]
    n_clusters = int(positive.max() + 1) if positive.size else 0
    return HDBSCANResult(
        labels=labels_arr,
        cluster_persistence=persistence_arr,
        probabilities=probs_arr,
        n_clusters=n_clusters,
    )


def _fit_cpu(
    x: NDArray[np.floating[Any]],
    *,
    min_cluster_size: int,
    min_samples: int | None,
    metric: str,
) -> HDBSCANResult:
    model, result = _fit_cpu_with_model(
        x,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
    )
    del model
    return result


def _fit_cpu_with_model(
    x: NDArray[np.floating[Any]],
    *,
    min_cluster_size: int,
    min_samples: int | None,
    metric: str,
) -> tuple[_hdbscan.HDBSCAN, HDBSCANResult]:
    model = _hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=None if min_samples is None else int(min_samples),
        metric=metric,
        gen_min_span_tree=True,
    )
    model.fit(x)
    result = _pack(model.labels_, model.cluster_persistence_, model.probabilities_)
    return model, result


def _relative_validity(model: _hdbscan.HDBSCAN) -> float:
    """Return HDBSCAN's MST-based DBCV proxy, or NaN if unavailable."""
    try:
        return float(model.relative_validity_)
    except (AttributeError, ValueError, ZeroDivisionError):
        return float("nan")


def _fit_cuml(
    x: NDArray[np.floating[Any]],
    *,
    min_cluster_size: int,
    min_samples: int | None,
    metric: str,
) -> HDBSCANResult:
    from cuml.cluster import HDBSCAN as CumlHDBSCAN  # noqa: N811, PLC0415

    model = CumlHDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=None if min_samples is None else int(min_samples),
        metric=metric,
        gen_min_span_tree=True,
    )
    model.fit(x)
    return _pack(model.labels_, model.cluster_persistence_, model.probabilities_)


def _fit(
    resolved: ResolvedEngine,
    x: NDArray[np.floating[Any]],
    *,
    min_cluster_size: int,
    min_samples: int | None,
    metric: str,
) -> HDBSCANResult:
    if resolved == "cpu":
        return _fit_cpu(
            x, min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric
        )
    return _fit_cuml(x, min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric)


def run_hdbscan(
    X: ArrayLike,
    *,
    min_cluster_size: int,
    min_samples: int | None = None,
    metric: str = "euclidean",
    engine: Engine = "auto",
) -> HDBSCANResult:
    """Cluster ``X`` with HDBSCAN and return persistence scores.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input feature matrix. Not standardised internally.
    min_cluster_size : int
        Smallest group of samples HDBSCAN will accept as a cluster.
        Must be at least 2.
    min_samples : int or None, default None
        How conservative the clustering is. ``None`` defers to the
        library default (``min_cluster_size``).
    metric : str, default ``"euclidean"``
        Distance metric. For ``engine="cuml"`` the metric must be one
        that cuml supports.
    engine : {"auto", "cpu", "cuml"}, default ``"auto"``
        Backend. ``"auto"`` prefers ``"cuml"`` when importable and
        falls back to ``"cpu"`` otherwise.

    Returns
    -------
    HDBSCANResult
        Cluster labels, per-cluster persistence, probabilities, and the
        number of clusters.

    Raises
    ------
    ValueError
        If ``X`` is not 2-D or ``min_cluster_size < 2``.
    ImportError
        If ``engine="cuml"`` is requested but :mod:`cuml` is missing.

    Examples
    --------
    >>> import numpy as np
    >>> from starfold.clustering import run_hdbscan
    >>> rng = np.random.default_rng(0)
    >>> X = np.vstack([rng.normal(loc, size=(80, 2)) for loc in (-5, 0, 5)])
    >>> result = run_hdbscan(X, min_cluster_size=10, engine="cpu")
    >>> result.n_clusters
    3
    """
    if min_cluster_size < 2:
        msg = f"min_cluster_size must be >= 2 (got {min_cluster_size})."
        raise ValueError(msg)
    if min_samples is not None and min_samples < 1:
        msg = f"min_samples must be >= 1 when set (got {min_samples})."
        raise ValueError(msg)
    resolved = _resolve_engine(engine)
    x = _as_2d_float(X)
    return _fit(
        resolved,
        x,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
    )


def _effective_mcs_bounds(
    n_samples: int,
    mcs_range: tuple[int, int],
) -> tuple[int, int]:
    low, high = mcs_range
    auto_cap = max(5, n_samples // 10)
    capped = min(high, auto_cap)
    return low, max(low, capped)


def _validate_search_inputs(
    n_trials: int,
    mcs_range: tuple[int, int],
    ms_range: tuple[int, int],
    objective: TrialObjective,
) -> None:
    if n_trials < 1:
        msg = f"n_trials must be >= 1 (got {n_trials})."
        raise ValueError(msg)
    if mcs_range[0] < 2 or mcs_range[1] < mcs_range[0]:
        msg = f"mcs_range must satisfy 2 <= low <= high (got {mcs_range})."
        raise ValueError(msg)
    if ms_range[0] < 1 or ms_range[1] < ms_range[0]:
        msg = f"ms_range must satisfy 1 <= low <= high (got {ms_range})."
        raise ValueError(msg)
    if objective not in ("persistence_sum", "combined_geom"):
        msg = (
            "objective must be 'persistence_sum' or 'combined_geom' "
            f"(got {objective!r})."
        )
        raise ValueError(msg)


def search_hdbscan(
    X: ArrayLike,
    *,
    n_trials: int = 100,
    mcs_range: tuple[int, int] = (5, 500),
    ms_range: tuple[int, int] = (1, 50),
    metric: str = "euclidean",
    random_state: int | None = None,
    engine: Engine = "auto",
    show_progress_bar: bool = False,
    objective: TrialObjective = "persistence_sum",
) -> OptunaSearchResult:
    """Tune HDBSCAN by maximising a per-trial objective.

    The default objective follows Neitzel et al. (2025) §3.3: for each
    ``(min_cluster_size, min_samples)`` pair, fit HDBSCAN and return the
    sum of :attr:`hdbscan.HDBSCAN.cluster_persistence_`. A second
    choice, ``"combined_geom"``, returns the geometric mean of the
    MST-based DBCV proxy (``relative_validity_``, clipped at 0) and the
    median per-cluster persistence -- a balanced "both must be good"
    score for clusterings where either weak persistence or weak
    internal validity should disqualify a trial. The search uses a
    :class:`optuna.samplers.TPESampler` seeded with ``random_state``,
    so identical seeds produce identical trial sequences on the CPU
    backend. Every trial records ``user_attrs`` with full per-trial
    metrics (``persistence_sum``, ``persistence_median``,
    ``persistence_max``, ``persistence_mean``, ``relative_validity``,
    ``n_clusters``, ``outlier_fraction``) so post-hoc selection and
    diagnostics are cheap.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input feature matrix.
    n_trials : int, default 100
        Number of Optuna trials.
    mcs_range : tuple of int, default ``(5, 500)``
        Inclusive log-uniform integer range for ``min_cluster_size``.
        The upper bound is automatically capped to
        ``max(5, n_samples // 10)`` when the user's upper bound exceeds
        that value, to keep the search meaningful on small samples.
    ms_range : tuple of int, default ``(1, 50)``
        Inclusive log-uniform integer range for ``min_samples``.
    metric : str, default ``"euclidean"``
        Distance metric, threaded through to every trial.
    random_state : int or None, default None
        Seed for Optuna's TPE sampler.
    engine : {"auto", "cpu", "cuml"}, default ``"auto"``
        Nominal backend. The search itself always runs HDBSCAN on the
        CPU because (a) the ``relative_validity_`` attribute used for
        diagnostic plots and the ``combined_geom`` objective is only
        computed by :class:`hdbscan.HDBSCAN`, and (b) on a 2-D embedding
        the CPU implementation is already faster than the GPU one.
        The setting is retained so it can be forwarded to the caller
        (:class:`starfold.pipeline.UnsupervisedPipeline` uses it for
        UMAP and the noise baseline).
    show_progress_bar : bool, default False
        Forwarded to :meth:`optuna.Study.optimize`.
    objective : {"persistence_sum", "combined_geom"}, default ``"persistence_sum"``
        What the TPE sampler maximises. ``"persistence_sum"`` matches
        the paper; ``"combined_geom"`` returns
        ``sqrt(max(DBCV, 0) * median_cluster_persistence)``, rewarding
        clusterings that are good on both stability and internal
        validity simultaneously.

    Returns
    -------
    OptunaSearchResult
        Best parameters, best objective value, the :class:`optuna.Study`,
        and a refit :class:`HDBSCANResult`.

    Raises
    ------
    ValueError
        If ``n_trials < 1`` or the ranges are ill-formed.

    Examples
    --------
    >>> import numpy as np
    >>> from starfold.clustering import search_hdbscan
    >>> rng = np.random.default_rng(0)
    >>> X = np.vstack([rng.normal(loc, size=(80, 2)) for loc in (-5, 0, 5)])
    >>> search = search_hdbscan(X, n_trials=10, random_state=0, engine="cpu")
    >>> search.hdbscan_result.n_clusters
    3
    """
    _validate_search_inputs(n_trials, mcs_range, ms_range, objective)

    x = _as_2d_float(X)
    # Trials and the final refit always use CPU HDBSCAN; see the
    # engine docstring note above. Resolve anyway so a missing cuml
    # with engine="cuml" still raises through resolve_engine.
    _ = _resolve_engine(engine)
    mcs_low, mcs_high = _effective_mcs_bounds(x.shape[0], mcs_range)
    ms_low, ms_high = ms_range

    def _attrs_from(model: _hdbscan.HDBSCAN | None, result: HDBSCANResult) -> dict[str, float]:
        labels = result.labels
        n_samples = int(labels.shape[0])
        persistence = np.asarray(result.cluster_persistence, dtype=np.float64)
        return {
            "relative_validity": (
                _relative_validity(model) if model is not None else float("nan")
            ),
            "n_clusters": float(result.n_clusters),
            "outlier_fraction": float((labels == -1).sum()) / float(max(n_samples, 1)),
            "persistence_sum": float(persistence.sum()) if persistence.size else 0.0,
            "persistence_max": float(persistence.max()) if persistence.size else 0.0,
            "persistence_mean": float(persistence.mean()) if persistence.size else 0.0,
            "persistence_median": (
                float(np.median(persistence)) if persistence.size else 0.0
            ),
        }

    def _score(attrs: dict[str, float]) -> float:
        if objective == "persistence_sum":
            return float(attrs["persistence_sum"])
        if objective == "combined_geom":
            dbcv_plus = max(0.0, float(attrs["relative_validity"]))
            if not np.isfinite(dbcv_plus):
                return 0.0
            return float(np.sqrt(dbcv_plus * float(attrs["persistence_median"])))
        msg = f"unknown objective {objective!r}."
        raise ValueError(msg)

    def objective_fn(trial: optuna.Trial) -> float:
        mcs = trial.suggest_int("min_cluster_size", mcs_low, mcs_high, log=True)
        ms = trial.suggest_int("min_samples", ms_low, ms_high, log=True)
        model, result = _fit_cpu_with_model(
            x, min_cluster_size=mcs, min_samples=ms, metric=metric
        )
        attrs = _attrs_from(model, result)
        for key, value in attrs.items():
            trial.set_user_attr(key, value)
        score = _score(attrs)
        # Release the fitted model (and its condensed tree + MST) before
        # the next trial so peak memory stays bounded to one live fit.
        del model, result
        gc.collect()
        return score

    sampler = optuna.samplers.TPESampler(seed=random_state)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=show_progress_bar)

    best_params = {k: int(v) for k, v in study.best_params.items()}
    best_model, best_result = _fit_cpu_with_model(
        x,
        min_cluster_size=best_params["min_cluster_size"],
        min_samples=best_params["min_samples"],
        metric=metric,
    )
    best_persistence_sum = float(np.sum(best_result.cluster_persistence))
    return OptunaSearchResult(
        best_params=best_params,
        best_persistence_sum=best_persistence_sum,
        study=study,
        hdbscan_result=best_result,
        model=best_model,
    )
