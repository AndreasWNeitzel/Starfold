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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import hdbscan as _hdbscan
import numpy as np
import optuna

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "Engine",
    "HDBSCANResult",
    "OptunaSearchResult",
    "run_hdbscan",
    "search_hdbscan",
]

Engine = Literal["auto", "cpu", "cuml"]


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
        and importance inspection.
    hdbscan_result
        :class:`HDBSCANResult` produced by refitting HDBSCAN at
        ``best_params``.
    """

    best_params: dict[str, int]
    best_persistence_sum: float
    study: optuna.Study
    hdbscan_result: HDBSCANResult


def _cuml_is_importable() -> bool:
    try:
        import cuml.cluster  # noqa: PLC0415
    except ImportError:
        return False
    return cuml.cluster is not None


def _resolve_engine(engine: Engine) -> Literal["cpu", "cuml"]:
    if engine == "cpu":
        return "cpu"
    if engine == "cuml":
        if not _cuml_is_importable():
            msg = "engine='cuml' requires the optional RAPIDS cuml dependency."
            raise ImportError(msg)
        return "cuml"
    if engine == "auto":
        return "cuml" if _cuml_is_importable() else "cpu"
    msg = f"engine must be 'auto', 'cpu', or 'cuml' (got {engine!r})."
    raise ValueError(msg)


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
    model = _hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=None if min_samples is None else int(min_samples),
        metric=metric,
        gen_min_span_tree=True,
    )
    model.fit(x)
    return _pack(model.labels_, model.cluster_persistence_, model.probabilities_)


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
    resolved: Literal["cpu", "cuml"],
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
) -> OptunaSearchResult:
    """Tune HDBSCAN by maximising the sum of cluster-persistence scores.

    The objective follows Neitzel et al. (2025) §3.3: for each
    ``(min_cluster_size, min_samples)`` pair, fit HDBSCAN and return the
    sum of :attr:`hdbscan.HDBSCAN.cluster_persistence_`. The search uses
    a :class:`optuna.samplers.TPESampler` seeded with ``random_state``,
    so identical seeds produce identical trial sequences on the CPU
    backend.

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
        Backend for each trial's HDBSCAN fit.
    show_progress_bar : bool, default False
        Forwarded to :meth:`optuna.Study.optimize`.

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
    if n_trials < 1:
        msg = f"n_trials must be >= 1 (got {n_trials})."
        raise ValueError(msg)
    if mcs_range[0] < 2 or mcs_range[1] < mcs_range[0]:
        msg = f"mcs_range must satisfy 2 <= low <= high (got {mcs_range})."
        raise ValueError(msg)
    if ms_range[0] < 1 or ms_range[1] < ms_range[0]:
        msg = f"ms_range must satisfy 1 <= low <= high (got {ms_range})."
        raise ValueError(msg)

    x = _as_2d_float(X)
    resolved = _resolve_engine(engine)
    mcs_low, mcs_high = _effective_mcs_bounds(x.shape[0], mcs_range)
    ms_low, ms_high = ms_range

    def objective(trial: optuna.Trial) -> float:
        mcs = trial.suggest_int("min_cluster_size", mcs_low, mcs_high, log=True)
        ms = trial.suggest_int("min_samples", ms_low, ms_high, log=True)
        result = _fit(resolved, x, min_cluster_size=mcs, min_samples=ms, metric=metric)
        return float(np.sum(result.cluster_persistence))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)

    best_params = {k: int(v) for k, v in study.best_params.items()}
    best_result = _fit(
        resolved,
        x,
        min_cluster_size=best_params["min_cluster_size"],
        min_samples=best_params["min_samples"],
        metric=metric,
    )
    return OptunaSearchResult(
        best_params=best_params,
        best_persistence_sum=float(study.best_value),
        study=study,
        hdbscan_result=best_result,
    )
