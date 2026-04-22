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

    best_params: dict[str, Any]
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
    cluster_selection_method: str = "eom",
    cluster_selection_epsilon: float = 0.0,
    alpha: float = 1.0,
) -> HDBSCANResult:
    model, result = _fit_cpu_with_model(
        x,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        alpha=alpha,
    )
    del model
    return result


def _fit_cpu_with_model(
    x: NDArray[np.floating[Any]],
    *,
    min_cluster_size: int,
    min_samples: int | None,
    metric: str,
    cluster_selection_method: str = "eom",
    cluster_selection_epsilon: float = 0.0,
    alpha: float = 1.0,
    prediction_data: bool = False,
) -> tuple[_hdbscan.HDBSCAN, HDBSCANResult]:
    model = _hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=None if min_samples is None else int(min_samples),
        metric=metric,
        cluster_selection_method=str(cluster_selection_method),
        cluster_selection_epsilon=float(cluster_selection_epsilon),
        alpha=float(alpha),
        gen_min_span_tree=True,
        prediction_data=bool(prediction_data),
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
    cluster_selection_method: str = "eom",
    cluster_selection_epsilon: float = 0.0,
    alpha: float = 1.0,
) -> HDBSCANResult:
    from cuml.cluster import HDBSCAN as CumlHDBSCAN  # noqa: N811, PLC0415

    model = CumlHDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=None if min_samples is None else int(min_samples),
        metric=metric,
        cluster_selection_method=str(cluster_selection_method),
        cluster_selection_epsilon=float(cluster_selection_epsilon),
        alpha=float(alpha),
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
    cluster_selection_method: str = "eom",
    cluster_selection_epsilon: float = 0.0,
    alpha: float = 1.0,
) -> HDBSCANResult:
    if resolved == "cpu":
        return _fit_cpu(
            x,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_method=cluster_selection_method,
            cluster_selection_epsilon=cluster_selection_epsilon,
            alpha=alpha,
        )
    return _fit_cuml(
        x,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        alpha=alpha,
    )


def run_hdbscan(
    X: ArrayLike,
    *,
    min_cluster_size: int,
    min_samples: int | None = None,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
    cluster_selection_epsilon: float = 0.0,
    alpha: float = 1.0,
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
    cluster_selection_method : {"eom", "leaf"}, default ``"eom"``
        How HDBSCAN picks the flat clustering off the condensed tree.
        ``"eom"`` (Excess of Mass) prefers fewer, larger, more stable
        clusters; ``"leaf"`` picks the leaves of the tree directly and
        so returns finer-grained clusters. Which is "right" depends on
        the density profile of the data and is not the library
        author's call; :func:`search_hdbscan` searches over both by
        default.
    cluster_selection_epsilon : float, default ``0.0``
        Distance threshold below which sibling leaves are merged into
        their parent before flat selection. ``0.0`` disables epsilon
        merging (HDBSCAN's native behaviour).
    alpha : float, default ``1.0``
        Scales MST edge distances before condensing. Values away from
        1 shift the density-contrast ratio at which branches survive;
        0.7--1.5 is a conservative useful range.
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
    if cluster_selection_method not in ("eom", "leaf"):
        msg = (
            "cluster_selection_method must be 'eom' or 'leaf' "
            f"(got {cluster_selection_method!r})."
        )
        raise ValueError(msg)
    if cluster_selection_epsilon < 0.0:
        msg = (
            "cluster_selection_epsilon must be >= 0.0 "
            f"(got {cluster_selection_epsilon})."
        )
        raise ValueError(msg)
    if alpha <= 0.0:
        msg = f"alpha must be > 0.0 (got {alpha})."
        raise ValueError(msg)
    resolved = _resolve_engine(engine)
    x = _as_2d_float(X)
    return _fit(
        resolved,
        x,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        alpha=alpha,
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
    cluster_selection_methods: tuple[str, ...],
    cluster_selection_epsilon_range: tuple[float, float],
    alpha_range: tuple[float, float],
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
    if not cluster_selection_methods:
        msg = "cluster_selection_methods must contain at least one method."
        raise ValueError(msg)
    invalid = [m for m in cluster_selection_methods if m not in ("eom", "leaf")]
    if invalid:
        msg = (
            "cluster_selection_methods entries must be 'eom' or 'leaf' "
            f"(got {cluster_selection_methods!r})."
        )
        raise ValueError(msg)
    eps_lo, eps_hi = cluster_selection_epsilon_range
    if eps_lo < 0.0 or eps_hi < eps_lo:
        msg = (
            "cluster_selection_epsilon_range must satisfy 0 <= low <= high "
            f"(got {cluster_selection_epsilon_range})."
        )
        raise ValueError(msg)
    alpha_lo, alpha_hi = alpha_range
    if alpha_lo <= 0.0 or alpha_hi < alpha_lo:
        msg = f"alpha_range must satisfy 0 < low <= high (got {alpha_range})."
        raise ValueError(msg)


def search_hdbscan(
    X: ArrayLike,
    *,
    n_trials: int = 100,
    mcs_range: tuple[int, int] = (5, 500),
    ms_range: tuple[int, int] = (1, 50),
    cluster_selection_methods: tuple[str, ...] = ("eom", "leaf"),
    cluster_selection_epsilon_range: tuple[float, float] = (0.0, 0.5),
    alpha_range: tuple[float, float] = (0.7, 1.5),
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
    cluster_selection_methods : tuple of str, default ``("eom", "leaf")``
        Which HDBSCAN flat-selection methods the Optuna trial may
        sample. ``("eom", "leaf")`` searches both; a single-element
        tuple (e.g. ``("eom",)``) pins the method and spends budget on
        the other axes. The two methods correspond to qualitatively
        different topologies (``"eom"`` = Excess of Mass, fewer larger
        clusters; ``"leaf"`` = tree leaves, finer clusters); not
        searching both leaves half the plausible cluster topology
        space invisible.
    cluster_selection_epsilon_range : tuple of float, default ``(0.0, 0.5)``
        Uniform float range for HDBSCAN's
        ``cluster_selection_epsilon``, the distance threshold below
        which sibling leaves merge before flat selection. ``0.0``
        disables epsilon merging (the library default). The 0--0.5
        range is a 2-D-embedding-appropriate default; enlarge when
        clustering raw high-dimensional features with larger typical
        distances. Set ``(0.0, 0.0)`` to pin epsilon off.
    alpha_range : tuple of float, default ``(0.7, 1.5)``
        Uniform float range for HDBSCAN's ``alpha`` (MST edge
        rescaling). Values around 1.0 are the library default;
        narrower ranges pin the knob (e.g. ``(1.0, 1.0)``).
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
    _validate_search_inputs(
        n_trials,
        mcs_range,
        ms_range,
        objective,
        cluster_selection_methods,
        cluster_selection_epsilon_range,
        alpha_range,
    )

    x = _as_2d_float(X)
    # Trials and the final refit always use CPU HDBSCAN; see the
    # engine docstring note above. Resolve anyway so a missing cuml
    # with engine="cuml" still raises through resolve_engine.
    _ = _resolve_engine(engine)
    mcs_low, mcs_high = _effective_mcs_bounds(x.shape[0], mcs_range)
    ms_low, ms_high = ms_range
    eps_lo, eps_hi = cluster_selection_epsilon_range
    alpha_lo, alpha_hi = alpha_range
    methods = tuple(cluster_selection_methods)

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
        if len(methods) > 1:
            csm = trial.suggest_categorical("cluster_selection_method", list(methods))
        else:
            csm = methods[0]
        cse = (
            trial.suggest_float("cluster_selection_epsilon", eps_lo, eps_hi)
            if eps_lo < eps_hi
            else eps_lo
        )
        alpha = (
            trial.suggest_float("alpha", alpha_lo, alpha_hi, log=True)
            if alpha_lo < alpha_hi
            else alpha_lo
        )
        model, result = _fit_cpu_with_model(
            x,
            min_cluster_size=mcs,
            min_samples=ms,
            metric=metric,
            cluster_selection_method=csm,
            cluster_selection_epsilon=cse,
            alpha=alpha,
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
    # Preserve the scalar objective used by this search so plotting and
    # downstream reports can label axes correctly without being told.
    study.set_user_attr("objective", objective)
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=show_progress_bar)

    # Pull the best trial's full parameter set, filling in any axis
    # that was pinned (single-element methods, zero-width epsilon /
    # alpha range) so the refit and the downstream config record match
    # the Optuna-selected trial exactly.
    raw_best = dict(study.best_params)
    best_csm = str(raw_best.get("cluster_selection_method", methods[0]))
    best_cse = float(raw_best.get("cluster_selection_epsilon", eps_lo))
    best_alpha = float(raw_best.get("alpha", alpha_lo))
    best_params: dict[str, Any] = {
        "min_cluster_size": int(raw_best["min_cluster_size"]),
        "min_samples": int(raw_best["min_samples"]),
        "cluster_selection_method": best_csm,
        "cluster_selection_epsilon": best_cse,
        "alpha": best_alpha,
    }
    # ``prediction_data=True`` on the final best-params refit so
    # downstream uncertainty propagation can call
    # ``hdbscan.approximate_predict`` without another fit. The Optuna
    # trials above leave prediction data off because (a) it adds a small
    # per-fit allocation and (b) trials are throwaway.
    best_model, best_result = _fit_cpu_with_model(
        x,
        min_cluster_size=best_params["min_cluster_size"],
        min_samples=best_params["min_samples"],
        metric=metric,
        cluster_selection_method=best_csm,
        cluster_selection_epsilon=best_cse,
        alpha=best_alpha,
        prediction_data=True,
    )
    best_persistence_sum = float(np.sum(best_result.cluster_persistence))
    return OptunaSearchResult(
        best_params=best_params,
        best_persistence_sum=best_persistence_sum,
        study=study,
        hdbscan_result=best_result,
        model=best_model,
    )
