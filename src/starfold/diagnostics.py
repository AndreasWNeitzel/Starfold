"""Public fit-diagnostic helpers shared by the pipeline.

This module collects the small, opinionated checks that protect users
from silent failures:

* :func:`validate_input_matrix` refuses ``X`` with NaN/inf or an
  ``n_samples`` so small that UMAP's neighbourhood geometry is not
  well-defined.
* :func:`diagnose_fit` walks a completed :class:`PipelineResult` and
  returns a list of human-readable warnings (``n_clusters=0``, high
  outlier fraction, low trustworthiness, flat Optuna search, noise-
  consistent fit, unavailable hierarchy) so :meth:`PipelineResult.summary`
  can surface them.
* :func:`auto_mcs_upper` picks a data-size-aware upper bound for the
  Optuna ``min_cluster_size`` search so the default is not silently
  wrong on large datasets.
* :func:`recommend_budget` prints a one-paragraph suggestion for
  ``n_trials`` / ``n_realisations`` / ``per_realisation_trials`` given
  ``n_samples``, so a new user does not have to guess.

These checks are deliberately conservative -- they warn but they do
not refuse. The only hard failures are invalid ``X`` (non-finite
entries, shape mismatch, or an ``n_samples`` that is below the
``n_neighbors`` UMAP needs to build a graph at all).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from starfold.pipeline import PipelineResult


__all__ = [
    "auto_mcs_upper",
    "diagnose_fit",
    "recommend_budget",
    "validate_input_matrix",
]


TRUSTWORTHINESS_ALERT_THRESHOLD: float = 0.90
OUTLIER_FRACTION_ALERT: float = 0.90
OPTUNA_FLAT_OBJECTIVE_ALERT: float = 1e-9
OPTUNA_PLATEAU_TAIL_FRACTION: float = 0.20


def validate_input_matrix(
    X: ArrayLike,
    *,
    n_neighbors: int,
    name: str = "X",
) -> NDArray[np.floating[Any]]:
    """Return ``X`` as a 2-D float64 array after hard sanity checks.

    Hard failures (ValueError):
      * ``X`` is not 2-D.
      * ``X`` contains NaN or inf.
      * ``n_samples <= n_neighbors`` (UMAP cannot build a k-NN graph).

    Soft warnings (issued via :mod:`warnings`):
      * ``n_samples < 3 * n_neighbors`` -- the neighbourhood statistics
        UMAP relies on are unstable in this regime; an experienced user
        may still want to run, but a new one deserves to know.

    Parameters
    ----------
    X
        The caller-provided feature matrix.
    n_neighbors
        UMAP's ``n_neighbors`` argument (from ``umap_kwargs``).
    name
        Name used in error messages (useful when the caller wraps the
        function on behalf of another API surface, e.g. an augmented
        matrix in the uncertainty-aware fit).

    Returns
    -------
    NDArray
        ``X`` coerced to ``float64``, 2-D, finite.
    """
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        msg = f"{name} must be a 2-D array (got shape {arr.shape})."
        raise ValueError(msg)
    if not np.all(np.isfinite(arr)):
        n_bad = int((~np.isfinite(arr)).sum())
        msg = (
            f"{name} contains {n_bad} non-finite entries (NaN or inf). "
            "UMAP and HDBSCAN require a fully finite feature matrix; "
            "impute or drop affected rows before calling fit."
        )
        raise ValueError(msg)
    n_samples = arr.shape[0]
    if n_samples <= n_neighbors:
        msg = (
            f"{name} has n_samples={n_samples} but UMAP needs "
            f"n_neighbors={n_neighbors} neighbours per point. Reduce "
            "n_neighbors (via umap_kwargs) or provide more samples."
        )
        raise ValueError(msg)
    if n_samples < 3 * n_neighbors:
        warnings.warn(
            f"{name} has n_samples={n_samples}, which is less than "
            f"3 x n_neighbors ({3 * n_neighbors}). UMAP's neighbourhood "
            "statistics are unstable in this regime; results should be "
            "interpreted with care.",
            UserWarning,
            stacklevel=3,
        )
    return arr


def auto_mcs_upper(n_samples: int) -> int:
    """Pick a data-size-aware upper bound for ``min_cluster_size``.

    The paper's ``(5, 500)`` default was tuned for tens-of-thousands
    of stars. It is too tight for million-sample embeddings (where
    natural clusters may contain thousands of points) and too loose for
    tiny samples. This helper returns:

    * ``max(5, n_samples // 10)`` on small samples (so Optuna has
      something to search over),
    * ``clip(n_samples // 20, 50, 5000)`` on large samples (so the
      range reaches a reasonable natural-cluster scale without
      exploding the trial-wise cost of HDBSCAN).

    The bound applies only when ``mcs_range`` is left at the default
    ``None`` sentinel; an explicit tuple is always honoured (but still
    capped so it does not exceed ``n_samples // 2``).
    """
    if n_samples < 50:
        return max(5, n_samples // 4)
    return int(np.clip(n_samples // 20, 50, 5000))


def recommend_budget(n_samples: int) -> dict[str, int]:
    """Return a starfold-recommended trial / realisation budget.

    These are *starting points*, not prescriptions: users who care
    about very tight credibility p-values should raise
    ``n_realisations``; users who know their cluster-size scale from
    prior science can reduce ``hdbscan_optuna_trials``.

    The recommendations follow three regimes:

    +----------------+----------------+---------------------+--------------------+
    | n_samples      | n_trials       | n_realisations      | per_realisation    |
    +================+================+=====================+====================+
    | <= 1k          | 50             | 500                 | 20                 |
    | 1k - 50k       | 100            | 1000                | 20                 |
    | 50k - 500k     | 120            | 500                 | 15                 |
    | > 500k         | 150            | 200                 | 10                 |
    +----------------+----------------+---------------------+--------------------+

    The total noise-baseline cost scales like
    ``n_realisations * per_realisation_trials`` UMAP+HDBSCAN fits, so
    a 1M-sample run with the large-scale recommendation is ~2000 fits
    vs 20 000 for the default. For production use, pair this with
    ``engine="cuml"`` or parallel ``n_jobs``.
    """
    if n_samples <= 1_000:
        return {"hdbscan_optuna_trials": 50, "n_realisations": 500, "per_realisation_trials": 20}
    if n_samples <= 50_000:
        return {"hdbscan_optuna_trials": 100, "n_realisations": 1000, "per_realisation_trials": 20}
    if n_samples <= 500_000:
        return {"hdbscan_optuna_trials": 120, "n_realisations": 500, "per_realisation_trials": 15}
    return {"hdbscan_optuna_trials": 150, "n_realisations": 200, "per_realisation_trials": 10}


def _optuna_plateau_warning(result: PipelineResult) -> str | None:
    """Return a plateau/flat-search warning for the Optuna study, or None."""
    study = result.search.study
    values = [t.value for t in study.trials if t.value is not None]
    if not values:
        return None
    best = float(max(values) if study.direction.name == "MAXIMIZE" else min(values))
    if abs(best) < OPTUNA_FLAT_OBJECTIVE_ALERT:
        return (
            f"Optuna best objective is ~0 ({best:.2e}); the search did not "
            "find any partition with non-trivial persistence. Consider "
            "widening mcs_range, raising n_trials, or reviewing the data."
        )
    n = len(values)
    tail = max(1, int(np.ceil(n * OPTUNA_PLATEAU_TAIL_FRACTION)))
    running = (
        np.maximum.accumulate(values)
        if study.direction.name == "MAXIMIZE"
        else np.minimum.accumulate(values)
    )
    # If the best was already reached before the tail started, Optuna
    # plateaued: the user is paying for trials that never improved.
    if running[-1] == running[-tail - 1 if tail < n else 0]:
        return (
            f"Optuna best was reached {n - tail} trials before the end of "
            f"the {n}-trial budget; consider halving n_trials on similar "
            "runs to save compute."
        )
    return None


def diagnose_fit(result: PipelineResult) -> list[str]:
    """Collect human-readable flags for a :class:`PipelineResult`.

    Returns a list of strings; each entry is one issue worth surfacing
    in :meth:`PipelineResult.summary`. Empty list means no flags.

    The checks are:

    * **degenerate fit**: ``n_clusters == 0`` (everything is outliers).
    * **high outlier fraction**: ``>90%`` of points in cluster ``-1``.
    * **low trustworthiness**: ``T(k) < 0.90`` -- the embedding does
      not preserve local neighbourhoods, so the clustering rests on
      shaky geometry.
    * **noise-consistent clustering**: at least one cluster exists but
      no cluster beats the 99.7-percentile noise baseline (which means
      the fit *might* be noise-level structure).
    * **unavailable hierarchy**: the fit was done on a backend that
      does not expose the condensed tree (cuml), so
      :attr:`PipelineResult.hierarchy.available` is ``False``. Users
      who care about sub-clustering or sibling-lambda queries must
      refit on CPU.
    * **Optuna plateau / flat**: the search either never escaped the
      zero-objective region or converged long before the trial budget.
    """
    flags: list[str] = []
    n = int(result.labels.shape[0])
    n_outliers = int(np.sum(result.labels < 0))
    outlier_frac = n_outliers / max(n, 1)

    if result.n_clusters == 0:
        flags.append(
            "DEGENERATE FIT: no clusters found (every point is an "
            "outlier). Possible causes: min_cluster_size too large for "
            "the data, UMAP collapsed the manifold (check trustworthiness), "
            "or the data genuinely has no density-based structure."
        )
    elif outlier_frac > OUTLIER_FRACTION_ALERT:
        flags.append(
            f"HIGH OUTLIER FRACTION: {outlier_frac:.1%} of points are in "
            "cluster -1. HDBSCAN is treating most of the data as noise; "
            "consider reducing min_cluster_size / min_samples or widening "
            "mcs_range."
        )

    if result.trustworthiness < TRUSTWORTHINESS_ALERT_THRESHOLD:
        flags.append(
            f"LOW TRUSTWORTHINESS: T(k)={result.trustworthiness:.3f} < "
            f"{TRUSTWORTHINESS_ALERT_THRESHOLD:.2f}. The 2-D embedding does "
            "not preserve local neighbourhoods well, so downstream cluster "
            "labels should be interpreted with care. Consider raising "
            "n_neighbors or n_epochs, or providing features with less "
            "redundancy."
        )

    if (
        result.significant is not None
        and result.n_clusters > 0
        and int(result.significant.sum()) == 0
    ):
        threshold = (
            result.noise_baseline.threshold
            if result.noise_baseline is not None
            else float("nan")
        )
        flags.append(
            "NOISE-CONSISTENT: no cluster beats the noise baseline "
            f"threshold ({threshold:.3f}). The fit may be indistinguishable "
            "from structureless noise at the requested percentile; treat "
            "cluster assignments as tentative."
        )

    if not result.hierarchy.available:
        flags.append(
            "HIERARCHY UNAVAILABLE: the backend used for this fit does "
            "not expose the HDBSCAN condensed tree (typically cuml). "
            "Calls to hierarchy.merge_lambda / sibling / subcluster_on "
            "will raise; refit with engine='cpu' for full tree access."
        )

    plateau = _optuna_plateau_warning(result)
    if plateau is not None:
        flags.append(plateau)

    return flags


def warn_fit_flags(flags: list[str]) -> None:
    """Emit each flag through :mod:`warnings` so logs capture them."""
    for line in flags:
        warnings.warn(line, UserWarning, stacklevel=3)
