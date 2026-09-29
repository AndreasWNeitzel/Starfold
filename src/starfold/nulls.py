"""Matched Monte Carlo null calibration on declared generator families.

Two families operate only on observed values, so no value in a null matrix
is invented:

- ``gaussian_copula``: per-column normal scores, Ledoit and Wolf 2004
  shrinkage correlation (doi:10.1016/S0047-259X(03)00096-4), one latent
  Gaussian draw, then each column's observed values rank-reordered to the
  latent ranks. Every marginal is preserved exactly and only one elliptical
  dependence survives. Null hypothesis: one elliptical population without
  multimodality.
- ``independent_marginals``: each column permuted independently, destroying
  all dependence while preserving every marginal (the permutation null of
  Lovdal et al. 2022, arXiv:2201.02404).

The statistic is supplied by the caller as a callable on a null matrix, so
this module never depends on the search. The p-value is the add-one Monte
Carlo estimate of Phipson and Smyth 2010 (doi:10.2202/1544-6115.1585) under
the sequential stopping rule of Besag and Clifford 1991
(doi:10.1093/biomet/78.2.301), with a Clopper and Pearson 1934 interval
(doi:10.1093/biomet/26.4.404) reported as nominal Monte Carlo precision.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import beta, norm, rankdata
from sklearn.covariance import LedoitWolf

from starfold._engine import derive_seed, parallel_map

NullProcedure = Literal["exploration", "full_search"]
NullStatistic = Callable[[NDArray[np.float64]], tuple[float, int]]

_FAMILIES: tuple[str, ...] = ("gaussian_copula", "independent_marginals")


@dataclass(frozen=True)
class NullEvidence:
    """Outcome of a matched null test over one or more generator families.

    Attributes
    ----------
    families : tuple of str
        Generator families in the order they were run.
    statistic_name : str
        Name of the statistic the caller supplied: ``exploration_design_max``
        or ``full_search_max``.
    observed : float
        Observed statistic on the data.
    values : tuple of tuple of float
        Null statistics per family, in generation order.
    n_realisations : tuple of int
        Realisations generated per family (the sequential rule may stop early).
    exceedances : tuple of int
        Count of null statistics at or above ``observed`` per family.
    p_values : tuple of float
        Add-one Monte Carlo p-value per family.
    p_intervals : tuple of tuple of float
        Nominal 95 percent Clopper-Pearson interval on the exceedance
        proportion per family, describing Monte Carlo precision only.
    p_value : float
        Composite p-value, the maximum over families (least favourable).
    alpha : float
        Declared level.
    structure_supported : bool
        ``p_value <= alpha``. Never a probability that groups are real.
    stopped_early : tuple of bool
        Whether the sequential rule stopped a family before ``n_null_max``.
    skipped_reason : str or None
        ``null_not_run`` when no family was declared, ``observed_statistic_zero``
        when the observed statistic is zero, else None.
    n_fits : int
        Fits spent on the realisations in ``values``.
    seconds : float
        Wall time of the stage.
    """

    families: tuple[str, ...]
    statistic_name: str
    observed: float
    values: tuple[tuple[float, ...], ...]
    n_realisations: tuple[int, ...]
    exceedances: tuple[int, ...]
    p_values: tuple[float, ...]
    p_intervals: tuple[tuple[float, float], ...]
    p_value: float
    alpha: float
    structure_supported: bool
    stopped_early: tuple[bool, ...]
    skipped_reason: str | None
    n_fits: int
    seconds: float


def _independent_marginals(x: NDArray[np.float64], rng: np.random.Generator) -> NDArray[np.float64]:
    """Permute every column independently."""
    result = np.empty_like(x)
    n = x.shape[0]
    for feature in range(x.shape[1]):
        result[:, feature] = x[rng.permutation(n), feature]
    return result


def _nearest_correlation(covariance: NDArray[np.float64]) -> NDArray[np.float64]:
    """Project a covariance onto the nearest positive definite correlation."""
    scale = np.sqrt(np.maximum(np.diag(covariance), np.finfo(np.float64).eps))
    correlation = covariance / np.outer(scale, scale)
    correlation = (correlation + correlation.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    positive = (eigenvectors * eigenvalues) @ eigenvectors.T
    positive_scale = np.sqrt(np.diag(positive))
    result = positive / np.outer(positive_scale, positive_scale)
    np.fill_diagonal(result, 1.0)
    return np.asarray(result, dtype=np.float64)


def _gaussian_copula(x: NDArray[np.float64], rng: np.random.Generator) -> NDArray[np.float64]:
    """Rank-reorder every column to one latent Gaussian draw."""
    n_samples, n_features = x.shape
    normal_scores = np.empty_like(x)
    for feature in range(n_features):
        ranks = rankdata(x[:, feature], method="average")
        normal_scores[:, feature] = norm.ppf((ranks - 0.5) / n_samples)
    covariance = np.asarray(LedoitWolf().fit(normal_scores).covariance_, dtype=np.float64)
    correlation = _nearest_correlation(covariance)
    latent = rng.multivariate_normal(
        mean=np.zeros(n_features),
        cov=correlation,
        size=n_samples,
        check_valid="raise",
    )
    result = np.empty_like(x)
    for feature in range(n_features):
        order = np.argsort(latent[:, feature], kind="mergesort")
        result[order, feature] = np.sort(x[:, feature])
    return result


def generate_null(x: NDArray[np.float64], family: str, seed: int) -> NDArray[np.float64]:
    """Draw one null matrix from a declared family.

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_features)
        Finite observed feature matrix.
    family : {"gaussian_copula", "independent_marginals"}
        Generator family.
    seed : int
        Seed of the generator draw.

    Returns
    -------
    ndarray of shape (n_samples, n_features)
        Null matrix whose every column holds exactly the observed values of
        the corresponding input column, reordered.

    Raises
    ------
    ValueError
        If ``family`` is unknown or ``x`` is not a finite 2-D matrix.
    """
    matrix = np.asarray(x, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        msg = "generate_null expects a finite 2-D matrix."
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    if family == "independent_marginals":
        return _independent_marginals(matrix, rng)
    if family == "gaussian_copula":
        return _gaussian_copula(matrix, rng)
    msg = f"Unknown null family {family!r}; expected one of {_FAMILIES}."
    raise ValueError(msg)


def monte_carlo_pvalue(exceedances: int, n_realisations: int) -> float:
    """Add-one Monte Carlo p-value ``(h + 1) / (L + 1)``.

    Parameters
    ----------
    exceedances : int
        Null statistics at or above the observed statistic.
    n_realisations : int
        Null realisations generated.

    Returns
    -------
    float
        The estimate of Phipson and Smyth 2010 (doi:10.2202/1544-6115.1585),
        or 1.0 when no realisation was generated.
    """
    if n_realisations <= 0:
        return 1.0
    return (exceedances + 1) / (n_realisations + 1)


def clopper_pearson(
    exceedances: int, n_realisations: int, level: float = 0.95
) -> tuple[float, float]:
    """Exact binomial interval on the exceedance proportion.

    Parameters
    ----------
    exceedances : int
        Successes.
    n_realisations : int
        Trials.
    level : float, default 0.95
        Two-sided coverage.

    Returns
    -------
    tuple of float
        Lower and upper bounds (Clopper and Pearson 1934,
        doi:10.1093/biomet/26.4.404), or ``(0.0, 1.0)`` when no trial was run.
    """
    if n_realisations <= 0:
        return (0.0, 1.0)
    tail = (1.0 - level) / 2.0
    lower = (
        0.0
        if exceedances == 0
        else float(beta.ppf(tail, exceedances, n_realisations - exceedances + 1))
    )
    upper = (
        1.0
        if exceedances == n_realisations
        else float(beta.ppf(1.0 - tail, exceedances + 1, n_realisations - exceedances))
    )
    return (lower, upper)


def _validate_settings(
    families: tuple[str, ...], n_null_max: int, n_exceedances: int, alpha: float, n_jobs: int
) -> None:
    """Reject settings under which the test could not reach ``alpha``."""
    for family in families:
        if family not in _FAMILIES:
            msg = f"Unknown null family {family!r}; expected one of {_FAMILIES}."
            raise ValueError(msg)
    if not 0.0 < alpha < 1.0:
        msg = "alpha must lie strictly between 0 and 1."
        raise ValueError(msg)
    minimum = int(np.ceil(1.0 / alpha)) - 1
    if n_null_max < minimum:
        msg = f"n_null_max={n_null_max} cannot reach alpha={alpha}; need at least {minimum}."
        raise ValueError(msg)
    if n_exceedances < 1:
        msg = "n_exceedances must be at least 1."
        raise ValueError(msg)
    if n_jobs < 1:
        msg = "n_jobs must be at least 1."
        raise ValueError(msg)


def _empty_evidence(
    families: tuple[str, ...],
    statistic_name: str,
    observed: float,
    alpha: float,
    reason: str,
) -> NullEvidence:
    """Evidence for a skipped stage."""
    return NullEvidence(
        families=families,
        statistic_name=statistic_name,
        observed=float(observed),
        values=tuple(() for _ in families),
        n_realisations=tuple(0 for _ in families),
        exceedances=tuple(0 for _ in families),
        p_values=tuple(1.0 for _ in families),
        p_intervals=tuple((0.0, 1.0) for _ in families),
        p_value=1.0,
        alpha=float(alpha),
        structure_supported=False,
        stopped_early=tuple(False for _ in families),
        skipped_reason=reason,
        n_fits=0,
        seconds=0.0,
    )


def _realisation(
    b: int,
    *,
    x: NDArray[np.float64],
    statistic: NullStatistic,
    family: str,
    family_index: int,
    random_state: int,
) -> tuple[float, int]:
    """Draw realisation ``b`` of one family and return its statistic and fits."""
    x_null = generate_null(x, family, derive_seed(random_state, 5, family_index, b))
    value, fits = statistic(x_null)
    return float(value), int(fits)


def _run_family(
    x: NDArray[np.float64],
    observed: float,
    statistic: NullStatistic,
    family: str,
    family_index: int,
    *,
    n_null_max: int,
    n_exceedances: int,
    random_state: int,
    n_jobs: int,
) -> tuple[tuple[float, ...], int, bool, int]:
    """Draw realisations of one family under the sequential rule.

    Returns the null values, the exceedance count, whether the rule stopped
    early, and the fits spent on the values kept.
    """
    draw = functools.partial(
        _realisation,
        x=x,
        statistic=statistic,
        family=family,
        family_index=family_index,
        random_state=random_state,
    )
    drawn: list[float] = []
    h = 0
    n_fits = 0
    for start in range(1, n_null_max + 1, n_jobs):
        batch = range(start, min(start + n_jobs, n_null_max + 1))
        outcomes = parallel_map(draw, batch, n_jobs=n_jobs)
        for b, (value, fits) in zip(batch, outcomes, strict=True):
            n_fits += fits
            drawn.append(value)
            if value >= observed:
                h += 1
            if h >= n_exceedances:
                return tuple(drawn), h, b < n_null_max, n_fits
    return tuple(drawn), h, False, n_fits


def null_test(
    x: NDArray[np.float64],
    observed: float,
    statistic: NullStatistic,
    *,
    families: Sequence[str],
    statistic_name: str,
    n_null_max: int,
    n_exceedances: int,
    alpha: float,
    random_state: int,
    n_jobs: int = 1,
) -> NullEvidence:
    """Run the sequential matched null test.

    Parameters
    ----------
    x : ndarray of shape (n_samples, n_features)
        Finite observed matrix (the rows the search used).
    observed : float
        Observed statistic on ``x``.
    statistic : callable
        ``statistic(x_null) -> (value, n_fits)``. To make observed and null
        values exchangeable, it must apply the identical procedure and seeds
        that produced ``observed``.
    families : sequence of str
        Generator families to run. An empty sequence skips the stage.
    statistic_name : str
        Recorded name of the statistic.
    n_null_max : int
        Maximum realisations per family.
    n_exceedances : int
        Sequential stopping count (Besag and Clifford 1991,
        doi:10.1093/biomet/78.2.301).
    alpha : float
        Declared level.
    random_state : int
        Base seed; realisation ``b`` of family ``f`` uses
        ``derive_seed(random_state, 5, f, b)``.
    n_jobs : int, default 1
        Realisations drawn at a time. Above 1, every realisation (a short
        final batch included) runs in one of ``n_jobs`` worker processes, so
        ``statistic`` must then pickle. The evidence, wall time aside, does
        not depend on ``n_jobs`` because the rule reads the realisations in
        order and discards those drawn past the stop.

    Returns
    -------
    NullEvidence
        Per-family values, p-values, intervals, the composite and the flag.

    Raises
    ------
    ValueError
        If a family is unknown, ``n_null_max < ceil(1 / alpha) - 1``,
        ``n_exceedances < 1`` or ``n_jobs < 1``.
    """
    start = time.perf_counter()
    family_tuple = tuple(str(f) for f in families)
    _validate_settings(family_tuple, n_null_max, n_exceedances, alpha, n_jobs)
    if not family_tuple:
        return _empty_evidence(family_tuple, statistic_name, observed, alpha, "null_not_run")
    if not observed > 0.0:
        return _empty_evidence(
            family_tuple, statistic_name, observed, alpha, "observed_statistic_zero"
        )
    values: list[tuple[float, ...]] = []
    exceed: list[int] = []
    early: list[bool] = []
    n_fits = 0
    for family_index, family in enumerate(family_tuple):
        drawn, h, stopped, fits = _run_family(
            x,
            observed,
            statistic,
            family,
            family_index,
            n_null_max=n_null_max,
            n_exceedances=n_exceedances,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        values.append(drawn)
        exceed.append(h)
        early.append(stopped)
        n_fits += fits
    counts = [len(v) for v in values]
    pvals = [monte_carlo_pvalue(h, c) for h, c in zip(exceed, counts, strict=True)]
    composite = max(pvals)
    return NullEvidence(
        families=family_tuple,
        statistic_name=statistic_name,
        observed=float(observed),
        values=tuple(values),
        n_realisations=tuple(counts),
        exceedances=tuple(exceed),
        p_values=tuple(pvals),
        p_intervals=tuple(clopper_pearson(h, c) for h, c in zip(exceed, counts, strict=True)),
        p_value=float(composite),
        alpha=float(alpha),
        structure_supported=bool(composite <= alpha),
        stopped_early=tuple(early),
        skipped_reason=None,
        n_fits=n_fits,
        seconds=time.perf_counter() - start,
    )
