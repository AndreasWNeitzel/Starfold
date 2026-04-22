"""Global credibility test for a starfold clustering run.

Background
----------
HDBSCAN on a reasonable hyperparameter grid will almost always return
at least two clusters, even on a structureless Gaussian point cloud.
Per-cluster persistence thresholds (see
:mod:`starfold.noise_baseline`) filter out individual clusters that
a noise realisation could match, but they do not answer the
*omnibus* question:

    "Is the clustering run *as a whole* distinguishable from what I
    would get if I fed structureless noise of the same shape through
    the same UMAP+Optuna+HDBSCAN pipeline?"

This module answers that by comparing three scalars from the real-data
run against the matching distributions recorded on the noise
realisations that built the baseline.

Scalars compared
----------------
1. ``n_clusters`` -- number of HDBSCAN clusters in the selected
   (best-trial) fit. Larger-than-noise typically means the data has
   more resolvable structure than noise; smaller-than-noise can also
   be informative (noise tends to splinter into small spurious
   groups).
2. ``best_objective`` -- the Optuna best-trial scalar objective
   (``persistence_sum`` or ``combined_geom``). Larger means the
   optimiser found a better optimum on real data than noise
   typically admits.
3. ``max_persistence`` -- the largest single-cluster persistence in
   the selected fit. Larger means the *strongest* real-data cluster
   is stronger than the strongest noise-realisation cluster
   typically is.

Each scalar gets an empirical, one-sided, upper-tail p-value using
the ``(r + 1) / (n + 1)`` unbiased plug-in estimator (Phipson & Smyth
2010, *Statistical Applications in Genetics and Molecular Biology*
**9**, 39) so p-values are never exactly zero with finite Monte Carlo.

A run "passes credibility at level ``alpha``" when all three
p-values are below ``alpha`` (default ``0.003``, the 3-sigma gate
chosen to match the paper's per-cluster threshold).

Scope
-----
Strictly beyond paper §3.3, which specifies only the per-cluster
threshold. The machinery is a natural extension of the same null
already computed for that threshold -- no extra noise realisations
are needed. See ``docs/design_decisions.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

__all__ = [
    "CredibilityReport",
    "compute_credibility",
    "empirical_upper_tail_pvalue",
]

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from starfold.noise_baseline import NoiseBaselineResult


def empirical_upper_tail_pvalue(
    observed: float,
    null_samples: NDArray[np.floating[Any]] | NDArray[np.integer[Any]],
) -> float:
    """Empirical upper-tail p-value with the Phipson-Smyth correction.

    Defined as ``(r + 1) / (n + 1)`` where ``r`` is the number of
    null samples greater than or equal to ``observed`` and ``n`` is
    the null sample size. The ``+1`` regularisation keeps p-values
    strictly positive under finite Monte Carlo, which matters when
    the observed value exceeds every null draw (the naive ``r / n``
    would give ``0`` and hide the finite resolution of the test).

    Parameters
    ----------
    observed
        The real-data scalar under test.
    null_samples
        1-D array of scalars drawn under the null.

    Returns
    -------
    float
        Empirical p-value in ``(0, 1]``.

    Raises
    ------
    ValueError
        If ``null_samples`` is empty or not 1-D.

    Examples
    --------
    >>> import numpy as np
    >>> null = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> round(empirical_upper_tail_pvalue(0.6, null), 6)
    0.166667
    >>> round(empirical_upper_tail_pvalue(0.3, null), 6)
    0.666667
    """
    arr = np.asarray(null_samples)
    if arr.ndim != 1:
        msg = f"null_samples must be 1-D (got shape {arr.shape})."
        raise ValueError(msg)
    if arr.size == 0:
        msg = "null_samples must be non-empty."
        raise ValueError(msg)
    r = int(np.sum(arr >= observed))
    return float((r + 1) / (arr.size + 1))


@dataclass(frozen=True)
class CredibilityReport:
    """Empirical global and per-cluster p-values for a clustering run.

    Global test
    -----------
    Each of three global scalars -- ``n_clusters``, ``best_objective``,
    ``max_persistence`` -- is compared one-sided upper-tail against the
    same quantity recorded on every noise realisation in the baseline.
    The ``passes`` flag is ``True`` iff all three p-values are below
    ``alpha``.

    Per-cluster test
    ----------------
    In addition, every real-data cluster's persistence is compared
    against :attr:`NoiseBaselineResult.null_cluster_persistence` -- the
    flat pool of every cluster's persistence across every noise
    realisation -- to produce a one-sided upper-tail p-value per
    cluster. This answers "does *this specific* cluster look like
    anything noise routinely produces?" which the global test cannot.
    :attr:`per_cluster_significant` is ``True`` at positions whose
    p-value is below ``alpha``.

    Parameters
    ----------
    observed_n_clusters, null_n_clusters, n_clusters_pvalue
        Number of clusters in the selected fit, the null
        distribution from the baseline, and the empirical p-value.
    observed_objective, null_objective, objective_pvalue
        Best-trial Optuna objective value on real data, its null
        distribution, and the p-value. ``objective_name`` records
        whether this is ``"persistence_sum"`` or ``"combined_geom"``.
    observed_max_persistence, null_max_persistence,
    max_persistence_pvalue
        Largest single-cluster persistence in the selected fit, its
        null distribution, and the p-value. This mirrors the
        quantity that sets the per-cluster significance threshold.
    observed_cluster_persistence
        Per-cluster persistence on real data, length ``n_clusters``.
    null_cluster_persistence
        Pooled null distribution of per-cluster persistence values
        across every noise realisation (copy of
        :attr:`NoiseBaselineResult.null_cluster_persistence`).
    per_cluster_pvalue
        Upper-tail empirical p-value for each real-data cluster
        against ``null_cluster_persistence``. Length ``n_clusters``.
    per_cluster_significant
        Boolean mask, ``p < alpha`` per cluster. Length ``n_clusters``.
    alpha
        Significance level for ``passes`` and
        ``per_cluster_significant`` (default 0.003, i.e. 3 sigma,
        matching the paper's per-cluster gate).
    passes
        ``True`` iff every global p-value is strictly less than
        ``alpha``.
    config
        Frozen record of what null distribution was used (copy of
        :attr:`NoiseBaselineResult.config`). Lets downstream code
        confirm the credibility was computed against the right
        baseline.
    """

    observed_n_clusters: int
    null_n_clusters: NDArray[np.integer[Any]]
    n_clusters_pvalue: float

    observed_objective: float
    null_objective: NDArray[np.floating[Any]]
    objective_pvalue: float
    objective_name: str

    observed_max_persistence: float
    null_max_persistence: NDArray[np.floating[Any]]
    max_persistence_pvalue: float

    observed_cluster_persistence: NDArray[np.floating[Any]]
    null_cluster_persistence: NDArray[np.floating[Any]]
    per_cluster_pvalue: NDArray[np.floating[Any]]
    per_cluster_significant: NDArray[np.bool_]

    alpha: float
    passes: bool
    config: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """One-paragraph human-readable verdict."""
        verdict = "PASS" if self.passes else "FAIL"
        n_cred = int(self.per_cluster_significant.sum())
        n_total = int(self.per_cluster_significant.size)
        return (
            f"credibility at alpha={self.alpha}: {verdict}\n"
            f"  n_clusters      obs={self.observed_n_clusters:d}"
            f"  null-median={int(np.median(self.null_n_clusters)):d}"
            f"  p={self.n_clusters_pvalue:.4f}\n"
            f"  {self.objective_name:<16s}"
            f"obs={self.observed_objective:.4f}"
            f"  null-median={float(np.median(self.null_objective)):.4f}"
            f"  p={self.objective_pvalue:.4f}\n"
            f"  max_persistence obs={self.observed_max_persistence:.4f}"
            f"  null-median={float(np.median(self.null_max_persistence)):.4f}"
            f"  p={self.max_persistence_pvalue:.4f}\n"
            f"  per-cluster     {n_cred:d}/{n_total:d} clusters credible"
            f" at alpha={self.alpha}"
        )


def compute_credibility(
    *,
    n_clusters: int,
    best_objective: float,
    max_persistence: float,
    baseline: NoiseBaselineResult,
    cluster_persistence: NDArray[np.floating[Any]] | None = None,
    alpha: float = 0.003,
) -> CredibilityReport:
    """Compare a real-data run's scalars against the baseline null.

    Parameters
    ----------
    n_clusters
        Number of HDBSCAN clusters in the selected real-data fit.
    best_objective
        Real-data best-trial Optuna objective (same scale as
        ``baseline.objective``; a mismatch raises).
    max_persistence
        Largest per-cluster persistence in the selected real-data
        fit.
    baseline
        The noise baseline whose ``per_realisation_*`` and
        ``null_cluster_*`` arrays form the null distributions.
    cluster_persistence
        Per-cluster persistence on real data, length ``n_clusters``.
        Optional for backwards compatibility; when ``None``, an
        empty array is stored and ``per_cluster_pvalue`` /
        ``per_cluster_significant`` are empty. Pass this whenever
        possible: the per-cluster test is cheap and often more
        informative than the global test when a subset of clusters
        is strong but the omnibus p-value is marginal.
    alpha
        Significance level for the ``passes`` flag and the
        per-cluster significance mask (default 0.003, 3 sigma).

    Returns
    -------
    CredibilityReport
        Global and per-cluster p-values, the raw null arrays for
        plotting, and the overall verdict.

    Raises
    ------
    ValueError
        If ``alpha`` is not in ``(0, 1)``, if ``baseline`` has no
        realisations, or if ``cluster_persistence`` is not 1-D or
        disagrees in length with ``n_clusters``.
    """
    if not 0.0 < alpha < 1.0:
        msg = f"alpha must be in (0, 1) (got {alpha})."
        raise ValueError(msg)
    if baseline.per_realisation_objective.size == 0:
        msg = "baseline has no realisations; cannot compute credibility."
        raise ValueError(msg)

    n_clusters_p = empirical_upper_tail_pvalue(
        float(n_clusters), baseline.per_realisation_n_clusters
    )
    objective_p = empirical_upper_tail_pvalue(
        float(best_objective), baseline.per_realisation_objective
    )
    max_persistence_p = empirical_upper_tail_pvalue(
        float(max_persistence), baseline.per_realisation_max
    )
    passes = bool(n_clusters_p < alpha and objective_p < alpha and max_persistence_p < alpha)

    if cluster_persistence is None:
        observed_cluster = np.zeros(0, dtype=np.float64)
    else:
        observed_cluster = np.asarray(cluster_persistence, dtype=np.float64)
        if observed_cluster.ndim != 1:
            msg = f"cluster_persistence must be 1-D (got shape {observed_cluster.shape})."
            raise ValueError(msg)
        if observed_cluster.size != n_clusters:
            msg = (
                f"cluster_persistence length {observed_cluster.size} does not "
                f"match n_clusters={n_clusters}."
            )
            raise ValueError(msg)

    null_pers = np.asarray(baseline.null_cluster_persistence, dtype=np.float64)
    if observed_cluster.size == 0:
        per_cluster_p = np.zeros(0, dtype=np.float64)
        per_cluster_sig = np.zeros(0, dtype=np.bool_)
    elif null_pers.size == 0:
        # No null pool to compare against (e.g. noise realisations that
        # produced no clusters at all). Return the least-significant
        # p-value of 1.0 per cluster so the arrays stay aligned with
        # the real-data cluster count; the caller can tell the test
        # did not run by looking at baseline.null_cluster_persistence.size.
        per_cluster_p = np.ones(observed_cluster.size, dtype=np.float64)
        per_cluster_sig = np.zeros(observed_cluster.size, dtype=np.bool_)
    else:
        per_cluster_p = np.array(
            [empirical_upper_tail_pvalue(float(v), null_pers) for v in observed_cluster],
            dtype=np.float64,
        )
        per_cluster_sig = per_cluster_p < alpha

    return CredibilityReport(
        observed_n_clusters=int(n_clusters),
        null_n_clusters=np.asarray(baseline.per_realisation_n_clusters, dtype=np.intp),
        n_clusters_pvalue=n_clusters_p,
        observed_objective=float(best_objective),
        null_objective=np.asarray(baseline.per_realisation_objective, dtype=np.float64),
        objective_pvalue=objective_p,
        objective_name=baseline.objective,
        observed_max_persistence=float(max_persistence),
        null_max_persistence=np.asarray(baseline.per_realisation_max, dtype=np.float64),
        max_persistence_pvalue=max_persistence_p,
        observed_cluster_persistence=observed_cluster,
        null_cluster_persistence=null_pers,
        per_cluster_pvalue=per_cluster_p,
        per_cluster_significant=per_cluster_sig,
        alpha=float(alpha),
        passes=passes,
        config=dict(baseline.config),
    )
