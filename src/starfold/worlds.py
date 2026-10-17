"""Possible worlds: reproducibility evidence for one reference partition.

A world is one refit of the selected configuration on a random subset of the
rows, with one measurement draw when per-value uncertainties or a sampler are
supplied. Following the clusterboot design of Hennig 2007
(doi:10.1016/j.csda.2006.11.025), the worlds never relabel the reference
partition and measure how often its groups and memberships recur. Every
statistic here is an algorithmic recurrence under the declared perturbation
design, not a posterior, and percentile bands over worlds are stability
intervals, not confidence intervals. :func:`summarize_worlds` gives the
definitions.

The evidence holds ``2 W n`` bytes of world labels and ``4 n G`` bytes of
support, and summarizing adds a float64 accumulator of ``8 n G`` bytes. No
statistic in this module allocates an ``n`` by ``n`` object.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from starfold._engine import ResolvedEngine, derive_seed, parallel_map, require_resolved
from starfold.backend import Params, fit_pipeline
from starfold.objective import pair_stability
from starfold.prep import MeasurementSampler, Scaling, measurement_draw

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

UNSAMPLED: int = -2
"""World label of a row the world did not sample."""

NOISE: int = -1
"""World and reference label of a noise row."""

MAX_WORLDS: int = 32767
"""Largest world count representable by the int16 and uint16 evidence arrays."""

_BAND_PERCENTILES: tuple[float, float] = (5.0, 95.0)


@dataclass(frozen=True)
class WorldEvidence:
    """Recurrence evidence of a reference partition over possible worlds.

    Attributes
    ----------
    world_labels : ndarray of shape (W, n), dtype int16
        Per world, -2 for a row the world did not sample, -1 for world noise,
        else the world cluster id.
    n_sampled : ndarray of shape (n,), dtype uint16
        Number of worlds that sampled each row (the evidence coverage).
    support : ndarray of shape (n, G), dtype float32
        Purity-weighted recurrence of each row landing with the members of
        each reference group, averaged over the worlds that sampled the row.
        NaN for rows no world sampled.
    unresolved : ndarray of shape (n,), dtype float32
        Mean mass a row placed on world noise or on reference-noise company,
        with ``support.sum(1) + unresolved == 1`` on every sampled row.
    own_support : ndarray of shape (n,), dtype float32
        ``support[i, labels[i]]`` for reference-assigned rows, NaN for
        reference noise.
    unresolved_mass : float
        Mean of ``unresolved`` over sampled rows.
    group_recurrence : ndarray of shape (G,), dtype float32
        Mean over worlds of the best Jaccard match of each reference group
        (Hennig 2007, doi:10.1016/j.csda.2006.11.025). NaN when no world
        sampled a member.
    group_recurrence_band : ndarray of shape (G, 2), dtype float32
        5th and 95th percentiles of the per-world Jaccard match, a stability
        interval.
    count_values : ndarray of shape (W,), dtype int16
        Number of clusters found in each world.
    count_interval : tuple of int
        Integer hull of the 5th and 95th percentiles of ``count_values``, a
        stability interval and never a confidence interval.
    count_reference : int
        Number of groups in the reference partition.
    property_median : ndarray of shape (G, d), dtype float64
        Median over worlds of the per-feature median of the world cluster
        matched to each group, in original units. NaN when never matched.
    property_band : ndarray of shape (G, d, 2), dtype float64
        5th and 95th percentiles over worlds of the same per-world medians.
    property_propagated : bool
        Whether the world features carried measurement draws.
    world_agreement : ndarray of shape (W,), dtype float32
        Adjusted Rand index, floored at 0, times joint coverage between each
        world and the reference on the world's rows.
    world_seeds : tuple of int
        UMAP seed of each world.
    world_errors : tuple of str
        Exception text of each world whose fit raised, empty otherwise.
    n_fits : int
        Number of world fits attempted.
    seconds : float
        Wall time of the stage.
    """

    world_labels: NDArray[np.int16]
    n_sampled: NDArray[np.uint16]
    support: NDArray[np.float32]
    unresolved: NDArray[np.float32]
    own_support: NDArray[np.float32]
    unresolved_mass: float
    group_recurrence: NDArray[np.float32]
    group_recurrence_band: NDArray[np.float32]
    count_values: NDArray[np.int16]
    count_interval: tuple[int, int]
    count_reference: int
    property_median: NDArray[np.float64]
    property_band: NDArray[np.float64]
    property_propagated: bool
    world_agreement: NDArray[np.float32]
    world_seeds: tuple[int, ...]
    world_errors: tuple[str, ...]
    n_fits: int
    seconds: float


@dataclass(frozen=True)
class _WorldTask:
    """Everything one world needs, shipped to a worker process as a unit."""

    world: int
    x: NDArray[np.float64]
    x_err: NDArray[np.float64] | None
    sampler: MeasurementSampler | None
    params: Params
    random_state: int
    engine: ResolvedEngine
    scaling: Scaling
    n_rows: int


@dataclass(frozen=True)
class _WorldOutcome:
    """What a world returns: its rows, labels, cluster medians and error text."""

    world: int
    rows: NDArray[np.intp]
    labels: NDArray[np.int32]
    cluster_medians: NDArray[np.float64]
    seed: int
    error: str


def _members(labels: NDArray[np.intp] | NDArray[np.int32]) -> list[NDArray[np.intp]]:
    """Positions of each distinct label, in sorted label order and ascending position."""
    order = np.argsort(labels, kind="stable")
    ids, starts = np.unique(labels[order], return_index=True)
    return np.split(order, starts[1:]) if ids.size else []


def _cluster_medians(x_w: NDArray[np.float64], labels: NDArray[np.int32]) -> NDArray[np.float64]:
    """Per-feature median of ``x_w`` over each world cluster, by sorted cluster id."""
    blocks = [block for block in _members(labels) if labels[block[0]] >= 0]
    medians = np.empty((len(blocks), x_w.shape[1]), dtype=np.float64)
    for index, block in enumerate(blocks):
        medians[index] = np.median(x_w[block], axis=0)
    return medians


def _fit_world(task: _WorldTask) -> _WorldOutcome:
    """Draw, perturb and fit one world, turning a raising fit into all noise plus its text."""
    rng = np.random.default_rng(derive_seed(task.random_state, 2, task.world))
    n = task.x.shape[0]
    rows = np.sort(rng.choice(n, size=task.n_rows, replace=False)).astype(np.intp)
    x_err_rows = None if task.x_err is None else task.x_err[rows]
    x_w = measurement_draw(task.x[rows], rows, rng, x_err_rows=x_err_rows, sampler=task.sampler)
    seed = int(derive_seed(task.random_state, 3, task.world))
    try:
        fit = fit_pipeline(x_w, task.params, seed=seed, engine=task.engine, scaling=task.scaling)
        labels = np.asarray(fit.labels, dtype=np.int32)
        if labels.shape != (task.n_rows,):
            msg = f"fit returned {labels.shape} labels for {task.n_rows} rows"
            raise ValueError(msg)  # noqa: TRY301
    except Exception as exc:  # noqa: BLE001
        labels = np.full(task.n_rows, NOISE, dtype=np.int32)
        error = f"{type(exc).__name__}: {exc}"
    else:
        error = ""
    return _WorldOutcome(
        world=task.world,
        rows=rows,
        labels=labels,
        cluster_medians=_cluster_medians(x_w, labels),
        seed=seed,
        error=error,
    )


def _reference_index(labels: NDArray[np.int32]) -> tuple[NDArray[np.intp], int]:
    """Map reference labels to ``0..G-1`` in sorted order of id, with ``G`` for noise."""
    assigned = labels >= 0
    ids = np.unique(labels[assigned])
    n_groups = int(ids.shape[0])
    ref = np.full(labels.shape[0], n_groups, dtype=np.intp)
    ref[assigned] = np.searchsorted(ids, labels[assigned])
    return ref, n_groups


def _band(values: NDArray[np.float64], axis: int = 0) -> NDArray[np.float64]:
    """5th and 95th percentiles along ``axis``, stacked on a trailing axis."""
    low, high = np.percentile(values, _BAND_PERCENTILES, axis=axis)
    return np.stack(
        [np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)], axis=-1
    )


@dataclass
class _Accumulator:
    """Running sums over worlds, filled world by world."""

    support: NDArray[np.float64]
    unresolved: NDArray[np.float64]
    recurrence: list[list[float]]
    properties: list[list[NDArray[np.float64]]]
    count_values: NDArray[np.int16]
    agreement: NDArray[np.float32]


def _accumulate_world(
    acc: _Accumulator,
    world: int,
    world_row: NDArray[np.int16],
    labels: NDArray[np.int32],
    ref: NDArray[np.intp],
    n_groups: int,
    x: NDArray[np.float64],
    medians: NDArray[np.float64] | None,
) -> None:
    """Add one world's contingency, Jaccard, property and agreement terms."""
    rows = np.flatnonzero(world_row != UNSAMPLED).astype(np.intp)
    lw = world_row[rows].astype(np.int32)
    assigned = lw >= 0
    assigned_rows = rows[assigned]
    ids, cluster = np.unique(lw[assigned], return_inverse=True)
    n_clusters = int(ids.shape[0])
    acc.count_values[world] = n_clusters
    ref_rows = ref[rows]

    contingency = np.zeros((n_clusters, n_groups + 1), dtype=np.float64)
    np.add.at(contingency, (cluster, ref_rows[assigned]), 1.0)
    cluster_size = contingency.sum(axis=1)
    share = contingency / np.maximum(cluster_size, 1.0)[:, None]
    blocks = _members(cluster)
    for index, block in enumerate(blocks):
        acc.support[assigned_rows[block]] += share[index, :n_groups]
    acc.unresolved[assigned_rows] += share[cluster, n_groups]
    acc.unresolved[rows[~assigned]] += 1.0

    members = np.bincount(ref_rows, minlength=n_groups + 1)[:n_groups].astype(np.float64)
    intersection = contingency[:, :n_groups]
    union = members[None, :] + cluster_size[:, None] - intersection
    jaccard = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    for group in range(n_groups):
        if members[group] == 0:
            continue
        if n_clusters == 0:
            acc.recurrence[group].append(0.0)
            continue
        best = int(np.argmax(jaccard[:, group]))
        acc.recurrence[group].append(float(jaccard[best, group]))
        if jaccard[best, group] <= 0.0:
            continue
        if medians is None:
            acc.properties[group].append(np.median(x[assigned_rows[blocks[best]]], axis=0))
        else:
            acc.properties[group].append(np.asarray(medians[best], dtype=np.float64))

    positions = np.arange(rows.shape[0], dtype=np.intp)
    ari, coverage = pair_stability(lw, labels[rows], positions, positions)
    acc.agreement[world] = ari * coverage


def summarize_worlds(
    world_labels: NDArray[np.int16],
    labels: NDArray[np.int32],
    x: NDArray[np.float64],
    *,
    propagated: bool,
    cluster_medians: Sequence[NDArray[np.float64]] | None = None,
) -> WorldEvidence:
    """Compute every world statistic from a world label matrix.

    Parameters
    ----------
    world_labels : ndarray of shape (W, n), dtype int16
        -2 for a row the world did not sample, -1 for world noise, else the
        world cluster id.
    labels : ndarray of shape (n,), dtype int32
        Reference partition with -1 for noise. Group ids are relabelled to
        ``0..G-1`` in sorted order of their original ids.
    x : ndarray of shape (n, d)
        Feature matrix in original units, used for the property statistics
        when ``cluster_medians`` is not given.
    propagated : bool
        Recorded as ``property_propagated``.
    cluster_medians : sequence of ndarray, optional
        Per world, the (K_w, d) per-feature medians of the world's own
        feature matrix over its clusters in sorted order of cluster id.
        :func:`run_worlds` supplies these, letting measurement draws enter
        the property bands although they are never persisted.

    Returns
    -------
    WorldEvidence
        Every statistic, with ``world_seeds`` and ``world_errors`` empty and
        ``n_fits`` and ``seconds`` zero until :func:`run_worlds` fills them.

    Raises
    ------
    ValueError
        On shape mismatches between ``world_labels``, ``labels``, ``x`` and
        ``cluster_medians``.

    Notes
    -----
    Support is purity-weighted. A row in world cluster ``c`` receives on
    group ``g`` the fraction of ``c`` that the reference assigns to ``g``, and
    on the unresolved column the fraction of ``c`` that the reference calls
    noise, while world noise places mass 1 on unresolved. Group recurrence is
    the maximum Jaccard similarity between the group's sampled members and any
    world cluster (Hennig 2007, doi:10.1016/j.csda.2006.11.025), averaged over
    the worlds that sampled at least one member. The rough guidelines of the
    R function ``fpc::clusterboot`` (Hennig, CRAN package fpc; a mean of 0.85
    or more highly stable, a value of 0.5 or less dissolved) are never
    applied as gates. Ties in the best match resolve to the lowest cluster id. Bands are
    5th and 95th percentiles over worlds with linear interpolation, and
    ``count_interval`` is the integer hull (floor, ceil) of those
    percentiles. ``world_agreement`` is the adjusted Rand index (Hubert and
    Arabie 1985, doi:10.1007/BF01908075) times joint coverage of
    :func:`~starfold.objective.pair_stability`.
    """
    world_labels = np.asarray(world_labels, dtype=np.int16)
    labels = np.asarray(labels, dtype=np.int32)
    x = np.asarray(x, dtype=np.float64)
    if world_labels.ndim != 2:
        msg = "world_labels must be a 2-D (W, n) matrix."
        raise ValueError(msg)
    n_worlds, n = world_labels.shape
    if labels.shape != (n,) or x.ndim != 2 or x.shape[0] != n:
        msg = "labels must have shape (n,) and x shape (n, d) matching world_labels."
        raise ValueError(msg)
    if cluster_medians is not None and len(cluster_medians) != n_worlds:
        msg = "cluster_medians must hold one (K_w, d) array per world."
        raise ValueError(msg)
    n_features = x.shape[1]
    ref, n_groups = _reference_index(labels)

    acc = _Accumulator(
        support=np.zeros((n, n_groups), dtype=np.float64),
        unresolved=np.zeros(n, dtype=np.float64),
        recurrence=[[] for _ in range(n_groups)],
        properties=[[] for _ in range(n_groups)],
        count_values=np.zeros(n_worlds, dtype=np.int16),
        agreement=np.zeros(n_worlds, dtype=np.float32),
    )
    for world in range(n_worlds):
        medians = None if cluster_medians is None else cluster_medians[world]
        _accumulate_world(acc, world, world_labels[world], labels, ref, n_groups, x, medians)

    n_sampled = (world_labels != UNSAMPLED).sum(axis=0)
    sampled = n_sampled > 0
    scale = np.where(sampled, n_sampled, 1).astype(np.float64)
    support = acc.support
    support /= scale[:, None]
    support[~sampled] = np.nan
    unresolved = acc.unresolved
    unresolved /= scale
    unresolved[~sampled] = np.nan
    own_support = np.full(n, np.nan, dtype=np.float64)
    reference_assigned = labels >= 0
    own_support[reference_assigned] = support[reference_assigned, ref[reference_assigned]]
    unresolved_mass = float(np.mean(unresolved[sampled])) if sampled.any() else float("nan")

    recurrence = np.full(n_groups, np.nan, dtype=np.float64)
    recurrence_band = np.full((n_groups, 2), np.nan, dtype=np.float64)
    property_median = np.full((n_groups, n_features), np.nan, dtype=np.float64)
    property_band = np.full((n_groups, n_features, 2), np.nan, dtype=np.float64)
    for group in range(n_groups):
        if acc.recurrence[group]:
            values = np.asarray(acc.recurrence[group], dtype=np.float64)
            recurrence[group] = values.mean()
            recurrence_band[group] = _band(values)
        if acc.properties[group]:
            stacked = np.stack(acc.properties[group], axis=0)
            property_median[group] = np.median(stacked, axis=0)
            property_band[group] = _band(stacked)

    low, high = np.percentile(acc.count_values.astype(np.float64), _BAND_PERCENTILES)
    count_interval = (int(np.floor(low)), int(np.ceil(high)))

    return WorldEvidence(
        world_labels=world_labels,
        n_sampled=n_sampled.astype(np.uint16),
        support=support.astype(np.float32),
        unresolved=unresolved.astype(np.float32),
        own_support=own_support.astype(np.float32),
        unresolved_mass=unresolved_mass,
        group_recurrence=recurrence.astype(np.float32),
        group_recurrence_band=recurrence_band.astype(np.float32),
        count_values=acc.count_values,
        count_interval=count_interval,
        count_reference=n_groups,
        property_median=property_median,
        property_band=property_band,
        property_propagated=propagated,
        world_agreement=acc.agreement,
        world_seeds=(),
        world_errors=(),
        n_fits=0,
        seconds=0.0,
    )


def _validate_run(
    x: NDArray[np.float64],
    labels: NDArray[np.int32],
    *,
    n_worlds: int,
    world_fraction: float,
    x_err: NDArray[np.float64] | None,
    has_sampler: bool,
) -> int:
    """Check the :func:`run_worlds` inputs and return the rows per world."""
    if x.ndim != 2:
        msg = "x must be a 2-D (n, d) matrix."
        raise ValueError(msg)
    n = int(x.shape[0])
    if labels.shape != (n,):
        msg = f"labels must have shape ({n},), got {labels.shape}."
        raise ValueError(msg)
    if n_worlds < 2 or n_worlds > MAX_WORLDS:
        msg = f"n_worlds must be between 2 and {MAX_WORLDS}, got {n_worlds}."
        raise ValueError(msg)
    if not 0.0 < world_fraction <= 1.0:
        msg = f"world_fraction must lie in (0, 1], got {world_fraction}."
        raise ValueError(msg)
    n_rows = round(world_fraction * n)
    if n_rows < 1:
        msg = f"world_fraction {world_fraction} draws no rows from n = {n}."
        raise ValueError(msg)
    if x_err is not None:
        if x_err.shape != x.shape:
            msg = f"x_err must have shape {x.shape}, got {x_err.shape}."
            raise ValueError(msg)
        if not np.all(np.isfinite(x_err)) or np.any(x_err < 0):
            msg = "x_err must be finite and non-negative."
            raise ValueError(msg)
        if has_sampler:
            msg = "x_err and sampler are exclusive; supply one perturbation."
            raise ValueError(msg)
    return n_rows


def run_worlds(
    x: NDArray[np.float64],
    params: Params,
    labels: NDArray[np.int32],
    *,
    n_worlds: int,
    world_fraction: float,
    random_state: int,
    engine: ResolvedEngine,
    scaling: Scaling,
    n_jobs: int = 1,
    x_err: NDArray[np.float64] | None = None,
    sampler: MeasurementSampler | None = None,
) -> WorldEvidence:
    """Refit one configuration over possible worlds and summarize the recurrence.

    Parameters
    ----------
    x : ndarray of shape (n, d)
        Feature matrix in original units.
    params : Params
        The configuration every world fits.
    labels : ndarray of shape (n,), dtype int32
        Reference partition with -1 for noise.
    n_worlds : int
        Number of worlds ``W``, between 2 and 32767.
    world_fraction : float
        Fraction of rows each world draws without replacement, in ``(0, 1]``.
    random_state : int
        Base seed. World ``w`` draws its rows and measurement with
        ``derive_seed(random_state, 2, w)`` and fits UMAP with
        ``derive_seed(random_state, 3, w)``.
    engine : {"cpu", "cuml"}
        Resolved backend. Any other value raises ``ValueError``.
    scaling : {"standard", "robust"}
        Scaler fitted inside each world on the world's rows only.
    n_jobs : int, default 1
        Worker processes over worlds. Above 1, ``sampler`` must be picklable.
    x_err : ndarray of shape (n, d), optional
        Measured per-value uncertainties. Each world adds one standard normal
        draw scaled by them.
    sampler : callable, optional
        Measurement sampler ``(x_rows, row_indices, rng) -> x_rows``, exclusive
        with ``x_err``.

    Returns
    -------
    WorldEvidence
        The world label matrix, all statistics, seeds, errors and wall time.

    Raises
    ------
    ValueError
        On shape mismatches, an invalid ``n_worlds``, ``world_fraction`` or
        ``engine``, or when both ``x_err`` and ``sampler`` are given.

    Notes
    -----
    A world whose fit raises is recorded as all noise on its rows, with its
    exception text kept in ``world_errors`` and no exception propagated; such
    a world still counts in ``n_sampled`` and adds unresolved mass. With seeds
    derived per world, the result does not depend on ``n_jobs``. Measurement
    draws are used inside the world and never persisted.
    """
    start = time.perf_counter()
    engine = require_resolved(engine)
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    x_err = None if x_err is None else np.asarray(x_err, dtype=np.float64)
    n_rows = _validate_run(
        x,
        labels,
        n_worlds=n_worlds,
        world_fraction=world_fraction,
        x_err=x_err,
        has_sampler=sampler is not None,
    )
    n = int(x.shape[0])

    tasks = [
        _WorldTask(
            world=world,
            x=x,
            x_err=x_err,
            sampler=sampler,
            params=params,
            random_state=random_state,
            engine=engine,
            scaling=scaling,
            n_rows=n_rows,
        )
        for world in range(n_worlds)
    ]
    outcomes: list[_WorldOutcome] = parallel_map(_fit_world, tasks, n_jobs=n_jobs)

    world_labels = np.full((n_worlds, n), UNSAMPLED, dtype=np.int16)
    medians: list[NDArray[np.float64]] = [np.empty((0, x.shape[1]))] * n_worlds
    seeds = [0] * n_worlds
    errors = [""] * n_worlds
    for outcome in outcomes:
        if outcome.labels.max(initial=NOISE) > np.iinfo(np.int16).max:
            msg = f"world {outcome.world} produced a cluster id beyond int16."
            raise ValueError(msg)
        world_labels[outcome.world, outcome.rows] = outcome.labels.astype(np.int16)
        medians[outcome.world] = outcome.cluster_medians
        seeds[outcome.world] = outcome.seed
        errors[outcome.world] = outcome.error

    summary = summarize_worlds(
        world_labels,
        labels,
        x,
        propagated=x_err is not None or sampler is not None,
        cluster_medians=medians,
    )
    return replace(
        summary,
        world_seeds=tuple(seeds),
        world_errors=tuple(errors),
        n_fits=n_worlds,
        seconds=time.perf_counter() - start,
    )
