"""Bounded label-invariant co-association check on the reference partition.

Evidence accumulation (Fred and Jain 2005, doi:10.1109/TPAMI.2005.113) and
consensus clustering (Monti et al. 2003, doi:10.1023/A:1023949509487) are
used only as a check on the emitted partition, never as the partition itself.
The pairwise matrices are the only ``n`` by ``n`` objects in the library and
stay inside the declared byte budget; above it the same cut runs on a seeded
subset of rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import HDBSCAN

from starfold._engine import require_resolved
from starfold.backend import rescale_params
from starfold.objective import pair_stability

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from starfold._engine import ResolvedEngine
    from starfold.backend import Params

BYTES_PER_PAIR: int = 16
"""Peak bytes per row pair of the dense check: the float64 matrix plus
HDBSCAN's float64 copy for core distances."""

_UINT8_MAX: int = 255


@dataclass(frozen=True)
class ConsensusCheck:
    """Outcome of the co-association cut and its agreement with the reference.

    Attributes
    ----------
    labels : ndarray of shape (n_rows,), dtype int32
        Consensus partition over ``rows`` with -1 for noise.
    rows : ndarray of shape (n_rows,), dtype intp
        Row indices the check used, sorted; every row when dense.
    agreement : float
        Adjusted Rand index times joint coverage between ``labels`` and the
        reference partition on ``rows``.
    subsampled : bool
        Whether the byte budget forced a seeded subset of rows.
    n_rows : int
        Number of rows used.
    bytes_used : int
        ``16 * n_rows ** 2``, the peak of the dense matrices.
    within_association : float
        Mean co-association of pairs the reference places in the same group:
        the fraction of worlds that clustered them together, over the worlds
        in which both were assigned. NaN when no such pair exists or when the
        matrix is not built.
    between_association : float
        The same quantity for pairs the reference places in different groups.
    contrast : float
        ``within_association - between_association``; label-invariant and
        computed whenever the matrix is built, even when the cut returns only
        noise; NaN when the cut is skipped because ``min_cluster_size`` exceeds
        ``n_rows``.
    """

    labels: NDArray[np.int32]
    rows: NDArray[np.intp]
    agreement: float
    subsampled: bool
    n_rows: int
    bytes_used: int
    within_association: float
    between_association: float
    contrast: float


def _count_dtype(n_worlds: int) -> type[np.unsignedinteger]:
    """Smallest unsigned integer dtype that holds a count of ``n_worlds``."""
    return np.uint8 if n_worlds <= _UINT8_MAX else np.uint16


def coassociation(
    world_labels: NDArray[np.int16],
    rows: NDArray[np.intp],
    *,
    engine: ResolvedEngine = "cpu",
) -> NDArray[np.float64]:
    """Co-association distance matrix over a set of rows.

    Parameters
    ----------
    world_labels : ndarray of shape (W, n), dtype int16
        -2 for a row the world did not sample, -1 for world noise, else the
        world cluster id.
    rows : ndarray of shape (n_rows,), dtype intp
        Row indices to compare.
    engine : {"cpu", "cuml"}, default "cpu"
        ``"cuml"`` accumulates the counts on the device in float32, which is
        exact for any count below 2**24, and returns the matrix to the host.
        Any other value raises ``ValueError``.

    Returns
    -------
    ndarray of shape (n_rows, n_rows), dtype float64
        ``D = max(1 - C / N, 1 / (2 W))`` where ``C`` counts the worlds in
        which both rows were sampled, both assigned and shared a label and
        ``N`` the worlds in which both were sampled and both assigned; 1.0
        where ``N`` is zero; zero diagonal.

    Notes
    -----
    ``C`` and ``N`` are accumulated per world from pairwise label equality
    over the rows sampled and assigned in that world, in uint8 when
    ``W <= 255`` and uint16 above, and are released before the distance is
    returned. The floor ``1 / (2 W)`` keeps perfectly recurrent pairs at a
    finite distance below the smallest resolvable non-zero co-association
    distance. A pair never jointly assigned carries no evidence of
    association and sits at the maximum distance; a midpoint value would make
    always-noise rows hubs through which single linkage merges separated
    groups.
    """
    engine = require_resolved(engine)
    world_labels = np.asarray(world_labels, dtype=np.int16)
    rows = np.asarray(rows, dtype=np.intp)
    if world_labels.ndim != 2 or rows.ndim != 1:
        msg = "world_labels must be (W, n) and rows one-dimensional."
        raise ValueError(msg)
    n_worlds = world_labels.shape[0]
    n_rows = rows.shape[0]
    if engine == "cuml":
        return _coassociation_device(world_labels[:, rows], n_worlds)
    dtype = _count_dtype(n_worlds)
    shared = np.zeros((n_rows, n_rows), dtype=dtype)
    joint = np.zeros((n_rows, n_rows), dtype=dtype)
    for world in range(n_worlds):
        labels = world_labels[world, rows]
        assigned = labels >= 0
        if not assigned.any():
            continue
        pair = np.logical_and.outer(assigned, assigned)
        joint += pair
        same = np.equal.outer(labels, labels)
        same &= pair
        shared += same
        del pair, same

    positive = joint > 0
    distance = np.empty((n_rows, n_rows), dtype=np.float64)
    np.divide(shared, joint, out=distance, where=positive)
    del shared, joint
    np.subtract(1.0, distance, out=distance, where=positive)
    np.maximum(distance, 1.0 / (2.0 * n_worlds), out=distance, where=positive)
    distance[~positive] = 1.0
    np.fill_diagonal(distance, 0.0)
    return distance


def _coassociation_device(labels_by_world: NDArray[np.int16], n_worlds: int) -> NDArray[np.float64]:
    """Return the distance of :func:`coassociation`, accumulated with CuPy."""
    import cupy  # noqa: PLC0415 - GPU dependency loaded only on the cuML backend

    world_labels = cupy.asarray(labels_by_world)
    n_rows = world_labels.shape[1]
    shared = cupy.zeros((n_rows, n_rows), dtype=cupy.float32)
    joint = cupy.zeros((n_rows, n_rows), dtype=cupy.float32)
    for world in range(n_worlds):
        labels = world_labels[world]
        assigned = labels >= 0
        if not bool(assigned.any()):
            continue
        _, cluster = cupy.unique(labels[assigned], return_inverse=True)
        one_hot = cupy.zeros((n_rows, int(cluster.max()) + 1), dtype=cupy.float32)
        one_hot[cupy.flatnonzero(assigned), cluster.ravel()] = 1.0
        shared += one_hot @ one_hot.T
        indicator = assigned.astype(cupy.float32)
        joint += cupy.outer(indicator, indicator)
    positive = joint > 0
    # Counts are exact in float32; the division is done in float64 so the
    # distances equal the CPU path's to the last bit.
    ratio = shared.astype(cupy.float64) / cupy.where(positive, joint, 1.0).astype(cupy.float64)
    del shared
    distance = cupy.where(positive, cupy.maximum(1.0 - ratio, 1.0 / (2.0 * n_worlds)), 1.0)
    del ratio, joint, positive
    cupy.fill_diagonal(distance, 0.0)
    return np.asarray(distance.get(), dtype=np.float64)


def association_contrast(
    distance: NDArray[np.float64], reference_labels: NDArray[np.int32]
) -> tuple[float, float]:
    """Mean co-association within and between the reference groups.

    Co-association is ``1 - distance``, the fraction of worlds that clustered a
    pair together among the worlds in which both were assigned. Only pairs the
    reference assigns are counted, and the diagonal is excluded. The sums run
    through a ``G`` by ``G`` table of group totals, so no further ``r`` by
    ``r`` array is allocated.

    Parameters
    ----------
    distance : ndarray of shape (r, r), float64
        Co-association distance from :func:`coassociation`.
    reference_labels : ndarray of shape (r,), int32
        Reference partition on the same rows, -1 for noise.

    Returns
    -------
    within, between : float
        Mean co-association of same-group and different-group pairs, NaN when
        no pair of the corresponding kind exists.
    """
    assigned = reference_labels >= 0
    if int(assigned.sum()) < 2:
        return math.nan, math.nan
    _, group = np.unique(reference_labels[assigned], return_inverse=True)
    members = np.zeros((reference_labels.shape[0], int(group.max()) + 1), dtype=np.float64)
    members[np.flatnonzero(assigned), group] = 1.0
    totals = members.T @ (distance @ members)
    sizes = members.sum(axis=0)
    diagonal = float(np.diagonal(distance)[assigned].sum())
    within_sum = float(np.trace(totals)) - diagonal
    between_sum = float(totals.sum()) - float(np.trace(totals))
    within_pairs = float(np.sum(sizes * (sizes - 1.0)))
    between_pairs = float(sizes.sum() ** 2 - np.sum(sizes**2))
    within = 1.0 - within_sum / within_pairs if within_pairs > 0 else math.nan
    between = 1.0 - between_sum / between_pairs if between_pairs > 0 else math.nan
    return within, between


def consensus_check(
    world_labels: NDArray[np.int16],
    params: Params,
    reference_labels: NDArray[np.int32],
    *,
    budget_bytes: int,
    check_rows: int,
    seed: int,
    engine: ResolvedEngine = "cpu",
) -> ConsensusCheck:
    """Cut the co-association distance and score it against the reference.

    Parameters
    ----------
    world_labels : ndarray of shape (W, n), dtype int16
        World label matrix from :func:`starfold.worlds.run_worlds`.
    params : Params
        Selected configuration at the full ``n`` rows; its
        ``min_cluster_size`` and ``selection_method`` drive the cut.
    reference_labels : ndarray of shape (n,), dtype int32
        Reference partition with -1 for noise.
    budget_bytes : int
        Largest ``16 * n_rows ** 2`` the check may allocate.
    check_rows : int
        Subset size used when all ``n`` rows exceed the budget.
    seed : int
        Seed of the subset draw.
    engine : {"cpu", "cuml"}, default "cpu"
        Backend of the co-association matrix. The cut itself is scikit-learn
        HDBSCAN on either backend, because cuML HDBSCAN accepts only Euclidean
        input and this cut runs on a precomputed distance.

    Returns
    -------
    ConsensusCheck
        Consensus labels, rows used, agreement with the reference, subsampling
        flag, bytes used and association contrast.

    Raises
    ------
    ValueError
        On shape mismatches, ``check_rows < 1``, an engine other than
        ``"cpu"`` or ``"cuml"``, or when even the subset exceeds
        ``budget_bytes``.

    Notes
    -----
    Dense over all rows only when ``16 n ** 2 <= budget_bytes``; otherwise a
    sorted seeded subset of ``min(check_rows, n)`` rows is used and
    ``params`` is rescaled to it with
    :func:`~starfold.backend.rescale_params`, so the cut asks for the same
    mass fraction as the reference. The cut is scikit-learn's HDBSCAN
    (Campello, Moulavi and Sander 2013, doi:10.1007/978-3-642-37456-2_14) with
    ``metric="precomputed"`` and ``copy=False``, which overwrites the matrix
    with mutual-reachability distances; the association contrast is therefore
    read before the cut. When ``min_cluster_size`` exceeds the rows used, no
    cluster can exist and every row is returned as noise with agreement 0 and
    NaN association, without building the matrix; the rescaling rules this out
    on a subset of two or more rows.
    """
    engine = require_resolved(engine)
    world_labels = np.asarray(world_labels, dtype=np.int16)
    reference_labels = np.asarray(reference_labels, dtype=np.int32)
    if world_labels.ndim != 2:
        msg = "world_labels must be a 2-D (W, n) matrix."
        raise ValueError(msg)
    n = world_labels.shape[1]
    if reference_labels.shape != (n,):
        msg = f"reference_labels must have shape ({n},), got {reference_labels.shape}."
        raise ValueError(msg)
    if check_rows < 1:
        msg = f"check_rows must be positive, got {check_rows}."
        raise ValueError(msg)

    if BYTES_PER_PAIR * n * n <= budget_bytes:
        rows = np.arange(n, dtype=np.intp)
        subsampled = False
    else:
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(n, size=min(check_rows, n), replace=False)).astype(np.intp)
        subsampled = True
        params = rescale_params(params, n, rows.shape[0])
    n_rows = rows.shape[0]
    bytes_used = BYTES_PER_PAIR * n_rows * n_rows
    if bytes_used > budget_bytes:
        msg = (
            f"the consensus subset of {n_rows} rows needs {bytes_used} bytes, above the "
            f"budget of {budget_bytes}; lower check_rows or raise the budget."
        )
        raise ValueError(msg)

    if params.min_cluster_size > n_rows:
        return ConsensusCheck(
            labels=np.full(n_rows, -1, dtype=np.int32),
            rows=rows,
            agreement=0.0,
            subsampled=subsampled,
            n_rows=n_rows,
            bytes_used=bytes_used,
            within_association=math.nan,
            between_association=math.nan,
            contrast=math.nan,
        )

    distance = coassociation(world_labels, rows, engine=engine)
    within, between = association_contrast(distance, reference_labels[rows])
    cut = HDBSCAN(
        metric="precomputed",
        copy=False,
        min_cluster_size=params.min_cluster_size,
        # Core distances over min_cluster_size rows, not the selected min_samples:
        # boundary rows co-clustered with a neighbouring group in every world sit
        # at the floor distance, and a core distance over few rows lets single
        # linkage merge the two groups through them.
        min_samples=params.min_cluster_size,
        cluster_selection_method=params.selection_method,
    )
    # The cut overwrites distance in place.
    labels = np.asarray(cut.fit(distance).labels_, dtype=np.int32)
    del distance, cut
    positions = np.arange(n_rows, dtype=np.intp)
    ari, coverage = pair_stability(labels, reference_labels[rows], positions, positions)
    return ConsensusCheck(
        labels=labels,
        rows=rows,
        agreement=float(ari * coverage),
        subsampled=subsampled,
        n_rows=n_rows,
        bytes_used=bytes_used,
        within_association=within,
        between_association=between,
        contrast=within - between,
    )
