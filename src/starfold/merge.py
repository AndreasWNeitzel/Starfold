"""Cluster-merge recommender.

HDBSCAN's condensed tree records *why* every pair of flat clusters
looks like two clusters rather than one: each pair has a ``merge_lambda``
(density threshold at which they rejoin into a common ancestor) and two
``birth_lambda`` values (density thresholds at which each detached from
its parent). When ``merge_lambda == max(birth_lambda_i, birth_lambda_j)``
the two clusters are *direct siblings* of one single split and the
ratio ``merge_lambda / max(birth_lambda_i, birth_lambda_j)`` hits 1.
For pairs that meet higher up in the tree (one is a cousin, a nephew,
or an ancestor-adjacent lineage) the merge happens at a smaller lambda
than the deeper cluster's birth, so the ratio falls below 1. A value
close to 1 therefore flags a near-sibling relationship that a user is
most likely to want to merge; a value close to 0 flags clusters that
only meet at a much sparser density level.

Note that ``min`` in the denominator is *not* informative: in any
cascaded condensed tree ``merge_lambda`` always equals the earlier-born
cluster's birth, so ``merge_lambda / min(birth_lambda_i, birth_lambda_j)
== 1`` for every pair. ``max`` is used instead for a non-degenerate signal.

That hierarchical signal alone is not quite enough: two siblings can
have a high cohesion ratio yet still sit cleanly apart in the embedding
(UMAP can spread a narrow density split into a wide geometric gap).
A paired geometric check -- centroid gap in the embedding divided by
intra-cluster RMS dispersion -- catches that failure mode. Both
signals must agree before a merge is *recommended*; both are returned
so users with stronger domain priors can override the default
thresholds.

This module provides the recommender as a pure function and as a thin
``PipelineResult.suggest_merges`` method. It does not mutate the flat
labelling for users -- merging is a scientific decision, not an
automatic one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from starfold.hierarchy import HierarchicalStructure

__all__ = [
    "MergeSuggestion",
    "suggest_merges",
]


@dataclass(frozen=True)
class MergeSuggestion:
    r"""One candidate cluster merge.

    Attributes
    ----------
    cluster_i, cluster_j
        The pair of flat cluster ids being compared. ``cluster_i <
        cluster_j`` by construction so the caller does not see both
        ``(a, b)`` and ``(b, a)``.
    merge_lambda
        HDBSCAN :math:`\\lambda` at which ``i`` and ``j`` rejoin the
        common ancestor in the condensed tree. Larger = denser-level
        merge.
    birth_lambda_i, birth_lambda_j
        :math:`\\lambda` at which each cluster first detached from its
        parent. Always :math:`\\ge` ``merge_lambda``.
    cohesion_ratio
        ``merge_lambda / max(birth_lambda_i, birth_lambda_j)`` in
        ``[0, 1]``. Equals 1 iff the pair are direct siblings of one
        condensed-tree split (both were born at the exact same lambda).
        Values below 1 mean the pair meet higher up in the tree -- the
        deeper cluster was already alive when the shallower one split
        off -- and the two are progressively less sibling-like as the
        ratio drops. Using ``max`` rather than ``min`` is important:
        ``min`` is degenerate on cascaded trees and yields 1 for every
        pair regardless of structure.
    centroid_gap
        Euclidean distance between the two cluster centroids in the
        2-D embedding. Units are embedding units.
    intra_dispersion
        ``max`` of the two per-cluster RMS distances from centroid. Used
        as the geometric scale the gap is compared against.
    gap_ratio
        ``centroid_gap / intra_dispersion``. Values below 1 mean the
        clusters overlap substantially in the embedding; values above
        ~2 mean they are clearly separated.
    recommended
        ``cohesion_ratio > cohesion_threshold`` *and* ``gap_ratio <
        gap_threshold``. Both signals agree the pair is one cluster.
    """

    cluster_i: int
    cluster_j: int
    merge_lambda: float
    birth_lambda_i: float
    birth_lambda_j: float
    cohesion_ratio: float
    centroid_gap: float
    intra_dispersion: float
    gap_ratio: float
    recommended: bool


def _cohesion_ratio(
    merge_lambda: float,
    birth_i: float,
    birth_j: float,
) -> float:
    """Return ``merge / max(birth_i, birth_j)`` or NaN if undefined.

    ``max`` is deliberate: ``min`` collapses to 1 on every cascaded
    condensed tree and gives no signal. ``max`` is 1 iff i and j are
    direct siblings and drops toward 0 as the deeper cluster's birth
    moves further past the shared split lambda.
    """
    if not (np.isfinite(merge_lambda) and np.isfinite(birth_i) and np.isfinite(birth_j)):
        return float("nan")
    denom = max(birth_i, birth_j)
    return float(merge_lambda / denom) if denom > 0 else float("nan")


def _geometry(
    ci: NDArray[np.floating[Any]],
    cj: NDArray[np.floating[Any]],
    rms_i: float,
    rms_j: float,
) -> tuple[float, float, float]:
    """Return (centroid_gap, intra_dispersion, gap_ratio)."""
    if np.any(np.isnan(ci)) or np.any(np.isnan(cj)):
        return float("nan"), float("nan"), float("nan")
    gap = float(np.linalg.norm(ci - cj))
    dispersion = float(max(rms_i, rms_j))
    gap_ratio = float(gap / dispersion) if dispersion > 0 else float("inf")
    return gap, dispersion, gap_ratio


def _one_suggestion(
    *,
    i: int,
    j: int,
    hierarchy: HierarchicalStructure,
    births: NDArray[np.floating[Any]],
    centroids: list[NDArray[np.floating[Any]]],
    rms: list[float],
    cohesion_threshold: float,
    gap_threshold: float,
) -> MergeSuggestion:
    m = hierarchy.merge_lambda(i, j)
    bi, bj = births[i], births[j]
    cohesion = _cohesion_ratio(m, bi, bj)
    gap, dispersion, gap_ratio = _geometry(centroids[i], centroids[j], rms[i], rms[j])
    recommended = bool(
        np.isfinite(cohesion)
        and np.isfinite(gap_ratio)
        and cohesion > cohesion_threshold
        and gap_ratio < gap_threshold
    )
    return MergeSuggestion(
        cluster_i=int(i),
        cluster_j=int(j),
        merge_lambda=float(m) if np.isfinite(m) else float("nan"),
        birth_lambda_i=float(bi) if np.isfinite(bi) else float("nan"),
        birth_lambda_j=float(bj) if np.isfinite(bj) else float("nan"),
        cohesion_ratio=cohesion,
        centroid_gap=gap,
        intra_dispersion=dispersion,
        gap_ratio=gap_ratio,
        recommended=recommended,
    )


def _birth_lambda_of(hierarchy: HierarchicalStructure, cluster_id: int) -> float:
    """Return the lambda on the edge that gives birth to ``cluster_id``.

    Falls back to NaN when the node is unknown (cuml, or a flat cluster
    whose node could not be recovered from the condensed tree).
    """
    node = int(hierarchy.flat_to_node[cluster_id])
    if node < 0:
        return float("nan")
    child_col = hierarchy.edges["child"]
    lam_col = hierarchy.edges["lambda_val"]
    hit = np.flatnonzero(child_col == node)
    if hit.size == 0:
        return float("nan")
    return float(lam_col[int(hit[0])])


def _cluster_centroid_and_rms(
    embedding: NDArray[np.floating[Any]],
    labels: NDArray[np.intp],
    cluster_id: int,
) -> tuple[NDArray[np.floating[Any]], float]:
    mask = labels == cluster_id
    points = embedding[mask]
    if points.shape[0] == 0:
        return np.full(embedding.shape[1], np.nan, dtype=np.float64), 0.0
    centroid = points.mean(axis=0)
    if points.shape[0] == 1:
        return centroid, 0.0
    rms = float(np.sqrt(np.mean(np.sum((points - centroid) ** 2, axis=1))))
    return centroid, rms


def suggest_merges(
    hierarchy: HierarchicalStructure,
    embedding: ArrayLike,
    *,
    cohesion_threshold: float = 0.9,
    gap_threshold: float = 2.0,
    sort_by: Literal["cohesion_ratio", "gap_ratio"] = "cohesion_ratio",
) -> list[MergeSuggestion]:
    """Rank every pair of flat clusters by how mergeable they look.

    Two signals are combined:

    * A *hierarchical* signal from HDBSCAN's condensed tree.
      ``cohesion_ratio = merge_lambda / max(birth_lambda_i,
      birth_lambda_j)`` equals 1 iff i and j are direct siblings of
      the same split and drops toward 0 as the pair's common ancestor
      moves further up in the tree. Only direct siblings can be
      merged tree-consistently; ratios near 1 flag those candidates.
    * A *geometric* signal from the 2-D embedding.
      ``gap_ratio = centroid_gap / max(rms_i, rms_j)`` measures whether
      the clusters actually sit on top of each other in the clustered
      space. A gap much smaller than the intra-cluster spread means the
      boundary is thin.

    A pair is ``recommended`` only when both signals agree. Callers
    with stronger priors (domain knowledge, a supervised validation
    set, ...) can inspect the returned signals and override the default
    thresholds. The thresholds are defaults, not laws -- document any
    non-default choice alongside the recommendation.

    Parameters
    ----------
    hierarchy : HierarchicalStructure
        The condensed-tree view from a :class:`PipelineResult`.
        Unavailable hierarchies (cuml fits) raise :class:`RuntimeError`.
    embedding : array-like of shape (n_samples, 2)
        The 2-D embedding the clustering was computed on. Used for the
        geometric signal.
    cohesion_threshold : float in (0, 1], default 0.9
        ``cohesion_ratio`` above which the hierarchical signal votes
        "merge". The default 0.9 is strict because the metric is only
        1 for *direct* siblings; 0.9 still admits near-sibling pairs
        from shallow cascaded splits. Lower it (say 0.6) if you want
        to surface more distant pairs for manual inspection.
    gap_threshold : float, default 2.0
        ``gap_ratio`` below which the geometric signal votes "merge".
        The default demands the centroids be closer than ~2x the
        intra-cluster RMS.
    sort_by : {"cohesion_ratio", "gap_ratio"}, default "cohesion_ratio"
        Sort key. ``cohesion_ratio`` descending puts tree-supported
        merges first; ``gap_ratio`` ascending puts geometrically
        overlapping pairs first.

    Returns
    -------
    list of MergeSuggestion
        One entry per unordered pair ``(i, j)`` of flat clusters, sorted
        by the chosen key. Empty if the hierarchy carries fewer than two
        flat clusters.

    Raises
    ------
    RuntimeError
        If ``hierarchy.available`` is ``False`` (cuml fits).
    ValueError
        If ``embedding`` has the wrong shape or sort_by is invalid.
    """
    from starfold.hierarchy import HierarchicalStructure  # noqa: F401, PLC0415

    if not hierarchy.available:
        msg = (
            "suggest_merges requires a HierarchicalStructure from a "
            "CPU-backend fit. Refit with engine='cpu' to enable merge "
            "recommendations."
        )
        raise RuntimeError(msg)
    if sort_by not in ("cohesion_ratio", "gap_ratio"):
        msg = f"sort_by must be 'cohesion_ratio' or 'gap_ratio' (got {sort_by!r})."
        raise ValueError(msg)
    if not 0.0 < cohesion_threshold <= 1.0:
        msg = f"cohesion_threshold must lie in (0, 1] (got {cohesion_threshold})."
        raise ValueError(msg)
    if gap_threshold <= 0.0:
        msg = f"gap_threshold must be > 0 (got {gap_threshold})."
        raise ValueError(msg)

    emb = np.asarray(embedding, dtype=np.float64)
    if emb.ndim != 2 or emb.shape[0] != hierarchy.n_samples:
        msg = (
            "embedding must be 2-D with the same n_samples the "
            "hierarchy was built from "
            f"(got shape {emb.shape}, expected ({hierarchy.n_samples}, d))."
        )
        raise ValueError(msg)

    n_flat = int(hierarchy.flat_to_node.size)
    if n_flat < 2:
        return []

    labels = hierarchy.labels
    births = np.array([_birth_lambda_of(hierarchy, k) for k in range(n_flat)], dtype=np.float64)
    centroids: list[NDArray[np.floating[Any]]] = []
    rms: list[float] = []
    for k in range(n_flat):
        c, r = _cluster_centroid_and_rms(emb, labels, k)
        centroids.append(c)
        rms.append(r)

    suggestions: list[MergeSuggestion] = [
        _one_suggestion(
            i=i,
            j=j,
            hierarchy=hierarchy,
            births=births,
            centroids=centroids,
            rms=rms,
            cohesion_threshold=cohesion_threshold,
            gap_threshold=gap_threshold,
        )
        for i in range(n_flat)
        for j in range(i + 1, n_flat)
    ]

    def _cohesion_key(s: MergeSuggestion) -> float:
        return -s.cohesion_ratio if np.isfinite(s.cohesion_ratio) else np.inf

    def _gap_key(s: MergeSuggestion) -> float:
        return s.gap_ratio if np.isfinite(s.gap_ratio) else np.inf

    suggestions.sort(key=_cohesion_key if sort_by == "cohesion_ratio" else _gap_key)
    return suggestions
