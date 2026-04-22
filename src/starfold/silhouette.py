r"""Chunked silhouette scores for HDBSCAN clusterings.

The silhouette coefficient compares, for every point :math:`x_i` inside a
cluster :math:`c`,

.. math::

    a(i) = \frac{1}{|c| - 1} \sum_{x_j \in c,\; j \neq i} d(x_i, x_j),
    \qquad
    b(i) = \min_{c' \neq c}\; \frac{1}{|c'|} \sum_{x_j \in c'} d(x_i, x_j),

and forms

.. math::
    s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}.

The naive computation builds an :math:`N \times N` pairwise-distance
matrix, which is 80 GB at :math:`N = 100\,000` and blows up before the
score is useful. The chunked implementation in this module streams the
distance rows in blocks of ``chunk`` samples at a time and accumulates a
compact ``(n_samples, n_clusters)`` matrix of per-cluster distance
sums. Peak memory is ``chunk * N * 8`` bytes for the block plus
``n_samples * n_clusters * 8`` bytes for the accumulator -- the latter
is ~24 MB at ``N = 100_000`` and ``n_clusters = 30``.

Outliers (HDBSCAN label ``-1``) are dropped from the score by
convention, matching :func:`sklearn.metrics.silhouette_score`. A
singleton cluster contributes :math:`s(i) = 0` (``a`` is undefined so
there is no within-cluster cohesion to measure). If the labelling
collapses to a single cluster the score is undefined and the function
raises :class:`ValueError`.

The result mirrors sklearn's silhouette_score at ``atol=1e-10`` on
small inputs (see ``tests/test_silhouette.py``).

Why this lives outside :mod:`starfold.clustering`: the silhouette is a
*post-hoc* quality check, not an HDBSCAN hyperparameter-search target
(DBCV-via-``relative_validity`` already plays that role). It is what
users reach for when deciding whether two clusters should be merged,
and that is exactly the question :func:`starfold.hierarchy.suggest_merges`
will answer in the next module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics import pairwise_distances

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "SilhouetteResult",
    "chunked_silhouette",
]

_DEFAULT_CHUNK_SIZE = 512


@dataclass(frozen=True)
class SilhouetteResult:
    """Outcome of :func:`chunked_silhouette`.

    Attributes
    ----------
    overall
        Mean silhouette coefficient across every non-outlier sample.
        ``NaN`` iff no samples are scored.
    per_sample
        Silhouette :math:`s(i)` for every input row. Outliers
        (label ``-1``) are given ``NaN`` so callers can compute
        non-outlier means easily.
    per_cluster
        Mean silhouette per cluster, indexed by non-negative label
        ``0..n_clusters - 1``. Singletons yield ``0.0`` by convention.
    cluster_sizes
        Size of each cluster, same shape as ``per_cluster``.
    n_outliers
        Number of samples with label ``-1``; dropped from ``overall``.
    """

    overall: float
    per_sample: NDArray[np.floating[Any]]
    per_cluster: NDArray[np.floating[Any]]
    cluster_sizes: NDArray[np.integer[Any]]
    n_outliers: int


def _validate_inputs(
    X: ArrayLike,
    labels: ArrayLike,
    metric: str,
    chunk_size: int,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.intp], int]:
    x = np.asarray(X, dtype=np.float64)
    lbl = np.asarray(labels, dtype=np.intp)
    if x.ndim != 2:
        msg = f"X must be a 2-D array (got shape {x.shape})."
        raise ValueError(msg)
    if lbl.ndim != 1:
        msg = f"labels must be a 1-D array (got shape {lbl.shape})."
        raise ValueError(msg)
    if lbl.shape[0] != x.shape[0]:
        msg = (
            "X and labels must have the same number of samples "
            f"(got {x.shape[0]} rows and {lbl.shape[0]} labels)."
        )
        raise ValueError(msg)
    if not isinstance(metric, str) or not metric:
        msg = f"metric must be a non-empty string (got {metric!r})."
        raise ValueError(msg)
    if chunk_size < 1:
        msg = f"chunk_size must be >= 1 (got {chunk_size})."
        raise ValueError(msg)
    return x, lbl, int(x.shape[0])


def _accumulate_sum_by_cluster(
    x_kept: NDArray[np.floating[Any]],
    remapped: NDArray[np.intp],
    n_clusters: int,
    metric: str,
    chunk_size: int,
) -> NDArray[np.floating[Any]]:
    """Stream the distance block and sum by cluster membership.

    Returns an ``(n_kept, n_clusters)`` matrix whose ``[i, c]`` entry is
    the sum of distances from kept row ``i`` to every kept row whose
    remapped label is ``c``.
    """
    n_kept = x_kept.shape[0]
    out = np.zeros((n_kept, n_clusters), dtype=np.float64)
    for start in range(0, n_kept, chunk_size):
        stop = min(start + chunk_size, n_kept)
        block = pairwise_distances(x_kept[start:stop], x_kept, metric=metric)
        for c in range(n_clusters):
            mask = remapped == c
            if not mask.any():
                continue
            out[start:stop, c] = block[:, mask].sum(axis=1)
    return out


def _silhouette_from_sums(
    sum_by_cluster: NDArray[np.floating[Any]],
    remapped: NDArray[np.intp],
    cluster_sizes: NDArray[np.integer[Any]],
) -> NDArray[np.floating[Any]]:
    """Compute the per-row silhouette from the per-cluster sum matrix."""
    n_kept = sum_by_cluster.shape[0]
    own = remapped
    size_own = cluster_sizes[own]
    a = np.where(
        size_own > 1,
        sum_by_cluster[np.arange(n_kept), own] / np.maximum(size_own - 1, 1),
        0.0,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        means = sum_by_cluster / cluster_sizes.astype(np.float64)[np.newaxis, :]
    means[np.arange(n_kept), own] = np.inf
    b = means.min(axis=1)
    denom = np.maximum(a, b)
    return np.where(
        (denom > 0) & (size_own > 1),
        (b - a) / np.where(denom > 0, denom, 1.0),
        0.0,
    )


def chunked_silhouette(
    X: ArrayLike,
    labels: ArrayLike,
    *,
    metric: str = "euclidean",
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> SilhouetteResult:
    r"""Silhouette coefficients computed without an ``N x N`` distance matrix.

    The silhouette :math:`s(i) = (b(i) - a(i)) / \max(a(i), b(i))`
    measures, for each sample, how much closer it is to its own
    cluster than to the nearest foreign cluster. Values near 1
    indicate well-separated clusters; values near 0 straddle a
    boundary; negative values flag samples that are likely in the
    wrong cluster.

    This implementation iterates rows in blocks of ``chunk_size`` to
    keep peak memory at ``chunk_size * N * 8`` bytes for the distance
    block plus ``N * n_clusters * 8`` bytes for the per-cluster
    accumulator. Outliers (``label == -1``) are excluded from the
    overall score and receive ``NaN`` in ``per_sample``.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Points to score. Typically the 2-D UMAP embedding from the
        pipeline, but the function is agnostic -- pass the original
        feature matrix if that is what you want to score.
    labels : array-like of shape (n_samples,)
        Cluster labels. ``-1`` marks outliers and is excluded.
    metric : str, default ``"euclidean"``
        Metric passed to :func:`sklearn.metrics.pairwise_distances`.
    chunk_size : int, default 512
        Number of rows per distance block. Controls the peak memory
        use: ``chunk_size * N * 8`` bytes.

    Returns
    -------
    SilhouetteResult
        Overall score, per-sample scores (NaN for outliers),
        per-cluster means, cluster sizes, and outlier count.

    Raises
    ------
    ValueError
        If fewer than two non-negative clusters are present.

    Examples
    --------
    >>> import numpy as np
    >>> from starfold.silhouette import chunked_silhouette
    >>> rng = np.random.default_rng(0)
    >>> X = np.vstack([rng.normal(c, 0.3, size=(30, 2)) for c in (-5.0, 5.0)])
    >>> labels = np.concatenate([np.zeros(30), np.ones(30)]).astype(int)
    >>> result = chunked_silhouette(X, labels, chunk_size=16)
    >>> bool(result.overall > 0.9)
    True
    """
    x, lbl, n = _validate_inputs(X, labels, metric, chunk_size)

    outlier_mask = lbl < 0
    n_outliers = int(outlier_mask.sum())
    kept_mask = ~outlier_mask
    kept_idx = np.flatnonzero(kept_mask).astype(np.intp)
    if kept_idx.size == 0:
        msg = "silhouette is undefined when every sample is an outlier."
        raise ValueError(msg)
    kept_labels = lbl[kept_idx]
    unique_labels = np.unique(kept_labels)
    n_clusters = int(unique_labels.shape[0])
    if n_clusters < 2:
        msg = (
            "silhouette requires at least two distinct non-outlier clusters "
            f"(got {n_clusters})."
        )
        raise ValueError(msg)
    # Remap potentially non-contiguous labels (e.g. 0, 2, 5) to
    # 0..n_clusters-1 so they index columns of the accumulator.
    remap = np.full(int(unique_labels.max()) + 1, -1, dtype=np.intp)
    remap[unique_labels] = np.arange(n_clusters, dtype=np.intp)
    remapped = remap[kept_labels]
    cluster_sizes = np.bincount(remapped, minlength=n_clusters).astype(np.intp)

    sum_by_cluster = _accumulate_sum_by_cluster(
        x[kept_idx], remapped, n_clusters, metric, chunk_size,
    )
    silhouette_kept = _silhouette_from_sums(sum_by_cluster, remapped, cluster_sizes)

    per_sample = np.full(n, np.nan, dtype=np.float64)
    per_sample[kept_idx] = silhouette_kept

    per_cluster = np.zeros(n_clusters, dtype=np.float64)
    for c in range(n_clusters):
        members = np.flatnonzero(remapped == c)
        per_cluster[c] = float(silhouette_kept[members].mean()) if members.size else 0.0

    # Re-project per_cluster back to the original (possibly non-contiguous)
    # label ordering so column c corresponds to label c -- but only when
    # the labels are already 0..n_clusters-1. Otherwise keep the remapped
    # order and expose the remap via unique_labels so callers can join.
    if np.array_equal(unique_labels, np.arange(n_clusters, dtype=unique_labels.dtype)):
        per_cluster_out = per_cluster
        cluster_sizes_out = cluster_sizes
    else:
        # Place the scores back under the original label as a dense array
        # of length max_label + 1; entries for missing labels are NaN.
        max_label = int(unique_labels.max())
        per_cluster_out = np.full(max_label + 1, np.nan, dtype=np.float64)
        per_cluster_out[unique_labels] = per_cluster
        cluster_sizes_out = np.zeros(max_label + 1, dtype=np.intp)
        cluster_sizes_out[unique_labels] = cluster_sizes

    overall = float(silhouette_kept.mean()) if silhouette_kept.size else float("nan")
    return SilhouetteResult(
        overall=overall,
        per_sample=per_sample,
        per_cluster=per_cluster_out,
        cluster_sizes=cluster_sizes_out,
        n_outliers=n_outliers,
    )
