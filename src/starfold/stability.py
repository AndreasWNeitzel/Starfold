"""Subsample-stability diagnostic for a chosen HDBSCAN fit.

Refitting HDBSCAN on random subsamples of the UMAP embedding tells you
how sensitive the clustering is to which points happened to be in the
sample. Three readouts matter:

* The distribution of ``n_clusters`` across subsamples -- a stable fit
  should concentrate on a single value.
* The Adjusted Rand Index (ARI) between each subsample's labels and the
  reference labels on the overlap -- a stable fit reports ARI close to
  1.
* The per-cluster persistence distribution, matched across subsamples
  by majority overlap with the reference clusters.

This module deliberately subsamples *the embedding*, not the raw
features: recomputing UMAP per subsample is far more expensive than the
clustering step and its stochastic layout differences would dominate
any signal from the clustering itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics import adjusted_rand_score

from starfold._engine import Engine, resolve_engine
from starfold.clustering import run_hdbscan

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = ["SubsampleStability", "compute_subsample_stability"]


@dataclass(frozen=True)
class SubsampleStability:
    """Outcome of :func:`compute_subsample_stability`.

    Attributes
    ----------
    n_subsamples
        Number of subsamples actually run.
    subsample_fraction
        Fraction of points retained in each subsample.
    n_clusters
        ``(n_subsamples,)`` integer array of cluster counts per
        subsample.
    ari
        ``(n_subsamples,)`` float array of Adjusted Rand Index values
        against the reference labels, computed on the overlap of each
        subsample.
    persistence_sum
        ``(n_subsamples,)`` float array of summed cluster persistence
        per subsample.
    persistence_per_cluster
        ``(n_subsamples, n_reference_clusters)`` float array. Entry
        ``[i, c]`` is the persistence of the subsample-``i`` cluster
        whose points most overlap with reference cluster ``c``, or
        ``NaN`` if no such subsample cluster exists.
    """

    n_subsamples: int
    subsample_fraction: float
    n_clusters: NDArray[np.intp]
    ari: NDArray[np.floating[Any]]
    persistence_sum: NDArray[np.floating[Any]]
    persistence_per_cluster: NDArray[np.floating[Any]]


def _match_persistence(
    subsample_labels: NDArray[np.intp],
    subsample_persistence: NDArray[np.floating[Any]],
    reference_labels_on_subsample: NDArray[np.intp],
    n_reference_clusters: int,
) -> NDArray[np.floating[Any]]:
    """Return per-reference-cluster persistence for this subsample."""
    out = np.full(n_reference_clusters, np.nan, dtype=np.float64)
    if n_reference_clusters == 0 or subsample_persistence.size == 0:
        return out
    for sub_c in range(int(subsample_persistence.shape[0])):
        mask = subsample_labels == sub_c
        if not mask.any():
            continue
        matching = reference_labels_on_subsample[mask]
        matching = matching[matching >= 0]
        if matching.size == 0:
            continue
        vals, counts = np.unique(matching, return_counts=True)
        winner = int(vals[int(np.argmax(counts))])
        if winner >= n_reference_clusters:
            continue
        if np.isnan(out[winner]) or subsample_persistence[sub_c] > out[winner]:
            out[winner] = float(subsample_persistence[sub_c])
    return out


def compute_subsample_stability(
    embedding: ArrayLike,
    reference_labels: ArrayLike,
    reference_persistence: ArrayLike,
    *,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_method: str = "eom",
    cluster_selection_epsilon: float = 0.0,
    alpha: float = 1.0,
    n_subsamples: int = 50,
    subsample_fraction: float = 0.8,
    metric: str = "euclidean",
    engine: Engine = "auto",
    random_state: int | None = None,
) -> SubsampleStability:
    """Refit HDBSCAN on random subsamples of ``embedding``.

    Parameters
    ----------
    embedding : array-like of shape ``(n_samples, 2)``
        2-D embedding whose clustering is being scrutinised.
    reference_labels : array-like of shape ``(n_samples,)``
        The canonical label vector, typically ``result.labels``.
    reference_persistence : array-like of shape ``(n_reference_clusters,)``
        The canonical per-cluster persistence, typically
        ``result.persistence``. Used only to determine
        ``n_reference_clusters``.
    min_cluster_size, min_samples : int
        HDBSCAN hyperparameters to hold fixed across subsamples.
        Typically ``result.best_params``.
    n_subsamples : int, default 50
        How many subsamples to draw.
    subsample_fraction : float, default 0.8
        Fraction of points retained per subsample.
    metric : str, default ``"euclidean"``
        Metric forwarded to each HDBSCAN fit.
    engine : {"auto", "cpu", "cuml"}, default ``"auto"``
        Backend for each HDBSCAN fit.
    random_state : int or None, default None
        Seed for the row-index RNG.

    Returns
    -------
    SubsampleStability
        Per-subsample n_clusters, ARI, persistence_sum, and matched
        per-reference-cluster persistence.
    """
    emb = np.asarray(embedding, dtype=np.float64)
    ref = np.asarray(reference_labels, dtype=np.intp)
    ref_pers = np.asarray(reference_persistence, dtype=np.float64)
    if emb.ndim != 2 or emb.shape[1] != 2:
        msg = f"embedding must have shape (n_samples, 2) (got {emb.shape})."
        raise ValueError(msg)
    if ref.shape[0] != emb.shape[0]:
        msg = "reference_labels must have one entry per embedding row."
        raise ValueError(msg)
    if not 0.1 <= subsample_fraction <= 1.0:
        msg = f"subsample_fraction must be in [0.1, 1.0] (got {subsample_fraction})."
        raise ValueError(msg)
    resolved = resolve_engine(engine)
    rng = np.random.default_rng(random_state)
    n_samples = emb.shape[0]
    k = max(round(subsample_fraction * n_samples), int(min_cluster_size) + 1)
    n_ref_clusters = int(ref_pers.shape[0])

    n_clusters = np.zeros(n_subsamples, dtype=np.intp)
    ari = np.zeros(n_subsamples, dtype=np.float64)
    persistence_sum = np.zeros(n_subsamples, dtype=np.float64)
    per_cluster = np.full((n_subsamples, n_ref_clusters), np.nan, dtype=np.float64)

    for i in range(n_subsamples):
        idx = rng.choice(n_samples, size=k, replace=False)
        emb_sub = emb[idx]
        ref_sub = ref[idx]
        sub = run_hdbscan(
            emb_sub,
            min_cluster_size=int(min_cluster_size),
            min_samples=int(min_samples),
            metric=metric,
            cluster_selection_method=cluster_selection_method,
            cluster_selection_epsilon=cluster_selection_epsilon,
            alpha=alpha,
            engine=resolved,
        )
        n_clusters[i] = int(sub.n_clusters)
        persistence_sum[i] = float(np.sum(sub.cluster_persistence))
        valid = ref_sub >= 0
        if int(valid.sum()) >= 2:
            ari[i] = float(adjusted_rand_score(ref_sub[valid], sub.labels[valid]))
        else:
            ari[i] = float("nan")
        per_cluster[i] = _match_persistence(
            sub.labels,
            np.asarray(sub.cluster_persistence, dtype=np.float64),
            ref_sub,
            n_ref_clusters,
        )

    return SubsampleStability(
        n_subsamples=int(n_subsamples),
        subsample_fraction=float(subsample_fraction),
        n_clusters=n_clusters,
        ari=ari,
        persistence_sum=persistence_sum,
        persistence_per_cluster=per_cluster,
    )
