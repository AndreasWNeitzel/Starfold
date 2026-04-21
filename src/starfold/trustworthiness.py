r"""Trustworthiness score for dimensionality-reduction embeddings.

Implements

.. math::

    T(k) = 1 - \frac{2}{N\,k\,(2N - 3k - 1)}
           \sum_{i=1}^{N} \sum_{x_j \in U_k(x_i)} \bigl(r(x_i, x_j) - k\bigr)

where :math:`N` is the number of samples, :math:`U_k(x_i)` is the set of
points that are among the :math:`k` nearest neighbours of :math:`x_i` in
the embedding but *not* among the :math:`k` nearest neighbours of
:math:`x_i` in the input, and :math:`r(x_i, x_j)` is the rank of
:math:`x_j` among all non-self points ordered by distance to
:math:`x_i` in the input (rank 1 = nearest non-self, rank :math:`N - 1` =
farthest, rank :math:`N` = self).

The formula is equivalent to :func:`sklearn.manifold.trustworthiness`; this
module is cross-tested against it. The score lies in ``[0, 1]``; 1 indicates
a perfectly faithful embedding.

The implementation streams the high-dimensional rank computation in
``_DEFAULT_CHUNK_SIZE`` rows at a time, so peak memory is
``chunk_size * N * 8`` bytes rather than ``N**2 * 8`` bytes. This keeps
``N = 25_000`` well under 1 GB at the default chunk size; the previous
dense implementation needed ~10 GB at that scale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import ArrayLike, NDArray

__all__ = ["trustworthiness", "trustworthiness_curve"]

_DEFAULT_CHUNK_SIZE = 512


def _validate_inputs(
    X_high: ArrayLike,
    X_low: ArrayLike,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], int]:
    x_high = np.asarray(X_high)
    x_low = np.asarray(X_low)
    if x_high.ndim != 2 or x_low.ndim != 2:
        msg = f"X_high and X_low must be 2-D arrays (got shapes {x_high.shape} and {x_low.shape})."
        raise ValueError(msg)
    if x_high.shape[0] != x_low.shape[0]:
        msg = (
            "X_high and X_low must have the same number of samples "
            f"(got {x_high.shape[0]} and {x_low.shape[0]})."
        )
        raise ValueError(msg)
    return x_high, x_low, int(x_high.shape[0])


def _validate_k(k: int, n: int) -> int:
    if not isinstance(k, (int, np.integer)):
        msg = f"k must be an integer (got {type(k).__name__})."
        raise TypeError(msg)
    k_int = int(k)
    if k_int < 1 or k_int >= n / 2:
        msg = (
            f"k must satisfy 1 <= k < n_samples / 2 to keep T(k) in [0, 1] "
            f"(got k={k_int}, n_samples={n})."
        )
        raise ValueError(msg)
    return k_int


def _topk_low(
    x_low: NDArray[np.floating[Any]],
    k_max: int,
    metric: str,
) -> NDArray[np.intp]:
    """Top-``k_max`` non-self neighbours of every low-D point.

    Uses :class:`sklearn.neighbors.NearestNeighbors`, which is O(N log N)
    on low-dimensional inputs and never materialises the full pairwise
    distance matrix.
    """
    nn = NearestNeighbors(n_neighbors=k_max + 1, metric=metric).fit(x_low)
    idx = nn.kneighbors(x_low, return_distance=False)
    # idx[:, 0] is the point itself (distance 0); drop it.
    return np.asarray(idx[:, 1 : k_max + 1], dtype=np.intp)


def _accumulate_penalty(
    x_high: NDArray[np.floating[Any]],
    nn_low: NDArray[np.intp],
    ks: list[int],
    metric: str,
    chunk_size: int,
) -> dict[int, int]:
    r"""Stream the high-D rank computation in row-chunks.

    For each chunk of ``chunk_size`` rows we materialise only a
    ``(chunk_size, N)`` distance block and its inverted-rank companion.
    The penalty
    :math:`\sum_{i \in \text{chunk}} \sum_{j \in U_k(x_i)} (r(x_i, x_j) - k)`
    is then accumulated for every requested ``k`` before the block is
    released, so memory peaks at ``2 * chunk_size * N * 8`` bytes.
    """
    n = x_high.shape[0]
    penalties: dict[int, int] = dict.fromkeys(ks, 0)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        chunk = x_high[start:stop]
        dist = pairwise_distances(chunk, x_high, metric=metric)
        local_rows = np.arange(stop - start)
        dist[local_rows, np.arange(start, stop)] = np.inf
        order = np.argsort(dist, axis=1)
        del dist
        inv = np.empty((stop - start, n), dtype=np.intp)
        rows = local_rows[:, None]
        inv[rows, order] = np.arange(1, n + 1, dtype=np.intp)[None, :]
        del order
        chunk_nn = nn_low[start:stop]
        for k in ks:
            ranks = inv[rows, chunk_nn[:, :k]] - k
            penalties[k] += int(np.sum(ranks[ranks > 0]))
        del inv
    return penalties


def _score_from_penalty(penalty: int, k: int, n: int) -> float:
    norm = n * k * (2.0 * n - 3.0 * k - 1.0)
    return 1.0 - penalty * (2.0 / norm)


def trustworthiness(
    X_high: ArrayLike,
    X_low: ArrayLike,
    *,
    k: int,
    metric: str = "euclidean",
) -> float:
    """Compute the trustworthiness score ``T(k)`` of an embedding.

    Parameters
    ----------
    X_high : array-like of shape (n_samples, n_features)
        Original high-dimensional data used to construct the embedding.
    X_low : array-like of shape (n_samples, n_components)
        Low-dimensional embedding of ``X_high``.
    k : int
        Number of nearest neighbours considered. Must satisfy
        ``1 <= k < n_samples / 2`` so that ``T(k)`` lies in ``[0, 1]``.
    metric : str, default ``"euclidean"``
        Distance metric passed to
        :func:`sklearn.metrics.pairwise_distances` for both spaces.

    Returns
    -------
    float
        Trustworthiness score in ``[0, 1]``. Higher is better; ``1.0`` means
        every point's ``k`` nearest neighbours in the embedding are also its
        ``k`` nearest neighbours in the input.

    Raises
    ------
    ValueError
        If the shapes are inconsistent or ``k`` is out of range.
    TypeError
        If ``k`` is not an integer.

    Examples
    --------
    >>> import numpy as np
    >>> from starfold.trustworthiness import trustworthiness
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(100, 5))
    >>> float(trustworthiness(X, X, k=5))
    1.0
    """
    x_high, x_low, n = _validate_inputs(X_high, X_low)
    k_int = _validate_k(k, n)
    nn_low = _topk_low(x_low, k_int, metric)
    penalties = _accumulate_penalty(x_high, nn_low, [k_int], metric, _DEFAULT_CHUNK_SIZE)
    return _score_from_penalty(penalties[k_int], k_int, n)


def trustworthiness_curve(
    X_high: ArrayLike,
    X_low: ArrayLike,
    *,
    k_values: Iterable[int],
    metric: str = "euclidean",
) -> dict[int, float]:
    """Compute trustworthiness at several values of ``k`` in one pass.

    The low-dimensional neighbour indices are computed once at the
    largest requested ``k``, and the high-dimensional rank stream is
    reused across every ``k`` inside each row-chunk, so the cost is
    roughly that of a single :func:`trustworthiness` call regardless of
    how many ``k`` values are supplied.

    Parameters
    ----------
    X_high : array-like of shape (n_samples, n_features)
        Original high-dimensional data.
    X_low : array-like of shape (n_samples, n_components)
        Low-dimensional embedding of ``X_high``.
    k_values : iterable of int
        Neighbourhood sizes to evaluate. Each value must satisfy
        ``1 <= k < n_samples / 2``. Duplicates are ignored; the first
        occurrence decides the order.
    metric : str, default ``"euclidean"``
        Distance metric for both spaces.

    Returns
    -------
    dict[int, float]
        Mapping ``k -> T(k)`` in the order ``k`` first appears in
        ``k_values``.

    Raises
    ------
    ValueError
        If the shapes are inconsistent or any ``k`` is out of range.

    Examples
    --------
    >>> import numpy as np
    >>> from starfold.trustworthiness import (
    ...     trustworthiness_curve,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(100, 5))
    >>> curve = trustworthiness_curve(X, X, k_values=[5, 15])
    >>> [float(curve[k]) for k in (5, 15)]
    [1.0, 1.0]
    """
    x_high, x_low, n = _validate_inputs(X_high, X_low)
    ordered_ks: list[int] = []
    for k_raw in k_values:
        k_int = _validate_k(k_raw, n)
        if k_int not in ordered_ks:
            ordered_ks.append(k_int)
    if not ordered_ks:
        return {}
    k_max = max(ordered_ks)
    nn_low = _topk_low(x_low, k_max, metric)
    penalties = _accumulate_penalty(x_high, nn_low, ordered_ks, metric, _DEFAULT_CHUNK_SIZE)
    return {k: _score_from_penalty(penalties[k], k, n) for k in ordered_ks}
