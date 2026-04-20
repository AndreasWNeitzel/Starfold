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
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics import pairwise_distances

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import ArrayLike, NDArray

__all__ = ["trustworthiness", "trustworthiness_curve"]


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


def _inverted_rank_index(
    X: NDArray[np.floating[Any]],
    metric: str,
) -> NDArray[np.intp]:
    n = X.shape[0]
    dist = pairwise_distances(X, metric=metric)
    np.fill_diagonal(dist, np.inf)
    order = np.argsort(dist, axis=1)
    del dist
    inv = np.empty((n, n), dtype=np.intp)
    rows = np.arange(n)[:, None]
    inv[rows, order] = np.arange(1, n + 1, dtype=np.intp)[None, :]
    return inv


def _topk_indices(
    X: NDArray[np.floating[Any]],
    k: int,
    metric: str,
) -> NDArray[np.intp]:
    n = X.shape[0]
    dist = pairwise_distances(X, metric=metric)
    np.fill_diagonal(dist, np.inf)
    if k >= n - 1:
        return np.argsort(dist, axis=1)[:, :k].astype(np.intp, copy=False)
    part = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
    return part.astype(np.intp, copy=False)


def _score(
    inv_high: NDArray[np.intp],
    nn_low: NDArray[np.intp],
    k: int,
    n: int,
) -> float:
    rows = np.arange(n)[:, None]
    ranks = inv_high[rows, nn_low] - k
    penalty = int(np.sum(ranks[ranks > 0]))
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
    inv_high = _inverted_rank_index(x_high, metric)
    nn_low = _topk_indices(x_low, k_int, metric)
    return _score(inv_high, nn_low, k_int, n)


def trustworthiness_curve(
    X_high: ArrayLike,
    X_low: ArrayLike,
    *,
    k_values: Iterable[int],
    metric: str = "euclidean",
) -> dict[int, float]:
    """Compute trustworthiness at several values of ``k`` in one pass.

    The inverted rank index of ``X_high`` and the sort order of ``X_low``
    are computed once and reused for every requested ``k``.

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
    inv_high = _inverted_rank_index(x_high, metric)
    dist_low = pairwise_distances(x_low, metric=metric)
    np.fill_diagonal(dist_low, np.inf)
    order_low = np.argsort(dist_low, axis=1)
    del dist_low

    out: dict[int, float] = {}
    for k_raw in k_values:
        k_int = _validate_k(k_raw, n)
        if k_int in out:
            continue
        nn_low = order_low[:, :k_int].astype(np.intp, copy=False)
        out[k_int] = _score(inv_high, nn_low, k_int, n)
    return out
