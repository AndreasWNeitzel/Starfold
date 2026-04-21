"""Thin, reproducible wrappers around UMAP, t-SNE, and PCA.

Each wrapper returns a plain ``numpy.ndarray`` of shape
``(n_samples, n_components)`` and accepts a ``random_state`` argument that
is threaded through to the underlying estimator. No wrapper standardises
its input; callers are responsible for scaling (e.g. via
:class:`sklearn.preprocessing.StandardScaler`). The
:class:`~starfold.pipeline.UnsupervisedPipeline` does the scaling
internally.

:func:`run_umap` also accepts an ``engine`` selector: ``"cpu"`` uses the
reference :mod:`umap-learn` (Euclidean CPU implementation), ``"cuml"``
uses :class:`cuml.manifold.UMAP` on the GPU, and ``"auto"`` picks cuml
when importable. t-SNE and PCA are CPU-only; they exist for
documentation diagnostics, not high-throughput embedding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from starfold._engine import Engine, ResolvedEngine, resolve_engine

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = ["run_pca", "run_tsne", "run_umap"]


def _as_2d_float(X: ArrayLike) -> NDArray[np.floating[Any]]:
    x = np.asarray(X, dtype=np.float64)
    if x.ndim != 2:
        msg = f"X must be a 2-D array (got shape {x.shape})."
        raise ValueError(msg)
    return x


def _run_umap_cpu(
    x: NDArray[np.floating[Any]],
    *,
    n_neighbors: int,
    min_dist: float,
    n_epochs: int,
    metric: str,
    n_components: int,
    random_state: int | None,
) -> NDArray[np.floating[Any]]:
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_epochs=n_epochs,
        metric=metric,
        n_components=n_components,
        random_state=random_state,
    )
    return np.asarray(reducer.fit_transform(x), dtype=np.float64)


def _run_umap_cuml(
    x: NDArray[np.floating[Any]],
    *,
    n_neighbors: int,
    min_dist: float,
    n_epochs: int,
    metric: str,
    n_components: int,
    random_state: int | None,
) -> NDArray[np.floating[Any]]:
    from cuml.manifold import UMAP as CumlUMAP  # noqa: N811, PLC0415

    reducer = CumlUMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_epochs=n_epochs,
        metric=metric,
        n_components=n_components,
        random_state=random_state,
        output_type="numpy",
    )
    # cuml accepts host arrays directly; it moves them to device internally.
    emb = reducer.fit_transform(x.astype(np.float32, copy=False))
    return np.asarray(emb, dtype=np.float64)


def run_umap(
    X: ArrayLike,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
    n_epochs: int = 10_000,
    metric: str = "euclidean",
    n_components: int = 2,
    random_state: int | None = None,
    engine: Engine = "auto",
) -> NDArray[np.floating[Any]]:
    """Compute a UMAP embedding of ``X``.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data. Not standardised internally; scale the features
        beforehand if they live on different magnitudes.
    n_neighbors : int, default 15
        Size of the local neighbourhood UMAP uses to build its fuzzy
        topological representation. Larger values emphasise global
        structure.
    min_dist : float, default 0.0
        Minimum distance between points in the embedding. 0.0 follows the
        paper and packs clusters tightly, which helps HDBSCAN separate them.
    n_epochs : int, default 10_000
        Number of optimisation epochs. UMAP typically converges after a few
        hundred epochs; 10 000 is conservative per the reference paper.
    metric : str, default ``"euclidean"``
        Distance metric in the input space. ``engine="cuml"`` only
        supports a subset -- check the cuml documentation.
    n_components : int, default 2
        Dimensionality of the embedding.
    random_state : int or None, default None
        Seed. On the CPU backend with an integer seed, :mod:`umap-learn`
        falls back to single-threaded execution to guarantee bit-identical
        output across runs. On ``engine="cuml"`` the seed is threaded
        through to cuml, but GPU reductions are not bit-reproducible; two
        runs with the same seed will be very similar, not identical.
    engine : {"auto", "cpu", "cuml"}, default ``"auto"``
        Backend. ``"auto"`` prefers ``"cuml"`` when the RAPIDS
        :mod:`cuml` package is importable, and ``"cpu"`` otherwise.
        ``"cuml"`` is strict -- it raises :class:`ImportError` if
        :mod:`cuml` is missing. ``"cpu"`` pins the reference
        :mod:`umap-learn` implementation.

    Returns
    -------
    ndarray of shape (n_samples, n_components)
        The low-dimensional embedding as a contiguous ``float64`` array,
        regardless of which backend produced it.

    Examples
    --------
    >>> import numpy as np
    >>> from starfold.embedding import run_umap
    >>> X = np.random.default_rng(0).normal(size=(100, 5))
    >>> emb = run_umap(X, n_epochs=50, random_state=0, engine="cpu")
    >>> emb.shape
    (100, 2)
    """
    x = _as_2d_float(X)
    resolved: ResolvedEngine = resolve_engine(engine)
    if resolved == "cuml":
        return _run_umap_cuml(
            x,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_epochs=n_epochs,
            metric=metric,
            n_components=n_components,
            random_state=random_state,
        )
    return _run_umap_cpu(
        x,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_epochs=n_epochs,
        metric=metric,
        n_components=n_components,
        random_state=random_state,
    )


def run_tsne(
    X: ArrayLike,
    *,
    perplexity: float = 30.0,
    n_iter: int = 10_000,
    metric: str = "euclidean",
    n_components: int = 2,
    random_state: int | None = None,
) -> NDArray[np.floating[Any]]:
    """Compute a t-SNE embedding of ``X``.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data. Not standardised internally.
    perplexity : float, default 30.0
        Effective number of nearest neighbours that t-SNE tries to preserve.
    n_iter : int, default 10_000
        Maximum number of gradient-descent iterations.
    metric : str, default ``"euclidean"``
        Distance metric in the input space.
    n_components : int, default 2
        Dimensionality of the embedding.
    random_state : int or None, default None
        Seed for reproducibility.

    Returns
    -------
    ndarray of shape (n_samples, n_components)
        The low-dimensional embedding.

    Examples
    --------
    >>> import numpy as np
    >>> from starfold.embedding import run_tsne
    >>> X = np.random.default_rng(0).normal(size=(100, 5))
    >>> emb = run_tsne(X, n_iter=250, random_state=0)
    >>> emb.shape
    (100, 2)
    """
    x = _as_2d_float(X)
    reducer = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        max_iter=n_iter,
        metric=metric,
        random_state=random_state,
    )
    return np.asarray(reducer.fit_transform(x), dtype=np.float64)


def run_pca(
    X: ArrayLike,
    *,
    n_components: int = 2,
    random_state: int | None = None,
) -> NDArray[np.floating[Any]]:
    """Compute a PCA projection of ``X``.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data. Not standardised internally.
    n_components : int, default 2
        Number of principal components to retain. Columns are ordered by
        decreasing variance explained.
    random_state : int or None, default None
        Seed for the randomised SVD solver (used by scikit-learn when
        ``n_samples`` is large).

    Returns
    -------
    ndarray of shape (n_samples, n_components)
        The projected data.

    Examples
    --------
    >>> import numpy as np
    >>> from starfold.embedding import run_pca
    >>> X = np.random.default_rng(0).normal(size=(100, 5))
    >>> emb = run_pca(X, n_components=2)
    >>> emb.shape
    (100, 2)
    """
    x = _as_2d_float(X)
    reducer = PCA(n_components=n_components, random_state=random_state)
    return np.asarray(reducer.fit_transform(x), dtype=np.float64)
