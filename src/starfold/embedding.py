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


def _fit_umap_with_model(
    X: ArrayLike,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
    n_epochs: int = 10_000,
    metric: str = "euclidean",
    n_components: int = 2,
    random_state: int | None = None,
    engine: Engine = "auto",
    low_memory: bool = False,
    n_jobs: int | None = None,
) -> tuple[NDArray[np.floating[Any]], umap.UMAP | None]:
    """Fit UMAP and return ``(embedding, reducer_or_None)``.

    Internal helper. The CPU backend returns the trained
    :class:`umap.UMAP` so downstream code can call
    :meth:`umap.UMAP.transform` on perturbed samples for uncertainty
    propagation. The cuml backend returns ``None`` for the reducer
    because this release does not wrap cuml's transform path through
    :func:`starfold.uncertainty.propagate_uncertainty`.
    """
    x = _as_2d_float(X)
    resolved: ResolvedEngine = resolve_engine(engine)
    if resolved == "cuml":
        emb = _run_umap_cuml(
            x,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_epochs=n_epochs,
            metric=metric,
            n_components=n_components,
            random_state=random_state,
        )
        return emb, None
    return _fit_umap_cpu(
        x,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_epochs=n_epochs,
        metric=metric,
        n_components=n_components,
        random_state=random_state,
        low_memory=low_memory,
        n_jobs=n_jobs,
    )


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
    low_memory: bool,
    n_jobs: int | None,
) -> NDArray[np.floating[Any]]:
    emb, _ = _fit_umap_cpu(
        x,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_epochs=n_epochs,
        metric=metric,
        n_components=n_components,
        random_state=random_state,
        low_memory=low_memory,
        n_jobs=n_jobs,
    )
    return emb


def _fit_umap_cpu(
    x: NDArray[np.floating[Any]],
    *,
    n_neighbors: int,
    min_dist: float,
    n_epochs: int,
    metric: str,
    n_components: int,
    random_state: int | None,
    low_memory: bool,
    n_jobs: int | None,
) -> tuple[NDArray[np.floating[Any]], umap.UMAP]:
    """Fit UMAP and return both the embedding and the trained reducer.

    The reducer supports :py:meth:`umap.UMAP.transform` so downstream
    code can project unseen samples (e.g. perturbations of the training
    matrix) through the same manifold without refitting.
    """
    # umap-learn forces single-threaded execution whenever random_state
    # is set so output is bit-reproducible. When the caller does not ask
    # for reproducibility (random_state is None), parallelise across all
    # cores by default -- this is the knn-graph bottleneck for large N.
    resolved_n_jobs = n_jobs if n_jobs is not None else (1 if random_state is not None else -1)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_epochs=n_epochs,
        metric=metric,
        n_components=n_components,
        random_state=random_state,
        low_memory=low_memory,
        n_jobs=resolved_n_jobs,
    )
    # Cast to float32 on the fly: umap-learn works internally in float32
    # and this halves peak kNN memory without changing output.
    x32 = x.astype(np.float32, copy=False)
    emb = np.asarray(reducer.fit_transform(x32), dtype=np.float64)
    return emb, reducer


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
    low_memory: bool = False,
    n_jobs: int | None = None,
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
    low_memory : bool, default False
        Forwarded to :class:`umap.UMAP`. Trades a modest CPU cost for
        a reduced kNN-graph memory footprint. Turn on if peak RSS is
        a concern on large ``n_samples``.
    n_jobs : int or None, default None
        CPU parallelism for the kNN build. ``None`` picks a sensible
        default: all cores when ``random_state`` is ``None``, single
        thread when a seed is supplied (so :mod:`umap-learn`'s
        bit-reproducibility guarantee is honoured).

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
        low_memory=low_memory,
        n_jobs=n_jobs,
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
    """Compute a t-SNE embedding of ``X`` (diagnostic comparison only).

    The starfold pipeline is built around UMAP: HDBSCAN, the noise
    baseline, and trustworthiness validation are all parameterised in
    terms of the UMAP embedding. :func:`run_tsne` is exposed so users
    can sanity-check whether a different manifold method shows the same
    gross structure (``plot_embedding_comparison``); it is *not*
    plumbed into :class:`~starfold.pipeline.UnsupervisedPipeline`, and
    wrapping the clustering step around a t-SNE embedding is not
    supported.

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
    """Compute a PCA projection of ``X`` (diagnostic comparison only).

    PCA is exposed as a cheap linear baseline for the embedding step:
    use it with :func:`starfold.plotting.plot_embedding_comparison` to
    see whether the non-linear UMAP manifold is actually picking up
    structure the linear projection misses. It is *not* a substitute
    for the UMAP step inside
    :class:`~starfold.pipeline.UnsupervisedPipeline`; HDBSCAN's density
    heuristics and the noise baseline are tuned for UMAP's output.

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
