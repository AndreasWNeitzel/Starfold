"""The only module that calls UMAP and HDBSCAN.

Neighbour graphs are built once per row set and sliced to the ``n_neighbors``
of each trial for UMAP's ``precomputed_knn``. UMAP (McInnes, Healy and
Melville 2018, arXiv:1802.03426) and HDBSCAN (Campello, Moulavi and Sander
2013, doi:10.1007/978-3-642-37456-2_14; McInnes and Healy 2017,
arXiv:1705.07321) run on the CPU reference libraries or on RAPIDS cuML.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import scipy.sparse
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors

from starfold._engine import (
    derive_seed,
    release_backend_caches,
    require_resolved,
    resolve_engine,
)
from starfold.prep import Scaling, scale, validate_matrix

if TYPE_CHECKING:
    from collections.abc import Mapping

    from starfold._engine import ResolvedEngine

__all__ = [
    "BackendAgreement",
    "Fit",
    "KnnGraph",
    "Params",
    "compare_backends",
    "default_params",
    "fit_hdbscan",
    "fit_pipeline",
    "fit_umap",
    "knn_graph",
    "neighbor_adjacency",
    "rescale_params",
]

CUML_MAX_MIN_SAMPLES = 1023

SelectionMethod = Literal["eom", "leaf"]
KnnGraph = tuple[NDArray[np.int32], NDArray[np.float32]]

_BRUTE_FORCE_MAX_ROWS = 20_000
_QUERY_CHUNK_BYTES = 64 * 2**20


@dataclass(frozen=True)
class Params:
    """One complete UMAP and HDBSCAN configuration.

    Attributes
    ----------
    n_neighbors : int
        UMAP neighbourhood size, at least 2.
    min_dist : float
        UMAP minimum embedding distance in ``[0, 1]``.
    n_components : int
        Embedding dimension, at least 1.
    min_cluster_size : int
        HDBSCAN minimum cluster size in rows, at least 2.
    min_samples : int
        HDBSCAN core-distance neighbour count in ``[1, min_cluster_size]``.
    selection_method : {"eom", "leaf"}
        HDBSCAN cluster selection rule.

    Raises
    ------
    ValueError
        If a field lies outside the range stated above.
    """

    n_neighbors: int
    min_dist: float
    n_components: int
    min_cluster_size: int
    min_samples: int
    selection_method: SelectionMethod

    def __post_init__(self) -> None:
        """Validate every field against the ranges of the umap-learn and hdbscan contracts."""
        checks = (
            (self.n_neighbors >= 2, "n_neighbors must be at least 2"),
            (0.0 <= self.min_dist <= 1.0, "min_dist must lie in [0, 1]"),
            (self.n_components >= 1, "n_components must be at least 1"),
            (self.min_cluster_size >= 2, "min_cluster_size must be at least 2"),
            (
                1 <= self.min_samples <= self.min_cluster_size,
                "min_samples must lie in [1, min_cluster_size]",
            ),
            (self.selection_method in ("eom", "leaf"), "selection_method must be 'eom' or 'leaf'"),
        )
        for passed, message in checks:
            if not passed:
                msg = f"{message} (got {self})."
                raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        """Return the fields as a JSON-ready mapping.

        Returns
        -------
        dict[str, object]
            Field names to values.
        """
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Params:
        """Rebuild a configuration from a mapping such as a JSON record.

        Parameters
        ----------
        d : mapping
            Field names to values, with integral floats accepted for int fields.

        Returns
        -------
        Params
            Validated configuration.

        Raises
        ------
        ValueError
            If a field is missing, non-numeric, non-integral, or out of range.
        """
        try:
            return cls(
                n_neighbors=_as_int(d["n_neighbors"]),
                min_dist=_as_float(d["min_dist"]),
                n_components=_as_int(d["n_components"]),
                min_cluster_size=_as_int(d["min_cluster_size"]),
                min_samples=_as_int(d["min_samples"]),
                selection_method=cast("SelectionMethod", d["selection_method"]),
            )
        except (KeyError, TypeError) as error:
            msg = f"Params mapping is invalid: {error}"
            raise ValueError(msg) from error


def _as_float(value: object) -> float:
    """Convert a real scalar to float, raising TypeError for bools and non-numeric objects."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        msg = f"expected a real number (got {value!r})."
        raise TypeError(msg)
    return float(value)


def _as_int(value: object) -> int:
    """Convert an integral scalar to int, rejecting fractions."""
    number = _as_float(value)
    if not number.is_integer():
        msg = f"expected an integer (got {value!r})."
        raise ValueError(msg)
    return int(number)


def default_params() -> Params:
    """Return the library defaults of umap-learn and hdbscan.

    Returns
    -------
    Params
        ``Params(15, 0.1, 2, 5, 5, "eom")``, used as trial 0 of every search
        after clipping into the declared space.
    """
    return Params(15, 0.1, 2, 5, 5, "eom")


def rescale_params(params: Params, from_rows: int, to_rows: int) -> Params:
    """Return the configuration at a different row count, keeping the mass fraction.

    ``min_cluster_size`` and ``min_samples`` scale by ``to_rows / from_rows``,
    with ``min_cluster_size`` clipped to ``[2, to_rows]`` and ``min_samples``
    to ``[1, min_cluster_size]`` and at most the cuML limit of 1023, so the
    smallest admissible population keeps the mass fraction the search
    validated. When ``to_rows`` is 1, the lower bound of 2 wins.
    ``n_neighbors``, ``min_dist`` and ``n_components`` describe local geometry
    and are unchanged.

    Parameters
    ----------
    params : Params
        Configuration validated on ``from_rows`` rows.
    from_rows, to_rows : int
        Row counts, both at least 1.

    Returns
    -------
    Params
        The rescaled configuration, unchanged when the row counts are equal.

    Raises
    ------
    ValueError
        If either row count is below 1.
    """
    if from_rows < 1 or to_rows < 1:
        msg = f"row counts must be at least 1 (got {from_rows} and {to_rows})."
        raise ValueError(msg)
    if from_rows == to_rows:
        return params
    factor = to_rows / from_rows
    size = max(2, min(round(params.min_cluster_size * factor), to_rows))
    samples = max(1, min(round(params.min_samples * factor), size, CUML_MAX_MIN_SAMPLES))
    return replace(params, min_cluster_size=size, min_samples=samples)


@dataclass(frozen=True, eq=False)
class Fit:
    """One UMAP and HDBSCAN fit on one row set.

    Attributes
    ----------
    labels : ndarray of shape (m,), int32
        Cluster ids with ``-1`` for noise.
    strengths : ndarray of shape (m,), float32
        HDBSCAN ``probabilities_``: within-fit strengths, not posteriors.
    embedding : ndarray of shape (m, n_components), float32
        UMAP coordinates.
    n_groups : int
        Number of distinct non-noise labels.
    seconds : float
        Total wall time of scaling, graph, UMAP and HDBSCAN.
    """

    labels: NDArray[np.int32]
    strengths: NDArray[np.float32]
    embedding: NDArray[np.float32]
    n_groups: int
    seconds: float


def _self_first(indices: NDArray[np.int32], distances: NDArray[np.float32]) -> KnnGraph:
    """Put each row's own index in column 0 at distance 0, as umap-learn expects."""
    rows = np.arange(indices.shape[0], dtype=np.int32)
    for i in np.flatnonzero(indices[:, 0] != rows):
        hits = np.flatnonzero(indices[i] == i)
        stop = int(hits[0]) if hits.size else indices.shape[1] - 1
        indices[i, 1 : stop + 1] = indices[i, :stop].copy()
        distances[i, 1 : stop + 1] = distances[i, :stop].copy()
        indices[i, 0] = i
        distances[i, 0] = 0.0
    return indices, distances


def _brute_knn(x: NDArray[np.float32], k: int, *, n_jobs: int) -> KnnGraph:
    """Exact Euclidean neighbours with chunked queries, never an m by m matrix."""
    m = x.shape[0]
    searcher = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean", n_jobs=n_jobs)
    searcher.fit(x)
    chunk = max(1, _QUERY_CHUNK_BYTES // (8 * m))
    indices = np.empty((m, k), dtype=np.int32)
    distances = np.empty((m, k), dtype=np.float32)
    for start in range(0, m, chunk):
        stop = min(start + chunk, m)
        dist, idx = searcher.kneighbors(x[start:stop])
        indices[start:stop] = idx
        distances[start:stop] = dist
    return indices, distances


def _descent_knn(x: NDArray[np.float32], k: int, *, seed: int, n_jobs: int) -> KnnGraph:
    """Seeded approximate neighbours (Dong, Moses and Li 2011, doi:10.1145/1963405.1963487)."""
    from pynndescent import NNDescent  # noqa: PLC0415 - compiled import deferred to first use

    index = NNDescent(x, n_neighbors=k, metric="euclidean", random_state=seed, n_jobs=n_jobs)
    indices, distances = index.neighbor_graph
    return np.asarray(indices, dtype=np.int32), np.asarray(distances, dtype=np.float32)


def _cuml_knn(x: NDArray[np.float32], k: int) -> KnnGraph:
    """Exact Euclidean neighbours on the GPU."""
    import cuml.neighbors  # noqa: PLC0415

    searcher = cuml.neighbors.NearestNeighbors(
        n_neighbors=k, algorithm="brute", metric="euclidean", output_type="numpy"
    )
    searcher.fit(x)
    distances, indices = searcher.kneighbors(x)
    return np.asarray(indices, dtype=np.int32), np.asarray(distances, dtype=np.float32)


def knn_graph(
    x_scaled: NDArray[np.float32],
    k: int,
    *,
    engine: ResolvedEngine,
    seed: int,
    n_jobs: int = 1,
) -> KnnGraph:
    """Build the Euclidean k-nearest-neighbour graph of one scaled row set.

    Parameters
    ----------
    x_scaled : ndarray of shape (m, d)
        Scaled rows.
    k : int
        Columns requested, capped at ``m``. Column 0 holds the row itself at
        distance 0, with distances ascending along each row.
    engine : {"cpu", "cuml"}
        CPU uses exact brute force in row chunks for ``m <= 20000`` and seeded
        pynndescent above, while cuML uses brute force on the device. Any other
        value raises ``ValueError``.
    seed : int
        Seed of the approximate search.
    n_jobs : int, default 1
        Worker count of the CPU search.

    Returns
    -------
    indices : ndarray of shape (m, k), int32
    distances : ndarray of shape (m, k), float32

    Raises
    ------
    ValueError
        If ``x_scaled`` is not two-dimensional, ``k < 1`` or ``engine`` is not
        resolved.
    """
    engine = require_resolved(engine)
    x = np.ascontiguousarray(x_scaled, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] < 1:
        msg = f"x_scaled must be a non-empty two-dimensional array (got shape {x.shape})."
        raise ValueError(msg)
    if k < 1:
        msg = f"k must be at least 1 (got {k})."
        raise ValueError(msg)
    k = min(k, x.shape[0])
    if engine == "cuml":
        indices, distances = _cuml_knn(x, k)
    elif x.shape[0] <= _BRUTE_FORCE_MAX_ROWS:
        indices, distances = _brute_knn(x, k, n_jobs=n_jobs)
    else:
        indices, distances = _descent_knn(x, k, seed=seed, n_jobs=n_jobs)
    return _self_first(indices, distances)


def neighbor_adjacency(knn: KnnGraph, k: int = 15) -> scipy.sparse.csr_matrix:
    """Return the symmetrised unweighted adjacency of the first ``k`` neighbours.

    Parameters
    ----------
    knn : KnnGraph
        Graph whose column 0 is the row itself. Columns ``1..k`` are used,
        giving each row ``k`` neighbours before symmetrisation (fewer when the
        graph is narrower).
    k : int, default 15
        Neighbours per row.

    Returns
    -------
    scipy.sparse.csr_matrix of shape (m, m), float32
        Entries are 1 where either row lists the other, with no self loops.
    """
    indices, _distances = knn
    m, width = indices.shape
    kk = max(0, min(k, width - 1))
    rows = np.repeat(np.arange(m, dtype=np.int32), kk)
    cols = indices[:, 1 : kk + 1].ravel()
    keep = (cols >= 0) & (cols != rows)
    data = np.ones(int(keep.sum()), dtype=np.float32)
    half = scipy.sparse.csr_matrix((data, (rows[keep], cols[keep])), shape=(m, m))
    full = half + half.T
    full.data[:] = 1.0
    full.sort_indices()
    return full.astype(np.float32)


def _slice_knn(knn: KnnGraph, n_neighbors: int, m: int) -> KnnGraph:
    """Copy the first ``n_neighbors`` columns so UMAP can never mutate the cache."""
    indices, distances = knn
    if indices.ndim != 2 or indices.shape != distances.shape:
        msg = f"knn arrays must share a 2-D shape (got {indices.shape} and {distances.shape})."
        raise ValueError(msg)
    if indices.shape[0] != m:
        msg = f"knn has {indices.shape[0]} rows but the data has {m}."
        raise ValueError(msg)
    if indices.shape[1] < n_neighbors:
        msg = f"knn has {indices.shape[1]} columns, fewer than n_neighbors={n_neighbors}."
        raise ValueError(msg)
    return (
        np.array(indices[:, :n_neighbors], dtype=np.int32, order="C"),
        np.array(distances[:, :n_neighbors], dtype=np.float32, order="C"),
    )


def _cuml_umap(
    x: NDArray[np.float32], params: Params, seed: int, graph: KnnGraph | None
) -> NDArray[np.float32]:
    """Deterministic cuML UMAP with an optional host precomputed graph."""
    import cuml.manifold  # noqa: PLC0415

    reducer = cuml.manifold.UMAP(
        n_neighbors=params.n_neighbors,
        n_components=params.n_components,
        min_dist=params.min_dist,
        metric="euclidean",
        random_state=seed,
        precomputed_knn=graph,
        output_type="numpy",
    )
    return np.ascontiguousarray(reducer.fit_transform(x), dtype=np.float32)


def fit_umap(
    x_scaled: NDArray[np.float32],
    params: Params,
    *,
    seed: int,
    engine: ResolvedEngine,
    knn: KnnGraph | None = None,
) -> NDArray[np.float32]:
    """Embed scaled rows with UMAP under a fixed seed.

    Parameters
    ----------
    x_scaled : ndarray of shape (m, d)
        Scaled rows.
    params : Params
        Configuration; only the UMAP fields are read.
    seed : int
        ``random_state`` of the layout; umap-learn then runs serially.
    engine : {"cpu", "cuml"}
        Backend. Any other value raises ``ValueError``.
    knn : KnnGraph, optional
        Graph sliced to ``params.n_neighbors`` and passed as ``precomputed_knn``
        on both backends. A silent fallback of the library to its own search
        is turned into an error.

    Returns
    -------
    ndarray of shape (m, n_components), float32
        Embedding.

    Raises
    ------
    ValueError
        If ``knn`` has fewer columns than ``n_neighbors`` or a row count that
        differs from ``x_scaled``, or ``engine`` is not resolved.
    """
    engine = require_resolved(engine)
    x = np.ascontiguousarray(x_scaled, dtype=np.float32)
    graph = None if knn is None else _slice_knn(knn, params.n_neighbors, x.shape[0])
    if engine == "cuml":
        return _cuml_umap(x, params, seed, graph)
    import umap  # noqa: PLC0415 - compiled import deferred to first use

    reducer = umap.UMAP(
        n_neighbors=params.n_neighbors,
        min_dist=params.min_dist,
        n_components=params.n_components,
        metric="euclidean",
        random_state=seed,
        n_jobs=1,
        precomputed_knn=(None, None, None) if graph is None else graph,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"precomputed_knn\[2\]")
        warnings.filterwarnings("error", message="precomputed_knn has")
        embedding = reducer.fit_transform(x)
    return np.ascontiguousarray(embedding, dtype=np.float32)


def fit_hdbscan(
    embedding: NDArray[np.float32], params: Params, *, engine: ResolvedEngine
) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    """Cluster an embedding with HDBSCAN.

    Parameters
    ----------
    embedding : ndarray of shape (m, n_components)
        UMAP coordinates.
    params : Params
        Configuration; only the HDBSCAN fields are read.
    engine : {"cpu", "cuml"}
        ``hdbscan.HDBSCAN`` (single-threaded core distances) or
        ``cuml.cluster.HDBSCAN``, both Euclidean with no spanning tree, no
        prediction data and ``allow_single_cluster=False``. Any other value
        raises ``ValueError``.

    Returns
    -------
    labels : ndarray of shape (m,), int32
        Cluster ids with ``-1`` for noise.
    strengths : ndarray of shape (m,), float32
        HDBSCAN ``probabilities_``.

    Raises
    ------
    ValueError
        If ``engine`` is not ``"cpu"`` or ``"cuml"``.
    """
    engine = require_resolved(engine)
    y = np.ascontiguousarray(embedding, dtype=np.float32)
    common = {
        "min_cluster_size": params.min_cluster_size,
        "min_samples": params.min_samples,
        "metric": "euclidean",
        "cluster_selection_method": params.selection_method,
        "allow_single_cluster": False,
        "gen_min_span_tree": False,
        "prediction_data": False,
    }
    if engine == "cuml":
        import cuml.cluster  # noqa: PLC0415

        clusterer = cuml.cluster.HDBSCAN(**common, output_type="numpy").fit(y)
    else:
        import hdbscan  # noqa: PLC0415 - compiled import deferred to first use

        clusterer = hdbscan.HDBSCAN(**common, core_dist_n_jobs=1).fit(y)
    labels = np.asarray(clusterer.labels_, dtype=np.int32)
    strengths = np.asarray(clusterer.probabilities_, dtype=np.float32)
    return labels, strengths


def fit_pipeline(
    x_rows: NDArray[np.float64],
    params: Params,
    *,
    seed: int,
    engine: ResolvedEngine,
    scaling: Scaling,
    knn: KnnGraph | None = None,
) -> Fit:
    """Scale, build or reuse the neighbour graph, embed and cluster one row set.

    Parameters
    ----------
    x_rows : ndarray of shape (m, d), float64
        Rows in original units. The scaler is fitted on these rows alone.
    params : Params
        Configuration.
    seed : int
        UMAP seed and, when a graph is built, the approximate-search seed.
    engine : {"cpu", "cuml"}
        Backend. Any other value raises ``ValueError`` before any work.
    scaling : {"standard", "robust"}
        Scaler family.
    knn : KnnGraph, optional
        Cached graph of the scaled rows, built at ``k = n_neighbors`` if absent.

    Returns
    -------
    Fit
        Labels, strengths, embedding, group count and wall seconds.

    Raises
    ------
    ValueError
        If ``engine`` is not ``"cpu"`` or ``"cuml"``. Exceptions from the
        scaler, UMAP and HDBSCAN propagate unchanged.
    """
    start = time.perf_counter()
    engine = require_resolved(engine)
    x_scaled = scale(x_rows, scaling)
    graph = (
        knn_graph(x_scaled, params.n_neighbors, engine=engine, seed=seed) if knn is None else knn
    )
    embedding = fit_umap(x_scaled, params, seed=seed, engine=engine, knn=graph)
    labels, strengths = fit_hdbscan(embedding, params, engine=engine)
    n_groups = int(np.unique(labels[labels >= 0]).size)
    return Fit(
        labels=labels,
        strengths=strengths,
        embedding=embedding,
        n_groups=n_groups,
        seconds=time.perf_counter() - start,
    )


@dataclass(frozen=True)
class BackendAgreement:
    """Partition agreement between the CPU and cuML backends.

    Attributes
    ----------
    agreement : tuple of float
        Per seed, adjusted Rand index times joint coverage between the two fits.
    mean_agreement : float
        Mean of ``agreement``.
    ari, coverage : tuple of float
        The two factors of ``agreement`` per seed. A low product can mean that
        the backends disagree or only that few rows are assigned by both.
    assigned_cpu, assigned_cuml : tuple of float
        Fraction of rows each backend assigned, per seed.
    normalised : tuple of float
        ``ari * coverage / min(assigned_cpu, assigned_cuml)`` per seed: agreement
        relative to the most the assigned fractions allow, so a configuration
        that leaves most rows as noise is not read as backend disagreement.
        Equal to 1 when neither backend assigns a row and 0 when only one does.
    mean_normalised : float
        Mean of ``normalised``, read by the ``backend_disagreement`` flag.
    n_groups_cpu, n_groups_cuml : tuple of int
        Per-seed group counts on each backend.
    """

    agreement: tuple[float, ...]
    mean_agreement: float
    ari: tuple[float, ...]
    coverage: tuple[float, ...]
    assigned_cpu: tuple[float, ...]
    assigned_cuml: tuple[float, ...]
    normalised: tuple[float, ...]
    mean_normalised: float
    n_groups_cpu: tuple[int, ...]
    n_groups_cuml: tuple[int, ...]


def compare_backends(
    x: NDArray[np.float64],
    params: Params,
    *,
    scaling: Scaling = "standard",
    n_seeds: int = 3,
    random_state: int = 0,
) -> BackendAgreement:
    """Fit the same configuration on both backends and report partition agreement.

    Backends are compared by the agreement of their partitions, never by exact
    labels, because their layouts differ in floating-point order.

    Parameters
    ----------
    x : ndarray of shape (n, d)
        Feature matrix; validated with ``validate_matrix``.
    params : Params
        Configuration fitted on the full row set.
    scaling : {"standard", "robust"}, default "standard"
        Scaler family.
    n_seeds : int, default 3
        Seeds ``derive_seed(random_state, 7, s)`` for ``s < n_seeds``.
    random_state : int, default 0
        Base seed.

    Returns
    -------
    BackendAgreement
        Per-seed agreement and group counts.

    Raises
    ------
    ImportError
        If cuML is not usable in this process.
    ValueError
        If ``n_seeds < 1`` or ``x`` fails validation.
    """
    from starfold.objective import pair_stability  # noqa: PLC0415

    resolve_engine("cuml")
    if n_seeds < 1:
        msg = f"n_seeds must be at least 1 (got {n_seeds})."
        raise ValueError(msg)
    rows, _kept = validate_matrix(x)
    positions = np.arange(rows.shape[0], dtype=np.intp)
    agreement: list[float] = []
    aris: list[float] = []
    coverages: list[float] = []
    assigned_cpu: list[float] = []
    assigned_cuml: list[float] = []
    normalised: list[float] = []
    counts_cpu: list[int] = []
    counts_cuml: list[int] = []
    for s in range(n_seeds):
        seed = derive_seed(random_state, 7, s)
        cpu = fit_pipeline(rows, params, seed=seed, engine="cpu", scaling=scaling)
        gpu = fit_pipeline(rows, params, seed=seed, engine="cuml", scaling=scaling)
        a, c = pair_stability(cpu.labels, gpu.labels, positions, positions)
        share_cpu = float(np.mean(cpu.labels >= 0))
        share_gpu = float(np.mean(gpu.labels >= 0))
        reachable = min(share_cpu, share_gpu)
        agreement.append(a * c)
        aris.append(a)
        coverages.append(c)
        assigned_cpu.append(share_cpu)
        assigned_cuml.append(share_gpu)
        if reachable > 0.0:
            normalised.append(a * c / reachable)
        else:
            # All noise on both backends is agreement; all noise on one is not.
            normalised.append(1.0 if max(share_cpu, share_gpu) == 0.0 else 0.0)
        counts_cpu.append(cpu.n_groups)
        counts_cuml.append(gpu.n_groups)
    release_backend_caches()
    return BackendAgreement(
        agreement=tuple(agreement),
        mean_agreement=float(np.mean(agreement)),
        ari=tuple(aris),
        coverage=tuple(coverages),
        assigned_cpu=tuple(assigned_cpu),
        assigned_cuml=tuple(assigned_cuml),
        normalised=tuple(normalised),
        mean_normalised=float(np.mean(normalised)),
        n_groups_cpu=tuple(counts_cpu),
        n_groups_cuml=tuple(counts_cuml),
    )
