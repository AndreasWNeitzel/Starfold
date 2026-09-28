"""Label-free trial objective: paired stability, joint coverage, input-space modularity.

Every trial receives one finite score by formula; degenerate outcomes score
zero without any gate, and the cluster count is never read. Stability alone
prefers the coarsest partition (von Luxburg 2010, arXiv:1007.1075); the
modularity of the partition on the input-space neighbour graph counters
that, and the geometric mean makes a zero in any lane fatal without a
threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler

from starfold._engine import parallel_map, require_resolved
from starfold.backend import Fit, fit_pipeline

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from numpy.typing import NDArray

    from starfold._engine import ResolvedEngine
    from starfold.backend import KnnGraph, Params
    from starfold.prep import ResamplePlan, Scaling

    FitJob = tuple[NDArray[np.float64], Params, int, ResolvedEngine, Scaling, KnnGraph]
    HeldoutNeighbours = Mapping[tuple[int, int], tuple[NDArray[np.intp], NDArray[np.intp]]]

__all__ = [
    "TrialPruned",
    "TrialScore",
    "combine",
    "evaluate",
    "heldout_neighbours",
    "modularity",
    "pair_stability",
    "score_fits",
    "transfer_labels",
]

_LEADING_FITS = 2
_TRANSFER_K = 15


def pair_stability(
    labels_a: NDArray[np.int32],
    labels_b: NDArray[np.int32],
    pos_a: NDArray[np.intp],
    pos_b: NDArray[np.intp],
) -> tuple[float, float]:
    """Return agreement and joint coverage of two label vectors on their overlap.

    With ``J`` the overlap rows assigned on both sides, the agreement is
    ``max(0, ARI)`` of the Hubert and Arabie adjusted Rand index
    (doi:10.1007/BF01908075) on ``J`` when ``|J| >= 2`` and both restricted
    vectors hold at least two distinct labels, else 0; coverage is
    ``|J| / |O|`` and 0 when the overlap is empty.

    Parameters
    ----------
    labels_a, labels_b : ndarray, int32
        Label vectors with ``-1`` for noise.
    pos_a, pos_b : ndarray, intp
        Aligned positions of the common rows in each vector.

    Returns
    -------
    agreement, coverage : float

    Raises
    ------
    ValueError
        If the position arrays differ in shape.
    """
    if pos_a.shape != pos_b.shape:
        msg = f"pos_a and pos_b must align (got shapes {pos_a.shape} and {pos_b.shape})."
        raise ValueError(msg)
    if pos_a.size == 0:
        return 0.0, 0.0
    la = labels_a[pos_a]
    lb = labels_b[pos_b]
    joint = (la >= 0) & (lb >= 0)
    n_joint = int(joint.sum())
    coverage = n_joint / pos_a.size
    if n_joint < 2:
        return 0.0, coverage
    la_j = la[joint]
    lb_j = lb[joint]
    if np.unique(la_j).size < 2 or np.unique(lb_j).size < 2:
        return 0.0, coverage
    return max(0.0, float(adjusted_rand_score(la_j, lb_j))), coverage


def modularity(
    labels: NDArray[np.int32],
    adjacency: scipy.sparse.csr_matrix,
    *,
    engine: ResolvedEngine = "cpu",
) -> float:
    """Return the Newman modularity of the non-noise clusters on a graph.

    ``Q = sum_c [e_c / E - (vol_c / 2E)^2]`` over clusters ``c >= 0`` (Newman
    2006, doi:10.1073/pnas.0601602103). Noise rows keep their degrees and
    edges in ``E`` but form no rewarded community; ``Q`` is raw and may be
    negative, and is 0 when the graph has no edges.

    Parameters
    ----------
    labels : ndarray of shape (m,), int32
        Cluster ids with ``-1`` for noise.
    adjacency : scipy.sparse.csr_matrix of shape (m, m)
        Symmetric weighted adjacency without self loops.
    engine : {"cpu", "cuml"}, default "cpu"
        ``"cuml"`` computes on the device with CuPy.

    Returns
    -------
    float
        Raw modularity.

    Raises
    ------
    ValueError
        If ``engine`` is not ``"cpu"`` or ``"cuml"``, or the adjacency shape
        does not match the labels.
    """
    engine = require_resolved(engine)
    m = labels.shape[0]
    if adjacency.shape != (m, m):
        msg = f"adjacency must have shape ({m}, {m}) (got {adjacency.shape})."
        raise ValueError(msg)
    xp = _array_module(engine)
    csr = scipy.sparse.csr_matrix(adjacency)
    data = xp.asarray(csr.data, dtype=xp.float64)
    two_e = float(data.sum())
    lab = xp.asarray(labels)
    assigned = lab >= 0
    if two_e <= 0.0 or not bool(assigned.any()):
        return 0.0
    indptr = xp.asarray(csr.indptr)
    row = xp.repeat(xp.arange(m), indptr[1:] - indptr[:-1])
    col = xp.asarray(csr.indices)
    n_clusters = int(lab[assigned].max()) + 1
    degrees = xp.bincount(row, weights=data, minlength=m)
    row_labels = lab[row]
    same = (row_labels == lab[col]) & (row_labels >= 0)
    inside = xp.bincount(row_labels[same], weights=data[same], minlength=n_clusters)
    volume = xp.bincount(lab[assigned], weights=degrees[assigned], minlength=n_clusters)
    return float(xp.sum(inside / two_e - (volume / two_e) ** 2))


def _array_module(engine: ResolvedEngine) -> Any:  # noqa: ANN401 - numpy or cupy module
    """Return NumPy on the CPU backend and CuPy on cuML, so both run the same arithmetic."""
    if engine == "cuml":
        import cupy  # noqa: PLC0415 - GPU dependency loaded only on the cuML backend

        return cupy
    return np


def combine(stability: float, coverage: float, modularity: float) -> float:
    """Return the geometric mean of the three clipped lanes.

    Parameters
    ----------
    stability, coverage, modularity : float
        Lane values; negatives are clipped to 0.

    Returns
    -------
    float
        ``(max(0, s) * max(0, c) * max(0, q)) ** (1/3)``.
    """
    product = max(0.0, stability) * max(0.0, coverage) * max(0.0, modularity)
    return float(product ** (1.0 / 3.0))


@dataclass(frozen=True)
class TrialScore:
    """Everything one trial records.

    Attributes
    ----------
    score : float
        Geometric mean of ``stability``, ``coverage`` and ``modularity``.
    stability, coverage : float
        Means over the ordered pairs of present fits, fitted resample first.
    modularity : float
        Mean over available fits of ``max(0, Q_r)``.
    pair_agreement, pair_coverage : tuple of float
        Per ordered pair ``(r, s)``, ``r != s``, both fits present: held-out
        prediction agreement of fit ``r`` on the rows of ``s`` it never saw.
    fit_counts : tuple of int
        Group count per present fit.
    fit_noise : tuple of float
        Noise fraction per present fit.
    error : str or None
        Exception text of a failed trial.
    """

    score: float
    stability: float
    coverage: float
    modularity: float
    pair_agreement: tuple[float, ...]
    pair_coverage: tuple[float, ...]
    fit_counts: tuple[int, ...]
    fit_noise: tuple[float, ...]
    error: str | None


def _neighbours(
    x_fit: NDArray[np.float64],
    x_query: NDArray[np.float64],
    *,
    scaling: Scaling,
    k: int,
    engine: ResolvedEngine,
) -> NDArray[np.intp]:
    """Return, per query row, the indices of its ``k`` nearest fitted rows after scaling."""
    if x_query.shape[0] == 0:
        return np.empty((0, min(k, x_fit.shape[0])), dtype=np.intp)
    scaler = StandardScaler() if scaling == "standard" else RobustScaler()
    fitted = np.asarray(scaler.fit_transform(x_fit), dtype=np.float32)
    queries = np.asarray(scaler.transform(x_query), dtype=np.float32)
    k_eff = min(k, fitted.shape[0])
    if engine == "cuml":
        import cuml.neighbors  # noqa: PLC0415 - GPU dependency loaded only on the cuML backend
        import cupy  # noqa: PLC0415

        index = cuml.neighbors.NearestNeighbors(
            n_neighbors=k_eff, algorithm="brute", metric="euclidean", output_type="cupy"
        ).fit(cupy.asarray(fitted))
        found = index.kneighbors(cupy.asarray(queries), return_distance=False).get()
    else:
        index = NearestNeighbors(n_neighbors=k_eff, algorithm="brute").fit(fitted)
        found = index.kneighbors(queries, return_distance=False)
    return np.asarray(found, dtype=np.intp)


def _vote(labels_fit: NDArray[np.int32], neighbours: NDArray[np.intp]) -> NDArray[np.int32]:
    """Return the most frequent neighbour label per row, ties to the smallest label."""
    if neighbours.shape[0] == 0:
        return np.empty(0, dtype=np.int32)
    votes = np.asarray(labels_fit, dtype=np.int32)[neighbours]
    # Mode of the k votes by pairwise comparison, ties to the smallest label; a
    # rows-by-labels count table would reach gigabytes with thousands of groups.
    counts = (votes[:, :, None] == votes[:, None, :]).sum(axis=2)
    best = counts.max(axis=1, keepdims=True)
    candidates = np.where(counts == best, votes, np.iinfo(np.int32).max)
    return np.asarray(candidates.min(axis=1), dtype=np.int32)


def transfer_labels(
    x_fit: NDArray[np.float64],
    labels_fit: NDArray[np.int32],
    x_query: NDArray[np.float64],
    *,
    scaling: Scaling,
    k: int = _TRANSFER_K,
    engine: ResolvedEngine = "cpu",
) -> NDArray[np.int32]:
    """Predict labels of unseen rows by majority vote of their input-space neighbours.

    The scaler is fitted on the fitted rows only, so the query rows are seen
    through the fit's own preprocessing; noise votes count, so a row whose
    neighbours are mostly noise is predicted as noise. This is the classifier
    step of cluster validation by prediction (Lange, Roth, Braun and Buhmann
    2004, doi:10.1162/089976604773717621; Tibshirani and Walther 2005,
    doi:10.1198/106186005X59243).

    Parameters
    ----------
    x_fit : ndarray of shape (m, d), float64
        Rows the partition was fitted on.
    labels_fit : ndarray of shape (m,), int32
        Their labels, ``-1`` for noise.
    x_query : ndarray of shape (h, d), float64
        Rows the fit never saw.
    scaling : {"standard", "robust"}
        Scaler family of the fit.
    k : int, default 15
        Neighbours per query row (capped at ``m``).
    engine : {"cpu", "cuml"}, default "cpu"
        ``"cuml"`` searches neighbours with ``cuml.neighbors.NearestNeighbors``.

    Returns
    -------
    ndarray of shape (h,), int32
        Predicted labels, ``-1`` for noise. A tied vote goes to the smallest
        label, noise first.

    Raises
    ------
    ValueError
        If ``engine`` is not ``"cpu"`` or ``"cuml"``.
    """
    engine = require_resolved(engine)
    return _vote(labels_fit, _neighbours(x_fit, x_query, scaling=scaling, k=k, engine=engine))


def _heldout_pair(
    x: NDArray[np.float64],
    plan: ResamplePlan,
    r: int,
    s: int,
    *,
    scaling: Scaling,
    engine: ResolvedEngine,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Return positions in ``s`` of the rows ``r`` never saw and their neighbours in ``r``."""
    rows, pos_s = plan.heldout(r, s)
    fitted = x[plan.indices[r]]
    return pos_s, _neighbours(fitted, x[rows], scaling=scaling, k=_TRANSFER_K, engine=engine)


def _heldout_agreement(
    labels_r: NDArray[np.int32],
    labels_s: NDArray[np.int32],
    pos_s: NDArray[np.intp],
    neighbours: NDArray[np.intp],
) -> tuple[float, float]:
    """Compare fit ``r``'s votes on the held-out rows with fit ``s``'s own labels."""
    predicted = _vote(labels_r, neighbours)
    return pair_stability(predicted, labels_s, np.arange(pos_s.size, dtype=np.intp), pos_s)


def heldout_neighbours(
    x: NDArray[np.float64],
    plan: ResamplePlan,
    *,
    scaling: Scaling,
    engine: ResolvedEngine = "cpu",
) -> HeldoutNeighbours:
    """Neighbour table of the held-out rows of every ordered resample pair.

    The table depends on the matrix, the plan, the scaling and the backend
    but on no configuration, so a search computes it once per matrix and plan
    and each trial only gathers labels and votes. Every entry is found exactly
    as :func:`transfer_labels` finds it.

    Parameters
    ----------
    x : ndarray of shape (n, d), float64
        Validated matrix.
    plan : ResamplePlan
        Row sets.
    scaling : {"standard", "robust"}
        Scaler family fitted on each resample.
    engine : {"cpu", "cuml"}, default "cpu"
        Backend of the neighbour search.

    Returns
    -------
    mapping of (r, s) to (pos_s, neighbours)
        For every ordered pair ``r != s``, the positions in resample ``s`` of
        the rows resample ``r`` never saw, and per such row the positions in
        resample ``r`` of its ``min(15, m)`` nearest rows.

    Raises
    ------
    ValueError
        If ``engine`` is not ``"cpu"`` or ``"cuml"``.
    """
    engine = require_resolved(engine)
    pairs = [(r, s) for r in range(plan.n_resamples) for s in range(plan.n_resamples) if r != s]
    return {(r, s): _heldout_pair(x, plan, r, s, scaling=scaling, engine=engine) for r, s in pairs}


class _Scorer:
    """Scores fit lists of one plan; each pair transfer and each ``Q_r`` is computed once.

    The memo is keyed on resample index, so it is valid while the fit of each
    resample stays the same, as it does while ``evaluate`` extends its list.
    """

    def __init__(
        self,
        x: NDArray[np.float64],
        plan: ResamplePlan,
        adjacency: Sequence[scipy.sparse.csr_matrix],
        *,
        scaling: Scaling,
        engine: ResolvedEngine,
        neighbours: HeldoutNeighbours | None,
    ) -> None:
        self.x, self.plan, self.adjacency = x, plan, adjacency
        self.scaling, self.engine, self.neighbours = scaling, engine, neighbours
        self.pairs: dict[tuple[int, int], tuple[float, float]] = {}
        self.q: dict[int, float] = {}

    def _pair(self, r: int, s: int, fit_r: Fit, fit_s: Fit) -> tuple[float, float]:
        if (r, s) not in self.pairs:
            if self.neighbours is None:
                entry = _heldout_pair(
                    self.x, self.plan, r, s, scaling=self.scaling, engine=self.engine
                )
            else:
                entry = self.neighbours[r, s]
            self.pairs[r, s] = _heldout_agreement(fit_r.labels, fit_s.labels, *entry)
        return self.pairs[r, s]

    def _modularity(self, r: int, fit: Fit) -> float:
        if r not in self.q:
            self.q[r] = modularity(fit.labels, self.adjacency[r], engine=self.engine)
        return self.q[r]

    def __call__(self, fits: Sequence[Fit | None]) -> TrialScore:
        present = [(r, fit) for r, fit in enumerate(fits) if fit is not None]
        pairs = [
            self._pair(r, s, fit_r, fit_s) for r, fit_r in present for s, fit_s in present if r != s
        ]
        q_values = [self._modularity(r, fit) for r, fit in present]
        agreements = [a for a, _c in pairs]
        coverages = [c for _a, c in pairs]
        stability = float(np.mean(agreements)) if agreements else 0.0
        coverage = float(np.mean(coverages)) if coverages else 0.0
        q = float(np.mean([max(0.0, v) for v in q_values])) if q_values else 0.0
        return TrialScore(
            score=combine(stability, coverage, q) if len(present) >= 2 else 0.0,
            stability=stability,
            coverage=coverage,
            modularity=q,
            pair_agreement=tuple(agreements),
            pair_coverage=tuple(coverages),
            fit_counts=tuple(fit.n_groups for _r, fit in present),
            fit_noise=tuple(float(np.mean(fit.labels < 0)) for _r, fit in present),
            error=None,
        )


def score_fits(
    fits: Sequence[Fit | None],
    plan: ResamplePlan,
    adjacency: Sequence[scipy.sparse.csr_matrix],
    *,
    x: NDArray[np.float64],
    scaling: Scaling,
    engine: ResolvedEngine = "cpu",
    neighbours: HeldoutNeighbours | None = None,
) -> TrialScore:
    """Score the fits present; a pair needs both fits, a lane needs one.

    Stability is the mean held-out prediction agreement over every ordered
    pair ``(r, s)`` of present fits: the rows of resample ``s`` absent from
    resample ``r`` receive labels transferred from fit ``r`` by
    :func:`transfer_labels`, and :func:`pair_stability` compares them with
    fit ``s``'s own labels on those rows. Coverage is the mean joint coverage
    of those predictions and modularity the mean clipped modularity of the
    fits on their resample graphs. Agreement on the overlap of two resamples
    cannot see structure fixed by the rows themselves, such as an embedding
    torn into stable micro-clusters; prediction on unseen rows can.

    Parameters
    ----------
    fits : sequence of Fit or None
        ``fits[r]`` belongs to resample ``r``; ``None`` entries are skipped.
    plan : ResamplePlan
        Provides the row sets and held-out rows.
    adjacency : sequence of scipy.sparse.csr_matrix
        ``adjacency[r]`` is the neighbour graph of resample ``r``.
    x : ndarray of shape (n, d), float64
        Validated matrix, needed to transfer labels to unseen rows.
    scaling : {"standard", "robust"}
        Scaler family of the fits.
    engine : {"cpu", "cuml"}, default "cpu"
        Backend of the label transfer and the modularity.
    neighbours : mapping, optional
        :func:`heldout_neighbours` of the same ``x``, ``plan``, ``scaling``
        and ``engine``; searched per pair when omitted, with the same result.

    Returns
    -------
    TrialScore
        Score 0 when fewer than two fits are present.

    Raises
    ------
    ValueError
        If ``engine`` is not ``"cpu"`` or ``"cuml"``, or fewer adjacency
        matrices than fits are given.
    """
    engine = require_resolved(engine)
    if len(adjacency) < len(fits):
        msg = f"{len(adjacency)} adjacency matrices for {len(fits)} fits."
        raise ValueError(msg)
    scorer = _Scorer(x, plan, adjacency, scaling=scaling, engine=engine, neighbours=neighbours)
    return scorer(fits)


class TrialPruned(Exception):  # noqa: N818 - the name is part of the public interface.
    """Raised by ``evaluate`` when ``should_prune`` accepts the intermediate score.

    Attributes
    ----------
    partial : float
        Score of the leading pair.
    fits : tuple of Fit
        Fits obtained before pruning, reusable through ``cached_fits``.
    """

    def __init__(self, partial: float, fits: tuple[Fit, ...]) -> None:
        super().__init__(f"trial pruned at intermediate score {partial:.4f}")
        self.partial = partial
        self.fits = fits


def _run_fit(job: FitJob) -> Fit | str:
    """Fit one resample, returning exception text instead of raising.

    Only labels and the group count leave the worker; the embedding and
    strengths are replaced by empty arrays, since the objective reads neither.
    """
    x_rows, params, seed, engine, scaling, knn = job
    try:
        fit = fit_pipeline(x_rows, params, seed=seed, engine=engine, scaling=scaling, knn=knn)
    except Exception as error:  # noqa: BLE001 - any library failure is a recorded trial outcome.
        return f"{type(error).__name__}: {error}"
    empty = np.empty(0, dtype=np.float32)
    return replace(fit, strengths=empty, embedding=empty.reshape(0, 0))


def _fit_until(
    fits: list[Fit],
    stop: int,
    x: NDArray[np.float64],
    params: Params,
    plan: ResamplePlan,
    graphs: Sequence[KnnGraph],
    engine: ResolvedEngine,
    scaling: Scaling,
    workers: int,
) -> str | None:
    """Fit resamples ``len(fits)..stop-1`` in order into ``fits``; return the first error text.

    One worker fits lazily in this process and stops at the first failure;
    several fit the whole range as one batch.
    """
    jobs: list[FitJob] = [
        (x[plan.indices[r]], params, plan.seeds[r], engine, scaling, graphs[r])
        for r in range(len(fits), stop)
    ]
    if not jobs:
        return None
    if workers == 1:
        outcomes: Iterator[Fit | str] = map(_run_fit, jobs)
    else:
        outcomes = iter(parallel_map(_run_fit, jobs, n_jobs=workers))
    for outcome in outcomes:
        if isinstance(outcome, str):
            return outcome
        fits.append(outcome)
    return None


def _failed(error: str) -> TrialScore:
    """Return the all-zero score of a trial whose fit raised."""
    return TrialScore(
        score=0.0,
        stability=0.0,
        coverage=0.0,
        modularity=0.0,
        pair_agreement=(),
        pair_coverage=(),
        fit_counts=(),
        fit_noise=(),
        error=error,
    )


def evaluate(
    x: NDArray[np.float64],
    params: Params,
    plan: ResamplePlan,
    *,
    graphs: Sequence[KnnGraph],
    adjacency: Sequence[scipy.sparse.csr_matrix],
    engine: ResolvedEngine,
    scaling: Scaling,
    n_jobs: int = 1,
    should_prune: Callable[[float], bool] | None = None,
    may_prune: bool = True,
    cached_fits: Sequence[Fit] | None = None,
    neighbours: HeldoutNeighbours | None = None,
) -> tuple[TrialScore, tuple[Fit, ...]]:
    """Evaluate one configuration on the resample plan with optional pruning.

    Once resamples 0 and 1 are fitted, their pair score is the intermediate
    value offered to ``should_prune``. A prunable trial fits them first;
    otherwise every resample is fitted in one batch and the leading pair is
    scored afterwards, so ``should_prune`` still sees it whenever both fits
    succeeded. The final score reuses the leading pair's transfers and
    modularities. Fits run in worker processes when ``n_jobs > 1`` on the CPU
    backend and stop at the first failure in this process otherwise; cuML
    always runs in this process.

    Parameters
    ----------
    x : ndarray of shape (n, d), float64
        Validated matrix.
    params : Params
        Configuration.
    plan : ResamplePlan
        Row sets and UMAP seeds.
    graphs : sequence of KnnGraph
        ``graphs[r]`` is the cached neighbour graph of the scaled rows of
        resample ``r`` under the same ``scaling``.
    adjacency : sequence of scipy.sparse.csr_matrix
        ``adjacency[r]`` is the modularity graph of resample ``r``.
    engine : {"cpu", "cuml"}
        Backend.
    scaling : {"standard", "robust"}
        Scaler family fitted inside each resample.
    n_jobs : int, default 1
        Worker processes over the resamples.
    should_prune : callable, optional
        Receives the intermediate score; ``True`` raises ``TrialPruned``.
    may_prune : bool, default True
        False only reports the intermediate score to ``should_prune`` and
        ignores its answer.
    cached_fits : sequence of Fit, optional
        Fits already available for the leading resamples.
    neighbours : mapping, optional
        :func:`heldout_neighbours` of ``x`` and ``plan`` under the same
        ``scaling`` and ``engine``; searched per pair when omitted.

    Returns
    -------
    score : TrialScore
        Full score, or score 0 with ``error`` set when a fit raised.
    fits : tuple of Fit
        All fits obtained, aligned with the leading resamples, with empty
        embeddings and strengths.

    Raises
    ------
    TrialPruned
        When ``should_prune`` returns ``True`` and ``may_prune`` is set.
    ValueError
        If ``engine`` is not ``"cpu"`` or ``"cuml"``, fewer graphs or
        adjacencies than resamples are given, or more cached fits than
        resamples.
    """
    engine = require_resolved(engine)
    n_resamples = plan.n_resamples
    if len(graphs) < n_resamples or len(adjacency) < n_resamples:
        msg = f"need {n_resamples} graphs and adjacencies (got {len(graphs)}, {len(adjacency)})."
        raise ValueError(msg)
    fits: list[Fit] = list(cached_fits) if cached_fits is not None else []
    if len(fits) > n_resamples:
        msg = f"{len(fits)} cached fits for {n_resamples} resamples."
        raise ValueError(msg)
    workers = 1 if engine == "cuml" else n_jobs
    scorer = _Scorer(x, plan, adjacency, scaling=scaling, engine=engine, neighbours=neighbours)

    def fit_until(stop: int) -> str | None:
        return _fit_until(fits, stop, x, params, plan, graphs, engine, scaling, workers)

    leading = min(_LEADING_FITS, n_resamples)
    staged = should_prune is not None and may_prune
    error = fit_until(leading if staged else n_resamples)
    if should_prune is not None and len(fits) >= leading:
        partial = scorer(fits[:leading]).score
        if should_prune(partial) and may_prune:
            raise TrialPruned(partial, tuple(fits))
    if error is None:
        error = fit_until(n_resamples)
    if error is not None:
        return _failed(error), tuple(fits)
    return scorer(fits), tuple(fits)
