"""Joint UMAP and HDBSCAN configuration search.

One Optuna study scores every candidate on a common resample plan. Trial 0
holds the library defaults clipped into the space, an Owen-scrambled Sobol
design covers the space and a multivariate TPE sampler converges, with median
pruning to save fits and with pruned near-winners completed from their cached
fits. With a declared null family, a trial's score is its gap over a fixed null
matrix. The top candidates and the default are re-scored on a fresh plan and
fresh null matrices before the largest fresh gap is selected. For this reason
the default comparison is not read off the maximised surface.
"""

from __future__ import annotations

import gc
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import optuna
from scipy.stats import qmc

from starfold._engine import derive_seed, require_resolved
from starfold.backend import Params, default_params, knn_graph, neighbor_adjacency
from starfold.nulls import generate_null
from starfold.objective import TrialPruned, combine, evaluate, heldout_neighbours
from starfold.prep import make_plan, scale

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from numpy.typing import NDArray
    from scipy.sparse import csr_matrix

    from starfold._engine import ResolvedEngine
    from starfold.api import Config, Space
    from starfold.backend import Fit, KnnGraph
    from starfold.objective import HeldoutNeighbours, TrialScore
    from starfold.prep import ResamplePlan, Scaling

K_MAX = 100
"""Neighbour-graph width cached once per resample and sliced per trial."""
ADJACENCY_K = 15
"""Width of the input-space modularity graph."""

Selection = Literal["eom", "leaf"]
TrialState = Literal["complete", "pruned", "failed"]
Phase = Literal["default", "exploration", "convergence", "completion"]
StopReason = Literal["patience", "budget", "exploration_only"]


@dataclass(frozen=True)
class ResolvedSpace:
    """Concrete inclusive bounds after the data-dependent rules."""

    n_neighbors: tuple[int, int]
    min_dist: tuple[float, float]
    n_components: tuple[int, int]
    min_cluster_size: tuple[int, int]
    min_samples_fraction: tuple[float, float]
    selection_method: tuple[Selection, ...]


@dataclass(frozen=True)
class TrialRecord:
    """One trial, whose ``score`` is the gap ``raw_score - null_score`` the sampler maximises.

    ``raw_score`` is the objective on the data and ``null_score`` the objective
    of the same configuration on the search's fixed null matrix. Without a null
    matrix, ``null_score`` is NaN and ``score`` equals ``raw_score``. A pruned
    trial has a NaN ``null_score`` and keeps its leading-pair data score
    (resamples 0 and 1) as ``score``, ``raw_score`` and ``partial_score``.

    Attributes
    ----------
    number : int
        Optuna trial number, 0 for the default.
    params : Params
        Configuration of the trial.
    state : {"complete", "pruned", "failed"}
        Outcome of the trial.
    score : float
        Gap ``raw_score - null_score``, or ``raw_score`` when no copy is scored.
    raw_score : float
        Objective on the data.
    null_score : float
        Objective of the same configuration on the search's structureless
        copy, NaN when none is scored.
    partial_score : float or None
        Leading-pair data score of a pruned trial, None otherwise.
    stability : float
        Held-out prediction agreement lane of the objective, NaN when pruned.
    coverage : float
        Joint coverage lane of the objective, NaN when pruned.
    modularity : float
        Input-space neighbourhood modularity lane, NaN when pruned.
    fit_counts : tuple of int
        Group count of each fit the trial ran.
    fit_noise : tuple of float
        Noise fraction of each fit the trial ran.
    seconds : float
        Wall time of the trial.
    error : str or None
        Error text of a failed trial, None otherwise.
    phase : {"default", "exploration", "convergence", "completion"}
        Search phase that produced the record.
    """

    number: int
    params: Params
    state: TrialState
    score: float
    raw_score: float
    null_score: float
    partial_score: float | None
    stability: float
    coverage: float
    modularity: float
    fit_counts: tuple[int, ...]
    fit_noise: tuple[float, ...]
    seconds: float
    error: str | None
    phase: Phase


@dataclass(frozen=True)
class Candidate:
    """A configuration re-scored on the fresh plan and fresh structureless copies.

    Attributes
    ----------
    params : Params
        Configuration of the candidate.
    trial_number : int
        Number of the search trial it comes from.
    score : float
        Search score of that trial on the common plan.
    fresh_score : float
        Objective on the data under the fresh plan.
    fresh_pair_scores : tuple of float
        Objective of each ordered resample pair under the fresh plan.
    fit_counts : tuple of int
        Group count of each fit on the fresh plan.
    null_scores : tuple of float
        Objective of the same configuration on each fresh structureless copy.
    gap : float
        ``fresh_score`` minus the mean of ``null_scores``, or ``fresh_score``
        when no copy is scored.
    """

    params: Params
    trial_number: int
    score: float
    fresh_score: float
    fresh_pair_scores: tuple[float, ...]
    fit_counts: tuple[int, ...]
    null_scores: tuple[float, ...]
    gap: float


@dataclass(frozen=True)
class SearchResult:
    """Outcome of :func:`search`. Every score is the objective, never agreement with truth.

    In a full search, ``gap_selected`` and ``gap_default`` are the fresh
    scores minus the mean score of the same configuration on the fresh null
    matrices, the gap statistic that selects the configuration (Tibshirani,
    Walther and Hastie 2001, doi:10.1111/1467-9868.00293). Both equal the
    fresh scores when no null matrix was scored. In exploration-only mode
    nothing is re-scored: both gaps, like ``score_selected`` and
    ``score_default``, are search scores on the common plan.

    Attributes
    ----------
    selected : Params
        Selected configuration.
    trials : tuple of TrialRecord
        Every trial, in trial-number order.
    candidates : tuple of Candidate
        Re-scored candidates, largest gap first; empty in exploration-only mode.
    score_selected : float
        Fresh score of the selected configuration, or its search score in
        exploration-only mode.
    score_default : float
        Fresh score of the default, or its search score in exploration-only
        mode.
    gap_selected : float
        Fresh gap of the selected configuration, or its search score in
        exploration-only mode.
    gap_default : float
        Fresh gap of the default, or its search score in exploration-only mode.
    paired_differences : tuple of float
        Fresh pair scores of the selected configuration minus those of the
        default; empty in exploration-only mode.
    beats_default_under_objective : bool
        Whether ``gap_selected`` exceeds ``gap_default``.
    selected_is_default : bool
        Whether the selected configuration is trial 0.
    exploration_max : float
        Largest complete search score among the default and design trials,
        0.0 when none completed.
    stop_reason : {"patience", "budget", "exploration_only"}
        Why the search stopped.
    n_completed : int
        Complete trials, completed pruned trials included.
    n_pruned : int
        Trials that remain pruned.
    n_completed_after_prune : int
        Pruned trials completed after the search.
    n_fits : int
        UMAP and HDBSCAN fits the search ran.
    seconds : float
        Wall time of the search.
    plan : ResamplePlan
        Common resample plan of the search scores.
    """

    selected: Params
    trials: tuple[TrialRecord, ...]
    candidates: tuple[Candidate, ...]
    score_selected: float
    score_default: float
    gap_selected: float
    gap_default: float
    paired_differences: tuple[float, ...]
    beats_default_under_objective: bool
    selected_is_default: bool
    exploration_max: float
    stop_reason: StopReason
    n_completed: int
    n_pruned: int
    n_completed_after_prune: int
    n_fits: int
    seconds: float
    plan: ResamplePlan


def _as_selection(value: str) -> Selection:
    if value in ("eom", "leaf"):
        return "eom" if value == "eom" else "leaf"
    msg = f"selection_method must be 'eom' or 'leaf', got {value!r}"
    raise ValueError(msg)


def resolve_space(space: Space, n: int, m: int) -> ResolvedSpace:
    """Apply the data-dependent bounds and check every range.

    ``n_neighbors`` is capped at ``m - 1``, and a declared bound above
    ``K_MAX`` is rejected given that every trial slices one cached graph of
    ``K_MAX`` neighbours. A ``None`` ``min_cluster_size`` becomes
    ``[5, min(max(50, n // 20), 1023)]``, with 5 the HDBSCAN default and 1023
    the cuML ``min_samples`` limit (github.com/rapidsai/cuml). CPU and GPU
    searches are thus the same procedure. Only these two rules read ``n`` and
    ``m``.

    Parameters
    ----------
    space : Space
        Declared bounds.
    n : int
        Rows of the full matrix.
    m : int
        Rows per resample.

    Returns
    -------
    ResolvedSpace
        Concrete inclusive bounds for every axis.

    Raises
    ------
    ValueError
        If ``n_neighbors`` reaches above ``K_MAX``, a range is empty, a
        log-uniform axis reaches zero, or a selection method is unknown.
    """
    if space.n_neighbors[1] > K_MAX:
        msg = (
            f"n_neighbors bounds {space.n_neighbors} reach above {K_MAX}, "
            "the width of the cached neighbour graph"
        )
        raise ValueError(msg)
    mcs = space.min_cluster_size or (5, min(max(50, n // 20), 1023))
    resolved = ResolvedSpace(
        (int(space.n_neighbors[0]), int(min(space.n_neighbors[1], m - 1))),
        (float(space.min_dist[0]), float(space.min_dist[1])),
        (int(space.n_components[0]), int(space.n_components[1])),
        (int(mcs[0]), int(mcs[1])),
        (float(space.min_samples_fraction[0]), float(space.min_samples_fraction[1])),
        tuple(_as_selection(s) for s in space.selection_method),
    )
    checks: tuple[tuple[str, tuple[float, float], bool], ...] = (
        ("n_neighbors", resolved.n_neighbors, True),
        ("min_dist", resolved.min_dist, False),
        ("n_components", resolved.n_components, False),
        ("min_cluster_size", resolved.min_cluster_size, True),
        ("min_samples_fraction", resolved.min_samples_fraction, True),
    )
    for name, (low, high), log in checks:
        if low > high or (log and low <= 0):
            msg = f"{name} bounds ({low}, {high}) are empty or reach zero on a log axis"
            raise ValueError(msg)
    return resolved


def default_in_space(space: ResolvedSpace) -> Params:
    """Return ``default_params()`` clipped into ``space``, trial 0 of every search.

    The result is the library default itself when the declared bounds contain
    it. When a declared bound excludes it (a floor on ``min_cluster_size``, a
    narrower ``n_neighbors`` range), each field is moved to the nearest bound
    and ``min_samples`` keeps the default's fraction of ``min_cluster_size``
    within the declared fraction bounds.

    Parameters
    ----------
    space : ResolvedSpace
        The resolved search space.

    Returns
    -------
    Params
        The default configuration inside ``space``.
    """
    d = default_params()
    mcs = int(np.clip(d.min_cluster_size, *space.min_cluster_size))
    lo, hi = space.min_samples_fraction
    fraction = float(np.clip(d.min_samples / d.min_cluster_size, lo, hi))
    offered = space.selection_method
    method = d.selection_method if d.selection_method in offered else offered[0]
    return Params(
        int(np.clip(d.n_neighbors, *space.n_neighbors)),
        float(np.clip(d.min_dist, *space.min_dist)),
        int(np.clip(d.n_components, *space.n_components)),
        mcs,
        # the integer nearest the fraction whose own fraction stays inside the declared bounds
        int(
            np.clip(
                round(fraction * mcs),
                max(1, int(np.ceil(lo * mcs))),
                max(1, int(np.floor(hi * mcs))),
            )
        ),
        method,
    )


def _params_from(row: dict[str, Any]) -> Params:
    mcs = int(row["min_cluster_size"])
    min_samples = int(np.clip(round(float(row["min_samples_fraction"]) * mcs), 1, mcs))
    return Params(
        n_neighbors=int(row["n_neighbors"]),
        min_dist=float(row["min_dist"]),
        n_components=int(row["n_components"]),
        min_cluster_size=mcs,
        min_samples=min_samples,
        selection_method=_as_selection(str(row["selection_method"])),
    )


def _values_of(params: Params) -> dict[str, Any]:
    return {
        "n_neighbors": params.n_neighbors,
        "min_dist": params.min_dist,
        "n_components": params.n_components,
        "min_cluster_size": params.min_cluster_size,
        "min_samples_fraction": params.min_samples / params.min_cluster_size,
        "selection_method": params.selection_method,
    }


def suggest_params(trial: optuna.Trial, space: ResolvedSpace) -> Params:
    """Draw one configuration from the sampler, or replay an enqueued one exactly.

    Given their multiplicative effect, ``n_neighbors``, ``min_cluster_size``
    and ``min_samples_fraction`` are log-uniform, while ``min_dist`` and
    ``n_components`` are uniform. The fraction is the stored parameter and
    sets ``min_samples = clip(round(f * mcs), 1, mcs)``. A fraction ``f = 1``
    thus reproduces HDBSCAN's ``min_samples=None``
    (github.com/scikit-learn-contrib/hdbscan).

    Parameters
    ----------
    trial : optuna.Trial
        Trial whose ``suggest_*`` calls define the distributions.
    space : ResolvedSpace
        Bounds from :func:`resolve_space`.

    Returns
    -------
    Params
        The configuration for this trial.
    """
    lo, hi = space.min_samples_fraction
    return _params_from(
        {
            "n_neighbors": trial.suggest_int("n_neighbors", *space.n_neighbors, log=True),
            "min_dist": trial.suggest_float("min_dist", *space.min_dist),
            "n_components": trial.suggest_int("n_components", *space.n_components),
            "min_cluster_size": trial.suggest_int(
                "min_cluster_size", *space.min_cluster_size, log=True
            ),
            "min_samples_fraction": trial.suggest_float("min_samples_fraction", lo, hi, log=True),
            "selection_method": trial.suggest_categorical(
                "selection_method", list(space.selection_method)
            ),
        }
    )


def _map_float(u: float, bounds: tuple[float, float], *, log: bool) -> float:
    low, high = bounds
    value = low * (high / low) ** u if log else low + u * (high - low)
    return float(min(max(value, low), high))


def _map_int(u: float, bounds: tuple[int, int], *, log: bool) -> int:
    low, high = bounds
    value = round(low * (high / low) ** u) if log else low + math.floor(u * (high - low + 1))
    return int(min(max(value, low), high))


def _sobol_rows(space: ResolvedSpace, n_points: int, seed: int) -> tuple[dict[str, Any], ...]:
    """Owen-scrambled Sobol exploration design over the five numeric axes.

    Points drawn from ``scipy.stats.qmc.Sobol(d=5, scramble=True, seed)``
    (Owen 1998, doi:10.1145/272991.273010) are mapped in the log or linear
    coordinate of each axis, and ``selection_method`` cycles through the
    offered methods. At equal budget, low-discrepancy coverage beats random
    points (Bergstra and Bengio 2012, JMLR 13:281). ``n_points`` must be zero
    or a power of two.
    """
    if n_points == 0:
        return ()
    exponent = n_points.bit_length() - 1
    if n_points < 0 or 1 << exponent != n_points:
        msg = f"n_points must be zero or a power of two, got {n_points}"
        raise ValueError(msg)
    unit = qmc.Sobol(d=5, scramble=True, seed=seed).random_base2(exponent)
    methods = space.selection_method
    return tuple(
        {
            "n_neighbors": _map_int(u[0], space.n_neighbors, log=True),
            "min_dist": _map_float(u[1], space.min_dist, log=False),
            "n_components": _map_int(u[2], space.n_components, log=False),
            "min_cluster_size": _map_int(u[3], space.min_cluster_size, log=True),
            "min_samples_fraction": _map_float(u[4], space.min_samples_fraction, log=True),
            "selection_method": methods[i % len(methods)],
        }
        for i, u in enumerate(unit)
    )


@dataclass(frozen=True)
class _Bundle:
    """A matrix with one plan's cached neighbour graphs, modularity graphs and transfer table."""

    x: NDArray[np.float64]
    plan: ResamplePlan
    graphs: tuple[KnnGraph, ...]
    adjacency: tuple[csr_matrix, ...]
    neighbours: HeldoutNeighbours


def _build(
    x: NDArray[np.float64],
    plan: ResamplePlan,
    scaling: Scaling,
    engine: ResolvedEngine,
    n_jobs: int,
) -> _Bundle:
    k = min(K_MAX, plan.m - 1)
    graphs = tuple(
        knn_graph(scale(x[rows], scaling), k, engine=engine, seed=seed, n_jobs=n_jobs)
        for rows, seed in zip(plan.indices, plan.seeds, strict=True)
    )
    adjacency = tuple(neighbor_adjacency(g, min(ADJACENCY_K, k)) for g in graphs)
    return _Bundle(
        x, plan, graphs, adjacency, heldout_neighbours(x, plan, scaling=scaling, engine=engine)
    )


def _fresh_bundles(
    x: NDArray[np.float64],
    fresh: ResamplePlan,
    config: Config,
    random_state: int,
    engine: ResolvedEngine,
    n_jobs: int,
) -> Iterator[_Bundle]:
    """Yield the fresh plan's bundle of the data, then one per null matrix, built on demand.

    Subtracting a configuration's score on a null matrix, a structureless
    surrogate with the observed marginals, cancels the scale preference of the
    modularity lane.
    """
    yield _build(x, fresh, config.scaling, engine, n_jobs)
    family = config.gap_family
    if family is None:
        return
    for b in range(config.n_null_rescore):
        x_null = generate_null(x, family, derive_seed(random_state, 8, b))
        yield _build(x_null, fresh, config.scaling, engine, n_jobs)


def _hook(trial: optuna.Trial) -> Callable[[float], bool]:
    def hook(value: float) -> bool:
        trial.report(value, 0)
        return trial.should_prune()

    return hook


def _phase(number: int, n_explore: int) -> Phase:
    if number == 0:
        return "default"
    return "exploration" if number <= n_explore else "convergence"


def _record(
    number: int,
    params: Params,
    s: TrialScore,
    phase: Phase,
    seconds: float,
    null: TrialScore | None = None,
) -> TrialRecord:
    state: TrialState = "failed" if s.error is not None else "complete"
    null_score = math.nan if null is None else null.score
    gap = s.score if null is None else s.score - null.score
    rest = (tuple(s.fit_counts), tuple(s.fit_noise), seconds, s.error, phase)
    return TrialRecord(
        number,
        params,
        state,
        gap,
        s.score,
        null_score,
        None,
        s.stability,
        s.coverage,
        s.modularity,
        *rest,
    )


def _pruned_record(
    number: int, params: Params, p: TrialPruned, phase: Phase, seconds: float
) -> TrialRecord:
    counts = tuple(int(fit.n_groups) for fit in p.fits)
    noise = tuple(float(np.mean(fit.labels < 0)) for fit in p.fits)
    nan = math.nan
    rest = (counts, noise, seconds, None, phase)
    return TrialRecord(
        number, params, "pruned", p.partial, p.partial, nan, p.partial, nan, nan, nan, *rest
    )


@dataclass
class _Run:
    """Mutable state of one study: records, cached fits of pruned trials, counters."""

    space: ResolvedSpace
    config: Config
    engine: ResolvedEngine
    n_jobs: int
    prune: bool
    records: dict[int, TrialRecord] = field(default_factory=dict)
    pruned_fits: dict[int, tuple[Fit, ...]] = field(default_factory=dict)
    n_fits: int = 0

    def score(
        self,
        params: Params,
        bundle: _Bundle,
        *,
        should_prune: Callable[[float], bool] | None = None,
        may_prune: bool = True,
        cached: tuple[Fit, ...] = (),
    ) -> tuple[TrialScore, float]:
        started = time.perf_counter()
        result, fits = evaluate(
            bundle.x,
            params,
            bundle.plan,
            graphs=bundle.graphs,
            adjacency=bundle.adjacency,
            engine=self.engine,
            scaling=self.config.scaling,
            n_jobs=self.n_jobs,
            should_prune=should_prune,
            may_prune=may_prune,
            cached_fits=cached or None,
            neighbours=bundle.neighbours,
        )
        self.n_fits += max(0, len(fits) - len(cached))
        return result, time.perf_counter() - started

    def null_score(self, params: Params, null: _Bundle | None) -> TrialScore | None:
        """Score a configuration on the fixed null matrix, or None when none is configured."""
        return None if null is None else self.score(params, null)[0]

    def objective(self, trial: optuna.Trial, main: _Bundle, null: _Bundle | None) -> float:
        params, number = suggest_params(trial, self.space), trial.number
        phase, started = _phase(number, self.config.n_explore), time.perf_counter()
        may_prune = number > self.config.n_explore
        hook = _hook(trial) if self.prune else None
        try:
            result, _seconds = self.score(params, main, should_prune=hook, may_prune=may_prune)
        except TrialPruned as pruned:
            self.n_fits += len(pruned.fits)
            # best() only grows, so a partial at or below it now can never be completed
            if pruned.partial > self.best(number):
                self.pruned_fits[number] = tuple(pruned.fits)
            seconds = time.perf_counter() - started
            self.records[number] = _pruned_record(number, params, pruned, phase, seconds)
            raise optuna.TrialPruned from pruned
        null_result = self.null_score(params, null)
        record = _record(number, params, result, phase, time.perf_counter() - started, null_result)
        self.records[number] = record
        return record.score

    def best(self, upto: int) -> float:
        """Best complete score (the gap) among trials ``0..upto``."""
        complete = [r for r in self.records.values() if r.state == "complete" and r.number <= upto]
        return max((r.score for r in complete), default=0.0)

    def on_trial_end(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        record = self.records.get(trial.number)
        first = self.config.n_explore + 1 + self.config.patience
        if record is None or record.state == "pruned" or trial.number < first:
            return
        window = self.best(trial.number) - self.best(trial.number - self.config.patience)
        if window < self.config.min_improvement:
            study.stop()

    def trials(self) -> tuple[TrialRecord, ...]:
        return tuple(self.records[number] for number in sorted(self.records))

    def complete_pruned(self, main: _Bundle, null: _Bundle | None) -> int:
        """Complete pruned trials whose leading-pair score exceeds the best complete gap.

        The leading-pair score estimates the trial's data score, an upper bound
        on its gap given that the null score is non-negative. At most
        ``n_complete_pruned`` trials are completed, highest leading-pair score
        first, and the cached fits of the rest are released.
        """
        best = self.best(max(self.records))
        pruned = [r for r in self.records.values() if r.state == "pruned"]
        eligible = [r for r in pruned if (r.partial_score or 0.0) > best]
        eligible.sort(key=lambda r: (-(r.partial_score or 0.0), r.number))
        chosen = eligible[: self.config.n_complete_pruned]
        for record in chosen:
            cached = self.pruned_fits.pop(record.number)
            result, seconds = self.score(record.params, main, cached=cached)
            seconds += record.seconds
            null_result = self.null_score(record.params, null)
            self.records[record.number] = _record(
                record.number, record.params, result, "completion", seconds, null_result
            )
        self.pruned_fits.clear()
        return len(chosen)

    def rescore(self, bundles: Iterator[_Bundle]) -> tuple[Candidate, ...]:
        """Re-score the top complete trials and the default, then rank by gap over the null.

        ``bundles`` yields the fresh data bundle, then one per null matrix,
        each scored for every candidate and released before the next is built.
        """
        complete = sorted(
            (r for r in self.records.values() if r.state == "complete"),
            key=lambda r: (-r.score, r.number),
        )
        chosen = complete[: self.config.n_rescore]
        if all(r.number != 0 for r in chosen):
            chosen.append(self.records[0])
        per_bundle: list[list[TrialScore]] = []
        for bundle in bundles:
            per_bundle.append([self.score(record.params, bundle)[0] for record in chosen])
            del bundle
        fresh, *nulls = per_bundle
        candidates = []
        for i, record in enumerate(chosen):
            result = fresh[i]
            null_scores = tuple(scores[i].score for scores in nulls)
            gap = result.score - float(np.mean(null_scores)) if null_scores else result.score
            pairs = zip(result.pair_agreement, result.pair_coverage, strict=True)
            candidate = Candidate(
                record.params,
                record.number,
                record.score,
                result.score,
                tuple(combine(a, c, result.modularity) for a, c in pairs),
                tuple(result.fit_counts),
                null_scores,
                gap,
            )
            candidates.append(candidate)
        return tuple(sorted(candidates, key=lambda c: (-c.gap, c.trial_number)))


def _optimize(run: _Run, main: _Bundle, null: _Bundle | None, seed: int, n_trials: int) -> None:
    config = run.config
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=0, seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=config.n_explore + 1, n_warmup_steps=0)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.enqueue_trial(_values_of(default_in_space(run.space)))
    for row in _sobol_rows(run.space, config.n_explore, seed):
        study.enqueue_trial(row)
    study.optimize(
        lambda trial: run.objective(trial, main, null),
        n_trials=n_trials,
        callbacks=[run.on_trial_end],
        catch=(),
    )


def search(
    x: NDArray[np.float64],
    *,
    config: Config,
    random_state: int,
    engine: ResolvedEngine,
    n_jobs: int = 1,
    exploration_only: bool = False,
) -> SearchResult:
    """Run the joint search and select by the fresh gap over the null.

    Every trial is scored on one resample plan with common random numbers
    (Glasserman and Yao 1992, doi:10.1287/mnsc.38.6.884). Trial 0 is
    :func:`default_in_space` and trials 1 to ``n_explore`` are the Sobol design,
    followed by multivariate TPE (Bergstra et al. 2011, NeurIPS 24; Akiba et al.
    2019, arXiv:1907.10902), all enqueued into one study for the surrogate to be
    built from complete evaluations. Convergence trials report the leading-pair
    score after two fits and may be median-pruned (median stopping rule, Golovin
    et al. 2017, doi:10.1145/3097983.3098043), while design trials report it
    too but are never pruned, giving the pruner a median from the first
    convergence trial onwards. The search stops when the best score improves by
    less than ``min_improvement`` over ``patience`` trials, or at ``n_trials``.
    Pruned trials whose leading-pair score exceeds the best complete gap are
    then completed, given that the data score bounds the gap from above (the
    null score is non-negative). The top ``n_rescore`` trials plus the default
    are re-scored on a fresh plan and fresh null matrices, and the largest fresh
    gap is selected. The fresh scores reduce the winner's curse of the search
    scores but do not remove it: the selected gap is still the largest of up to
    ``n_rescore + 1`` fresh gaps, the default's included, so it remains biased
    upwards relative to the default's.

    Parameters
    ----------
    x : ndarray of shape (n, d)
        Validated feature matrix.
    config : Config
        Search controls.
    random_state : int
        Seed of both plans, the null matrices, the Sobol scrambling and the
        sampler.
    engine : {"cpu", "cuml"}
        Resolved backend.
    n_jobs : int, default 1
        Worker processes inside a trial.
    exploration_only : bool, default False
        Run trials 0 to ``n_explore`` only, without pruning, completion or
        fresh re-scoring. ``selected`` is then their best, the reported scores
        are search scores and ``candidates`` is empty. The ``exploration`` null
        procedure uses this mode.

    Returns
    -------
    SearchResult
        Selected configuration, trial table, candidates and counters.

    Raises
    ------
    ValueError
        If ``engine`` is not ``"cpu"`` or ``"cuml"``, ``x`` is not
        two-dimensional, ``n_trials < n_explore + 1``, or
        :func:`resolve_space` rejects the space.
    ImportError
        If ``engine="cuml"`` and the cuML libraries cannot be imported.
    """
    started = time.perf_counter()
    engine = require_resolved(engine)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    x = np.ascontiguousarray(x, dtype=np.float64)
    if x.ndim != 2:
        msg = f"x must be two-dimensional, got shape {x.shape}"
        raise ValueError(msg)
    n_design = config.n_explore + 1
    n_trials = n_design if exploration_only else config.n_trials
    if n_trials < n_design:
        msg = f"n_trials ({config.n_trials}) must cover the default and the design ({n_design})"
        raise ValueError(msg)
    n = x.shape[0]
    plan = make_plan(
        n, n_resamples=config.n_resamples, fraction=config.resample_fraction, seed=random_state
    )
    space = resolve_space(config.space, n, plan.m)
    main = _build(x, plan, config.scaling, engine, n_jobs)
    null: _Bundle | None = None
    if config.gap_family is not None and config.n_null_rescore > 0:
        x_null = generate_null(x, config.gap_family, derive_seed(random_state, 9))
        null = _build(x_null, plan, config.scaling, engine, n_jobs)
    run = _Run(space, config, engine, n_jobs, config.prune and not exploration_only)
    _optimize(run, main, null, random_state, n_trials)
    trials = run.trials()
    candidates: tuple[Candidate, ...] = ()
    paired: tuple[float, ...] = ()
    stop_reason: StopReason = "exploration_only"
    n_after = 0
    if exploration_only:
        complete = [r for r in trials if r.state == "complete"]
        best = max(complete, key=lambda r: (r.score, -r.number), default=trials[0])
        selected, scores, is_default = best.params, (best.score, trials[0].score), best.number == 0
        gaps = scores
    else:
        n_after = run.complete_pruned(main, null)
        # each pruned trial's traceback holds both bundles in a reference cycle
        del main, null
        gc.collect()
        fresh_seed = derive_seed(random_state, 6)
        fresh = make_plan(
            n, n_resamples=config.n_resamples, fraction=config.resample_fraction, seed=fresh_seed
        )
        candidates = run.rescore(_fresh_bundles(x, fresh, config, random_state, engine, n_jobs))
        top, default = candidates[0], next(c for c in candidates if c.trial_number == 0)
        selected, scores, is_default = (
            top.params,
            (top.fresh_score, default.fresh_score),
            top.trial_number == 0,
        )
        gaps = (top.gap, default.gap)
        paired = tuple(
            s - d for s, d in zip(top.fresh_pair_scores, default.fresh_pair_scores, strict=True)
        )
        stop_reason = "patience" if len(trials) < n_trials else "budget"
        trials = run.trials()
    return SearchResult(
        selected=selected,
        trials=trials,
        candidates=candidates,
        score_selected=scores[0],
        score_default=scores[1],
        gap_selected=gaps[0],
        gap_default=gaps[1],
        paired_differences=paired,
        beats_default_under_objective=gaps[0] > gaps[1],
        selected_is_default=is_default,
        exploration_max=max(
            (r.score for r in trials[:n_design] if r.state == "complete"), default=0.0
        ),
        stop_reason=stop_reason,
        n_completed=sum(r.state == "complete" for r in trials),
        n_pruned=sum(r.state == "pruned" for r in trials),
        n_completed_after_prune=n_after,
        n_fits=run.n_fits,
        seconds=time.perf_counter() - started,
        plan=plan,
    )
