"""Top-level API: configuration, ``recover``, and the stand-alone matched null.

``recover`` runs, in order: validation, joint search, the reference and
default full-data fits, possible worlds, the bounded consensus check, the
matched null, the optional backend comparison, flag derivation, and
provenance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from starfold import nulls as _nulls
from starfold._engine import derive_seed, resolve_engine
from starfold.backend import (
    _BRUTE_FORCE_MAX_ROWS,
    compare_backends,
    default_params,
    fit_pipeline,
    rescale_params,
)
from starfold.consensus import consensus_check
from starfold.prep import validate_matrix
from starfold.result import Result, build_provenance, derive_flags
from starfold.search import resolve_space
from starfold.search import search as _search
from starfold.worlds import run_worlds

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from starfold._engine import Engine, ResolvedEngine
    from starfold.nulls import NullEvidence, NullProcedure
    from starfold.prep import MeasurementSampler, Scaling
    from starfold.search import SearchResult


@dataclass(frozen=True)
class Space:
    """Search space bounds; trial 0 is the library default clipped into them.

    Attributes
    ----------
    n_neighbors : tuple of int
        Log-uniform bounds; the upper bound may not exceed 100, the width
        of the cached neighbour graph, and is capped at ``m - 1`` per fit.
    min_dist : tuple of float
        Uniform bounds.
    n_components : tuple of int
        Uniform integer bounds.
    min_cluster_size : tuple of int or None
        Log-uniform bounds; None means ``[5, min(max(50, n // 20), 1023)]``.
    min_samples_fraction : tuple of float
        Log-uniform bounds of ``min_samples / min_cluster_size``.
    selection_method : tuple of str
        Categorical choices among ``eom`` and ``leaf``.
    """

    n_neighbors: tuple[int, int] = (5, 100)
    min_dist: tuple[float, float] = (0.0, 0.5)
    n_components: tuple[int, int] = (2, 10)
    min_cluster_size: tuple[int, int] | None = None
    min_samples_fraction: tuple[float, float] = (0.05, 1.0)
    selection_method: tuple[str, ...] = ("eom", "leaf")


@dataclass(frozen=True)
class Config:
    """Declared budgets and rules of one ``recover`` run.

    ``null_families`` is empty by default, so ``recover`` skips the matched
    null: it costs 102 fits per realisation, up to 19 realisations, about six
    times the rest of a run. Run :func:`null_test` once on the result you
    report, or name the families here to run it inside ``recover``.

    Construction raises ``ValueError`` when a range is empty or reaches zero
    on a log axis, ``n_explore`` is not a power of two, ``n_null_max`` cannot
    reach ``alpha``, or a budget or option is outside its allowed values.
    """

    space: Space = field(default_factory=Space)
    scaling: Scaling = "standard"
    n_trials: int = 48
    n_explore: int = 16
    n_resamples: int = 3
    resample_fraction: float = 0.8
    patience: int = 16
    min_improvement: float = 0.01
    prune: bool = True
    n_complete_pruned: int = 3
    n_rescore: int = 3
    n_null_rescore: int = 3
    gap_family: str | None = "gaussian_copula"
    n_worlds: int = 32
    world_fraction: float = 0.8
    null_families: tuple[str, ...] = ()
    null_procedure: NullProcedure = "exploration"
    n_null_max: int = 19
    null_exceedances: int = 3
    alpha: float = 0.05
    consensus_budget_bytes: int = 512 * 2**20
    consensus_check_rows: int = 4096
    check_backends: bool = False

    def __post_init__(self) -> None:
        """Validate every declared budget."""
        _validate_config(self)


def _validate_search_budgets(config: Config) -> None:
    """Raise ``ValueError`` for an invalid search budget."""
    checks: tuple[tuple[bool, str], ...] = (
        (config.scaling in ("standard", "robust"), "scaling must be 'standard' or 'robust'."),
        (
            config.n_explore >= 2 and not config.n_explore & (config.n_explore - 1),
            "n_explore must be a power of two of at least 2 (Sobol design).",
        ),
        (config.n_trials > config.n_explore, "n_trials must exceed n_explore."),
        (config.n_resamples >= 2, "n_resamples must be at least 2."),
        (
            0.0 < config.resample_fraction < 1.0,
            "resample_fraction must lie strictly between 0 and 1.",
        ),
        (
            config.patience >= 1 and config.min_improvement >= 0.0,
            "patience must be at least 1 and min_improvement non-negative.",
        ),
        (
            config.n_complete_pruned >= 0 and config.n_rescore >= 1,
            "n_complete_pruned must be non-negative and n_rescore at least 1.",
        ),
        (config.n_null_rescore >= 0, "n_null_rescore must be non-negative."),
        (
            config.gap_family in (None, "gaussian_copula", "independent_marginals"),
            "gap_family must be None, 'gaussian_copula' or 'independent_marginals'.",
        ),
    )
    for passed, message in checks:
        if not passed:
            raise ValueError(message)
    resolve_space(config.space, n=1000, m=800)


def _validate_evidence_budgets(config: Config) -> None:
    """Raise ``ValueError`` for an invalid world, consensus, or null budget."""
    minimum_null = int(np.ceil(1.0 / config.alpha)) - 1 if 0.0 < config.alpha < 1.0 else 0
    checks: tuple[tuple[bool, str], ...] = (
        (2 <= config.n_worlds <= 32767, "n_worlds must lie between 2 and 32767."),
        (
            0.0 < config.world_fraction < 1.0,
            "world_fraction must lie strictly between 0 and 1.",
        ),
        (
            config.null_procedure in ("exploration", "full_search"),
            "null_procedure must be 'exploration' or 'full_search'.",
        ),
        (0.0 < config.alpha < 1.0, "alpha must lie strictly between 0 and 1."),
        (
            config.n_null_max >= minimum_null,
            "n_null_max cannot reach alpha; raise n_null_max or alpha.",
        ),
        (config.null_exceedances >= 1, "null_exceedances must be at least 1."),
        (
            config.consensus_budget_bytes >= 0 and config.consensus_check_rows >= 2,
            "consensus_budget_bytes must be non-negative and consensus_check_rows at least 2.",
        ),
        (
            16 * config.consensus_check_rows**2 <= config.consensus_budget_bytes,
            "consensus_check_rows squared times 16 bytes must fit consensus_budget_bytes.",
        ),
        (
            all(f in ("gaussian_copula", "independent_marginals") for f in config.null_families),
            "null_families may contain only 'gaussian_copula' and 'independent_marginals'.",
        ),
    )
    for passed, message in checks:
        if not passed:
            raise ValueError(message)


def _validate_config(config: Config) -> None:
    """Raise ``ValueError`` for any invalid configuration value."""
    _validate_search_budgets(config)
    _validate_evidence_budgets(config)


def _prepare(
    X: ArrayLike,
    X_err: ArrayLike | None,
) -> tuple[NDArray[np.float64], NDArray[np.intp], NDArray[np.float64] | None, int]:
    """Validate the matrix and align the optional uncertainties with kept rows."""
    n_input = int(np.asarray(X).shape[0]) if np.asarray(X).ndim >= 1 else 0
    x, kept = validate_matrix(X)
    x_err: NDArray[np.float64] | None = None
    if X_err is not None:
        errors = np.asarray(X_err, dtype=np.float64)
        if errors.shape != np.asarray(X).shape:
            msg = "X_err must have the same shape as X."
            raise ValueError(msg)
        x_err = np.ascontiguousarray(errors[kept])
        if not np.isfinite(x_err).all() or (x_err < 0).any():
            msg = "X_err must be finite and non-negative on every kept row."
            raise ValueError(msg)
    return x, kept, x_err, n_input


def _resolve_jobs(engine: ResolvedEngine, n_jobs: int) -> tuple[int, bool]:
    """Return the effective job count and whether cuML forced it to one."""
    if n_jobs < 1:
        msg = "n_jobs must be at least 1."
        raise ValueError(msg)
    if engine == "cuml" and n_jobs != 1:
        return 1, True
    return n_jobs, False


def _null_stage(
    x: NDArray[np.float64],
    search_result: SearchResult,
    *,
    config: Config,
    random_state: int,
    engine: ResolvedEngine,
    n_jobs: int,
) -> NullEvidence:
    """Run the matched null with the statistic the configuration declares.

    On the CPU with at most ``_BRUTE_FORCE_MAX_ROWS`` rows per resample, where
    every neighbour graph is exact, realisations run ``n_jobs`` at a time and
    each replay runs whole in one process, single-threaded when ``n_jobs``
    exceeds 1; the evidence equals the serial run's. Larger resamples use
    pynndescent, whose graph depends on its thread count, so those replays
    keep the search's ``n_jobs`` and run one at a time, as on cuML.
    """
    concurrent = engine == "cpu" and search_result.plan.m <= _BRUTE_FORCE_MAX_ROWS
    replay_jobs, null_jobs = (1, n_jobs) if concurrent else (n_jobs, 1)
    if config.null_procedure == "exploration":
        observed = search_result.exploration_max
        name = "exploration_design_max"

        def statistic(x_null: NDArray[np.float64]) -> tuple[float, int]:
            replay = _search(
                x_null,
                config=config,
                random_state=random_state,
                engine=engine,
                n_jobs=replay_jobs,
                exploration_only=True,
            )
            return replay.exploration_max, replay.n_fits

    else:
        observed = search_result.score_selected
        name = "full_search_max"

        def statistic(x_null: NDArray[np.float64]) -> tuple[float, int]:
            replay = _search(
                x_null, config=config, random_state=random_state, engine=engine, n_jobs=replay_jobs
            )
            return replay.score_selected, replay.n_fits

    return _nulls.null_test(
        x,
        observed,
        statistic,
        families=config.null_families,
        statistic_name=name,
        n_null_max=config.n_null_max,
        n_exceedances=config.null_exceedances,
        alpha=config.alpha,
        random_state=random_state,
        n_jobs=null_jobs,
    )


def recover(
    X: ArrayLike,
    *,
    X_err: ArrayLike | None = None,
    measurement_sampler: MeasurementSampler | None = None,
    config: Config | None = None,
    random_state: int = 0,
    engine: Engine = "auto",
    n_jobs: int = 1,
) -> Result:
    """Recover recurrent density populations from a feature matrix.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix; rows with non-finite values are rejected and counted.
    X_err : array-like of shape (n_samples, n_features), optional
        Per-value standard deviations; each world draws ``X + X_err * N(0, 1)``.
    measurement_sampler : callable, optional
        Alternative measurement model ``sampler(rows, indices, rng)``.
    config : Config, optional
        Declared budgets and rules; defaults to ``Config()``.
    random_state : int, default 0
        Base seed of every stochastic stage.
    engine : {"auto", "cpu", "cuml"}, default "auto"
        Backend; ``auto`` prefers cuML when it imports with a device.
    n_jobs : int, default 1
        Worker processes inside a trial and over worlds. On the CPU with at
        most 20000 rows per resample, null realisations also run ``n_jobs``
        at a time. Forced to 1 on cuML.

    Returns
    -------
    Result
        The emitted partition, world evidence, consensus check, null
        calibration, flags and provenance.

    Raises
    ------
    ValueError
        For an invalid matrix, uncertainties, or job count.
    ImportError
        If ``engine="cuml"`` or ``config.check_backends`` is set and cuML
        cannot be used; raised before any stage runs.
    """
    started = time.perf_counter()
    cfg = Config() if config is None else config
    if X_err is not None and measurement_sampler is not None:
        msg = "Give X_err or measurement_sampler, not both."
        raise ValueError(msg)
    x, kept, x_err, n_input = _prepare(X, X_err)
    resolved = resolve_engine(engine)
    if cfg.check_backends:
        resolve_engine("cuml")
    jobs, forced = _resolve_jobs(resolved, n_jobs)
    seconds: dict[str, float] = {}
    fits: dict[str, int] = {}

    stage = time.perf_counter()
    search_result = _search(x, config=cfg, random_state=random_state, engine=resolved, n_jobs=jobs)
    seconds["search"] = time.perf_counter() - stage
    fits["search"] = search_result.n_fits

    stage = time.perf_counter()
    deployed = rescale_params(search_result.selected, search_result.plan.m, x.shape[0])
    reference = fit_pipeline(
        x,
        deployed,
        seed=derive_seed(random_state, 4),
        engine=resolved,
        scaling=cfg.scaling,
    )
    default_fit = fit_pipeline(
        x,
        default_params(),
        seed=derive_seed(random_state, 4, 1),
        engine=resolved,
        scaling=cfg.scaling,
    )
    seconds["reference"] = time.perf_counter() - stage
    fits["reference"] = 2

    stage = time.perf_counter()
    world_rows = round(cfg.world_fraction * x.shape[0])
    world_params = rescale_params(search_result.selected, search_result.plan.m, max(world_rows, 1))
    evidence = run_worlds(
        x,
        world_params,
        reference.labels,
        n_worlds=cfg.n_worlds,
        world_fraction=cfg.world_fraction,
        random_state=random_state,
        engine=resolved,
        scaling=cfg.scaling,
        n_jobs=jobs,
        x_err=x_err,
        sampler=measurement_sampler,
    )
    seconds["worlds"] = time.perf_counter() - stage
    fits["worlds"] = evidence.n_fits

    stage = time.perf_counter()
    consensus = consensus_check(
        evidence.world_labels,
        deployed,
        reference.labels,
        budget_bytes=cfg.consensus_budget_bytes,
        check_rows=cfg.consensus_check_rows,
        seed=derive_seed(random_state, 7),
        engine=resolved,
    )
    seconds["consensus"] = time.perf_counter() - stage

    stage = time.perf_counter()
    null = _null_stage(
        x, search_result, config=cfg, random_state=random_state, engine=resolved, n_jobs=jobs
    )
    seconds["null"] = time.perf_counter() - stage
    fits["null"] = null.n_fits

    backends = None
    if cfg.check_backends:
        stage = time.perf_counter()
        backends = compare_backends(x, deployed, scaling=cfg.scaling, random_state=random_state)
        seconds["backends"] = time.perf_counter() - stage
        fits["backends"] = 2 * len(backends.agreement)

    seconds["total"] = time.perf_counter() - started
    flags = derive_flags(
        labels=reference.labels,
        search=search_result,
        worlds=evidence,
        consensus=consensus,
        null=null,
        backends=backends,
        rows_rejected=n_input - int(x.shape[0]),
        n_jobs_forced=forced,
    )
    provenance = build_provenance(
        x=x,
        config=cfg,
        random_state=random_state,
        engine_requested=str(engine),
        engine_resolved=str(resolved),
        n_jobs=jobs,
        n_jobs_forced=forced,
        n_input_rows=n_input,
        stage_seconds=seconds,
        stage_fits=fits,
    )
    return Result(
        labels=reference.labels,
        strengths=reference.strengths,
        default_labels=default_fit.labels,
        kept_rows=kept,
        params=search_result.selected,
        deployed_params=deployed,
        default_params=default_params(),
        search=search_result,
        worlds=evidence,
        consensus=consensus,
        null=null,
        backends=backends,
        flags=flags,
        provenance=provenance,
    )


def null_test(
    X: ArrayLike,
    search_result: SearchResult,
    *,
    config: Config | None = None,
    families: tuple[str, ...] = ("gaussian_copula",),
    random_state: int = 0,
    engine: Engine = "auto",
    n_jobs: int = 1,
) -> NullEvidence:
    """Run the matched null for a completed search, separately from ``recover``.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The finite matrix the search used.
    search_result : SearchResult
        The search whose observed statistic is calibrated.
    config, random_state, engine
        Must match the search so the replay is matched.
    families : tuple of str, default ("gaussian_copula",)
        Generator families, replacing ``config.null_families``.
    n_jobs : int, default 1
        As in ``recover``; above 20000 rows per resample it must also match
        the search, because the approximate neighbour graph depends on it.

    Returns
    -------
    NullEvidence
        Matched null outcome per generator family.
    """
    cfg = replace(Config() if config is None else config, null_families=tuple(families))
    x, kept, _, _ = _prepare(X, None)
    if kept.size != np.asarray(X).shape[0]:
        msg = "null_test requires the finite matrix the search used."
        raise ValueError(msg)
    resolved = resolve_engine(engine)
    jobs, _ = _resolve_jobs(resolved, n_jobs)
    return _null_stage(
        x, search_result, config=cfg, random_state=random_state, engine=resolved, n_jobs=jobs
    )


__all__ = [
    "Config",
    "Space",
    "null_test",
    "recover",
]
