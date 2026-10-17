"""Behavioural tests for ``starfold.search``.

``evaluate``, ``scale`` and the neighbour-graph builders are replaced by
deterministic fakes, the objective keyed on ``Params``, so the unit tests run no
UMAP fit. Every array here is a synthetic fixture built inside this file.
"""

from __future__ import annotations

import importlib
import math
import weakref
from dataclasses import dataclass, field, replace

import numpy as np
import optuna
import pytest
import scipy.sparse as sps

from starfold import backend, objective

# The package exports the function ``search`` under the module's name.
sm = importlib.import_module("starfold.search")


@dataclass(frozen=True)
class Space:
    n_neighbors: tuple = (5, 100)
    min_dist: tuple = (0.0, 0.5)
    n_components: tuple = (2, 10)
    min_cluster_size: tuple | None = None
    min_samples_fraction: tuple = (0.05, 1.0)
    selection_method: tuple = ("eom", "leaf")


@dataclass(frozen=True)
class Config:
    space: Space = field(default_factory=Space)
    scaling: str = "standard"
    n_trials: int = 8
    n_explore: int = 2
    n_resamples: int = 3
    resample_fraction: float = 0.8
    patience: int = 100
    min_improvement: float = 0.01
    prune: bool = False
    n_complete_pruned: int = 3
    n_rescore: int = 3
    n_null_rescore: int = 0
    gap_family: str | None = None


@dataclass(frozen=True)
class Call:
    params: object
    fresh: bool
    n_cached: int
    pruned: bool
    hooked: bool
    may_prune: bool


def _fit(m, r):
    labels = np.zeros(m, dtype=np.int32)
    labels[: m // 10] = -1
    labels[m // 2 :] = 1
    strengths = np.ones(m, dtype=np.float32)
    embedding = np.zeros((m, 2), dtype=np.float32)
    return backend.Fit(
        labels=labels, strengths=strengths, embedding=embedding, n_groups=2 + r, seconds=0.0
    )


class FakeEvaluator:
    """Stand-in for ``objective.evaluate`` keyed on ``Params``.

    ``fixed`` maps design points to ``(intermediate, final)``; a configuration seen
    for the first time draws the next pair from ``sequence``; ``fresh_shift`` is
    added to the final score of non-design points on the fresh plan; ``errors``
    marks points whose evaluation reports a failed fit.
    """

    def __init__(self, fixed, sequence=(), *, fresh_shift=0.0, errors=()):
        self.fixed = dict(fixed)
        self.assigned = {}
        self.sequence = list(sequence)
        self.fresh_shift = fresh_shift
        self.errors = set(errors)
        self.main_seed = None
        self.calls = []

    def lookup(self, params, *, fresh):
        if params in self.fixed:
            return self.fixed[params]
        if params not in self.assigned:
            self.assigned[params] = self.sequence.pop(0) if self.sequence else (0.3, 0.3)
        intermediate, final = self.assigned[params]
        return intermediate, final + (self.fresh_shift if fresh else 0.0)

    def __call__(
        self,
        x,
        params,
        plan,
        *,
        graphs,
        adjacency,
        engine,
        scaling,
        n_jobs=1,
        should_prune=None,
        may_prune=True,
        cached_fits=None,
        neighbours=None,
    ):
        assert x.shape[0] == plan.n
        assert len(graphs) == plan.n_resamples
        assert len(adjacency) == plan.n_resamples
        assert (engine, scaling, n_jobs) == ("cpu", "standard", 1)
        pairs = range(plan.n_resamples)
        assert set(neighbours) == {(r, s) for r in pairs for s in pairs if r != s}
        if self.main_seed is None:
            self.main_seed = plan.seed
        fresh = plan.seed != self.main_seed
        intermediate, final = self.lookup(params, fresh=fresh)
        fits = list(cached_fits or ())
        while len(fits) < 2:
            fits.append(_fit(plan.m, len(fits)))
        pruned = should_prune is not None and should_prune(intermediate) and may_prune
        n_cached = len(cached_fits or ())
        hooked = should_prune is not None
        self.calls.append(Call(params, fresh, n_cached, pruned, hooked, may_prune))
        if pruned:
            raise objective.TrialPruned(intermediate, tuple(fits))
        while len(fits) < plan.n_resamples:
            fits.append(_fit(plan.m, len(fits)))
        r = plan.n_resamples
        n_pairs = r * (r - 1) // 2
        error = "fit raised" if params in self.errors else None
        score = objective.TrialScore(
            score=0.0 if error else final,
            stability=final,
            coverage=1.0,
            modularity=0.5,
            pair_agreement=(final,) * n_pairs,
            pair_coverage=(1.0,) * n_pairs,
            fit_counts=tuple(fit.n_groups for fit in fits),
            fit_noise=(0.1,) * r,
            error=error,
        )
        return score, tuple(fits)


class NullAware(FakeEvaluator):
    """Scores 0.2 on the all-zero null matrix that the patched ``generate_null`` returns."""

    def __call__(self, x, params, plan, **kwargs):
        score, fits = super().__call__(x, params, plan, **kwargs)
        if not x.any():
            score = replace(score, score=0.2)
        return score, fits


def _zero_null(x, _family, _seed):
    return np.zeros_like(x)


@dataclass
class GraphLog:
    knn: list = field(default_factory=list)
    adjacency_k: list = field(default_factory=list)


def _patch(monkeypatch, evaluator):
    log = GraphLog()

    def knn_graph(x_scaled, k, *, engine, seed, n_jobs=1):
        m = x_scaled.shape[0]
        log.knn.append((m, k, engine, seed, n_jobs))
        return np.zeros((m, k + 1), dtype=np.int32), np.zeros((m, k + 1), dtype=np.float32)

    def neighbor_adjacency(knn, k=15):
        log.adjacency_k.append(k)
        m = knn[0].shape[0]
        return sps.csr_matrix((m, m), dtype=np.float32)

    def scale(x_rows, scaling):
        assert scaling == "standard"
        return x_rows.astype(np.float32)

    monkeypatch.setattr(sm, "knn_graph", knn_graph)
    monkeypatch.setattr(sm, "neighbor_adjacency", neighbor_adjacency)
    monkeypatch.setattr(sm, "scale", scale)
    monkeypatch.setattr(sm, "evaluate", evaluator)
    return log


def _matrix(n=120, d=3):
    return np.random.default_rng(7).standard_normal((n, d))


def _design(config, n, seed):
    plan = sm.make_plan(
        n, n_resamples=config.n_resamples, fraction=config.resample_fraction, seed=seed
    )
    space = sm.resolve_space(config.space, n, plan.m)
    return (sm.default_params(), *_sobol(space, config.n_explore, seed))


def _sobol(space, n_points, seed):
    rows = sm._sobol_rows(space, n_points, seed)  # noqa: SLF001 - the design has no public entry
    return tuple(sm._params_from(row) for row in rows)  # noqa: SLF001


def _run(monkeypatch, config, evaluator, *, seed=11, **kwargs):
    log = _patch(monkeypatch, evaluator)
    result = sm.search(_matrix(), config=config, random_state=seed, engine="cpu", **kwargs)
    return result, log


@pytest.mark.parametrize(
    ("space", "n", "m", "attribute", "expected"),
    [
        (Space(), 1000, 800, "n_neighbors", (5, 100)),
        (Space(), 100, 80, "n_neighbors", (5, 79)),
        (Space(), 1000, 800, "min_cluster_size", (5, 50)),
        (Space(), 40000, 32000, "min_cluster_size", (5, 1023)),
        (Space(min_cluster_size=(5, 30)), 1000, 800, "min_cluster_size", (5, 30)),
        (Space(), 1000, 800, "min_samples_fraction", (0.05, 1.0)),
        (Space(), 1000, 800, "selection_method", ("eom", "leaf")),
    ],
)
def test_resolve_space_applies_data_rules(space, n, m, attribute, expected):
    assert getattr(sm.resolve_space(space, n, m), attribute) == expected


@pytest.mark.parametrize(
    ("space", "m", "message"),
    [
        (Space(selection_method=("eom", "centroid")), 800, "selection_method"),
        (Space(min_samples_fraction=(0.0, 1.0)), 800, "log axis"),
        (Space(), 5, "empty"),
        (Space(n_neighbors=(5, 200)), 800, "cached neighbour graph"),
    ],
)
def test_resolve_space_rejects_bad_spaces(space, m, message):
    with pytest.raises(ValueError, match=message):
        sm.resolve_space(space, 1000, m)


@pytest.mark.parametrize(
    "space",
    [
        Space(n_neighbors=(20, 100)),
        Space(min_dist=(0.2, 0.5)),
        Space(n_components=(3, 10)),
        Space(min_cluster_size=(10, 50)),
        Space(min_samples_fraction=(0.05, 0.5)),
        Space(selection_method=("leaf",)),
    ],
)
def test_bounds_excluding_the_default_resolve_and_clip_trial_zero(space):
    resolved = sm.resolve_space(space, 1000, 800)
    d = sm.default_in_space(resolved)
    assert resolved.n_neighbors[0] <= d.n_neighbors <= resolved.n_neighbors[1]
    assert resolved.min_dist[0] <= d.min_dist <= resolved.min_dist[1]
    assert resolved.n_components[0] <= d.n_components <= resolved.n_components[1]
    assert resolved.min_cluster_size[0] <= d.min_cluster_size <= resolved.min_cluster_size[1]
    assert d.selection_method in resolved.selection_method
    assert 1 <= d.min_samples <= d.min_cluster_size


def test_sobol_design_is_seeded_and_inside_the_space():
    space = sm.resolve_space(Space(), 1000, 800)
    design = _sobol(space, 16, 3)
    assert len(design) == 16
    assert design == _sobol(space, 16, 3)
    assert design != _sobol(space, 16, 4)
    assert [p.selection_method for p in design] == ["eom", "leaf"] * 8
    for p in design:
        assert 5 <= p.n_neighbors <= 100
        assert 0.0 <= p.min_dist <= 0.5
        assert 2 <= p.n_components <= 10
        assert 5 <= p.min_cluster_size <= 50
        assert 1 <= p.min_samples <= p.min_cluster_size
    assert min(p.n_neighbors for p in design) < 11
    assert max(p.n_neighbors for p in design) > 47
    assert {2, 10} <= {p.n_components for p in design}
    assert _sobol(space, 0, 3) == ()


@pytest.mark.parametrize("n_points", [6, -4])
def test_sobol_design_needs_a_power_of_two(n_points):
    space = sm.resolve_space(Space(), 1000, 800)
    with pytest.raises(ValueError, match="power of two"):
        _sobol(space, n_points, 3)


@pytest.mark.parametrize(
    ("fraction", "mcs", "expected"),
    [(0.05, 5, 1), (1.0, 5, 5), (0.3, 5, 2), (0.5, 7, 4), (0.05, 40, 2)],
)
def test_suggest_params_clips_min_samples(fraction, mcs, expected):
    space = sm.resolve_space(Space(), 1000, 800)
    values = {
        "n_neighbors": 15,
        "min_dist": 0.1,
        "n_components": 2,
        "min_cluster_size": mcs,
        "min_samples_fraction": fraction,
        "selection_method": "leaf",
    }
    params = sm.suggest_params(optuna.trial.FixedTrial(values), space)
    assert params.min_samples == expected
    assert params.min_cluster_size == mcs
    assert params.selection_method == "leaf"
    assert (params.n_neighbors, params.min_dist, params.n_components) == (15, 0.1, 2)


def test_search_runs_default_then_design_then_tpe(monkeypatch):
    config = Config(n_trials=8, n_explore=4, n_rescore=2)
    design = _design(config, 120, 11)
    fixed = {p: (0.4 + 0.05 * i, 0.4 + 0.05 * i) for i, p in enumerate(design)}
    evaluator = FakeEvaluator(fixed, sequence=[(0.2, 0.2)] * 10)
    result, log = _run(monkeypatch, config, evaluator)
    assert [r.params for r in result.trials[:5]] == list(design)
    assert result.trials[0].params == sm.default_params()
    assert [r.number for r in result.trials] == list(range(8))
    assert [r.phase for r in result.trials] == [
        "default",
        *["exploration"] * 4,
        *["convergence"] * 3,
    ]
    assert {r.state for r in result.trials} == {"complete"}
    assert result.stop_reason == "budget"
    assert result.exploration_max == pytest.approx(0.6)
    assert result.plan.n == 120
    assert result.plan.n_resamples == 3
    assert [(m, k, engine) for m, k, engine, _, _ in log.knn] == [(96, 95, "cpu")] * 6
    assert log.adjacency_k == [15] * 6
    main_seeds = [seed for _, _, _, seed, _ in log.knn[:3]]
    assert main_seeds == list(result.plan.seeds)
    assert set(main_seeds).isdisjoint(seed for _, _, _, seed, _ in log.knn[3:])
    assert not any(call.hooked for call in evaluator.calls)
    record = result.trials[1]
    assert record.fit_counts == (2, 3, 4)
    assert record.fit_noise == (0.1, 0.1, 0.1)
    assert record.partial_score is None
    assert record.error is None
    assert record.seconds >= 0.0


def test_search_selects_by_fresh_rescoring(monkeypatch):
    config = Config(n_trials=8, n_explore=4, n_rescore=2)
    design = _design(config, 120, 11)
    fixed = {p: (0.4 + 0.05 * i, 0.4 + 0.05 * i) for i, p in enumerate(design)}
    evaluator = FakeEvaluator(fixed, sequence=[(0.2, 0.2)] * 10)
    result, _log = _run(monkeypatch, config, evaluator)
    assert [c.trial_number for c in result.candidates] == [4, 3, 0]
    assert result.selected == design[4]
    assert result.score_selected == pytest.approx(0.6)
    assert result.score_default == pytest.approx(0.4)
    assert result.beats_default_under_objective
    assert not result.selected_is_default
    expected = math.cbrt(0.6 * 0.5) - math.cbrt(0.4 * 0.5)
    assert result.paired_differences == pytest.approx((expected,) * 3)
    assert result.candidates[0].fresh_pair_scores == pytest.approx((math.cbrt(0.3),) * 3)
    assert result.candidates[0].fit_counts == (2, 3, 4)
    assert sum(call.fresh for call in evaluator.calls) == 3
    assert result.n_fits == 8 * 3 + 3 * 3
    assert (result.n_completed, result.n_pruned, result.n_completed_after_prune) == (8, 0, 0)


def test_search_prunes_and_completes_the_near_winner(monkeypatch):
    config = Config(n_trials=8, n_explore=2, prune=True, n_complete_pruned=1, n_rescore=2)
    design = _design(config, 120, 11)
    fixed = dict.fromkeys(design, (0.5, 0.2))
    sequence = [(0.9, 0.25), (0.45, 0.6), (0.9, 0.25), (0.45, 0.6), (0.9, 0.25)]
    evaluator = FakeEvaluator(fixed, sequence=sequence)
    result, _log = _run(monkeypatch, config, evaluator)
    states = [r.state for r in result.trials]
    assert states == ["complete"] * 6 + ["pruned", "complete"]
    completed = result.trials[4]
    assert completed.phase == "completion"
    assert completed.score == pytest.approx(0.6)
    assert completed.partial_score is None
    pruned = result.trials[6]
    assert pruned.phase == "convergence"
    assert pruned.score == pytest.approx(0.45)
    assert pruned.partial_score == pytest.approx(0.45)
    assert math.isnan(pruned.stability)
    assert pruned.fit_counts == (2, 3)
    assert pruned.fit_noise == pytest.approx((9 / 96,) * 2)
    assert (result.n_completed, result.n_pruned, result.n_completed_after_prune) == (7, 1, 1)
    main_calls = [call for call in evaluator.calls if not call.fresh]
    assert all(call.hooked for call in main_calls if call.n_cached == 0)
    hooked = [call.may_prune for call in main_calls if call.hooked]
    assert hooked == [False] * 3 + [True] * (len(hooked) - 3)
    assert [call.params for call in main_calls if call.pruned] == [completed.params, pruned.params]
    assert [call.params for call in main_calls if call.n_cached == 2] == [completed.params]
    assert [c.trial_number for c in result.candidates] == [4, 3, 0]
    assert result.selected == completed.params
    assert result.n_fits == 3 * 3 + 3 * 3 + 2 * 2 + 1 + 3 * 3
    assert result.stop_reason == "budget"


@pytest.mark.parametrize(("step", "reason", "n_ran"), [(0.0, "patience", 6), (0.05, "budget", 8)])
def test_search_stop_reasons(monkeypatch, step, reason, n_ran):
    config = Config(n_trials=8, n_explore=2, patience=2, min_improvement=0.01)
    fixed = dict.fromkeys(_design(config, 120, 11), (0.5, 0.5))
    sequence = [(0.3, 0.5 + step * (i + 1)) for i in range(8)]
    evaluator = FakeEvaluator(fixed, sequence=sequence)
    result, _log = _run(monkeypatch, config, evaluator)
    assert len(result.trials) == n_ran
    assert result.stop_reason == reason


def test_patience_on_the_final_trial_reports_the_budget(monkeypatch):
    config = Config(n_trials=6, n_explore=2, patience=2, min_improvement=0.01)
    fixed = dict.fromkeys(_design(config, 120, 11), (0.5, 0.5))
    evaluator = FakeEvaluator(fixed, sequence=[(0.3, 0.5)] * 6)
    result, _log = _run(monkeypatch, config, evaluator)
    assert len(result.trials) == 6
    assert result.stop_reason == "budget"


def test_search_keeps_the_default_on_a_fresh_tie(monkeypatch):
    config = Config(n_trials=6, n_explore=2, n_rescore=2)
    design = _design(config, 120, 11)
    fixed = {design[0]: (0.5, 0.5), design[1]: (0.4, 0.4), design[2]: (0.45, 0.45)}
    sequence = [(0.8, 0.8), (0.7, 0.7), (0.65, 0.65)]
    evaluator = FakeEvaluator(fixed, sequence=sequence, fresh_shift=-0.3)
    result, _log = _run(monkeypatch, config, evaluator)
    assert [c.trial_number for c in result.candidates] == [0, 3, 4]
    assert result.candidates[1].score == pytest.approx(0.8)
    assert result.candidates[1].fresh_score == pytest.approx(0.5)
    assert result.selected == sm.default_params()
    assert result.selected_is_default
    assert not result.beats_default_under_objective
    assert result.score_selected == pytest.approx(result.score_default)
    assert result.paired_differences == pytest.approx((0.0,) * 3)


def test_search_exploration_only_runs_the_design_alone(monkeypatch):
    config = Config(n_trials=48, n_explore=4, prune=True)
    design = _design(config, 120, 11)
    scores = [0.5, 0.4, 0.7, 0.3, 0.6]
    fixed = {p: (s, s) for p, s in zip(design, scores, strict=True)}
    evaluator = FakeEvaluator(fixed)
    result, log = _run(monkeypatch, config, evaluator, exploration_only=True)
    assert [r.params for r in result.trials] == list(design)
    assert result.stop_reason == "exploration_only"
    assert result.candidates == ()
    assert result.paired_differences == ()
    assert result.selected == design[2]
    assert result.score_selected == pytest.approx(0.7)
    assert result.score_default == pytest.approx(0.5)
    assert result.exploration_max == pytest.approx(0.7)
    assert result.beats_default_under_objective
    assert not result.selected_is_default
    assert result.n_fits == 15
    assert len(log.knn) == 3
    assert not any(call.fresh or call.hooked for call in evaluator.calls)


def test_search_records_a_failed_trial(monkeypatch):
    config = Config(n_trials=4, n_explore=2)
    design = _design(config, 120, 11)
    fixed = {design[0]: (0.5, 0.5), design[1]: (0.9, 0.9), design[2]: (0.4, 0.4)}
    evaluator = FakeEvaluator(fixed, errors=[design[1]])
    result, _log = _run(monkeypatch, config, evaluator)
    failed = result.trials[1]
    assert failed.state == "failed"
    assert failed.score == 0.0
    assert failed.error == "fit raised"
    assert result.exploration_max == pytest.approx(0.5)
    assert 1 not in {c.trial_number for c in result.candidates}
    assert result.n_completed == 3


@pytest.mark.parametrize(
    ("partial", "n_complete_pruned", "n_after"), [(0.3, 3, 0), (0.5, 3, 3), (0.5, 1, 1)]
)
def test_completion_compares_the_leading_pair_with_the_best_gap(
    monkeypatch, partial, n_complete_pruned, n_after
):
    """Design trials score raw 0.6 and gap 0.4; a partial above the gap is completed.

    The data score bounds the gap, so 0.5, below the best data score, can still
    win on gap, and does: it completes at 0.9 - 0.2 = 0.7. Only trials above the
    best gap keep their fits, and none are held after completion.
    """
    config = Config(
        n_trials=6,
        n_explore=2,
        prune=True,
        n_complete_pruned=n_complete_pruned,
        n_rescore=2,
        n_null_rescore=1,
        gap_family="gaussian_copula",
    )
    monkeypatch.setattr(sm, "generate_null", _zero_null)
    held = []
    complete_pruned = sm._Run.complete_pruned  # noqa: SLF001 - the study state has no public entry

    def logged(self, main, null):
        held.append(set(self.pruned_fits))
        n = complete_pruned(self, main, null)
        held.append(set(self.pruned_fits))
        return n

    monkeypatch.setattr(sm._Run, "complete_pruned", logged)  # noqa: SLF001
    design = _design(config, 120, 11)
    evaluator = NullAware(dict.fromkeys(design, (0.7, 0.6)), sequence=[(partial, 0.9)] * 3)
    result, _log = _run(monkeypatch, config, evaluator)
    assert all(r.score == pytest.approx(0.4) for r in result.trials[:3])
    assert all(r.raw_score == pytest.approx(0.6) for r in result.trials[:3])
    assert result.n_completed_after_prune == n_after
    assert result.n_pruned == 3 - n_after
    completed = [r for r in result.trials if r.phase == "completion"]
    assert all(r.score == pytest.approx(0.7) for r in completed)
    cached = {3, 4, 5} if partial > 0.4 else set()
    assert held == [cached, set()]
    assert {r.number for r in completed} <= cached


def test_rescore_builds_one_bundle_at_a_time(monkeypatch):
    config = Config(
        n_trials=4, n_explore=2, n_rescore=2, n_null_rescore=2, gap_family="gaussian_copula"
    )
    monkeypatch.setattr(sm, "generate_null", _zero_null)
    events = []

    class Logged(NullAware):
        def __call__(self, x, params, plan, **kwargs):
            events.append("s")
            return super().__call__(x, params, plan, **kwargs)

    evaluator = Logged(dict.fromkeys(_design(config, 120, 11), (0.5, 0.5)))
    _patch(monkeypatch, evaluator)
    fake_knn = sm.knn_graph

    def knn_graph(*args, **kwargs):
        events.append("b")
        return fake_knn(*args, **kwargs)

    monkeypatch.setattr(sm, "knn_graph", knn_graph)
    result = sm.search(_matrix(), config=config, random_state=11, engine="cpu")
    n = len(result.candidates)
    assert "".join(events) == "b" * 6 + "s" * 8 + ("bbb" + "s" * n) * 3


def test_rescore_starts_after_the_study_bundles_are_released(monkeypatch):
    config = Config(
        n_trials=6,
        n_explore=2,
        prune=True,
        n_rescore=2,
        n_null_rescore=1,
        gap_family="gaussian_copula",
    )
    monkeypatch.setattr(sm, "generate_null", _zero_null)
    built = []
    alive = []

    class Probe(NullAware):
        def __call__(self, x, params, plan, **kwargs):
            if self.main_seed is not None and plan.seed != self.main_seed and not alive:
                alive.append(sum(ref() is not None for ref in built[:2]))
            return super().__call__(x, params, plan, **kwargs)

    design = _design(config, 120, 11)
    evaluator = Probe(dict.fromkeys(design, (0.7, 0.6)), sequence=[(0.3, 0.9)] * 3)
    _patch(monkeypatch, evaluator)
    build = sm._build  # noqa: SLF001 - the bundle builder has no public entry

    def logged_build(*args):
        bundle = build(*args)
        built.append(weakref.ref(bundle))
        return bundle

    monkeypatch.setattr(sm, "_build", logged_build)
    result = sm.search(_matrix(), config=config, random_state=11, engine="cpu")
    assert result.n_pruned == 3
    assert alive == [0]


def test_search_rejects_an_unresolved_engine(monkeypatch):
    evaluator = FakeEvaluator({})
    _patch(monkeypatch, evaluator)
    with pytest.raises(ValueError, match="engine must be"):
        sm.search(_matrix(), config=Config(), random_state=0, engine="auto")
    assert evaluator.calls == []


def test_search_rejects_short_budgets_and_flat_input(monkeypatch):
    evaluator = FakeEvaluator({})
    _patch(monkeypatch, evaluator)
    with pytest.raises(ValueError, match="n_trials"):
        sm.search(_matrix(), config=Config(n_trials=2, n_explore=2), random_state=0, engine="cpu")
    with pytest.raises(ValueError, match="two-dimensional"):
        sm.search(np.zeros(120), config=Config(), random_state=0, engine="cpu")
    assert evaluator.calls == []


@pytest.mark.slow
def test_search_real_two_blobs():
    rng = np.random.default_rng(20260905)
    x = np.concatenate([rng.normal(-5.0, 1.0, (120, 3)), rng.normal(5.0, 1.0, (120, 3))])
    space = Space(n_neighbors=(5, 30), n_components=(2, 2), min_cluster_size=(5, 20))
    config = Config(
        space=space, n_trials=4, n_explore=2, n_resamples=2, patience=1, prune=True, n_rescore=2
    )
    result = sm.search(x, config=config, random_state=0, engine="cpu")
    assert len(result.trials) == 4
    assert result.trials[0].params == sm.default_params()
    assert {r.params.n_components for r in result.trials} == {2}
    assert result.stop_reason == "budget"
    assert 0.0 <= result.score_selected <= 1.0
    assert result.n_fits >= 4 * 2 + 2 * 2
    assert result.beats_default_under_objective == (result.score_selected > result.score_default)
    assert len(result.candidates) >= 2


def test_search_scores_each_trial_against_a_null_matrix(monkeypatch):
    config = Config(
        n_trials=4, n_explore=2, n_rescore=2, n_null_rescore=2, gap_family="gaussian_copula"
    )
    draws = []

    def generate_null(x, family, seed):
        draws.append((family, seed))
        return np.zeros_like(x)

    monkeypatch.setattr(sm, "generate_null", generate_null)
    design = _design(config, 120, 11)
    evaluator = NullAware(dict.fromkeys(design, (0.5, 0.6)), sequence=[(0.5, 0.9)])
    result, _log = _run(monkeypatch, config, evaluator)
    complete = [r for r in result.trials if r.state == "complete"]
    assert len(complete) == 4
    assert all(r.null_score == pytest.approx(0.2) for r in complete)
    assert all(r.score == pytest.approx(r.raw_score - 0.2) for r in complete)
    assert draws[0][0] == "gaussian_copula"
    assert len(draws) == 3
    assert len({seed for _family, seed in draws}) == 3
    for candidate in result.candidates:
        assert len(candidate.null_scores) == 2
        assert candidate.gap == pytest.approx(candidate.fresh_score - 0.2)
    assert result.gap_selected == pytest.approx(result.score_selected - 0.2)
    assert result.gap_default == pytest.approx(result.score_default - 0.2)
    assert result.selected == complete[3].params


def test_user_bounds_may_exclude_the_library_default():
    """A floor on min_cluster_size above HDBSCAN's 5 resolves; trial 0 is the clipped default."""
    declared = Space(min_cluster_size=(1000, 2000), min_samples_fraction=(0.05, 0.25))
    space = sm.resolve_space(declared, n=90_000, m=72_000)
    d = sm.default_in_space(space)
    assert space.min_cluster_size == (1000, 2000)
    assert d.min_cluster_size == 1000
    assert 50 <= d.min_samples <= 250
    assert d.n_neighbors == 15
    assert d.selection_method == "eom"


def test_clipped_default_min_samples_keeps_its_fraction_inside_the_bounds():
    """Rounding the clipped fraction times min_cluster_size must not step past the fraction cap."""
    declared = Space(min_cluster_size=(1547, 9959), min_samples_fraction=(0.0010082, 0.010082))
    space = sm.resolve_space(declared, n=199_190, m=159_352)
    d = sm.default_in_space(space)
    lo, hi = space.min_samples_fraction
    assert lo <= d.min_samples / d.min_cluster_size <= hi
