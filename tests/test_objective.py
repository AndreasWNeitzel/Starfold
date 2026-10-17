from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
import scipy.sparse

from starfold import objective
from starfold.backend import Fit, Params, knn_graph, neighbor_adjacency
from starfold.objective import (
    TrialPruned,
    TrialScore,
    combine,
    evaluate,
    heldout_neighbours,
    modularity,
    pair_stability,
    score_fits,
    transfer_labels,
)
from starfold.prep import make_plan, scale, validate_matrix

IDENTITY = np.arange(100, dtype=np.intp)
PARAMS = Params(15, 0.1, 2, 10, 5, "eom")


def _labels(values) -> np.ndarray:
    return np.asarray(values, dtype=np.int32)


def _fake_fit(labels: np.ndarray) -> Fit:
    labels = _labels(labels)
    return Fit(
        labels=labels,
        strengths=np.ones(labels.size, dtype=np.float32),
        embedding=np.zeros((labels.size, 2), dtype=np.float32),
        n_groups=int(np.unique(labels[labels >= 0]).size),
        seconds=0.25,
    )


def _two_triangles_with_bridge() -> scipy.sparse.csr_matrix:
    edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]
    rows = [a for a, b in edges] + [b for a, b in edges]
    cols = [b for a, b in edges] + [a for a, b in edges]
    return scipy.sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(6, 6)
    )


def test_pair_stability_all_noise_both_sides():
    noise = _labels(np.full(100, -1))
    assert pair_stability(noise, noise, IDENTITY, IDENTITY) == (0.0, 0.0)


def test_pair_stability_one_cluster_holding_every_row():
    one = _labels(np.zeros(100))
    assert pair_stability(one, one, IDENTITY, IDENTITY) == (0.0, 1.0)


def test_pair_stability_one_tiny_cluster_plus_noise():
    tiny = _labels(np.full(100, -1))
    tiny[:3] = 0
    assert pair_stability(tiny, tiny, IDENTITY, IDENTITY) == (0.0, 0.03)


def test_pair_stability_two_recurrent_tiny_clusters():
    a = _labels(np.full(100, -1))
    a[:3] = 0
    a[3:6] = 1
    b = a.copy()
    b[:3] = 5
    b[3:6] = 2
    assert pair_stability(a, b, IDENTITY, IDENTITY) == (1.0, 0.06)


def test_pair_stability_single_joint_row_scores_zero():
    a = _labels(np.full(100, -1))
    a[0] = 0
    a[1] = 1
    b = _labels(np.full(100, -1))
    b[0] = 0
    b[2] = 1
    assert pair_stability(a, b, IDENTITY, IDENTITY) == (0.0, 0.01)


def test_pair_stability_empty_overlap():
    empty = np.zeros(0, dtype=np.intp)
    assert pair_stability(_labels([0, 1]), _labels([0, 1]), empty, empty) == (0.0, 0.0)


def test_pair_stability_clips_negative_ari_and_uses_positions():
    a = _labels([0, 0, 1, 1, 0, 1, -1, 9])
    b = _labels([1, 0, 1, 0, 9, 9, -1, 9])
    pos_a = np.array([0, 1, 2, 3], dtype=np.intp)
    pos_b = np.array([0, 2, 1, 3], dtype=np.intp)
    agreement, coverage = pair_stability(a, b, pos_a, pos_b)
    assert agreement == 1.0
    assert coverage == 1.0
    worse = pair_stability(a, b, np.array([0, 1, 2, 3], dtype=np.intp), pos_a)
    assert worse[0] == 0.0
    assert worse[1] == 1.0
    partial = pair_stability(a, b, np.array([0, 1, 6, 7], dtype=np.intp), np.arange(4, 8))
    assert partial == (0.0, 0.75)


def test_pair_stability_rejects_misaligned_positions():
    with pytest.raises(ValueError, match="must align"):
        pair_stability(_labels([0, 1]), _labels([0, 1]), IDENTITY[:2], IDENTITY[:3])


def test_modularity_hand_computed_six_node_graph():
    graph = _two_triangles_with_bridge()
    assert modularity(_labels([0, 0, 0, 1, 1, 1]), graph) == pytest.approx(float(Fraction(5, 14)))
    assert modularity(_labels([0, 0, 0, 1, 1, -1]), graph) == pytest.approx(float(Fraction(19, 98)))
    assert modularity(_labels([0, 1, 0, 1, 0, 1]), graph) == pytest.approx(float(Fraction(-3, 14)))
    assert modularity(_labels([0, 0, 0, 0, 0, 0]), graph) == pytest.approx(0.0)
    assert modularity(_labels([-1] * 6), graph) == 0.0
    assert modularity(_labels([3, 3, 3, 7, 7, 7]), graph) == pytest.approx(float(Fraction(5, 14)))


def test_modularity_empty_graph_and_shape_check():
    empty = scipy.sparse.csr_matrix((6, 6), dtype=np.float32)
    assert modularity(_labels([0, 0, 0, 1, 1, 1]), empty) == 0.0
    with pytest.raises(ValueError, match="adjacency must have shape"):
        modularity(_labels([0, 1]), _two_triangles_with_bridge())


def test_combine_geometric_mean_and_clipping():
    assert combine(1.0, 1.0, 1.0) == 1.0
    assert combine(0.5, 0.5, 0.5) == pytest.approx(0.5)
    assert combine(0.0, 1.0, 1.0) == 0.0
    assert combine(1.0, 1.0, -0.2) == 0.0
    assert combine(0.8, 0.5, 0.3) == pytest.approx((0.8 * 0.5 * 0.3) ** (1 / 3))


def _naive_modularity(labels, dense) -> float:
    """Newman's Q by the textbook double loop over ordered pairs."""
    two_e = float(dense.sum())
    if two_e == 0.0:
        return 0.0
    degrees = dense.sum(axis=1)
    total = 0.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            if labels[i] >= 0 and labels[i] == labels[j]:
                total += dense[i, j] - degrees[i] * degrees[j] / two_e
    return total / two_e


def _line_setup():
    rng = np.random.default_rng(12)
    x = np.concatenate([rng.normal(-5.0, 0.2, (50, 2)), rng.normal(5.0, 0.2, (50, 2))]).astype(
        np.float64
    )
    truth = np.repeat(np.array([0, 1], dtype=np.int32), 50)
    plan = make_plan(100, n_resamples=3, fraction=0.8, seed=4)
    adjacency = [
        neighbor_adjacency(knn_graph(scale(x[rows], "standard"), 6, engine="cpu", seed=0), k=5)
        for rows in plan.indices
    ]
    return x, truth, plan, adjacency


def test_score_fits_on_recurrent_true_partition():
    x, truth, plan, adjacency = _line_setup()
    fits = [_fake_fit(truth[rows]) for rows in plan.indices]
    score = score_fits(fits, plan, adjacency, x=x, scaling="standard")
    assert score.error is None
    assert score.pair_agreement == (1.0,) * 6
    assert score.pair_coverage == (1.0,) * 6
    assert score.fit_counts == (2, 2, 2)
    assert score.fit_noise == (0.0, 0.0, 0.0)
    pairs = list(zip(plan.indices, adjacency, strict=True))
    expected_q = [
        _naive_modularity(truth[rows], adj.toarray().astype(np.float64)) for rows, adj in pairs
    ]
    q_values = [modularity(truth[rows], adj) for rows, adj in pairs]
    np.testing.assert_allclose(q_values, expected_q, rtol=1e-12)
    assert all(0.4 < q <= 0.5 for q in expected_q)
    assert score.modularity == pytest.approx(np.mean(expected_q))
    assert score.score == pytest.approx(combine(1.0, 1.0, float(np.mean(expected_q))))


def test_score_fits_degenerate_and_partial_inputs():
    x, truth, plan, adjacency = _line_setup()
    kw = {"x": x, "scaling": "standard"}
    noise = [_fake_fit(np.full(plan.m, -1)) for _ in range(3)]
    assert score_fits(noise, plan, adjacency, **kw).score == 0.0
    assert score_fits(noise, plan, adjacency, **kw).coverage == 0.0
    fits = [_fake_fit(truth[rows]) for rows in plan.indices]
    q = [modularity(fit.labels, adj) for fit, adj in zip(fits, adjacency, strict=True)]
    partial = score_fits([fits[0], None, fits[2]], plan, adjacency, **kw)
    assert len(partial.fit_counts) == 2
    assert len(partial.pair_agreement) == 2
    assert partial.modularity == pytest.approx(np.mean([q[0], q[2]]))
    assert partial.score == pytest.approx(combine(1.0, 1.0, float(np.mean([q[0], q[2]]))))
    single = score_fits(fits[:1], plan, adjacency, **kw)
    assert single.score == 0.0
    assert len(single.fit_counts) == 1
    assert single.pair_agreement == ()
    assert single.modularity == pytest.approx(q[0])
    with pytest.raises(ValueError, match="adjacency matrices"):
        score_fits(fits, plan, adjacency[:2], **kw)


def test_heldout_rows_and_transfer_on_separated_blobs():
    x, truth, plan, adjacency = _line_setup()
    rows, pos = plan.heldout(0, 1)
    assert rows.size > 0
    assert np.array_equal(plan.indices[1][pos], rows)
    assert not np.isin(rows, plan.indices[0]).any()
    predicted = transfer_labels(
        x[plan.indices[0]], truth[plan.indices[0]], x[rows], scaling="standard"
    )
    assert np.array_equal(predicted, truth[rows])
    kw = {"x": x, "scaling": "standard"}
    fits = [_fake_fit(truth[plan.indices[0]]), _fake_fit(truth[plan.indices[1]])]
    exact = score_fits(fits, plan, adjacency, **kw)
    assert (exact.pair_agreement, exact.pair_coverage) == ((1.0, 1.0), (1.0, 1.0))
    torn = truth[plan.indices[1]].copy()
    torn[: torn.size // 2] = np.arange(torn.size // 2) % 7 + 2
    worse = score_fits([fits[0], _fake_fit(torn)], plan, adjacency, **kw)
    assert worse.pair_agreement[0] < 1.0


def test_trial_pruned_carries_partial_and_fits():
    fit = _fake_fit([0, 1])
    error = TrialPruned(0.25, (fit,))
    assert error.partial == 0.25
    assert error.fits == (fit,)
    assert "0.2500" in str(error)
    assert isinstance(error, Exception)


def test_evaluate_reports_fit_errors_without_fitting(two_blobs):
    x, _kept = validate_matrix(two_blobs)
    plan = make_plan(x.shape[0], n_resamples=3, fraction=0.8, seed=3)
    graphs = [
        knn_graph(scale(x[rows], "standard"), 8, engine="cpu", seed=seed)
        for rows, seed in zip(plan.indices, plan.seeds, strict=True)
    ]
    adjacency = [neighbor_adjacency(graph, k=5) for graph in graphs]
    score, fits = evaluate(
        x, PARAMS, plan, graphs=graphs, adjacency=adjacency, engine="cpu", scaling="standard"
    )
    assert isinstance(score, TrialScore)
    assert score.score == 0.0
    assert score.error is not None
    assert score.error.startswith("ValueError")
    assert score.fit_counts == ()
    assert fits == ()
    with pytest.raises(ValueError, match="graphs and adjacencies"):
        evaluate(
            x,
            PARAMS,
            plan,
            graphs=graphs[:2],
            adjacency=adjacency,
            engine="cpu",
            scaling="standard",
        )


def test_evaluate_prunes_then_completes_from_cache(two_blobs):
    x, _kept = validate_matrix(two_blobs)
    plan = make_plan(x.shape[0], n_resamples=3, fraction=0.8, seed=3)
    graphs = [
        knn_graph(scale(x[rows], "standard"), 20, engine="cpu", seed=seed)
        for rows, seed in zip(plan.indices, plan.seeds, strict=True)
    ]
    adjacency = [neighbor_adjacency(graph, k=15) for graph in graphs]
    table = heldout_neighbours(x, plan, scaling="standard")
    offered: list[float] = []

    def prune_everything(value: float) -> bool:
        offered.append(value)
        return True

    with pytest.raises(TrialPruned) as info:
        evaluate(
            x,
            PARAMS,
            plan,
            graphs=graphs,
            adjacency=adjacency,
            engine="cpu",
            scaling="standard",
            should_prune=prune_everything,
        )
    pruned = info.value
    assert len(pruned.fits) == 2
    assert offered == [pruned.partial]
    assert 0.0 <= pruned.partial <= 1.0
    assert all(fit.labels.shape == (plan.m,) for fit in pruned.fits)
    assert all(fit.embedding.size == fit.strengths.size == 0 for fit in pruned.fits)

    score, fits = evaluate(
        x,
        PARAMS,
        plan,
        graphs=graphs,
        adjacency=adjacency,
        engine="cpu",
        scaling="standard",
        should_prune=lambda _value: False,
        cached_fits=pruned.fits,
        neighbours=table,
    )
    assert len(fits) == 3
    assert fits[0] is pruned.fits[0]
    assert fits[1] is pruned.fits[1]
    assert score.error is None
    assert score == score_fits(fits, plan, adjacency, x=x, scaling="standard")
    assert len(score.pair_agreement) == 6
    assert score.fit_counts == (2, 2, 2)
    assert score.stability > 0.9
    assert score.coverage > 0.9
    assert score.modularity > 0.4
    assert score.score > 0.5
    assert score.score == pytest.approx(combine(score.stability, score.coverage, score.modularity))
    assert pruned.partial == pytest.approx(
        score_fits(fits[:2], plan, adjacency, x=x, scaling="standard").score
    )


def _counting_fits(monkeypatch, plan, fail=()):
    """Replace UMAP and HDBSCAN by a sign rule; record parallel batch sizes and fitted resamples."""
    batches: list[int] = []
    fitted: list[int] = []

    def fit_pipeline(x_rows, _params, *, seed, engine, scaling, knn):
        assert (engine, scaling, knn) == ("cpu", "standard", None)
        r = plan.seeds.index(seed)
        fitted.append(r)
        if r in fail:
            msg = f"resample {r}"
            raise ValueError(msg)
        labels = (x_rows[:, 0] > 0).astype(np.int32)
        return _fake_fit(labels)

    def parallel_map(fn, items, *, n_jobs):
        assert n_jobs == 3
        batches.append(len(items))
        return [fn(item) for item in items]

    monkeypatch.setattr(objective, "fit_pipeline", fit_pipeline)
    monkeypatch.setattr(objective, "parallel_map", parallel_map)
    return batches, fitted


def _evaluate(x, plan, adjacency, **kwargs):
    graphs = [None] * plan.n_resamples
    return evaluate(
        x,
        PARAMS,
        plan,
        graphs=graphs,
        adjacency=adjacency,
        engine="cpu",
        scaling="standard",
        **kwargs,
    )


@pytest.mark.parametrize(
    ("answer", "may_prune", "batches"),
    [(None, True, [3]), (False, True, [2, 1]), (True, False, [3])],
)
def test_evaluate_batches_every_fit_unless_it_may_prune(monkeypatch, answer, may_prune, batches):
    x, _truth, plan, adjacency = _line_setup()
    seen, _fitted = _counting_fits(monkeypatch, plan)
    offered: list[float] = []

    def report(value: float) -> bool:
        offered.append(value)
        return bool(answer)

    hook = None if answer is None else report
    score, fits = _evaluate(x, plan, adjacency, n_jobs=3, should_prune=hook, may_prune=may_prune)
    assert seen == batches
    assert score == score_fits(fits, plan, adjacency, x=x, scaling="standard")
    assert all(fit.embedding.size == fit.strengths.size == 0 for fit in fits)
    leading = score_fits(fits[:2], plan, adjacency, x=x, scaling="standard").score
    assert offered == ([] if hook is None else [leading])


def test_evaluate_prunes_after_the_leading_batch(monkeypatch):
    x, _truth, plan, adjacency = _line_setup()
    seen, fitted = _counting_fits(monkeypatch, plan)
    with pytest.raises(TrialPruned) as info:
        _evaluate(x, plan, adjacency, n_jobs=3, should_prune=lambda _value: True)
    assert seen == [2]
    assert fitted == [0, 1]
    assert len(info.value.fits) == 2


@pytest.mark.parametrize(("fail", "reported"), [(2, True), (1, False), (0, False)])
def test_evaluate_reports_the_leading_pair_only_when_both_fitted(monkeypatch, fail, reported):
    x, _truth, plan, adjacency = _line_setup()
    _counting_fits(monkeypatch, plan, fail={fail})
    offered: list[float] = []

    def report(value: float) -> bool:
        offered.append(value)
        return False

    score, fits = _evaluate(x, plan, adjacency, n_jobs=3, should_prune=report, may_prune=False)
    assert score.error == f"ValueError: resample {fail}"
    assert score.score == 0.0
    assert len(fits) == fail
    assert len(offered) == int(reported)


def test_evaluate_in_one_process_stops_at_the_first_failure(monkeypatch):
    x, _truth, plan, adjacency = _line_setup()
    batches, fitted = _counting_fits(monkeypatch, plan, fail={0})
    score, fits = _evaluate(x, plan, adjacency)
    assert score.error == "ValueError: resample 0"
    assert fitted == [0]
    assert batches == []
    assert fits == ()


def test_heldout_neighbours_give_the_uncached_scores():
    x, truth, plan, adjacency = _line_setup()
    rng = np.random.default_rng(3)
    fits = []
    for rows in plan.indices:
        labels = truth[rows].copy()
        flip = rng.random(rows.size) < 0.2
        labels[flip] = rng.integers(-1, 3, size=int(flip.sum()))
        fits.append(_fake_fit(labels))
    table = heldout_neighbours(x, plan, scaling="robust")
    assert set(table) == {(r, s) for r in range(3) for s in range(3) if r != s}
    for (r, s), (pos_s, neighbours) in table.items():
        rows, expected = plan.heldout(r, s)
        np.testing.assert_array_equal(pos_s, expected)
        assert neighbours.shape == (rows.size, 15)
        assert neighbours.dtype == np.intp
    cached = score_fits(fits, plan, adjacency, x=x, scaling="robust", neighbours=table)
    assert cached == score_fits(fits, plan, adjacency, x=x, scaling="robust")
    assert 0.0 < cached.stability < 1.0
    held = []
    for r, s in table:
        rows, pos_s = plan.heldout(r, s)
        fitted = plan.indices[r]
        predicted = transfer_labels(x[fitted], fits[r].labels, x[rows], scaling="robust")
        own = np.arange(rows.size, dtype=np.intp)
        held.append(pair_stability(predicted, fits[s].labels, own, pos_s))
    assert cached.pair_agreement == tuple(a for a, _c in held)
    assert cached.pair_coverage == tuple(c for _a, c in held)


def test_stage_functions_reject_an_unresolved_engine():
    x, truth, plan, adjacency = _line_setup()
    fits = [_fake_fit(truth[rows]) for rows in plan.indices]
    labels = truth[plan.indices[0]]
    calls = [
        lambda: modularity(labels, adjacency[0], engine="auto"),
        lambda: transfer_labels(
            x[plan.indices[0]], labels, x[:5], scaling="standard", engine="gpu"
        ),
        lambda: heldout_neighbours(x, plan, scaling="standard", engine="cuML"),
        lambda: score_fits(fits, plan, adjacency, x=x, scaling="standard", engine="auto"),
        lambda: evaluate(
            x,
            PARAMS,
            plan,
            graphs=[None] * 3,
            adjacency=adjacency,
            engine="auto",
            scaling="standard",
        ),
    ]
    for call in calls:
        with pytest.raises(ValueError, match="engine must be"):
            call()


def test_score_fits_one_group_across_resamples_has_zero_stability():
    x, _truth, plan, adjacency = _line_setup()
    fits = [_fake_fit(np.zeros(plan.m, dtype=np.int32)) for _ in plan.indices]

    score = score_fits(fits, plan, adjacency, x=x, scaling="standard")

    assert score.fit_counts == (1, 1, 1)
    assert score.stability == 0.0
    assert score.coverage == 1.0
    assert score.score == 0.0
