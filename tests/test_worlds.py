"""Behavioural tests for the possible-world statistics and the world runner."""

from __future__ import annotations

import math
import statistics
import sys
from dataclasses import dataclass

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from starfold import worlds
from starfold._engine import derive_seed
from starfold.backend import Fit, Params
from starfold.worlds import WorldEvidence, run_worlds, summarize_worlds

PARAMS = Params(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    min_cluster_size=5,
    min_samples=5,
    selection_method="eom",
)


def _naive_summary(world_labels, labels, x):  # noqa: C901, PLR0912, PLR0915
    """Every world statistic by plain Python loops over sets and lists.

    Kept as one flat loop on purpose: the value of this reference is that it
    shares no structure with the vectorised implementation it checks.
    """
    n_worlds, n = world_labels.shape
    d = x.shape[1]
    ids = sorted({int(v) for v in labels if v >= 0})
    index = {v: k for k, v in enumerate(ids)}
    n_groups = len(ids)
    support = [[0.0] * n_groups for _ in range(n)]
    unresolved = [0.0] * n
    n_sampled = [0] * n
    recurrence = {g: [] for g in range(n_groups)}
    properties = {g: [] for g in range(n_groups)}
    counts = []
    agreement = []
    for w in range(n_worlds):
        rows = [i for i in range(n) if world_labels[w, i] != -2]
        for i in rows:
            n_sampled[i] += 1
        clusters = {}
        for i in rows:
            c = int(world_labels[w, i])
            if c >= 0:
                clusters.setdefault(c, []).append(i)
            else:
                unresolved[i] += 1.0
        counts.append(len(clusters))
        for members in clusters.values():
            size = len(members)
            for i in members:
                for j in members:
                    if labels[j] >= 0:
                        support[i][index[int(labels[j])]] += 1.0 / size
                    else:
                        unresolved[i] += 1.0 / size
        for g in range(n_groups):
            members_g = {i for i in rows if labels[i] >= 0 and index[int(labels[i])] == g}
            if not members_g:
                continue
            best_jaccard, best_cluster = 0.0, None
            for c in sorted(clusters):
                members_c = set(clusters[c])
                jaccard = len(members_g & members_c) / len(members_g | members_c)
                if jaccard > best_jaccard:
                    best_jaccard, best_cluster = jaccard, c
            recurrence[g].append(best_jaccard)
            if best_cluster is not None:
                properties[g].append(
                    [statistics.median(x[i, f] for i in clusters[best_cluster]) for f in range(d)]
                )
        joint = [i for i in rows if world_labels[w, i] >= 0 and labels[i] >= 0]
        coverage = len(joint) / len(rows) if rows else 0.0
        if (
            len(joint) >= 2
            and len({int(world_labels[w, i]) for i in joint}) >= 2
            and len({int(labels[i]) for i in joint}) >= 2
        ):
            ari = max(0.0, adjusted_rand_score([world_labels[w, i] for i in joint], labels[joint]))
        else:
            ari = 0.0
        agreement.append(ari * coverage)
    out_support = np.full((n, n_groups), np.nan)
    out_unresolved = np.full(n, np.nan)
    for i in range(n):
        if n_sampled[i]:
            out_support[i] = np.asarray(support[i]) / n_sampled[i]
            out_unresolved[i] = unresolved[i] / n_sampled[i]
    own = np.full(n, np.nan)
    for i in range(n):
        if labels[i] >= 0 and n_sampled[i]:
            own[i] = out_support[i, index[int(labels[i])]]
    group_recurrence = np.full(n_groups, np.nan)
    band = np.full((n_groups, 2), np.nan)
    prop_median = np.full((n_groups, d), np.nan)
    prop_band = np.full((n_groups, d, 2), np.nan)
    for g in range(n_groups):
        if recurrence[g]:
            group_recurrence[g] = statistics.fmean(recurrence[g])
            band[g] = np.percentile(recurrence[g], [5, 95])
        if properties[g]:
            stack = np.asarray(properties[g])
            prop_median[g] = np.median(stack, axis=0)
            prop_band[g] = np.percentile(stack, [5, 95], axis=0).T
    low, high = np.percentile(counts, [5, 95])
    return {
        "n_sampled": np.asarray(n_sampled),
        "support": out_support,
        "unresolved": out_unresolved,
        "own_support": own,
        "unresolved_mass": statistics.fmean(out_unresolved[~np.isnan(out_unresolved)]),
        "group_recurrence": group_recurrence,
        "group_recurrence_band": band,
        "count_values": np.asarray(counts),
        "count_interval": (math.floor(low), math.ceil(high)),
        "count_reference": n_groups,
        "property_median": prop_median,
        "property_band": prop_band,
        "world_agreement": np.asarray(agreement),
    }


def _hand_built_worlds():
    """Forty rows, three reference groups with unsorted ids, seven varied worlds."""
    rng = np.random.default_rng(7)
    labels = np.asarray([3] * 12 + [7] * 12 + [9] * 10 + [-1] * 6, dtype=np.int32)
    n = labels.shape[0]
    group_index = {3: 0, 7: 1, 9: 2}
    x = (
        rng.normal(size=(n, 3))
        + np.asarray([group_index.get(int(v), 3) * 5.0 for v in labels])[:, None]
    )
    n_worlds = 7
    world_labels = np.full((n_worlds, n), -2, dtype=np.int16)
    for w in range(n_worlds):
        rows = np.sort(rng.choice(38, size=32, replace=False))
        for i in rows:
            if w == 5:
                world_labels[w, i] = -1
                continue
            if labels[i] < 0:
                world_labels[w, i] = -1 if rng.random() < 0.5 else int(rng.integers(0, 3))
                continue
            group = group_index[int(labels[i])]
            if w == 6 and group in (0, 1):
                group = 0
            cluster = (group + w) % 3
            draw = rng.random()
            if draw < 0.1:
                cluster = (cluster + 1) % 3
            elif draw < 0.2:
                cluster = -1
            world_labels[w, i] = cluster
    return world_labels, labels, x


def test_summary_matches_naive_loops():
    world_labels, labels, x = _hand_built_worlds()
    summary = summarize_worlds(world_labels, labels, x, propagated=False)
    naive = _naive_summary(world_labels, labels, x)
    for key, expected in naive.items():
        got = getattr(summary, key)
        if isinstance(expected, np.ndarray):
            assert np.allclose(got, expected, equal_nan=True, atol=1e-6), key
        else:
            assert got == pytest.approx(expected), key
    assert summary.property_propagated is False
    assert summary.world_labels.dtype == np.int16


def test_summary_dtypes_and_shapes():
    world_labels, labels, x = _hand_built_worlds()
    summary = summarize_worlds(world_labels, labels, x, propagated=True)
    n_worlds, n = world_labels.shape
    assert summary.n_sampled.dtype == np.uint16
    assert summary.support.dtype == np.float32
    assert summary.support.shape == (n, 3)
    assert summary.unresolved.shape == (n,)
    assert summary.own_support.shape == (n,)
    assert summary.group_recurrence.dtype == np.float32
    assert summary.group_recurrence_band.shape == (3, 2)
    assert summary.count_values.dtype == np.int16
    assert summary.count_values.shape == (n_worlds,)
    assert summary.property_median.shape == (3, 3)
    assert summary.property_band.shape == (3, 3, 2)
    assert summary.world_agreement.dtype == np.float32
    assert summary.world_agreement.shape == (n_worlds,)
    assert summary.property_propagated is True
    assert isinstance(summary.count_interval, tuple)
    assert all(isinstance(v, int) for v in summary.count_interval)


def test_support_identity_on_sampled_rows():
    world_labels, labels, x = _hand_built_worlds()
    summary = summarize_worlds(world_labels, labels, x, propagated=False)
    sampled = summary.n_sampled > 0
    total = summary.support.sum(axis=1) + summary.unresolved
    assert np.allclose(total[sampled], 1.0, atol=1e-6)
    assert summary.unresolved_mass == pytest.approx(float(summary.unresolved[sampled].mean()))


def test_unsampled_rows_and_reference_noise_are_nan():
    world_labels, labels, x = _hand_built_worlds()
    summary = summarize_worlds(world_labels, labels, x, propagated=False)
    never = np.flatnonzero(summary.n_sampled == 0)
    assert never.tolist() == [38, 39]
    assert np.isnan(summary.support[never]).all()
    assert np.isnan(summary.unresolved[never]).all()
    assert np.isnan(summary.own_support[never]).all()
    assert np.isnan(summary.own_support[labels < 0]).all()
    sampled_assigned = (summary.n_sampled > 0) & (labels >= 0)
    assert np.isfinite(summary.own_support[sampled_assigned]).all()
    assert np.isfinite(summary.support[summary.n_sampled > 0]).all()


def test_band_brackets_mean_and_reference_groups_are_relabelled():
    world_labels, labels, x = _hand_built_worlds()
    summary = summarize_worlds(world_labels, labels, x, propagated=False)
    band = summary.group_recurrence_band
    assert (band[:, 0] <= summary.group_recurrence + 1e-6).all()
    assert (band[:, 1] >= summary.group_recurrence - 1e-6).all()
    assert summary.count_reference == 3
    group_index = {3: 0, 7: 1, 9: 2}
    for i in np.flatnonzero((labels >= 0) & (summary.n_sampled > 0)):
        assert summary.own_support[i] == summary.support[i, group_index[int(labels[i])]]


def test_count_interval_is_integer_hull_of_percentiles():
    n = 12
    counts = [2, 2, 2, 3, 3, 1, 4]
    labels = np.zeros(n, dtype=np.int32)
    world_labels = np.full((len(counts), n), -1, dtype=np.int16)
    for w, k in enumerate(counts):
        world_labels[w, :k] = np.arange(k)
    summary = summarize_worlds(world_labels, labels, np.zeros((n, 2)), propagated=False)
    assert summary.count_values.tolist() == counts
    low, high = np.percentile(counts, [5, 95])
    assert summary.count_interval == (math.floor(low), math.ceil(high))
    assert summary.count_interval == (1, 4)


def test_property_median_on_known_matrix():
    x = np.asarray([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0], [6.0, 60.0]])
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32)
    world_labels = np.asarray(
        [[0, 0, 0, 1, 1, 1], [1, 1, -2, 0, 0, -2]],
        dtype=np.int16,
    )
    summary = summarize_worlds(world_labels, labels, x, propagated=False)
    expected_median = np.asarray([[1.75, 17.5], [4.75, 47.5]])
    assert np.allclose(summary.property_median, expected_median)
    expected_band = np.stack(
        [
            np.percentile([[2.0, 20.0], [1.5, 15.0]], [5, 95], axis=0).T,
            np.percentile([[5.0, 50.0], [4.5, 45.0]], [5, 95], axis=0).T,
        ]
    )
    assert np.allclose(summary.property_band, expected_band)
    assert np.allclose(summary.group_recurrence, [1.0, 1.0])
    assert np.allclose(summary.world_agreement, [1.0, 1.0])


def test_property_skips_worlds_without_a_match():
    x = np.arange(12, dtype=np.float64).reshape(6, 2)
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32)
    world_labels = np.asarray(
        [[0, 0, 0, 1, 1, 1], [-1, -1, -1, 0, 0, 0], [-1, -1, -1, -1, -1, -1]],
        dtype=np.int16,
    )
    summary = summarize_worlds(world_labels, labels, x, propagated=False)
    assert np.allclose(summary.property_median[0], np.median(x[:3], axis=0))
    assert np.allclose(summary.property_median[1], np.median(x[3:], axis=0))
    assert np.allclose(summary.group_recurrence, [1.0 / 3.0, 2.0 / 3.0])
    assert summary.count_values.tolist() == [2, 1, 0]
    assert np.allclose(summary.world_agreement, [1.0, 0.0, 0.0])
    assert np.allclose(summary.unresolved, [2 / 3, 2 / 3, 2 / 3, 1 / 3, 1 / 3, 1 / 3])


def test_world_agreement_cases():
    labels = np.asarray([0] * 5 + [1] * 5 + [-1] * 2, dtype=np.int32)
    identical = labels.astype(np.int16)
    permuted = np.where(labels >= 0, 1 - labels, -1).astype(np.int16)
    all_noise = np.full(12, -1, dtype=np.int16)
    noise_assigned = np.asarray([0] * 5 + [1] * 5 + [1, 1], dtype=np.int16)
    world_labels = np.stack([identical, permuted, all_noise, noise_assigned])
    summary = summarize_worlds(world_labels, labels, np.zeros((12, 2)), propagated=False)
    assert np.allclose(summary.world_agreement, [10 / 12, 10 / 12, 0.0, 10 / 12])
    expected_support = [0.75] * 5 + [(2.0 + 5.0 / 7.0) / 4.0] * 5
    expected_unresolved = [0.25] * 5 + [0.25 * (1.0 + 2.0 / 7.0)] * 5
    assert np.allclose(summary.support[:10].sum(axis=1), expected_support, atol=1e-6)
    assert np.allclose(summary.unresolved[:10], expected_unresolved, atol=1e-6)
    assert np.allclose(summary.support[:10].sum(axis=1) + summary.unresolved[:10], 1.0, atol=1e-6)


def test_zero_reference_groups():
    labels = np.full(8, -1, dtype=np.int32)
    world_labels = np.asarray(
        [[0, 0, 0, 1, 1, 1, -1, -2], [0, 0, 1, 1, -1, -1, -2, -2]], dtype=np.int16
    )
    summary = summarize_worlds(world_labels, labels, np.zeros((8, 3)), propagated=False)
    assert summary.count_reference == 0
    assert summary.support.shape == (8, 0)
    assert summary.group_recurrence.shape == (0,)
    assert summary.group_recurrence_band.shape == (0, 2)
    assert summary.property_median.shape == (0, 3)
    assert summary.property_band.shape == (0, 3, 2)
    assert np.isnan(summary.own_support).all()
    assert np.allclose(summary.unresolved[:7], 1.0)
    assert np.isnan(summary.unresolved[7])
    assert summary.unresolved_mass == pytest.approx(1.0)
    assert summary.count_values.tolist() == [2, 2]
    assert np.allclose(summary.world_agreement, 0.0)


def test_one_reference_group():
    labels = np.zeros(8, dtype=np.int32)
    world_labels = np.asarray(
        [[0] * 8, [0, 0, 0, 0, 1, 1, 1, 1], [-1] * 4 + [0] * 4], dtype=np.int16
    )
    summary = summarize_worlds(world_labels, labels, np.zeros((8, 2)), propagated=False)
    assert summary.count_reference == 1
    assert np.allclose(
        summary.support[:, 0],
        [2 / 3] * 4 + [1.0] * 4,
    )
    assert np.allclose(summary.group_recurrence, [(1.0 + 0.5 + 0.5) / 3.0])
    assert np.allclose(summary.world_agreement, 0.0)
    assert summary.count_interval == (1, 2)


def test_summary_carries_no_run_metadata():
    world_labels, labels, x = _hand_built_worlds()
    summary = summarize_worlds(world_labels, labels, x, propagated=False)
    assert isinstance(summary, WorldEvidence)
    assert summary.world_seeds == ()
    assert summary.world_errors == ()
    assert summary.n_fits == 0
    assert summary.seconds == 0.0


def test_summarize_rejects_shape_mismatch():
    world_labels = np.zeros((2, 5), dtype=np.int16)
    with pytest.raises(ValueError, match="labels must have shape"):
        summarize_worlds(
            world_labels, np.zeros(4, dtype=np.int32), np.zeros((5, 2)), propagated=False
        )
    with pytest.raises(ValueError, match="2-D"):
        summarize_worlds(
            np.zeros(5, dtype=np.int16),
            np.zeros(5, dtype=np.int32),
            np.zeros((5, 2)),
            propagated=False,
        )


@dataclass
class _Recorder:
    """Deterministic stand-in for the fit; labels follow the sign of feature 0."""

    fail_seeds: set[int]
    calls: list[dict]

    def __call__(self, x_rows, params, *, seed, engine, scaling, knn=None):
        self.calls.append(
            {
                "x_rows": np.array(x_rows, copy=True),
                "params": params,
                "seed": seed,
                "engine": engine,
                "scaling": scaling,
                "knn": knn,
            }
        )
        if seed in self.fail_seeds:
            msg = "boom"
            raise RuntimeError(msg)
        labels = (x_rows[:, 0] > 0.0).astype(np.int32)
        labels[-1] = -1
        return Fit(
            labels=labels,
            strengths=np.ones(labels.shape[0], dtype=np.float32),
            embedding=np.zeros((labels.shape[0], 2), dtype=np.float32),
            n_groups=2,
            seconds=0.0,
        )


def _two_blob_matrix(n=60, d=3, seed=11):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    x[: n // 2, 0] -= 4.0
    x[n // 2 :, 0] += 4.0
    labels = (x[:, 0] > 0.0).astype(np.int32)
    return x, labels


def _serial_map_recorder(record):
    def parallel_map(fn, items, *, n_jobs):
        record.append((fn, n_jobs))
        return [fn(item) for item in items]

    return parallel_map


def test_run_worlds_plumbing_seeds_and_error_capture(monkeypatch):
    x, labels = _two_blob_matrix()
    n = x.shape[0]
    random_state = 5
    n_worlds = 4
    fail_seed = derive_seed(random_state, 3, 1)
    recorder = _Recorder(fail_seeds={fail_seed}, calls=[])
    map_calls = []
    monkeypatch.setattr(worlds, "fit_pipeline", recorder)
    monkeypatch.setattr(worlds, "parallel_map", _serial_map_recorder(map_calls))
    evidence = run_worlds(
        x,
        PARAMS,
        labels,
        n_worlds=n_worlds,
        world_fraction=0.8,
        random_state=random_state,
        engine="cpu",
        scaling="robust",
        n_jobs=3,
    )
    assert isinstance(evidence, WorldEvidence)
    assert evidence.n_fits == n_worlds
    assert evidence.seconds >= 0.0
    assert evidence.property_propagated is False
    assert evidence.world_labels.dtype == np.int16
    assert evidence.world_labels.shape == (n_worlds, n)
    assert evidence.world_seeds == tuple(derive_seed(random_state, 3, w) for w in range(n_worlds))
    assert evidence.world_errors[1].startswith("RuntimeError: boom")
    assert all(err == "" for w, err in enumerate(evidence.world_errors) if w != 1)
    assert len(recorder.calls) == n_worlds
    assert [call["seed"] for call in recorder.calls] == list(evidence.world_seeds)
    assert all(call["engine"] == "cpu" for call in recorder.calls)
    assert all(call["scaling"] == "robust" for call in recorder.calls)
    assert all(call["params"] is PARAMS for call in recorder.calls)
    assert all(call["knn"] is None for call in recorder.calls)
    assert len(map_calls) == 1
    fn, n_jobs = map_calls[0]
    assert n_jobs == 3
    assert getattr(sys.modules[fn.__module__], fn.__name__) is fn
    m = round(0.8 * n)
    for w in range(n_worlds):
        rng = np.random.default_rng(derive_seed(random_state, 2, w))
        rows = np.sort(rng.choice(n, size=m, replace=False))
        assert np.array_equal(recorder.calls[w]["x_rows"], x[rows])
        world = evidence.world_labels[w]
        assert (world[rows] != -2).all()
        assert (np.delete(world, rows) == -2).all()
        if w == 1:
            assert (world[rows] == -1).all()
        else:
            assert np.array_equal(world[rows][:-1], labels[rows][:-1])
            assert world[rows][-1] == -1
    assert np.array_equal(evidence.n_sampled, (evidence.world_labels != -2).sum(axis=0))
    assert evidence.count_values.tolist() == [2, 0, 2, 2]
    assert evidence.count_reference == 2
    sampled = evidence.n_sampled > 0
    assert np.allclose(evidence.support[sampled].sum(axis=1) + evidence.unresolved[sampled], 1.0)
    assert evidence.world_agreement[1] == 0.0
    assert (evidence.world_agreement[[0, 2, 3]] > 0.9).all()


def test_run_worlds_is_deterministic_in_random_state(monkeypatch):
    x, labels = _two_blob_matrix()
    monkeypatch.setattr(worlds, "fit_pipeline", _Recorder(fail_seeds=set(), calls=[]))
    common = {"n_worlds": 3, "world_fraction": 0.7, "engine": "cpu", "scaling": "standard"}
    first = run_worlds(x, PARAMS, labels, random_state=1, **common)
    second = run_worlds(x, PARAMS, labels, random_state=1, **common)
    other = run_worlds(x, PARAMS, labels, random_state=2, **common)
    assert np.array_equal(first.world_labels, second.world_labels)
    assert first.world_seeds == second.world_seeds
    assert not np.array_equal(first.world_labels, other.world_labels)
    assert first.world_seeds != other.world_seeds


def test_run_worlds_propagates_measurement_draws(monkeypatch):
    x, labels = _two_blob_matrix()
    n, d = x.shape
    x_err = np.full(x.shape, 0.05)
    recorder = _Recorder(fail_seeds=set(), calls=[])
    monkeypatch.setattr(worlds, "fit_pipeline", recorder)
    random_state = 3
    evidence = run_worlds(
        x,
        PARAMS,
        labels,
        n_worlds=3,
        world_fraction=0.8,
        random_state=random_state,
        engine="cpu",
        scaling="standard",
        x_err=x_err,
    )
    assert evidence.property_propagated is True
    m = round(0.8 * n)
    per_world = []
    for w, call in enumerate(recorder.calls):
        rng = np.random.default_rng(derive_seed(random_state, 2, w))
        rows = np.sort(rng.choice(n, size=m, replace=False))
        expected = x[rows] + x_err[rows] * rng.standard_normal((m, d))
        assert np.allclose(call["x_rows"], expected)
        assert not np.allclose(call["x_rows"], x[rows])
        world_rows = evidence.world_labels[w, rows]
        per_world.append([np.median(call["x_rows"][world_rows == g], axis=0) for g in range(2)])
    stacked = np.asarray(per_world)
    assert np.allclose(evidence.property_median, np.median(stacked, axis=0))
    assert np.allclose(
        evidence.property_band, np.percentile(stacked, [5, 95], axis=0).transpose(1, 2, 0)
    )


def _shift_second_feature(x_rows, row_indices, rng):
    del row_indices, rng
    shifted = x_rows.copy()
    shifted[:, 1] += 100.0
    return shifted


def test_run_worlds_sampler_enters_property_bands(monkeypatch):
    x, labels = _two_blob_matrix()
    monkeypatch.setattr(worlds, "fit_pipeline", _Recorder(fail_seeds=set(), calls=[]))
    common = {
        "n_worlds": 3,
        "world_fraction": 0.8,
        "random_state": 9,
        "engine": "cpu",
        "scaling": "standard",
    }
    plain = run_worlds(x, PARAMS, labels, **common)
    shifted = run_worlds(x, PARAMS, labels, sampler=_shift_second_feature, **common)
    assert plain.property_propagated is False
    assert shifted.property_propagated is True
    assert np.array_equal(plain.world_labels, shifted.world_labels)
    assert np.allclose(shifted.property_median[:, 1] - plain.property_median[:, 1], 100.0)
    assert np.allclose(shifted.property_median[:, 0], plain.property_median[:, 0])
    assert np.allclose(shifted.property_band[:, 1, :] - plain.property_band[:, 1, :], 100.0)


def test_run_worlds_cluster_medians_agree_with_summarize(monkeypatch):
    x, labels = _two_blob_matrix()
    monkeypatch.setattr(worlds, "fit_pipeline", _Recorder(fail_seeds=set(), calls=[]))
    evidence = run_worlds(
        x,
        PARAMS,
        labels,
        n_worlds=3,
        world_fraction=0.8,
        random_state=4,
        engine="cpu",
        scaling="standard",
    )
    summary = summarize_worlds(evidence.world_labels, labels, x, propagated=False)
    assert np.allclose(evidence.property_median, summary.property_median)
    assert np.allclose(evidence.property_band, summary.property_band)
    assert np.allclose(evidence.support, summary.support, equal_nan=True)


def test_run_worlds_rejects_bad_inputs(monkeypatch):
    x, labels = _two_blob_matrix()
    monkeypatch.setattr(worlds, "fit_pipeline", _Recorder(fail_seeds=set(), calls=[]))
    common = {"random_state": 0, "engine": "cpu", "scaling": "standard"}
    with pytest.raises(ValueError, match="n_worlds"):
        run_worlds(x, PARAMS, labels, n_worlds=1, world_fraction=0.8, **common)
    with pytest.raises(ValueError, match="n_worlds"):
        run_worlds(x, PARAMS, labels, n_worlds=40000, world_fraction=0.8, **common)
    with pytest.raises(ValueError, match="world_fraction"):
        run_worlds(x, PARAMS, labels, n_worlds=2, world_fraction=0.0, **common)
    with pytest.raises(ValueError, match="world_fraction"):
        run_worlds(x, PARAMS, labels, n_worlds=2, world_fraction=1.5, **common)
    with pytest.raises(ValueError, match="labels must have shape"):
        run_worlds(x, PARAMS, labels[:-1], n_worlds=2, world_fraction=0.8, **common)
    with pytest.raises(ValueError, match="x_err must have shape"):
        run_worlds(x, PARAMS, labels, n_worlds=2, world_fraction=0.8, x_err=np.ones(3), **common)
    with pytest.raises(ValueError, match="non-negative"):
        run_worlds(
            x, PARAMS, labels, n_worlds=2, world_fraction=0.8, x_err=-np.ones_like(x), **common
        )
    with pytest.raises(ValueError, match="exclusive"):
        run_worlds(
            x,
            PARAMS,
            labels,
            n_worlds=2,
            world_fraction=0.8,
            x_err=np.ones_like(x),
            sampler=_shift_second_feature,
            **common,
        )
    with pytest.raises(ValueError, match="2-D"):
        run_worlds(x[:, 0], PARAMS, labels, n_worlds=2, world_fraction=0.8, **common)


@pytest.mark.parametrize("engine", ["auto", "gpu", "CUML"])
def test_run_worlds_rejects_an_unresolved_engine(monkeypatch, engine):
    x, labels = _two_blob_matrix()
    recorder = _Recorder(fail_seeds=set(), calls=[])
    monkeypatch.setattr(worlds, "fit_pipeline", recorder)
    with pytest.raises(ValueError, match="engine must be"):
        run_worlds(
            x,
            PARAMS,
            labels,
            n_worlds=2,
            world_fraction=0.8,
            random_state=0,
            engine=engine,
            scaling="standard",
        )
    assert recorder.calls == []


def _oversized_fit(x_rows, params, *, seed, engine, scaling, knn=None):
    del params, seed, engine, scaling, knn
    labels = np.full(x_rows.shape[0], 40000, dtype=np.int32)
    return Fit(
        labels=labels,
        strengths=np.ones(labels.shape[0], dtype=np.float32),
        embedding=np.zeros((labels.shape[0], 2), dtype=np.float32),
        n_groups=1,
        seconds=0.0,
    )


def test_run_worlds_rejects_cluster_ids_beyond_int16(monkeypatch):
    x, labels = _two_blob_matrix()
    monkeypatch.setattr(worlds, "fit_pipeline", _oversized_fit)
    with pytest.raises(ValueError, match="int16"):
        run_worlds(
            x,
            PARAMS,
            labels,
            n_worlds=2,
            world_fraction=0.8,
            random_state=0,
            engine="cpu",
            scaling="standard",
        )


@pytest.mark.slow
def test_run_worlds_real_fit_two_blobs():
    rng = np.random.default_rng(0)
    half = 120
    x = np.concatenate([rng.normal(-6.0, 1.0, (half, 4)), rng.normal(6.0, 1.0, (half, 4))])
    labels = np.asarray([0] * half + [1] * half, dtype=np.int32)
    params = Params(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        min_cluster_size=10,
        min_samples=5,
        selection_method="eom",
    )
    evidence = run_worlds(
        x,
        params,
        labels,
        n_worlds=3,
        world_fraction=0.8,
        random_state=0,
        engine="cpu",
        scaling="standard",
    )
    assert evidence.world_errors == ("", "", "")
    assert evidence.n_fits == 3
    assert evidence.world_labels.shape == (3, 2 * half)
    assert evidence.count_reference == 2
    sampled = evidence.n_sampled > 0
    assert np.allclose(evidence.support[sampled].sum(axis=1) + evidence.unresolved[sampled], 1.0)
    assert (evidence.group_recurrence >= 0.5).all()
    assert (evidence.world_agreement > 0.3).all()
    assert np.isfinite(evidence.property_median).all()
