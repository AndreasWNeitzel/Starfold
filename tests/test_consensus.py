"""Behavioural tests for the bounded co-association check."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from starfold.backend import Params
from starfold.consensus import (
    ConsensusCheck,
    association_contrast,
    coassociation,
    consensus_check,
)


def _params(min_cluster_size=5, min_samples=3, selection_method="eom"):
    return Params(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        selection_method=selection_method,
    )


def _naive_distance(world_labels, rows):
    n_worlds = world_labels.shape[0]
    r = len(rows)
    distance = np.zeros((r, r))
    for a in range(r):
        for b in range(r):
            if a == b:
                continue
            shared = joint = 0
            for w in range(n_worlds):
                la, lb = world_labels[w, rows[a]], world_labels[w, rows[b]]
                if la >= 0 and lb >= 0:
                    joint += 1
                    shared += la == lb
            if joint == 0:
                distance[a, b] = 1.0
            else:
                distance[a, b] = max(1.0 - shared / joint, 1.0 / (2.0 * n_worlds))
    return distance


def _hand_built_worlds(n_worlds=6, n=14, seed=3):
    rng = np.random.default_rng(seed)
    world_labels = np.full((n_worlds, n), -2, dtype=np.int16)
    for w in range(n_worlds):
        rows = rng.choice(n - 2, size=10, replace=False)
        for i in rows:
            block = 0 if i < 6 else 1
            draw = rng.random()
            if draw < 0.15:
                world_labels[w, i] = -1
            elif draw < 0.25:
                world_labels[w, i] = 1 - block
            else:
                world_labels[w, i] = block
    return world_labels


def test_coassociation_matches_naive_loops():
    world_labels = _hand_built_worlds()
    rows = np.arange(world_labels.shape[1], dtype=np.intp)
    distance = coassociation(world_labels, rows)
    assert distance.dtype == np.float64
    assert distance.shape == (14, 14)
    assert np.allclose(distance, _naive_distance(world_labels, rows))
    assert np.allclose(distance, distance.T)
    assert (np.diag(distance) == 0.0).all()
    never_sampled = [12, 13]
    assert (distance[never_sampled][:, :12] == 1.0).all()
    assert distance[12, 13] == 1.0


def test_coassociation_over_a_row_subset():
    world_labels = _hand_built_worlds()
    rows = np.asarray([1, 4, 7, 9, 12], dtype=np.intp)
    distance = coassociation(world_labels, rows)
    assert distance.shape == (5, 5)
    assert np.allclose(distance, _naive_distance(world_labels, rows))


def test_coassociation_floor_and_never_joint_value():
    n_worlds = 8
    always_together = np.zeros((n_worlds, 2), dtype=np.int16)
    never_joint = np.asarray([[0, -1]] * 4 + [[-1, 0]] * 4, dtype=np.int16)
    world_labels = np.concatenate([always_together, never_joint], axis=1)
    distance = coassociation(world_labels, np.arange(4, dtype=np.intp))
    assert distance[0, 1] == pytest.approx(1.0 / (2.0 * n_worlds))
    assert distance[2, 3] == 1.0
    assert distance[0, 2] == pytest.approx(1.0 / (2.0 * n_worlds))
    assert (np.diag(distance) == 0.0).all()


@pytest.mark.parametrize("n_worlds", [255, 256, 300])
def test_coassociation_counts_do_not_overflow_at_the_dtype_switch(n_worlds):
    n = 5
    world_labels = np.zeros((n_worlds, n), dtype=np.int16)
    world_labels[:, 3] = 1
    world_labels[:, 4] = -1
    distance = coassociation(world_labels, np.arange(n, dtype=np.intp))
    floor = 1.0 / (2.0 * n_worlds)
    assert distance[0, 1] == pytest.approx(floor)
    assert distance[0, 3] == pytest.approx(1.0)
    assert distance[0, 4] == 1.0
    assert np.allclose(distance, _naive_distance(world_labels, np.arange(n)))


def test_coassociation_rejects_bad_shapes():
    with pytest.raises(ValueError, match="one-dimensional"):
        coassociation(np.zeros((2, 4), dtype=np.int16), np.zeros((2, 2), dtype=np.intp))
    with pytest.raises(ValueError, match="one-dimensional"):
        coassociation(np.zeros(4, dtype=np.int16), np.arange(4, dtype=np.intp))


@pytest.mark.parametrize("engine", ["auto", "gpu", "CPU"])
def test_unresolved_engine_is_rejected(engine):
    world_labels, reference = _two_block_worlds()
    rows = np.arange(world_labels.shape[1], dtype=np.intp)
    with pytest.raises(ValueError, match="engine must be"):
        coassociation(world_labels, rows, engine=engine)
    with pytest.raises(ValueError, match="engine must be"):
        consensus_check(
            world_labels,
            _params(),
            reference,
            budget_bytes=10**9,
            check_rows=10,
            seed=0,
            engine=engine,
        )


def _two_block_worlds(n_worlds=8, n=40, seed=0):
    rng = np.random.default_rng(seed)
    world_labels = np.full((n_worlds, n), -2, dtype=np.int16)
    for w in range(n_worlds):
        rows = rng.choice(n, size=32, replace=False)
        world_labels[w, rows] = np.where(rows < n // 2, (w % 2), 1 - (w % 2))
    reference = np.asarray([0] * (n // 2) + [1] * (n // 2), dtype=np.int32)
    return world_labels, reference


def test_dense_check_recovers_two_blocks():
    world_labels, reference = _two_block_worlds()
    n = world_labels.shape[1]
    check = consensus_check(
        world_labels, _params(), reference, budget_bytes=16 * n * n, check_rows=10, seed=0
    )
    assert isinstance(check, ConsensusCheck)
    assert check.subsampled is False
    assert check.n_rows == n
    assert check.bytes_used == 16 * n * n
    assert np.array_equal(check.rows, np.arange(n))
    assert check.labels.dtype == np.int32
    assert check.labels.shape == (n,)
    assert set(check.labels.tolist()) == {0, 1}
    assert adjusted_rand_score(check.labels, reference) == pytest.approx(1.0)
    assert check.agreement == pytest.approx(1.0)


def test_subset_switch_exactly_at_the_byte_bound():
    world_labels, reference = _two_block_worlds()
    n = world_labels.shape[1]
    dense = consensus_check(
        world_labels, _params(), reference, budget_bytes=16 * n * n, check_rows=12, seed=1
    )
    subset = consensus_check(
        world_labels, _params(), reference, budget_bytes=16 * n * n - 1, check_rows=12, seed=1
    )
    assert dense.subsampled is False
    assert subset.subsampled is True
    assert subset.n_rows == 12
    assert subset.bytes_used == 16 * 12 * 12
    assert subset.labels.shape == (12,)
    assert subset.rows.dtype == np.intp
    assert np.array_equal(subset.rows, np.sort(subset.rows))
    assert np.unique(subset.rows).size == 12
    expected_rows = np.sort(np.random.default_rng(1).choice(n, size=12, replace=False))
    assert np.array_equal(subset.rows, expected_rows)
    other_seed = consensus_check(
        world_labels, _params(), reference, budget_bytes=16 * n * n - 1, check_rows=12, seed=2
    )
    assert not np.array_equal(other_seed.rows, subset.rows)
    assert subset.agreement == pytest.approx(1.0)


def test_subset_is_capped_at_n_and_refused_above_budget():
    world_labels, reference = _two_block_worlds()
    n = world_labels.shape[1]
    with pytest.raises(ValueError, match="above the budget"):
        consensus_check(
            world_labels,
            _params(),
            reference,
            budget_bytes=16 * n * n - 1,
            check_rows=n + 5,
            seed=0,
        )
    with pytest.raises(ValueError, match="check_rows must be positive"):
        consensus_check(
            world_labels, _params(), reference, budget_bytes=10**9, check_rows=0, seed=0
        )


def test_all_noise_when_min_cluster_size_exceeds_rows():
    world_labels, reference = _two_block_worlds()
    n = world_labels.shape[1]
    check = consensus_check(
        world_labels,
        _params(min_cluster_size=n + 1, min_samples=3),
        reference,
        budget_bytes=16 * n * n,
        check_rows=10,
        seed=0,
    )
    assert (check.labels == -1).all()
    assert check.labels.dtype == np.int32
    assert check.agreement == 0.0
    assert check.n_rows == n
    assert np.isnan(check.contrast)


def test_subset_cut_keeps_the_mass_fraction():
    world_labels, reference = _two_block_worlds()
    n = world_labels.shape[1]
    params = _params(min_cluster_size=16, min_samples=5)
    dense = consensus_check(
        world_labels, params, reference, budget_bytes=16 * n * n, check_rows=12, seed=1
    )
    subset = consensus_check(
        world_labels, params, reference, budget_bytes=16 * 12 * 12, check_rows=12, seed=1
    )
    assert dense.agreement == pytest.approx(1.0)
    assert subset.subsampled is True
    assert subset.n_rows == 12
    # 16 of 40 rows is 4.8 of 12: the cut asks for 5 rows, not 16.
    assert set(subset.labels.tolist()) == {0, 1}
    assert subset.agreement == pytest.approx(1.0)


def test_consensus_check_rejects_bad_shapes():
    world_labels, reference = _two_block_worlds()
    with pytest.raises(ValueError, match="reference_labels must have shape"):
        consensus_check(
            world_labels, _params(), reference[:-1], budget_bytes=10**9, check_rows=10, seed=0
        )
    with pytest.raises(ValueError, match="2-D"):
        consensus_check(
            world_labels[0], _params(), reference, budget_bytes=10**9, check_rows=10, seed=0
        )


def test_agreement_uses_joint_coverage_against_reference():
    world_labels, reference = _two_block_worlds()
    n = world_labels.shape[1]
    noisy_reference = reference.copy()
    noisy_reference[:4] = -1
    check = consensus_check(
        world_labels, _params(), noisy_reference, budget_bytes=16 * n * n, check_rows=10, seed=0
    )
    assert check.agreement == pytest.approx((n - 4) / n)


def test_association_contrast_separates_recurrent_groups():
    n_worlds, n = 16, 8
    labels = np.zeros((n_worlds, n), dtype=np.int16)
    labels[:, 4:] = 1
    distance = coassociation(labels, np.arange(n, dtype=np.intp))
    reference = np.array([0] * 4 + [1] * 4, dtype=np.int32)
    within, between = association_contrast(distance, reference)
    assert within == pytest.approx(1.0 - 1.0 / (2 * n_worlds))
    assert between == pytest.approx(0.0)


def test_association_contrast_ignores_reference_noise_and_degenerate_input():
    n_worlds, n = 8, 6
    labels = np.zeros((n_worlds, n), dtype=np.int16)
    distance = coassociation(labels, np.arange(n, dtype=np.intp))
    reference = np.array([0, 0, 0, 0, -1, -1], dtype=np.int32)
    within, between = association_contrast(distance, reference)
    assert within == pytest.approx(1.0 - 1.0 / (2 * n_worlds))
    assert np.isnan(between)
    lonely = np.full(n, -1, dtype=np.int32)
    assert all(np.isnan(v) for v in association_contrast(distance, lonely))


def test_consensus_check_reports_the_contrast():
    n = 60
    rng = np.random.default_rng(3)
    reference = np.where(np.arange(n) < 30, 0, 1).astype(np.int32)
    world_labels = np.full((12, n), -2, dtype=np.int16)
    for w in range(12):
        rows = np.sort(rng.choice(n, size=48, replace=False))
        world_labels[w, rows] = reference[rows]
    check = consensus_check(
        world_labels,
        Params(15, 0.1, 2, 5, 5, "eom"),
        reference,
        budget_bytes=512 * 2**20,
        check_rows=4096,
        seed=0,
    )
    assert check.within_association > 0.9
    assert check.between_association == pytest.approx(0.0)
    assert check.contrast == pytest.approx(check.within_association - check.between_association)


def test_contrast_is_read_before_the_cut_rewrites_the_matrix():
    n_worlds, n = 10, 40
    world_labels = np.zeros((n_worlds, n), dtype=np.int16)
    world_labels[:, 20:] = 1
    world_labels[1::2, 10:20] = 2
    reference = np.repeat(np.asarray([0, 1], dtype=np.int32), 20)
    # Rows 0-9 and 10-19 are together in half the worlds; with core distances
    # over 15 rows the cut's mutual reachability lifts every group-0 pair to 0.5.
    floor = 1.0 / (2 * n_worlds)
    within_total = 2 * 90 * (1.0 - floor) + 200 * 0.5 + 380 * (1.0 - floor)
    check = consensus_check(
        world_labels,
        _params(min_cluster_size=15, min_samples=5),
        reference,
        budget_bytes=16 * n * n,
        check_rows=10,
        seed=0,
    )
    assert check.within_association == pytest.approx(within_total / 760)
    assert check.between_association == pytest.approx(0.0)
    fresh = coassociation(world_labels, np.arange(n, dtype=np.intp))
    assert (check.within_association, check.between_association) == pytest.approx(
        association_contrast(fresh, reference)
    )


def test_association_contrast_matches_pairwise_means():
    rng = np.random.default_rng(4)
    world_labels = rng.integers(-2, 4, size=(12, 50)).astype(np.int16)
    distance = coassociation(world_labels, np.arange(50, dtype=np.intp))
    reference = rng.integers(-1, 3, size=50).astype(np.int32)
    before = distance.copy()
    within, between = association_contrast(distance, reference)
    assert np.array_equal(distance, before)
    assigned = np.flatnonzero(reference >= 0)
    same = [(i, j) for i in assigned for j in assigned if i != j and reference[i] == reference[j]]
    other = [(i, j) for i in assigned for j in assigned if reference[i] != reference[j]]
    assert within == pytest.approx(np.mean([1.0 - distance[i, j] for i, j in same]), abs=1e-12)
    assert between == pytest.approx(np.mean([1.0 - distance[i, j] for i, j in other]), abs=1e-12)


@pytest.mark.parametrize(
    ("reference_value", "world_value", "cut_value", "expected_within"),
    [(-1, -1, -1, None), (0, 0, 0, 0.875)],
)
def test_consensus_check_degenerate_reference_groups(
    monkeypatch, reference_value, world_value, cut_value, expected_within
):
    n_worlds, n_rows = 4, 8
    reference = np.full(n_rows, reference_value, dtype=np.int32)
    world_labels = np.full((n_worlds, n_rows), world_value, dtype=np.int16)
    cut_labels = np.full(n_rows, cut_value, dtype=np.int32)

    class FixedCut:
        def __init__(self, **_kwargs):
            self.labels_ = cut_labels.copy()

        def fit(self, distance):
            assert distance.shape == (n_rows, n_rows)
            return self

    monkeypatch.setattr("starfold.consensus.HDBSCAN", FixedCut)
    check = consensus_check(
        world_labels,
        _params(min_cluster_size=2, min_samples=1),
        reference,
        budget_bytes=16 * n_rows * n_rows,
        check_rows=n_rows,
        seed=0,
    )

    assert check.agreement == 0.0
    assert np.isnan(check.between_association)
    assert np.isnan(check.contrast)
    if expected_within is None:
        assert np.isnan(check.within_association)
    else:
        assert check.within_association == pytest.approx(expected_within)
