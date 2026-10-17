from __future__ import annotations

import numpy as np
import pytest

from starfold._engine import derive_seed
from starfold.prep import make_plan, measurement_draw, scale, validate_matrix


def test_validate_matrix_drops_nonfinite_rows(single_gaussian):
    x = single_gaussian.copy()
    x[3, 0] = np.nan
    x[10, 4] = np.inf
    x[11, 2] = -np.inf
    rows, kept = validate_matrix(x)
    assert rows.shape == (197, 5)
    assert rows.dtype == np.float64
    assert rows.flags.c_contiguous
    assert kept.dtype == np.intp
    assert set(range(200)) - set(kept.tolist()) == {3, 10, 11}
    np.testing.assert_array_equal(rows, x[kept])
    assert np.isfinite(rows).all()


def test_validate_matrix_accepts_lists_and_keeps_order(single_gaussian):
    rows, kept = validate_matrix(single_gaussian.tolist())
    np.testing.assert_array_equal(rows, single_gaussian)
    np.testing.assert_array_equal(kept, np.arange(200))


def test_validate_matrix_rejects_wrong_ndim(single_gaussian):
    with pytest.raises(ValueError, match="two-dimensional"):
        validate_matrix(single_gaussian[:, 0])
    with pytest.raises(ValueError, match="two-dimensional"):
        validate_matrix(single_gaussian[:, :, None])


def test_validate_matrix_rejects_single_column(single_gaussian):
    with pytest.raises(ValueError, match="at least 2 columns"):
        validate_matrix(single_gaussian[:, :1])


def test_validate_matrix_rejects_too_few_finite_rows(single_gaussian):
    x = single_gaussian.copy()
    x[49:, 1] = np.nan
    with pytest.raises(ValueError, match="at least 50 finite rows"):
        validate_matrix(x)
    with pytest.raises(ValueError, match="at least 50 finite rows"):
        validate_matrix(single_gaussian[:49])


def test_validate_matrix_rejects_non_numeric():
    with pytest.raises(ValueError, match="numeric"):
        validate_matrix([["a", "b"]] * 60)


def test_scale_standard(single_gaussian):
    scaled = scale(single_gaussian, "standard")
    assert scaled.dtype == np.float32
    assert scaled.flags.c_contiguous
    np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(scaled.std(axis=0), 1.0, atol=1e-5)


def test_scale_robust_centres_on_median(single_gaussian):
    scaled = scale(single_gaussian, "robust")
    assert scaled.dtype == np.float32
    np.testing.assert_allclose(np.median(scaled, axis=0), 0.0, atol=1e-5)


def test_scale_is_fitted_on_given_rows_only(single_gaussian):
    head = scale(single_gaussian[:60], "standard")
    full = scale(single_gaussian, "standard")
    assert not np.allclose(head, full[:60])


def test_scale_rejects_unknown_family(single_gaussian):
    with pytest.raises(ValueError, match="scaling must be"):
        scale(single_gaussian, "minmax")  # type: ignore[arg-type]


def test_make_plan_is_deterministic_and_seeded():
    first = make_plan(100, n_resamples=3, fraction=0.8, seed=11)
    second = make_plan(100, n_resamples=3, fraction=0.8, seed=11)
    other = make_plan(100, n_resamples=3, fraction=0.8, seed=12)
    assert first.n == 100
    assert first.m == 80
    assert first.n_resamples == 3
    assert first.seeds == second.seeds
    assert first.seeds == tuple(derive_seed(11, 1, r) for r in range(3))
    for a, b in zip(first.indices, second.indices, strict=True):
        np.testing.assert_array_equal(a, b)
    assert any(not np.array_equal(a, b) for a, b in zip(first.indices, other.indices, strict=True))


def test_make_plan_indices_follow_the_declared_draw():
    plan = make_plan(100, n_resamples=2, fraction=0.8, seed=5)
    for r, rows in enumerate(plan.indices):
        rng = np.random.default_rng(derive_seed(5, 0, r))
        expected = np.sort(rng.choice(100, size=80, replace=False))
        np.testing.assert_array_equal(rows, expected)
        assert rows.dtype == np.intp
        assert np.all(np.diff(rows) > 0)


def test_make_plan_rounds_and_allows_full_fraction():
    assert make_plan(101, n_resamples=2, fraction=0.5, seed=0).m == round(50.5)
    full = make_plan(60, n_resamples=2, fraction=1.0, seed=0)
    np.testing.assert_array_equal(full.indices[0], np.arange(60))


@pytest.mark.parametrize(
    ("n", "n_resamples", "fraction", "message"),
    [
        (100, 1, 0.8, "n_resamples"),
        (100, 2, 0.0, "fraction"),
        (100, 2, 1.5, "fraction"),
        (2, 2, 0.5, "at least 2"),
    ],
)
def test_make_plan_rejects_bad_arguments(n, n_resamples, fraction, message):
    with pytest.raises(ValueError, match=message):
        make_plan(n, n_resamples=n_resamples, fraction=fraction, seed=0)


def test_measurement_draw_copies_without_perturbation(single_gaussian):
    rows = np.arange(200, dtype=np.intp)
    drawn = measurement_draw(single_gaussian, rows, np.random.default_rng(0))
    np.testing.assert_array_equal(drawn, single_gaussian)
    assert drawn is not single_gaussian
    assert not np.shares_memory(drawn, single_gaussian)


def test_measurement_draw_with_uncertainties(single_gaussian):
    rows = np.arange(200, dtype=np.intp)
    err = np.full_like(single_gaussian, 0.3)
    drawn = measurement_draw(single_gaussian, rows, np.random.default_rng(9), x_err_rows=err)
    expected = single_gaussian + err * np.random.default_rng(9).standard_normal((200, 5))
    np.testing.assert_allclose(drawn, expected)
    assert not np.array_equal(drawn, single_gaussian)


def test_measurement_draw_with_sampler(single_gaussian):
    rows = np.arange(200, dtype=np.intp)
    calls = []

    def sampler(x, indices, rng):
        calls.append((x, indices, rng))
        x += 1.0
        return x

    rng = np.random.default_rng(1)
    drawn = measurement_draw(single_gaussian, rows, rng, sampler=sampler)
    np.testing.assert_allclose(drawn, single_gaussian + 1.0)
    assert len(calls) == 1
    assert calls[0][2] is rng
    assert calls[0][1] is rows
    assert not np.shares_memory(calls[0][0], single_gaussian)


def test_measurement_draw_rejects_both(single_gaussian):
    rows = np.arange(200, dtype=np.intp)
    with pytest.raises(ValueError, match="not both"):
        measurement_draw(
            single_gaussian,
            rows,
            np.random.default_rng(0),
            x_err_rows=np.zeros_like(single_gaussian),
            sampler=lambda x, _i, _r: x,
        )


def test_measurement_draw_rejects_bad_shapes_and_values(single_gaussian):
    rows = np.arange(200, dtype=np.intp)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="row_indices"):
        measurement_draw(single_gaussian, rows[:10], rng)
    with pytest.raises(ValueError, match="x_err_rows shape"):
        measurement_draw(single_gaussian, rows, rng, x_err_rows=np.zeros((200, 4)))
    bad_err = np.zeros_like(single_gaussian)
    bad_err[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-negative"):
        measurement_draw(single_gaussian, rows, rng, x_err_rows=-np.ones_like(single_gaussian))
    with pytest.raises(ValueError, match="x_err_rows must be finite"):
        measurement_draw(single_gaussian, rows, rng, x_err_rows=bad_err)
    with pytest.raises(ValueError, match="sampler returned shape"):
        measurement_draw(single_gaussian, rows, rng, sampler=lambda x, _i, _r: x[:, :2])

    def infinite(x, _indices, _rng):
        x[0, 0] = np.inf
        return x

    with pytest.raises(ValueError, match="non-finite"):
        measurement_draw(single_gaussian, rows, rng, sampler=infinite)
