from dataclasses import replace

import numpy as np
import pytest

from starfold.nulls import (
    NullEvidence,
    clopper_pearson,
    generate_null,
    monte_carlo_pvalue,
    null_test,
)


def _matrix(seed: int = 0, n: int = 120, d: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, d))
    base[:, 1] = 0.8 * base[:, 0] + 0.6 * base[:, 1]
    return base


@pytest.mark.parametrize("family", ["gaussian_copula", "independent_marginals"])
def test_generate_null_preserves_every_marginal_exactly(family: str) -> None:
    x = _matrix()
    null = generate_null(x, family, seed=3)
    assert null.shape == x.shape
    for column in range(x.shape[1]):
        assert np.array_equal(np.sort(null[:, column]), np.sort(x[:, column]))


@pytest.mark.parametrize("family", ["gaussian_copula", "independent_marginals"])
def test_generate_null_is_seeded(family: str) -> None:
    x = _matrix()
    first = generate_null(x, family, seed=11)
    second = generate_null(x, family, seed=11)
    third = generate_null(x, family, seed=12)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)


def test_independent_marginals_destroys_dependence() -> None:
    x = _matrix(n=400)
    null = generate_null(x, "independent_marginals", seed=5)
    assert abs(np.corrcoef(x[:, 0], x[:, 1])[0, 1]) > 0.5
    assert abs(np.corrcoef(null[:, 0], null[:, 1])[0, 1]) < 0.2


def test_gaussian_copula_keeps_rank_dependence() -> None:
    x = _matrix(n=400)
    null = generate_null(x, "gaussian_copula", seed=5)
    observed = np.corrcoef(x[:, 0], x[:, 1])[0, 1]
    reproduced = np.corrcoef(null[:, 0], null[:, 1])[0, 1]
    assert abs(observed - reproduced) < 0.2


def test_generate_null_rejects_unknown_family_and_bad_matrix() -> None:
    x = _matrix()
    with pytest.raises(ValueError, match="Unknown null family"):
        generate_null(x, "elliptical_gaussian", seed=0)
    with pytest.raises(ValueError, match="finite 2-D"):
        generate_null(np.array([1.0, 2.0]), "gaussian_copula", seed=0)


def test_pvalue_and_interval_arithmetic() -> None:
    assert monte_carlo_pvalue(0, 19) == pytest.approx(0.05)
    assert monte_carlo_pvalue(3, 5) == pytest.approx(4 / 6)
    assert monte_carlo_pvalue(0, 0) == 1.0
    assert clopper_pearson(0, 0) == (0.0, 1.0)
    lower, upper = clopper_pearson(0, 19)
    assert lower == 0.0
    assert 0.1 < upper < 0.2
    lower, upper = clopper_pearson(19, 19)
    assert upper == 1.0
    assert 0.8 < lower < 0.9


def _counting_statistic(values: list[float]):
    calls = {"n": 0}

    def statistic(_x: np.ndarray) -> tuple[float, int]:
        value = values[calls["n"] % len(values)]
        calls["n"] += 1
        return value, 2

    return statistic, calls


def test_null_test_sequential_stop_and_pvalue() -> None:
    x = _matrix()
    statistic, calls = _counting_statistic([0.9, 0.1, 0.9, 0.9, 0.1])
    evidence = null_test(
        x,
        0.5,
        statistic,
        families=("independent_marginals",),
        statistic_name="exploration_design_max",
        n_null_max=19,
        n_exceedances=3,
        alpha=0.05,
        random_state=0,
    )
    assert isinstance(evidence, NullEvidence)
    assert evidence.n_realisations == (4,)
    assert evidence.exceedances == (3,)
    assert evidence.stopped_early == (True,)
    assert evidence.p_values == (pytest.approx(4 / 5),)
    assert evidence.p_value == pytest.approx(4 / 5)
    assert evidence.structure_supported is False
    assert evidence.n_fits == 8
    assert calls["n"] == 4


def test_null_test_runs_to_budget_and_supports_structure() -> None:
    x = _matrix()
    statistic, _ = _counting_statistic([0.1])
    evidence = null_test(
        x,
        0.5,
        statistic,
        families=("gaussian_copula", "independent_marginals"),
        statistic_name="exploration_design_max",
        n_null_max=19,
        n_exceedances=3,
        alpha=0.05,
        random_state=1,
    )
    assert evidence.n_realisations == (19, 19)
    assert evidence.exceedances == (0, 0)
    assert evidence.stopped_early == (False, False)
    assert evidence.p_value == pytest.approx(0.05)
    assert evidence.structure_supported is True
    assert evidence.p_intervals[0][0] == 0.0


def test_null_test_skips_when_nothing_to_test() -> None:
    x = _matrix()
    statistic, calls = _counting_statistic([0.1])
    skipped = null_test(
        x,
        0.5,
        statistic,
        families=(),
        statistic_name="exploration_design_max",
        n_null_max=19,
        n_exceedances=3,
        alpha=0.05,
        random_state=0,
    )
    assert skipped.skipped_reason == "null_not_run"
    assert skipped.structure_supported is False
    zero = null_test(
        x,
        0.0,
        statistic,
        families=("gaussian_copula",),
        statistic_name="exploration_design_max",
        n_null_max=19,
        n_exceedances=3,
        alpha=0.05,
        random_state=0,
    )
    assert zero.skipped_reason == "observed_statistic_zero"
    assert zero.n_realisations == (0,)
    assert calls["n"] == 0


def _first_value(x_null: np.ndarray) -> tuple[float, int]:
    return float(x_null[0, 0]), 2


@pytest.mark.parametrize("observed", [1.2, 10.0])
def test_null_test_batches_leave_the_evidence_unchanged(observed: float) -> None:
    """Batches of five give the serial evidence at a mid-batch stop and at the budget."""
    x = _matrix()
    common = {
        "families": ("independent_marginals", "gaussian_copula"),
        "statistic_name": "exploration_design_max",
        "n_null_max": 19,
        "n_exceedances": 3,
        "alpha": 0.05,
        "random_state": 0,
    }
    serial = null_test(x, observed, _first_value, **common)
    batched = null_test(x, observed, _first_value, n_jobs=5, **common)
    assert replace(batched, seconds=0.0) == replace(serial, seconds=0.0)
    if observed < 5.0:
        assert serial.n_realisations == (17, 12)
        assert serial.n_fits == 2 * (17 + 12)
    else:
        assert serial.n_realisations == (19, 19)


def test_null_test_keeps_one_worker_count_for_every_batch(monkeypatch) -> None:
    """A short final batch reuses the full pool instead of resizing it."""
    sizes: list[int] = []

    def recording_map(fn, items, *, n_jobs):
        sizes.append(n_jobs)
        return [fn(item) for item in items]

    monkeypatch.setattr("starfold.nulls.parallel_map", recording_map)
    evidence = null_test(
        _matrix(),
        10.0,
        _first_value,
        families=("gaussian_copula",),
        statistic_name="exploration_design_max",
        n_null_max=19,
        n_exceedances=3,
        alpha=0.05,
        random_state=0,
        n_jobs=4,
    )
    assert evidence.n_realisations == (19,)
    assert sizes == [4, 4, 4, 4, 4]


def test_null_test_rejects_underpowered_budget() -> None:
    x = _matrix()
    statistic, _ = _counting_statistic([0.1])
    with pytest.raises(ValueError, match="cannot reach alpha"):
        null_test(
            x,
            0.5,
            statistic,
            families=("gaussian_copula",),
            statistic_name="exploration_design_max",
            n_null_max=10,
            n_exceedances=3,
            alpha=0.05,
            random_state=0,
        )
    with pytest.raises(ValueError, match="n_jobs"):
        null_test(
            x,
            0.5,
            statistic,
            families=("gaussian_copula",),
            statistic_name="exploration_design_max",
            n_null_max=19,
            n_exceedances=3,
            alpha=0.05,
            random_state=0,
            n_jobs=0,
        )
