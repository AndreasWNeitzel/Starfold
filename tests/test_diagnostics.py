"""Tests for ``starfold.diagnostics``.

Covers:

* ``validate_input_matrix`` refuses NaN/inf and tiny-n inputs with
  clear error messages, and warns on the marginal 2*n_neighbors regime.
* ``auto_mcs_upper`` produces the documented regime boundaries.
* ``recommend_budget`` returns the advertised dict for each sample-size
  bucket.
* ``diagnose_fit`` surfaces the expected flags on synthetic degenerate
  / low-trust / noise-consistent / hierarchy-unavailable results.
* ``warn_fit_flags`` threads every flag through ``warnings``.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from starfold.diagnostics import (
    auto_mcs_upper,
    diagnose_fit,
    recommend_budget,
    validate_input_matrix,
    warn_fit_flags,
)

# ---------------------------------------------------------------- validate


def test_validate_accepts_clean_matrix() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    out = validate_input_matrix(X, n_neighbors=15)
    assert out.shape == (120, 4)
    assert out.dtype == np.float64


def test_validate_rejects_nan() -> None:
    X = np.ones((50, 3))
    X[3, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_input_matrix(X, n_neighbors=5)


def test_validate_rejects_inf() -> None:
    X = np.ones((50, 3))
    X[7, 2] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        validate_input_matrix(X, n_neighbors=5)


def test_validate_rejects_1d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        validate_input_matrix(np.zeros(10), n_neighbors=3)


def test_validate_rejects_n_samples_le_n_neighbors() -> None:
    X = np.zeros((10, 2))
    with pytest.raises(ValueError, match="n_neighbors"):
        validate_input_matrix(X, n_neighbors=15)


def test_validate_warns_on_small_n() -> None:
    X = np.random.default_rng(0).normal(size=(20, 2))
    with pytest.warns(UserWarning, match="unstable"):
        validate_input_matrix(X, n_neighbors=10)


def test_validate_does_not_warn_on_large_enough_n() -> None:
    X = np.random.default_rng(0).normal(size=(200, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = validate_input_matrix(X, n_neighbors=10)
    assert out.shape == (200, 2)


def test_validate_custom_name_in_error() -> None:
    with pytest.raises(ValueError, match="X_aug"):
        validate_input_matrix(np.zeros(10), n_neighbors=3, name="X_aug")


# ---------------------------------------------------------------- auto_mcs


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (20, 5),  # max(5, 20//4=5)
        (40, 10),  # max(5, 40//4=10)
        (49, 12),  # small-N branch: max(5, 49//4=12)
        (50, 50),  # clip lower bound hits at n//20=2 -> 50
        (500, 50),  # n//20=25 -> 50
        (1_000, 50),
        (10_000, 500),  # n//20=500
        (50_000, 2_500),
        (100_000, 5_000),
        (10_000_000, 5_000),  # upper clip
    ],
)
def test_auto_mcs_upper_boundaries(n: int, expected: int) -> None:
    assert auto_mcs_upper(n) == expected


# ---------------------------------------------------------------- budget


@pytest.mark.parametrize(
    ("n", "trials", "reals", "per"),
    [
        (800, 50, 500, 20),
        (1_000, 50, 500, 20),
        (10_000, 100, 1_000, 20),
        (50_000, 100, 1_000, 20),
        (200_000, 120, 500, 15),
        (500_000, 120, 500, 15),
        (1_000_000, 150, 200, 10),
    ],
)
def test_recommend_budget_regimes(n: int, trials: int, reals: int, per: int) -> None:
    rec = recommend_budget(n)
    assert rec["hdbscan_optuna_trials"] == trials
    assert rec["n_realisations"] == reals
    assert rec["per_realisation_trials"] == per


# ---------------------------------------------------------------- diagnose


def _fake_result(
    *,
    labels: np.ndarray,
    n_clusters: int,
    trustworthiness: float,
    significant: np.ndarray | None = None,
    noise_threshold: float | None = None,
    hierarchy_available: bool = True,
    optuna_values: list[float] | None = None,
) -> SimpleNamespace:
    trials: list[Any] = []
    if optuna_values is not None:
        trials.extend(SimpleNamespace(value=v) for v in optuna_values)
    study = SimpleNamespace(
        trials=trials,
        direction=SimpleNamespace(name="MAXIMIZE"),
    )
    noise_ns = None if noise_threshold is None else SimpleNamespace(threshold=noise_threshold)
    return SimpleNamespace(
        labels=labels,
        n_clusters=n_clusters,
        trustworthiness=trustworthiness,
        significant=significant,
        noise_baseline=noise_ns,
        hierarchy=SimpleNamespace(available=hierarchy_available),
        search=SimpleNamespace(study=study),
    )


def test_diagnose_flags_degenerate_fit() -> None:
    result = _fake_result(
        labels=np.full(50, -1, dtype=np.intp),
        n_clusters=0,
        trustworthiness=0.95,
    )
    flags = diagnose_fit(result)  # type: ignore[arg-type]
    assert any("DEGENERATE FIT" in f for f in flags)


def test_diagnose_flags_high_outlier_fraction() -> None:
    labels = np.full(100, -1, dtype=np.intp)
    labels[:5] = 0  # 95% outliers
    result = _fake_result(labels=labels, n_clusters=1, trustworthiness=0.95)
    flags = diagnose_fit(result)  # type: ignore[arg-type]
    assert any("HIGH OUTLIER FRACTION" in f for f in flags)


def test_diagnose_flags_low_trustworthiness() -> None:
    result = _fake_result(
        labels=np.zeros(50, dtype=np.intp),
        n_clusters=1,
        trustworthiness=0.75,
    )
    flags = diagnose_fit(result)  # type: ignore[arg-type]
    assert any("LOW TRUSTWORTHINESS" in f for f in flags)


def test_diagnose_flags_noise_consistent() -> None:
    labels = np.zeros(50, dtype=np.intp)
    significant = np.array([False, False])
    result = _fake_result(
        labels=labels,
        n_clusters=2,
        trustworthiness=0.95,
        significant=significant,
        noise_threshold=0.42,
    )
    flags = diagnose_fit(result)  # type: ignore[arg-type]
    assert any("NOISE-CONSISTENT" in f for f in flags)


def test_diagnose_flags_hierarchy_unavailable() -> None:
    result = _fake_result(
        labels=np.zeros(50, dtype=np.intp),
        n_clusters=1,
        trustworthiness=0.95,
        hierarchy_available=False,
    )
    flags = diagnose_fit(result)  # type: ignore[arg-type]
    assert any("HIERARCHY UNAVAILABLE" in f for f in flags)


def test_diagnose_flags_optuna_plateau() -> None:
    # 20 trials, best is reached on trial 2 and never improves
    values = [0.1, 0.9] + [0.9] * 18
    result = _fake_result(
        labels=np.zeros(50, dtype=np.intp),
        n_clusters=1,
        trustworthiness=0.95,
        optuna_values=values,
    )
    flags = diagnose_fit(result)  # type: ignore[arg-type]
    assert any("plateau" in f.lower() or "before the end" in f for f in flags)


def test_diagnose_flags_optuna_flat_zero() -> None:
    values = [0.0] * 10
    result = _fake_result(
        labels=np.full(50, -1, dtype=np.intp),
        n_clusters=0,
        trustworthiness=0.95,
        optuna_values=values,
    )
    flags = diagnose_fit(result)  # type: ignore[arg-type]
    assert any("did not find any partition" in f for f in flags)


def test_diagnose_empty_on_healthy_result() -> None:
    values = [0.1 + 0.01 * i for i in range(20)]  # monotone improving
    result = _fake_result(
        labels=np.zeros(100, dtype=np.intp),
        n_clusters=3,
        trustworthiness=0.98,
        significant=np.array([True, True, True]),
        noise_threshold=0.1,
        optuna_values=values,
    )
    flags = diagnose_fit(result)  # type: ignore[arg-type]
    assert flags == []


def test_warn_fit_flags_emits_each_flag() -> None:
    flags = ["alpha", "beta"]
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        warn_fit_flags(flags)
    messages = [str(w.message) for w in captured]
    assert "alpha" in messages
    assert "beta" in messages
