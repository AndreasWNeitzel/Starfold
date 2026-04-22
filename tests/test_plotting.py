"""Smoke tests for the plotting utilities.

Uses the non-interactive Agg backend so the tests run on headless CI.
Every plot function must:

* return a :class:`matplotlib.axes.Axes` (or a Figure/Axes tuple for
  :func:`plot_embedding_comparison`),
* save to a PNG without raising,
* accept an externally-created Axes and reuse it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optuna

from starfold.clustering import search_hdbscan
from starfold.credibility import compute_credibility
from starfold.noise_baseline import NoiseBaselineResult
from starfold.plotting import (
    plot_credibility,
    plot_embedding,
    plot_embedding_comparison,
    plot_optuna_history,
    plot_optuna_param_importance,
    plot_persistence_vs_baseline,
    plot_trustworthiness_curve,
)

if TYPE_CHECKING:
    from pathlib import Path


def _random_embedding(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(120, 2))


def _random_labels(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(low=-1, high=3, size=120, dtype=np.intp)


def test_plot_embedding_returns_axes_and_saves(tmp_path: Path) -> None:
    emb = _random_embedding()
    labels = _random_labels()
    ax = plot_embedding(emb, labels, title="demo")
    assert ax.figure is not None
    out = tmp_path / "embedding.png"
    ax.figure.savefig(out)
    assert out.exists()
    assert out.stat().st_size > 0
    plt.close(ax.figure)


def test_plot_embedding_rejects_non_2d() -> None:
    import pytest  # noqa: PLC0415

    with pytest.raises(ValueError, match="shape"):
        plot_embedding(np.zeros((10, 3)))


def test_plot_trustworthiness_curve_returns_axes() -> None:
    scores = {5: 0.92, 10: 0.94, 15: 0.95}
    ax = plot_trustworthiness_curve(scores)
    assert ax.lines
    plt.close(ax.figure)


def test_plot_persistence_vs_baseline_colours_significant_clusters() -> None:
    persistence = np.array([0.2, 0.7, 0.9])
    ax = plot_persistence_vs_baseline(persistence, baseline=0.5)
    bar_colors = {patch.get_facecolor() for patch in ax.patches}
    assert len(bar_colors) >= 2
    plt.close(ax.figure)


def test_plot_persistence_vs_baseline_accepts_per_realisation_max() -> None:
    persistence = np.array([0.6])
    per_realisation_max = np.array([0.1, 0.2, 0.3, 0.4])
    ax = plot_persistence_vs_baseline(
        persistence, baseline=0.5, per_realisation_max=per_realisation_max
    )
    plt.close(ax.figure)


def test_plot_optuna_history_on_real_study() -> None:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(loc, size=(50, 2)) for loc in (-5, 0, 5)])
    search = search_hdbscan(
        X,
        n_trials=8,
        mcs_range=(5, 20),
        ms_range=(1, 5),
        random_state=0,
        engine="cpu",
    )
    ax = plot_optuna_history(search.study)
    assert ax.collections  # scatter created a collection
    # Default objective is "persistence_sum" so the derived label wins over
    # a generic fallback; the plot must identify which objective it shows.
    assert "persistence" in ax.get_ylabel().lower()
    plt.close(ax.figure)


def test_plot_optuna_history_labels_combined_geom() -> None:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(loc, size=(50, 2)) for loc in (-5, 0, 5)])
    search = search_hdbscan(
        X,
        n_trials=8,
        mcs_range=(5, 20),
        ms_range=(1, 5),
        random_state=0,
        engine="cpu",
        objective="combined_geom",
    )
    # search_hdbscan must stamp the objective onto the study itself so
    # downstream plots do not need to be told separately.
    assert search.study.user_attrs.get("objective") == "combined_geom"
    ax = plot_optuna_history(search.study)
    label = ax.get_ylabel()
    # The combined_geom label contains a LaTeX sqrt and the DBCV token;
    # it must not fall back to the persistence-sum phrase.
    assert "sqrt" in label.lower() or r"\sqrt" in label
    assert "sum" not in label.lower()
    plt.close(ax.figure)


def test_plot_optuna_history_empty_study() -> None:
    study = optuna.create_study(direction="maximize")
    ax = plot_optuna_history(study)
    assert ax.texts
    plt.close(ax.figure)


def test_plot_optuna_param_importance_handles_small_study() -> None:
    study = optuna.create_study(direction="maximize")
    ax = plot_optuna_param_importance(study)
    assert ax.texts
    plt.close(ax.figure)


def test_plot_embedding_comparison_returns_fig_and_axes() -> None:
    emb_a = _random_embedding(0)
    emb_b = _random_embedding(1)
    labels = _random_labels()
    fig, axes = plot_embedding_comparison({"A": emb_a, "B": emb_b}, labels)
    assert len(list(axes)) == 2
    plt.close(fig)


def test_plot_embedding_comparison_rejects_empty_dict() -> None:
    import pytest  # noqa: PLC0415

    with pytest.raises(ValueError, match="at least one"):
        plot_embedding_comparison({})


def test_plot_embedding_accepts_external_axes() -> None:
    emb = _random_embedding()
    fig, ax = plt.subplots()
    returned = plot_embedding(emb, ax=ax)
    assert returned is ax
    plt.close(fig)


def test_plot_credibility_returns_three_axes() -> None:
    rng = np.random.default_rng(0)
    null_pool = rng.uniform(0.0, 0.3, size=100)
    baseline = NoiseBaselineResult(
        threshold=0.5,
        per_realisation_max=rng.uniform(0.0, 0.3, size=50),
        per_realisation_n_clusters=rng.integers(1, 4, size=50, dtype=np.intp),
        per_realisation_objective=rng.uniform(0.0, 1.0, size=50),
        null_cluster_persistence=null_pool,
        null_cluster_size=np.full(100, 10, dtype=np.intp),
        null_cluster_realisation=rng.integers(0, 50, size=100, dtype=np.intp),
        percentile=99.7,
        config={"objective": "persistence_sum"},
    )
    report = compute_credibility(
        n_clusters=6, best_objective=5.0, max_persistence=0.9,
        baseline=baseline, alpha=0.05,
    )
    axes = plot_credibility(report)
    assert len(axes) == 3
    for ax in axes:
        assert ax.patches or ax.texts
    plt.close(axes[0].figure)


def test_plot_credibility_rejects_wrong_axes_count() -> None:
    import pytest  # noqa: PLC0415

    baseline = NoiseBaselineResult(
        threshold=0.0,
        per_realisation_max=np.zeros(3),
        per_realisation_n_clusters=np.zeros(3, dtype=np.intp),
        per_realisation_objective=np.zeros(3),
        null_cluster_persistence=np.zeros(0, dtype=np.float64),
        null_cluster_size=np.zeros(0, dtype=np.intp),
        null_cluster_realisation=np.zeros(0, dtype=np.intp),
        percentile=99.7,
        config={"objective": "persistence_sum"},
    )
    report = compute_credibility(
        n_clusters=2, best_objective=0.5, max_persistence=0.5,
        baseline=baseline,
    )
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="exactly three"):
        plot_credibility(report, axes=[ax])
    plt.close(fig)


def test_plot_per_cluster_credibility_smoke() -> None:
    from starfold.plotting import plot_per_cluster_credibility  # noqa: PLC0415

    rng = np.random.default_rng(0)
    null_pool = rng.uniform(0.0, 0.3, size=100)
    baseline = NoiseBaselineResult(
        threshold=0.5,
        per_realisation_max=rng.uniform(0.0, 0.3, size=20),
        per_realisation_n_clusters=rng.integers(1, 4, size=20, dtype=np.intp),
        per_realisation_objective=rng.uniform(0.0, 1.0, size=20),
        null_cluster_persistence=null_pool,
        null_cluster_size=np.full(100, 10, dtype=np.intp),
        null_cluster_realisation=rng.integers(0, 20, size=100, dtype=np.intp),
        percentile=99.7,
        config={"objective": "persistence_sum"},
    )
    observed = np.array([0.9, 0.2, 0.05])
    report = compute_credibility(
        n_clusters=3, best_objective=3.0, max_persistence=0.9,
        baseline=baseline, cluster_persistence=observed, alpha=0.05,
    )
    ax = plot_per_cluster_credibility(report)
    # One bar per cluster.
    assert len(ax.patches) == 3
    plt.close(ax.figure)


def test_plot_per_cluster_credibility_handles_no_clusters() -> None:
    from starfold.plotting import plot_per_cluster_credibility  # noqa: PLC0415

    baseline = NoiseBaselineResult(
        threshold=0.0,
        per_realisation_max=np.zeros(3),
        per_realisation_n_clusters=np.zeros(3, dtype=np.intp),
        per_realisation_objective=np.zeros(3),
        null_cluster_persistence=np.zeros(0, dtype=np.float64),
        null_cluster_size=np.zeros(0, dtype=np.intp),
        null_cluster_realisation=np.zeros(0, dtype=np.intp),
        percentile=99.7,
        config={"objective": "persistence_sum"},
    )
    report = compute_credibility(
        n_clusters=0, best_objective=0.0, max_persistence=0.0,
        baseline=baseline,
    )
    ax = plot_per_cluster_credibility(report)
    # Empty bar collection; text placeholder drawn.
    assert not ax.patches
    plt.close(ax.figure)


def test_plot_uncertainty_map_smoke() -> None:
    from starfold.plotting import plot_uncertainty_map  # noqa: PLC0415
    from starfold.uncertainty import UncertaintyPropagation  # noqa: PLC0415

    rng = np.random.default_rng(3)
    emb = rng.normal(size=(50, 2))
    membership = rng.dirichlet(np.ones(4), size=50)
    consensus = np.argmax(membership, axis=1).astype(np.intp)
    instab = 1.0 - membership.max(axis=1)
    prop = UncertaintyPropagation(
        membership=membership,
        consensus_label=consensus,
        instability=instab,
        n_draws=10,
        sigma_shape="scalar",
    )
    ax = plot_uncertainty_map(emb, prop)
    assert len(ax.collections) >= 1
    plt.close(ax.figure)


def test_plot_uncertainty_map_rejects_shape_mismatch() -> None:
    from starfold.plotting import plot_uncertainty_map  # noqa: PLC0415
    from starfold.uncertainty import UncertaintyPropagation  # noqa: PLC0415

    rng = np.random.default_rng(4)
    emb = rng.normal(size=(50, 2))
    membership = rng.dirichlet(np.ones(3), size=10)  # wrong n
    prop = UncertaintyPropagation(
        membership=membership,
        consensus_label=np.zeros(10, dtype=np.intp),
        instability=np.zeros(10),
        n_draws=5,
        sigma_shape="scalar",
    )
    import pytest  # noqa: PLC0415

    with pytest.raises(ValueError, match="samples"):
        plot_uncertainty_map(emb, prop)
