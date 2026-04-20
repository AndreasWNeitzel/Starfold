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
from starfold.plotting import (
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
