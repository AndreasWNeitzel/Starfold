"""Plotting utilities for starfold results.

Two categories:

* **Core plots** (paper-faithful, required by the public API):
  :func:`plot_embedding` and :func:`plot_trustworthiness_curve`.
* **Diagnostic plots** (for README / tutorial figures, not strictly
  required by the methodology): :func:`plot_persistence_vs_baseline`,
  :func:`plot_optuna_history`, :func:`plot_optuna_param_importance`,
  :func:`plot_embedding_comparison`.

Every function accepts an optional ``ax`` argument and returns the
:class:`matplotlib.axes.Axes` it drew on, so callers can compose
panels without rebuilding figures. Default styling avoids the
matplotlib warning about missing backends on headless CI machines by
deferring the figure creation to the caller when ``ax`` is supplied.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import optuna
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike

__all__ = [
    "plot_embedding",
    "plot_embedding_comparison",
    "plot_optuna_history",
    "plot_optuna_param_importance",
    "plot_persistence_vs_baseline",
    "plot_trustworthiness_curve",
]


def _get_ax(ax: Axes | None, figsize: tuple[float, float]) -> Axes:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if ax is not None:
        return ax
    _, new_ax = plt.subplots(figsize=figsize)
    return new_ax


def plot_embedding(
    embedding: ArrayLike,
    labels: ArrayLike | None = None,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    point_size: float = 8.0,
    outlier_color: str = "lightgrey",
    cmap: str = "tab20",
    figsize: tuple[float, float] = (6.0, 5.5),
) -> Axes:
    """Scatter-plot a 2-D embedding coloured by cluster label.

    Parameters
    ----------
    embedding
        2-D array of shape ``(n_samples, 2)``.
    labels
        Cluster labels of shape ``(n_samples,)``. ``-1`` is treated as
        outlier and drawn in ``outlier_color``. Pass ``None`` to
        ignore labels.
    ax
        Existing axes. If ``None``, a new figure/axes are created.
    title
        Optional title.
    point_size
        Scatter marker size.
    outlier_color
        Colour used for ``-1`` labels.
    cmap
        Matplotlib colormap name used for non-outlier labels.
    figsize
        Size of the new figure if ``ax`` is ``None``.

    Returns
    -------
    Axes
        The axes with the scatter plot.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    emb = np.asarray(embedding, dtype=np.float64)
    if emb.ndim != 2 or emb.shape[1] != 2:
        msg = f"embedding must have shape (n_samples, 2) (got {emb.shape})."
        raise ValueError(msg)
    axis = _get_ax(ax, figsize)

    if labels is None:
        axis.scatter(emb[:, 0], emb[:, 1], s=point_size, color="tab:blue")
    else:
        lab = np.asarray(labels).astype(np.intp, copy=False)
        outliers = lab < 0
        if outliers.any():
            axis.scatter(
                emb[outliers, 0],
                emb[outliers, 1],
                s=point_size,
                color=outlier_color,
                label="outlier",
                alpha=0.6,
            )
        unique = np.unique(lab[~outliers])
        colormap = plt.get_cmap(cmap)
        for i, u in enumerate(unique):
            mask = lab == u
            axis.scatter(
                emb[mask, 0],
                emb[mask, 1],
                s=point_size,
                color=colormap(i % colormap.N),
                label=f"cluster {int(u)}",
            )

    axis.set_xlabel("component 1")
    axis.set_ylabel("component 2")
    if title is not None:
        axis.set_title(title)
    axis.set_aspect("equal", adjustable="datalim")
    return axis


def plot_trustworthiness_curve(
    scores: Mapping[int, float],
    *,
    ax: Axes | None = None,
    threshold: float | None = 0.9,
    figsize: tuple[float, float] = (5.5, 4.0),
) -> Axes:
    """Plot :math:`T(k)` vs :math:`k`.

    Parameters
    ----------
    scores
        Mapping from ``k`` to :math:`T(k)`. Typically the output of
        :func:`starfold.trustworthiness.trustworthiness_curve`.
    ax
        Existing axes.
    threshold
        Horizontal reference line. Defaults to 0.9, the paper's
        acceptance heuristic. Pass ``None`` to hide it.
    figsize
        Figure size when ``ax`` is ``None``.

    Returns
    -------
    Axes
        The axes with the curve drawn.
    """
    axis = _get_ax(ax, figsize)
    ks = sorted(scores.keys())
    ts = [scores[k] for k in ks]
    axis.plot(ks, ts, marker="o")
    if threshold is not None:
        axis.axhline(
            threshold, color="grey", linestyle="--", linewidth=1.0, label=f"T = {threshold}"
        )
        axis.legend(loc="lower right")
    axis.set_xlabel("k (nearest neighbours)")
    axis.set_ylabel("trustworthiness T(k)")
    axis.set_ylim(0.0, 1.01)
    return axis


def plot_persistence_vs_baseline(
    persistence: ArrayLike,
    *,
    baseline: float | None = None,
    per_realisation_max: ArrayLike | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (6.0, 4.0),
) -> Axes:
    """Bar-plot per-cluster persistence against a noise baseline.

    Parameters
    ----------
    persistence
        Per-cluster persistence values, length ``n_clusters``.
    baseline
        Horizontal line at the noise threshold. Typically
        :attr:`NoiseBaselineResult.threshold`.
    per_realisation_max
        Optional per-realisation maxima to overlay as a violin on the
        right-hand side, giving a sense of the baseline's spread.
    ax
        Existing axes.
    figsize
        Figure size.

    Returns
    -------
    Axes
        The axes with the bars.
    """
    axis = _get_ax(ax, figsize)
    values = np.asarray(persistence, dtype=np.float64)
    xs = np.arange(values.shape[0])
    if baseline is not None:
        colors = ["tab:green" if v > baseline else "tab:grey" for v in values]
    else:
        colors = ["tab:blue"] * values.shape[0]
    axis.bar(xs, values, color=colors, label="cluster persistence")
    if baseline is not None:
        axis.axhline(
            baseline,
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
            label=f"noise threshold = {baseline:.3f}",
        )
    if per_realisation_max is not None:
        per = np.asarray(per_realisation_max, dtype=np.float64)
        right = float(values.shape[0]) + 0.5
        axis.violinplot([per], positions=[right], widths=0.6, showmeans=True)
    axis.set_xlabel("cluster index")
    axis.set_ylabel("persistence")
    axis.legend(loc="upper right")
    return axis


def plot_optuna_history(
    study: optuna.Study,
    *,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (6.0, 4.0),
) -> Axes:
    """Plot Optuna trial values and the running-best objective."""
    axis = _get_ax(ax, figsize)
    values = np.array([t.value for t in study.trials if t.value is not None], dtype=np.float64)
    if values.size == 0:
        axis.text(0.5, 0.5, "no completed trials", ha="center", va="center")
        return axis
    trial_numbers = np.arange(1, values.shape[0] + 1)
    axis.scatter(trial_numbers, values, s=25, color="tab:blue", label="trial")
    best = (
        np.maximum.accumulate(values)
        if study.direction.name == "MAXIMIZE"
        else np.minimum.accumulate(values)
    )
    axis.plot(trial_numbers, best, color="tab:red", linewidth=1.5, label="running best")
    axis.set_xlabel("trial #")
    axis.set_ylabel("objective (sum cluster persistence)")
    axis.legend(loc="lower right")
    return axis


def plot_optuna_param_importance(
    study: optuna.Study,
    *,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (5.0, 3.0),
) -> Axes:
    """Horizontal bar plot of Optuna parameter importances.

    Gracefully degrades to a text annotation if the study has too few
    completed trials to fit an importance estimator.
    """
    import optuna  # noqa: PLC0415

    axis = _get_ax(ax, figsize)
    try:
        importances: dict[str, float] = optuna.importance.get_param_importances(study)
    except (ValueError, RuntimeError):
        axis.text(
            0.5,
            0.5,
            "importance unavailable\n(too few trials)",
            ha="center",
            va="center",
        )
        return axis
    names = list(importances.keys())
    values = list(importances.values())
    axis.barh(names, values, color="tab:purple")
    axis.set_xlabel("importance")
    axis.set_xlim(0.0, 1.0)
    return axis


def plot_embedding_comparison(
    embeddings: Mapping[str, ArrayLike],
    labels: ArrayLike | None = None,
    *,
    figsize: tuple[float, float] = (15.0, 4.5),
    point_size: float = 8.0,
) -> tuple[Figure, Sequence[Axes]]:
    """Side-by-side scatter of multiple embeddings sharing one label vector.

    Parameters
    ----------
    embeddings
        Mapping from method name (e.g. ``"PCA"``) to 2-D embedding.
    labels
        Optional single label vector applied to every panel.
    figsize
        Size of the combined figure.
    point_size
        Scatter marker size.

    Returns
    -------
    tuple of Figure and list of Axes
        The created figure and its axes in insertion order.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    n = len(embeddings)
    if n == 0:
        msg = "embeddings must contain at least one entry."
        raise ValueError(msg)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axes_flat: list[Axes] = list(axes.flat)
    for axis, (name, emb) in zip(axes_flat, embeddings.items(), strict=True):
        plot_embedding(emb, labels, ax=axis, title=name, point_size=point_size)
        axis.legend().set_visible(False)
    fig.tight_layout()
    return fig, axes_flat
