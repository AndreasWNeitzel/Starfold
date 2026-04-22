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

    import hdbscan as _hdbscan_typing
    import optuna
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike

    from starfold.credibility import CredibilityReport
    from starfold.stability import SubsampleStability
    from starfold.uncertainty import UncertaintyPropagation

__all__ = [
    "plot_condensed_tree",
    "plot_credibility",
    "plot_embedding",
    "plot_embedding_comparison",
    "plot_granularity_stability",
    "plot_membership_confidence",
    "plot_optuna_history",
    "plot_optuna_hyperparam_landscape",
    "plot_optuna_parallel",
    "plot_optuna_param_importance",
    "plot_optuna_pareto",
    "plot_per_cluster_credibility",
    "plot_persistence_vs_baseline",
    "plot_subsample_stability",
    "plot_trustworthiness_curve",
    "plot_uncertainty_map",
]


_OBJECTIVE_Y_LABELS: dict[str, str] = {
    "persistence_sum": "objective (sum of cluster persistence)",
    "combined_geom": r"objective ($\sqrt{\max(\mathrm{DBCV},0)\cdot\widetilde{p}}$)",
}


def _objective_label(study: optuna.Study) -> str:
    """Human-readable y-axis label for the scalar Optuna objective.

    Reads ``study.user_attrs["objective"]`` (set by
    :func:`starfold.clustering.search_hdbscan`) and falls back to the
    generic ``"objective"`` when absent, so plots still work for
    studies built outside starfold.
    """
    name = str(study.user_attrs.get("objective", ""))
    return _OBJECTIVE_Y_LABELS.get(name, "objective")


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
    continuity_scores: Mapping[int, float] | None = None,
    ax: Axes | None = None,
    threshold: float | None = 0.9,
    figsize: tuple[float, float] = (5.5, 4.0),
) -> Axes:
    """Plot :math:`T(k)` vs :math:`k`, optionally overlaid with :math:`C(k)`.

    Parameters
    ----------
    scores
        Mapping from ``k`` to :math:`T(k)`. Typically the output of
        :func:`starfold.trustworthiness.trustworthiness_curve`.
    continuity_scores
        Optional mapping from ``k`` to :math:`C(k)`. When supplied,
        the continuity curve is drawn alongside trustworthiness on the
        same axes -- the two together bracket the embedding's fidelity
        (T bounds the false-neighbour rate, C bounds the lost-neighbour
        rate). Typically the output of
        :func:`starfold.trustworthiness.continuity_curve`.
    ax
        Existing axes.
    threshold
        Horizontal reference line. Defaults to 0.9, the paper's
        acceptance heuristic for trustworthiness. Pass ``None`` to hide
        it.
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
    axis.plot(ks, ts, marker="o", label="T(k) trustworthiness")
    if continuity_scores is not None:
        cs = [continuity_scores[k] for k in ks if k in continuity_scores]
        cks = [k for k in ks if k in continuity_scores]
        axis.plot(cks, cs, marker="s", linestyle="--", label="C(k) continuity")
    if threshold is not None:
        axis.axhline(
            threshold, color="grey", linestyle=":", linewidth=1.0, label=f"threshold {threshold}"
        )
    axis.legend(loc="lower right")
    axis.set_xlabel("k (nearest neighbours)")
    axis.set_ylabel("T(k) / C(k)" if continuity_scores is not None else "trustworthiness T(k)")
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


def _credibility_panel(
    axis: Axes,
    observed: float,
    null: np.ndarray,
    pvalue: float,
    *,
    xlabel: str,
    alpha: float,
    integer: bool = False,
) -> None:
    if null.size == 0:
        axis.text(0.5, 0.5, "no null samples", ha="center", va="center")
        axis.set_xlabel(xlabel)
        return
    if integer:
        lo, hi = int(np.min(null)), int(np.max(null))
        edges: list[float] = [float(x) for x in np.arange(lo, hi + 2) - 0.5]
        axis.hist(
            null, bins=edges, color="tab:grey", edgecolor="white", alpha=0.8, label="noise null"
        )
    else:
        axis.hist(null, bins=30, color="tab:grey", edgecolor="white", alpha=0.8, label="noise null")
    colour = "tab:green" if pvalue < alpha else "tab:red"
    axis.axvline(
        observed,
        color=colour,
        linewidth=2.0,
        label=f"observed = {observed:.3f}" if not integer else f"observed = {int(observed)}",
    )
    axis.set_xlabel(xlabel)
    axis.set_ylabel("noise-realisation count")
    axis.set_title(f"p = {pvalue:.4f}", fontsize=10)
    axis.legend(loc="upper right", fontsize=8)


def plot_credibility(
    report: CredibilityReport,
    *,
    axes: Sequence[Axes] | None = None,
    figsize: tuple[float, float] = (13.0, 4.0),
) -> Sequence[Axes]:
    """Three-panel null-distribution histogram with observed overlay.

    Each panel shows the noise-realisation null distribution of one
    run-level scalar (number of clusters, best Optuna objective,
    maximum cluster persistence) with the real-data observation
    overlaid as a vertical line. The panel title is the upper-tail
    p-value; the overlay line is green when ``p < alpha`` and red
    otherwise, mirroring the ``passes`` criterion on the report.

    Parameters
    ----------
    report
        A :class:`CredibilityReport` from
        :func:`starfold.credibility.compute_credibility`.
    axes
        Three existing axes, left-to-right. When ``None``, a new
        1x3 figure is created.
    figsize
        Figure size used when ``axes`` is ``None``.

    Returns
    -------
    Sequence[Axes]
        The three axes drawn on.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if axes is None:
        _, axes_arr = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
        panel: list[Axes] = list(axes_arr)
    else:
        panel = list(axes)
    if len(panel) != 3:
        msg = f"plot_credibility needs exactly three axes (got {len(panel)})."
        raise ValueError(msg)

    _credibility_panel(
        panel[0],
        observed=float(report.observed_n_clusters),
        null=np.asarray(report.null_n_clusters, dtype=np.float64),
        pvalue=report.n_clusters_pvalue,
        xlabel="n_clusters",
        alpha=report.alpha,
        integer=True,
    )
    _credibility_panel(
        panel[1],
        observed=float(report.observed_objective),
        null=np.asarray(report.null_objective, dtype=np.float64),
        pvalue=report.objective_pvalue,
        xlabel=f"best-trial objective ({report.objective_name})",
        alpha=report.alpha,
    )
    _credibility_panel(
        panel[2],
        observed=float(report.observed_max_persistence),
        null=np.asarray(report.null_max_persistence, dtype=np.float64),
        pvalue=report.max_persistence_pvalue,
        xlabel="max cluster persistence",
        alpha=report.alpha,
    )
    verdict = "PASS" if report.passes else "FAIL"
    panel[0].figure.suptitle(
        f"credibility vs noise null: {verdict} at alpha={report.alpha}",
        fontsize=12,
    )
    return panel


def plot_per_cluster_credibility(
    report: CredibilityReport,
    *,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (7.0, 4.5),
) -> Axes:
    """Bar chart of per-cluster persistence with credibility shading.

    Overlays each cluster's observed persistence (bars, green when
    ``p < alpha`` and red otherwise) against the noise null's 50th,
    99.7th, and 99.97th percentiles (dashed horizontal lines). This
    is often the most honest single figure: it shows exactly which
    clusters stand out from noise, not just whether the run-level
    scalars do.

    Parameters
    ----------
    report
        A :class:`CredibilityReport` whose
        ``observed_cluster_persistence``,
        ``null_cluster_persistence``, and
        ``per_cluster_significant`` fields are populated.
    ax
        Existing axis to draw on. Creates a new figure when ``None``.
    figsize
        Figure size used when ``ax`` is ``None``.

    Returns
    -------
    Axes
        The axis drawn on.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    obs = np.asarray(report.observed_cluster_persistence, dtype=np.float64)
    sig = np.asarray(report.per_cluster_significant, dtype=bool)
    pvals = np.asarray(report.per_cluster_pvalue, dtype=np.float64)
    null = np.asarray(report.null_cluster_persistence, dtype=np.float64)
    if obs.size == 0:
        ax.text(0.5, 0.5, "no clusters", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("cluster id")
        ax.set_ylabel("persistence")
        return ax
    colours = ["tab:green" if s else "tab:red" for s in sig]
    indices = np.arange(obs.size)
    ax.bar(indices, obs, color=colours, edgecolor="black", linewidth=0.4)
    for i, (val, p) in enumerate(zip(obs, pvals, strict=True)):
        ax.text(
            i,
            val,
            f"p={p:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
        )
    if null.size:
        for pct, style in [(50.0, ":"), (99.7, "--"), (99.97, "-.")]:
            level = float(np.percentile(null, pct))
            ax.axhline(
                level,
                color="tab:grey",
                linestyle=style,
                linewidth=1.0,
                label=f"null {pct:.2f} pct = {level:.3f}",
            )
        ax.legend(loc="upper right", fontsize=8)
    n_cred = int(sig.sum())
    ax.set_title(f"per-cluster credibility: {n_cred}/{obs.size} at alpha={report.alpha}")
    ax.set_xlabel("cluster id")
    ax.set_ylabel("persistence")
    ax.set_xticks(indices)
    return ax


def plot_uncertainty_map(
    embedding: ArrayLike,
    propagation: UncertaintyPropagation,
    *,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (7.0, 5.5),
    cmap: str = "magma",
    s: float = 10.0,
) -> Axes:
    """Scatter the 2-D embedding coloured by per-sample instability.

    Instability is ``1 - max(membership, axis=1)``. A sample that lands
    in the same cluster in every Monte Carlo draw has instability 0;
    a sample that splits 50/50 between two clusters has instability
    0.5. The resulting map is the clearest visual answer to "which of
    my samples are sitting on a cluster boundary?".

    Parameters
    ----------
    embedding
        2-D UMAP embedding of shape ``(n_samples, 2)``, typically
        ``result.embedding``.
    propagation
        An :class:`~starfold.uncertainty.UncertaintyPropagation`
        returned by :meth:`PipelineResult.propagate_uncertainty`.
    ax
        Existing axis to draw on. A new figure is created when ``None``.
    figsize
        Figure size used when ``ax`` is ``None``.
    cmap
        Matplotlib colormap for the instability colour scale.
    s
        Marker size for the scatter points.

    Returns
    -------
    Axes
        The axis drawn on.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    emb = np.asarray(embedding, dtype=np.float64)
    if emb.ndim != 2 or emb.shape[1] != 2:
        msg = f"embedding must have shape (n_samples, 2) (got {emb.shape})."
        raise ValueError(msg)
    instab = np.asarray(propagation.instability, dtype=np.float64)
    if instab.shape[0] != emb.shape[0]:
        msg = (
            f"embedding has {emb.shape[0]} samples but propagation has "
            f"{instab.shape[0]}; these must match."
        )
        raise ValueError(msg)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    scatter = ax.scatter(
        emb[:, 0],
        emb[:, 1],
        c=instab,
        cmap=cmap,
        s=s,
        vmin=0.0,
        vmax=max(0.5, float(instab.max()) if instab.size else 0.5),
        edgecolors="none",
    )
    ax.figure.colorbar(scatter, ax=ax, label="instability  (1 - max membership)")
    mean_instab = float(instab.mean()) if instab.size else 0.0
    frac_high = float((instab > 0.5).mean()) if instab.size else 0.0
    ax.set_title(
        f"uncertainty map  ({propagation.n_draws} draws, "
        f"mean={mean_instab:.3f}, >0.5: {frac_high:.1%})"
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    return ax


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
    axis.set_ylabel(_objective_label(study))
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


_ATTR_KEYS: tuple[str, ...] = (
    "relative_validity",
    "n_clusters",
    "outlier_fraction",
    "persistence_sum",
    "persistence_max",
    "persistence_mean",
    "persistence_median",
)


_METHOD_CODE = {"eom": 0.0, "leaf": 1.0}


def _trial_frame(study: optuna.Study) -> dict[str, np.ndarray]:
    """Extract (mcs, ms, value, user_attrs, extra params) arrays from a study.

    Returned keys include every metric recorded by
    :func:`starfold.clustering.search_hdbscan`, the study ``value``
    under the key ``"value"``, and the three optional Optuna axes
    ``alpha``, ``cluster_selection_epsilon`` and
    ``cluster_selection_method`` (the last encoded numerically:
    ``eom=0``, ``leaf=1``) so plotting can surface them even when the
    caller did not request categorical sampling. Trials that did not
    sample a given axis (pinned ranges) get ``NaN`` in that column.
    The legacy key ``"persistence"`` is aliased to ``"persistence_sum"``
    for backward-compatible plotting code, and ``"dbcv"`` aliases
    ``"relative_validity"``.
    """
    buckets: dict[str, list[float]] = {k: [] for k in _ATTR_KEYS}
    mcs: list[float] = []
    ms: list[float] = []
    alpha: list[float] = []
    eps: list[float] = []
    method: list[float] = []
    value: list[float] = []
    number: list[int] = []
    for t in study.trials:
        if t.value is None or t.params.get("min_cluster_size") is None:
            continue
        mcs.append(float(t.params["min_cluster_size"]))
        ms.append(float(t.params["min_samples"]))
        alpha.append(float(t.params.get("alpha", float("nan"))))
        eps.append(float(t.params.get("cluster_selection_epsilon", float("nan"))))
        method_raw = t.params.get("cluster_selection_method")
        method.append(_METHOD_CODE[method_raw] if method_raw in _METHOD_CODE else float("nan"))
        value.append(float(t.value))
        for key in _ATTR_KEYS:
            buckets[key].append(float(t.user_attrs.get(key, float("nan"))))
        number.append(int(t.number))
    frame: dict[str, np.ndarray] = {
        "mcs": np.array(mcs, dtype=np.float64),
        "ms": np.array(ms, dtype=np.float64),
        "alpha": np.array(alpha, dtype=np.float64),
        "cluster_selection_epsilon": np.array(eps, dtype=np.float64),
        "cluster_selection_method": np.array(method, dtype=np.float64),
        "value": np.array(value, dtype=np.float64),
        "number": np.array(number, dtype=np.intp),
    }
    for key in _ATTR_KEYS:
        frame[key] = np.array(buckets[key], dtype=np.float64)
    # Aliases: older callers ask for "persistence" (the sum) and "dbcv".
    frame["persistence"] = frame["persistence_sum"]
    frame["dbcv"] = frame["relative_validity"]
    return frame


def _pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return a boolean mask of maximise-maximise Pareto-optimal points."""
    n = x.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated = (x >= x[i]) & (y >= y[i]) & ((x > x[i]) | (y > y[i]))
        if np.any(dominated):
            keep[i] = False
    return keep


def _best_trial_row(study: optuna.Study, data: dict[str, np.ndarray]) -> int | None:
    """Row index into ``_trial_frame`` for ``study.best_trial``.

    Returns ``None`` if the study has no best trial (no completed trial)
    or the best trial was dropped from the frame (e.g. its value is
    None). The match is on ``trial.number``.
    """
    try:
        best_number = int(study.best_trial.number)
    except (ValueError, AttributeError):
        return None
    numbers = data["number"]
    hit = np.flatnonzero(numbers == best_number)
    if hit.size == 0:
        return None
    return int(hit[0])


_METRIC_LABELS: dict[str, str] = {
    "persistence_sum": "sum of cluster persistence",
    "persistence_median": "median cluster persistence",
    "persistence_mean": "mean cluster persistence",
    "persistence_max": "max cluster persistence",
    "relative_validity": "relative validity (DBCV proxy)",
    "dbcv": "relative validity (DBCV proxy)",
    "persistence": "sum of cluster persistence",
    "n_clusters": "n_clusters",
    "outlier_fraction": "outlier fraction",
}


def plot_optuna_pareto(
    study: optuna.Study,
    *,
    x_metric: str = "persistence_sum",
    y_metric: str = "relative_validity",
    ax: Axes | None = None,
    figsize: tuple[float, float] = (5.5, 4.5),
    selected_index: int | None = None,
    selected_label: str | None = None,
) -> Axes:
    """Scatter trials in (x_metric, y_metric) space with a Pareto frontier.

    Both axes are treated as maximise-direction. Any trial-level metric
    recorded in :attr:`optuna.trial.Trial.user_attrs` by
    :func:`starfold.clustering.search_hdbscan` is a valid choice -- e.g.
    ``"persistence_sum"``, ``"persistence_median"``,
    ``"relative_validity"``. A point is on the frontier when no other
    trial dominates it on both axes simultaneously. Trials whose
    chosen metric evaluates to NaN are dropped before plotting.

    Parameters
    ----------
    study
        The Optuna study.
    x_metric, y_metric
        Which user_attr to place on each axis. Aliases ``"persistence"``
        (sum) and ``"dbcv"`` (relative_validity) are honoured.
    ax, figsize
        Standard plotting arguments.
    selected_index
        Index into the filtered (finite on both axes) points that the
        caller wants starred as "the selected trial". When ``None``,
        defaults to ``argmax(x_metric)`` to retain the original
        behaviour.
    selected_label
        Legend label for the starred point. ``None`` derives one from
        ``x_metric``.
    """
    axis = _get_ax(ax, figsize)
    data = _trial_frame(study)
    if x_metric not in data or y_metric not in data:
        msg = f"x_metric/y_metric must be a key of _trial_frame: got {x_metric!r}, {y_metric!r}."
        raise ValueError(msg)
    x_all = data[x_metric]
    y_all = data[y_metric]
    finite = np.isfinite(x_all) & np.isfinite(y_all)
    if finite.sum() == 0:
        axis.text(
            0.5,
            0.5,
            f"no trials with finite {x_metric} and {y_metric}",
            ha="center",
            va="center",
        )
        return axis
    x = x_all[finite]
    y = y_all[finite]
    axis.scatter(x, y, s=22, color="tab:blue", alpha=0.6, label="trial")
    mask = _pareto_mask(x, y)
    order = np.argsort(x[mask])
    axis.plot(
        x[mask][order],
        y[mask][order],
        color="tab:red",
        linewidth=1.5,
        marker="o",
        markersize=5,
        label="Pareto frontier",
    )
    if selected_index is None:
        row = _best_trial_row(study, data)
        if row is not None and finite[row]:
            # Map absolute row -> index in the filtered (finite) array.
            sel = int(np.sum(finite[:row]))
            auto_label = "selected (Optuna best)"
        else:
            sel = int(np.argmax(x))
            auto_label = f"selected (arg max {x_metric})"
    else:
        sel = int(selected_index)
        auto_label = f"selected (arg max {x_metric})"
    if 0 <= sel < x.shape[0]:
        label = selected_label if selected_label is not None else auto_label
        axis.scatter(
            x[sel],
            y[sel],
            s=220,
            marker="*",
            color="tab:orange",
            edgecolor="black",
            linewidth=1.0,
            zorder=5,
            label=label,
        )
    axis.set_xlabel(_METRIC_LABELS.get(x_metric, x_metric))
    axis.set_ylabel(_METRIC_LABELS.get(y_metric, y_metric))
    axis.legend(loc="best", fontsize=8)
    return axis


def plot_optuna_hyperparam_landscape(
    study: optuna.Study,
    *,
    metric: str = "persistence",
    ax: Axes | None = None,
    figsize: tuple[float, float] = (5.5, 4.5),
) -> Axes:
    """Scatter trials in (log mcs, log ms) colored by ``metric``.

    ``metric`` is one of ``"persistence"``, ``"dbcv"``,
    ``"n_clusters"``, or ``"outlier_fraction"``. The best trial by the
    Optuna objective is marked with a star.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    axis = _get_ax(ax, figsize)
    data = _trial_frame(study)
    if data["mcs"].size == 0:
        axis.text(0.5, 0.5, "no completed trials", ha="center", va="center")
        return axis
    if metric not in data:
        msg = f"metric must be one of {sorted(data.keys())!r}, got {metric!r}."
        raise ValueError(msg)
    colour_values = data[metric]
    finite = np.isfinite(colour_values)
    sc = axis.scatter(
        data["mcs"][finite],
        data["ms"][finite],
        c=colour_values[finite],
        s=38,
        cmap="viridis",
        edgecolor="white",
        linewidth=0.4,
    )
    plt.colorbar(sc, ax=axis, label=metric)
    best_idx = _best_trial_row(study, data)
    if best_idx is None:
        best_idx = int(np.argmax(data["persistence"]))
    axis.scatter(
        data["mcs"][best_idx],
        data["ms"][best_idx],
        s=240,
        marker="*",
        color="tab:orange",
        edgecolor="black",
        linewidth=1.0,
        zorder=5,
        label="selected",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("min_cluster_size (log)")
    axis.set_ylabel("min_samples (log)")
    axis.legend(loc="best", fontsize=8)
    return axis


def plot_granularity_stability(
    study: optuna.Study,
    *,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (5.5, 4.5),
) -> Axes:
    """Granularity-stability scatter: n_clusters vs persistence, coloured by DBCV.

    Each trial is a point with horizontal position = number of
    clusters, vertical position = summed persistence. Colour = DBCV.
    The best trial is starred. Exposes the classic HDBSCAN
    trade-off between "many weakly-stable clusters" and "few
    strongly-stable clusters".
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    axis = _get_ax(ax, figsize)
    data = _trial_frame(study)
    if data["mcs"].size == 0:
        axis.text(0.5, 0.5, "no completed trials", ha="center", va="center")
        return axis
    finite = np.isfinite(data["dbcv"])
    sc = axis.scatter(
        data["n_clusters"][finite],
        data["persistence"][finite],
        c=data["dbcv"][finite],
        cmap="coolwarm",
        s=38,
        edgecolor="white",
        linewidth=0.4,
    )
    plt.colorbar(sc, ax=axis, label="DBCV proxy")
    if (~finite).any():
        axis.scatter(
            data["n_clusters"][~finite],
            data["persistence"][~finite],
            color="lightgrey",
            s=20,
            label="DBCV = NaN",
        )
    best_idx = _best_trial_row(study, data)
    if best_idx is None:
        best_idx = int(np.argmax(data["persistence"]))
    axis.scatter(
        data["n_clusters"][best_idx],
        data["persistence"][best_idx],
        s=240,
        marker="*",
        color="tab:orange",
        edgecolor="black",
        linewidth=1.0,
        zorder=5,
        label="selected",
    )
    axis.set_xlabel("n_clusters")
    axis.set_ylabel("sum of cluster persistence")
    axis.legend(loc="best", fontsize=8)
    return axis


def plot_optuna_parallel(  # noqa: PLR0915
    study: optuna.Study,
    *,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (8.0, 4.5),
) -> Axes:
    """Parallel-coordinates plot of trials.

    Axes (left to right): ``min_cluster_size``, ``min_samples``,
    ``alpha``, ``cluster_selection_epsilon`` (``eps``),
    ``cluster_selection_method`` (``method`` — eom vs leaf, shown as a
    discrete 0/1 axis with tick labels), ``n_clusters``,
    ``outlier_fraction``, ``persistence``, ``DBCV``. Axes that were
    pinned (a constant in the search) are dropped automatically so the
    plot never carries a redundant column. Each trial is a polyline;
    colour encodes TPE iteration (viridis). Shows search progress at a
    glance.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.collections import LineCollection  # noqa: PLC0415

    axis = _get_ax(ax, figsize)
    data = _trial_frame(study)
    n = int(data["mcs"].size)
    if n == 0:
        axis.text(0.5, 0.5, "no completed trials", ha="center", va="center")
        return axis
    candidates = [
        ("mcs", "min_cluster_size"),
        ("ms", "min_samples"),
        ("alpha", "alpha"),
        ("cluster_selection_epsilon", "eps"),
        ("cluster_selection_method", "method"),
        ("n_clusters", "n_clusters"),
        ("outlier_fraction", "outlier_fraction"),
        ("persistence", "persistence"),
        ("dbcv", "DBCV"),
    ]
    # Drop axes that were pinned (all-NaN) or constant — they carry no
    # information and would otherwise render as flat lines.
    keys: list[str] = []
    labels: list[str] = []
    for key, label in candidates:
        col = data[key]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            continue
        if finite.size > 1 and np.allclose(finite, finite[0]):
            continue
        keys.append(key)
        labels.append(label)
    cols = np.array([data[k] for k in keys], dtype=np.float64).T  # (n, k)
    col_min = np.nanmin(cols, axis=0)
    col_max = np.nanmax(cols, axis=0)
    spread = np.where(col_max > col_min, col_max - col_min, 1.0)
    norm = (cols - col_min) / spread
    xs = np.arange(len(keys))

    order = np.argsort(data["number"])
    cmap = plt.get_cmap("viridis")
    tpe_colour = cmap(np.linspace(0.0, 1.0, n))
    segments = []
    colours = []
    for rank, i in enumerate(order):
        row = norm[i]
        if np.any(~np.isfinite(row)):
            continue
        segments.append(np.column_stack([xs, row]))
        colours.append(tpe_colour[rank])
    lc = LineCollection(segments, colors=colours, linewidths=0.9, alpha=0.7)
    axis.add_collection(lc)
    axis.set_xticks(xs)
    axis.set_xticklabels(labels, rotation=30, ha="right")
    axis.set_yticks([])
    axis.set_ylim(-0.05, 1.05)
    axis.set_xlim(-0.3, len(keys) - 0.7)
    from matplotlib.colors import Normalize  # noqa: PLC0415

    for x in xs:
        axis.axvline(float(x), color="black", linewidth=0.5, alpha=0.3)
    # Label the two ends of the categorical method axis so readers
    # know which extreme is EOM vs leaf.
    if "cluster_selection_method" in keys:
        idx = keys.index("cluster_selection_method")
        axis.text(
            float(idx) + 0.05,
            -0.08,
            "eom",
            ha="left",
            va="top",
            fontsize=8,
            color="0.4",
        )
        axis.text(
            float(idx) + 0.05,
            1.08,
            "leaf",
            ha="left",
            va="bottom",
            fontsize=8,
            color="0.4",
        )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(1, max(n, 1)))
    sm.set_array([])
    plt.colorbar(sm, ax=axis, label="trial number")
    return axis


def plot_condensed_tree(
    model: _hdbscan_typing.HDBSCAN | None,
    *,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (6.5, 5.0),
    select_clusters: bool = True,
) -> Axes:
    """Draw HDBSCAN's condensed tree with selected clusters highlighted.

    Parameters
    ----------
    model
        A fitted :class:`hdbscan.HDBSCAN` instance (as returned on the
        CPU backend in :attr:`OptunaSearchResult.model`).
    ax
        Target axes.
    figsize
        Figure size when ``ax`` is ``None``.
    select_clusters
        Whether to outline the selected (flat) clusters.
    """
    axis = _get_ax(ax, figsize)
    if model is None:
        axis.text(
            0.5,
            0.5,
            "condensed tree unavailable\n(cuml backend?)",
            ha="center",
            va="center",
        )
        return axis
    try:
        model.condensed_tree_.plot(
            axis=axis,
            select_clusters=select_clusters,
            selection_palette=None,
        )
    except (AttributeError, ValueError) as exc:
        axis.text(0.5, 0.5, f"condensed tree unavailable\n({exc})", ha="center", va="center")
        return axis
    axis.set_xlabel("")
    return axis


def plot_membership_confidence(
    embedding: ArrayLike,
    labels: ArrayLike,
    probabilities: ArrayLike,
    *,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (6.0, 5.5),
    outlier_color: str = "lightgrey",
) -> Axes:
    """Scatter the 2-D embedding with colour = HDBSCAN membership probability.

    Outliers are drawn in ``outlier_color``; clustered points are
    coloured by their ``probabilities_`` in viridis. Exposes how
    confident each assignment is and which cluster boundaries are
    fuzzy.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    axis = _get_ax(ax, figsize)
    emb = np.asarray(embedding, dtype=np.float64)
    lab = np.asarray(labels).astype(np.intp, copy=False)
    prob = np.asarray(probabilities, dtype=np.float64)
    outliers = lab < 0
    if outliers.any():
        axis.scatter(
            emb[outliers, 0],
            emb[outliers, 1],
            s=6,
            color=outlier_color,
            alpha=0.5,
            label="outlier",
        )
    sc = axis.scatter(
        emb[~outliers, 0],
        emb[~outliers, 1],
        c=prob[~outliers],
        cmap="viridis",
        s=8,
        vmin=0.0,
        vmax=1.0,
    )
    plt.colorbar(sc, ax=axis, label="cluster-membership probability")
    axis.set_xlabel("component 1")
    axis.set_ylabel("component 2")
    axis.set_aspect("equal", adjustable="datalim")
    return axis


def plot_subsample_stability(
    stability: SubsampleStability,
    reference_persistence: ArrayLike,
    *,
    axes: Sequence[Axes] | None = None,
    figsize: tuple[float, float] = (13.0, 3.8),
) -> Sequence[Axes]:
    """Three-panel summary of :class:`SubsampleStability`.

    Panels: (left) histogram of n_clusters; (middle) histogram of
    ARI vs reference on the overlap; (right) box-plot of per-cluster
    persistence across subsamples with reference persistence overlaid
    in orange.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if axes is None:
        _, ax_arr = plt.subplots(1, 3, figsize=figsize)
        panel = list(ax_arr)
    else:
        panel = list(axes)
    if len(panel) != 3:
        msg = f"plot_subsample_stability needs 3 axes, got {len(panel)}."
        raise ValueError(msg)
    nc = np.asarray(stability.n_clusters, dtype=np.intp)
    ari = np.asarray(stability.ari, dtype=np.float64)
    per = np.asarray(stability.persistence_per_cluster, dtype=np.float64)
    ref = np.asarray(reference_persistence, dtype=np.float64)

    panel[0].hist(
        nc, bins=np.arange(nc.min(), nc.max() + 2) - 0.5, color="tab:blue", edgecolor="white"
    )
    panel[0].set_xlabel("n_clusters")
    panel[0].set_ylabel("subsample count")
    panel[0].set_title("cluster count under subsampling")

    panel[1].hist(ari[np.isfinite(ari)], bins=20, color="tab:green", edgecolor="white")
    panel[1].axvline(1.0, color="tab:red", linestyle="--", linewidth=1.0, label="ARI = 1")
    panel[1].set_xlabel("Adjusted Rand Index (vs reference)")
    panel[1].set_ylabel("subsample count")
    panel[1].set_title("label stability")
    panel[1].legend(loc="upper left", fontsize=8)

    if per.shape[1] > 0:
        data = [per[np.isfinite(per[:, c]), c] for c in range(per.shape[1])]
        positions = np.arange(1, per.shape[1] + 1)
        panel[2].boxplot(
            data,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            boxprops={"facecolor": "tab:blue", "alpha": 0.45},
            medianprops={"color": "black"},
        )
        panel[2].scatter(
            positions,
            ref[: per.shape[1]],
            color="tab:orange",
            zorder=4,
            s=40,
            label="selected fit",
        )
        panel[2].set_xticks(positions)
        panel[2].set_xticklabels([str(c) for c in range(per.shape[1])])
        panel[2].set_xlabel("reference cluster index")
        panel[2].set_ylabel("persistence")
        panel[2].set_title("per-cluster persistence distribution")
        panel[2].legend(loc="best", fontsize=8)
    return panel


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
