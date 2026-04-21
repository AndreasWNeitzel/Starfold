"""End-to-end pipeline: scale -> UMAP -> Optuna+HDBSCAN -> baseline + trust.

:class:`UnsupervisedPipeline` orchestrates a single run of the
methodology described in Neitzel et al. (2025) §3. It standardises the
input features with :class:`sklearn.preprocessing.StandardScaler`,
computes a 2-D UMAP embedding, runs a TPE-sampled Optuna search for
HDBSCAN hyperparameters, compares the real clusters' persistence to a
structureless-noise baseline, and reports a trustworthiness score for
the embedding.

The pipeline returns a :class:`PipelineResult` dataclass holding every
artefact needed to reproduce, inspect, or plot the run.

The paper's two-run workflow (first run on the full sample, second run
on each major component separately) is *not* baked in: it is a
scientific choice for the astronomy application, not part of the
methodology. Users who want the second run filter ``result.labels``
down to a subcluster and call :meth:`UnsupervisedPipeline.fit` again on
that subset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from starfold.clustering import (
    Engine,
    OptunaSearchResult,
    TrialObjective,
    search_hdbscan,
)
from starfold.embedding import run_umap
from starfold.noise_baseline import NoiseBaselineResult, compute_noise_baseline
from starfold.trustworthiness import trustworthiness

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike, NDArray

    from starfold.stability import SubsampleStability

__all__ = ["PipelineResult", "UnsupervisedPipeline"]


@dataclass(frozen=True)
class PipelineResult:
    r"""Artefact of a single :meth:`UnsupervisedPipeline.fit` call.

    Attributes
    ----------
    embedding
        2-D UMAP embedding of the scaled input, shape
        ``(n_samples, 2)``.
    labels
        HDBSCAN cluster labels. ``-1`` flags outliers.
    probabilities
        Per-sample cluster-membership strength in ``[0, 1]``.
    persistence
        Per-cluster persistence score.
    significant
        Boolean mask, ``persistence > noise_baseline.threshold`` for
        each cluster. ``None`` if the noise baseline was skipped.
    trustworthiness
        :math:`T(k = n_\text{neighbors})` of the embedding.
    n_clusters
        Number of distinct non-negative cluster labels.
    best_params
        ``{"min_cluster_size": ..., "min_samples": ...}`` selected by
        the search.
    search
        The :class:`OptunaSearchResult` from the Optuna run, retained
        for trial-history and importance inspection.
    noise_baseline
        The :class:`NoiseBaselineResult` used to set ``significant``,
        or ``None`` if skipped.
    scaler
        Fitted :class:`StandardScaler` so the exact transform can be
        applied to new data.
    config
        Frozen record of pipeline arguments, useful for saving.
    """

    embedding: NDArray[np.floating[Any]]
    labels: NDArray[np.intp]
    probabilities: NDArray[np.floating[Any]]
    persistence: NDArray[np.floating[Any]]
    significant: NDArray[np.bool_] | None
    trustworthiness: float
    n_clusters: int
    best_params: dict[str, int]
    search: OptunaSearchResult
    noise_baseline: NoiseBaselineResult | None
    scaler: StandardScaler
    config: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a short printable metrics table."""
        n = int(self.labels.shape[0])
        n_outliers = int(np.sum(self.labels < 0))
        objective = str(self.config.get("hdbscan_objective", "persistence_sum"))
        persistence_sum = float(self.persistence.sum()) if self.persistence.size else 0.0
        persistence_median = (
            float(np.median(self.persistence)) if self.persistence.size else 0.0
        )
        best_trial = None
        try:
            best_trial = self.search.study.best_trial
        except (ValueError, AttributeError):
            best_trial = None
        dbcv = (
            float(best_trial.user_attrs.get("relative_validity", float("nan")))
            if best_trial is not None
            else float("nan")
        )
        lines = [
            "starfold pipeline result",
            "-" * 32,
            f"n_samples        {n}",
            f"n_clusters       {self.n_clusters}",
            f"n_outliers       {n_outliers}  ({n_outliers / n:.1%})",
            f"trustworthiness  {self.trustworthiness:.4f}",
            f"objective        {objective}",
            f"best_params      {self.best_params}",
            f"persistence_sum  {persistence_sum:.4f}",
            f"persistence_med  {persistence_median:.4f}",
            f"DBCV (MST proxy) {dbcv:.4f}",
        ]
        if self.noise_baseline is not None:
            lines.append(f"noise_threshold  {self.noise_baseline.threshold:.4f}")
            if self.significant is not None:
                lines.append(f"significant      {int(self.significant.sum())}/{self.n_clusters}")
        lines.append("persistence      " + np.array2string(self.persistence, precision=3))
        return "\n".join(lines)

    def save(self, directory: Path | str) -> Path:
        """Save the result to ``directory`` and return the path."""
        from starfold.io import save_pipeline_result  # noqa: PLC0415

        return save_pipeline_result(self, directory)

    def plot_tuning_dashboard(
        self,
        *,
        figsize: tuple[float, float] = (22.0, 10.5),
    ) -> Figure:
        """Eight-panel HDBSCAN tuning dashboard for this run.

        Panels: (a) Optuna TPE history of the objective, (b) Pareto
        frontier in sum-persistence vs DBCV, (c) Pareto in median
        persistence vs DBCV, (d) hyperparameter landscape, (e)
        granularity-stability, (f) parallel coordinates, (g) the
        HDBSCAN condensed tree of the selected fit, (h) fANOVA
        parameter importance. The star on every panel marks the
        selected trial (the Optuna best, which for
        ``hdbscan_objective="combined_geom"`` is the geometric mean of
        DBCV and median persistence rather than simply arg-max of
        persistence).

        Parameters
        ----------
        figsize
            Size of the combined figure.

        Returns
        -------
        matplotlib.figure.Figure
            The assembled figure, ready to ``savefig`` or ``show``.
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415

        from starfold.plotting import (  # noqa: PLC0415
            plot_condensed_tree,
            plot_granularity_stability,
            plot_optuna_history,
            plot_optuna_hyperparam_landscape,
            plot_optuna_parallel,
            plot_optuna_param_importance,
            plot_optuna_pareto,
        )

        study = self.search.study
        fig, axes = plt.subplots(2, 4, figsize=figsize, constrained_layout=True)
        plot_optuna_history(study, ax=axes[0, 0])
        axes[0, 0].set_title("(a) Optuna TPE history (objective)")
        plot_optuna_pareto(
            study,
            x_metric="persistence_sum",
            y_metric="relative_validity",
            ax=axes[0, 1],
        )
        axes[0, 1].set_title("(b) Pareto: sum persistence vs DBCV")
        plot_optuna_pareto(
            study,
            x_metric="persistence_median",
            y_metric="relative_validity",
            ax=axes[0, 2],
        )
        axes[0, 2].set_title("(c) Pareto: median persistence vs DBCV")
        plot_optuna_hyperparam_landscape(
            study, metric="persistence_sum", ax=axes[0, 3],
        )
        axes[0, 3].set_title("(d) landscape (colour = sum persistence)")
        plot_granularity_stability(study, ax=axes[1, 0])
        axes[1, 0].set_title("(e) granularity-stability trade-off")
        plot_optuna_parallel(study, ax=axes[1, 1])
        axes[1, 1].set_title("(f) parallel coordinates (colour = trial #)")
        plot_condensed_tree(self.search.model, ax=axes[1, 2])
        axes[1, 2].set_title("(g) HDBSCAN condensed tree")
        plot_optuna_param_importance(study, ax=axes[1, 3])
        axes[1, 3].set_title("(h) parameter importance (fANOVA)")
        objective = str(self.config.get("hdbscan_objective", "persistence_sum"))
        fig.suptitle(
            "starfold HDBSCAN tuning dashboard -- "
            f"objective = {objective}, best = {self.best_params}, "
            f"sum persistence = {float(self.persistence.sum()):.3f}",
            fontsize=13,
        )
        return fig

    def plot_quality_dashboard(
        self,
        X: ArrayLike,
        *,
        stability: SubsampleStability | None = None,
        n_subsamples: int = 30,
        subsample_fraction: float = 0.8,
        k_values: tuple[int, ...] = (5, 10, 15, 30, 50, 100),
        figsize: tuple[float, float] = (17.0, 9.5),
        random_state: int | None = 0,
    ) -> Figure:
        """Six-panel pipeline-quality dashboard.

        Panels: (a) HDBSCAN membership-probability map, (b) fANOVA
        parameter importance, (c) trustworthiness T(k), (d)
        n_clusters across subsamples, (e) ARI vs reference labels, (f)
        per-cluster persistence distribution across subsamples.

        Parameters
        ----------
        X
            The *raw*, un-standardised input matrix. This is re-scaled
            internally with the pipeline's fitted scaler so the
            trustworthiness curve uses the same high-dimensional space
            the pipeline saw.
        stability
            Optional pre-computed :class:`SubsampleStability`. When
            ``None`` one is computed on the fly via
            :func:`compute_subsample_stability`.
        n_subsamples, subsample_fraction
            Forwarded to :func:`compute_subsample_stability` when
            ``stability`` is ``None``. 30 subsamples x 80% is a reasonable
            tutorial default.
        k_values
            k grid for :func:`trustworthiness_curve`.
        figsize
            Figure size.
        random_state
            Seed threaded into :func:`compute_subsample_stability` for
            reproducibility.

        Returns
        -------
        matplotlib.figure.Figure
            The assembled figure.
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415

        from starfold.plotting import (  # noqa: PLC0415
            plot_membership_confidence,
            plot_optuna_param_importance,
            plot_subsample_stability,
            plot_trustworthiness_curve,
        )
        from starfold.stability import compute_subsample_stability  # noqa: PLC0415
        from starfold.trustworthiness import trustworthiness_curve  # noqa: PLC0415

        x = np.asarray(X, dtype=np.float64)
        if x.ndim != 2:
            msg = f"X must be a 2-D array (got shape {x.shape})."
            raise ValueError(msg)
        x_scaled = self.scaler.transform(x)

        if stability is None:
            stability = compute_subsample_stability(
                self.embedding,
                self.labels,
                self.persistence,
                min_cluster_size=self.best_params["min_cluster_size"],
                min_samples=self.best_params["min_samples"],
                n_subsamples=n_subsamples,
                subsample_fraction=subsample_fraction,
                engine=self.config.get("engine", "auto"),
                random_state=random_state,
            )

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])

        ax_conf = fig.add_subplot(gs[0, 0])
        plot_membership_confidence(
            self.embedding, self.labels, self.probabilities, ax=ax_conf,
        )
        ax_conf.set_title("(a) HDBSCAN membership-probability map")

        ax_imp = fig.add_subplot(gs[0, 1])
        plot_optuna_param_importance(self.search.study, ax=ax_imp)
        ax_imp.set_title("(b) parameter importance (fANOVA)")

        ax_trust = fig.add_subplot(gs[0, 2])
        scores = trustworthiness_curve(x_scaled, self.embedding, k_values=k_values)
        plot_trustworthiness_curve(scores, ax=ax_trust, threshold=0.9)
        ax_trust.set_title("(c) trustworthiness T(k)")

        ax_sb1 = fig.add_subplot(gs[1, 0])
        ax_sb2 = fig.add_subplot(gs[1, 1])
        ax_sb3 = fig.add_subplot(gs[1, 2])
        plot_subsample_stability(
            stability, self.persistence, axes=[ax_sb1, ax_sb2, ax_sb3],
        )
        ax_sb1.set_title(
            f"(d) n_clusters across {stability.n_subsamples} subsamples "
            f"({int(stability.subsample_fraction * 100)}%)"
        )
        ax_sb2.set_title("(e) ARI vs reference labels")
        ax_sb3.set_title("(f) per-cluster persistence distribution")

        fig.suptitle("starfold pipeline-quality dashboard", fontsize=13)
        return fig


class UnsupervisedPipeline:
    """UMAP -> Optuna-tuned HDBSCAN -> noise baseline -> trustworthiness.

    Parameters
    ----------
    umap_kwargs : dict, optional
        Passed to :func:`starfold.embedding.run_umap`. Defaults to the
        paper's settings (``n_neighbors=15``, ``min_dist=0.0``,
        ``n_epochs=10_000``).
    hdbscan_optuna_trials : int, default 100
        Number of Optuna trials.
    mcs_range, ms_range : tuple of int
        Search ranges forwarded to :func:`search_hdbscan`.
    metric : str, default ``"euclidean"``
        HDBSCAN metric, threaded through to every trial.
    engine : {"auto", "cpu", "cuml"}, default ``"auto"``
        Backend for both UMAP and HDBSCAN. ``"auto"`` prefers the RAPIDS
        :mod:`cuml` GPU implementations when importable and falls back
        to CPU otherwise. The same selector is threaded into
        :func:`starfold.noise_baseline.compute_noise_baseline` unless
        explicitly overridden in ``noise_baseline_kwargs``.
    noise_baseline_kwargs : dict, optional
        Forwarded to :func:`compute_noise_baseline`. Pass
        ``{"n_realisations": 0}`` or set ``skip_noise_baseline=True``
        to bypass the (expensive) baseline.
    skip_noise_baseline : bool, default False
        When ``True``, the pipeline does not compute the baseline;
        ``result.significant`` will be ``None``.
    random_state : int or None, default None
        Seed threaded through to every stochastic step (StandardScaler
        is deterministic; UMAP, Optuna, and the noise baseline all
        consume this).

    Examples
    --------
    >>> import numpy as np
    >>> from starfold.pipeline import UnsupervisedPipeline
    >>> rng = np.random.default_rng(0)
    >>> X = np.vstack([rng.normal(loc, size=(60, 2)) for loc in (-5, 5)])
    >>> pipe = UnsupervisedPipeline(
    ...     umap_kwargs={"n_epochs": 100},
    ...     hdbscan_optuna_trials=8,
    ...     skip_noise_baseline=True,
    ...     random_state=0,
    ... )
    >>> result = pipe.fit(X)
    >>> result.embedding.shape
    (120, 2)
    """

    def __init__(
        self,
        *,
        umap_kwargs: dict[str, Any] | None = None,
        hdbscan_optuna_trials: int = 100,
        mcs_range: tuple[int, int] = (5, 500),
        ms_range: tuple[int, int] = (1, 50),
        metric: str = "euclidean",
        engine: Engine = "auto",
        noise_baseline_kwargs: dict[str, Any] | None = None,
        skip_noise_baseline: bool = False,
        random_state: int | None = None,
        hdbscan_objective: TrialObjective = "persistence_sum",
    ) -> None:
        self.umap_kwargs: dict[str, Any] = dict(umap_kwargs or {})
        self.hdbscan_optuna_trials = int(hdbscan_optuna_trials)
        self.mcs_range: tuple[int, int] = (int(mcs_range[0]), int(mcs_range[1]))
        self.ms_range: tuple[int, int] = (int(ms_range[0]), int(ms_range[1]))
        self.metric = metric
        self.engine: Engine = engine
        self.noise_baseline_kwargs: dict[str, Any] = dict(noise_baseline_kwargs or {})
        self.skip_noise_baseline = bool(skip_noise_baseline)
        self.random_state = random_state
        self.hdbscan_objective: TrialObjective = hdbscan_objective

    def _as_2d_float(self, X: ArrayLike) -> NDArray[np.floating[Any]]:
        x = np.asarray(X, dtype=np.float64)
        if x.ndim != 2:
            msg = f"X must be a 2-D array (got shape {x.shape})."
            raise ValueError(msg)
        return x

    def fit(self, X: ArrayLike) -> PipelineResult:
        """Run the full pipeline on ``X`` and return a :class:`PipelineResult`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix. Standardised internally with
            :class:`StandardScaler`.

        Returns
        -------
        PipelineResult
            Every artefact of the run.
        """
        x = self._as_2d_float(X)
        n_samples, n_features = x.shape

        scaler = StandardScaler().fit(x)
        x_scaled = scaler.transform(x)

        umap_kwargs = dict(self.umap_kwargs)
        umap_kwargs.setdefault("engine", self.engine)
        embedding = run_umap(x_scaled, random_state=self.random_state, **umap_kwargs)

        search = search_hdbscan(
            embedding,
            n_trials=self.hdbscan_optuna_trials,
            mcs_range=self.mcs_range,
            ms_range=self.ms_range,
            metric=self.metric,
            random_state=self.random_state,
            engine=self.engine,
            objective=self.hdbscan_objective,
        )
        hdbscan_result = search.hdbscan_result

        baseline: NoiseBaselineResult | None = None
        significant: NDArray[np.bool_] | None = None
        if not self.skip_noise_baseline:
            baseline_kwargs = dict(self.noise_baseline_kwargs)
            baseline_kwargs.setdefault("random_state", self.random_state)
            baseline_kwargs.setdefault("engine", self.engine)
            baseline = compute_noise_baseline(
                n_samples=n_samples,
                n_features=n_features,
                umap_kwargs=self.umap_kwargs,
                **baseline_kwargs,
            )
            significant = hdbscan_result.cluster_persistence > baseline.threshold

        k = int(self.umap_kwargs.get("n_neighbors", 15))
        k_eff = min(k, max(1, (n_samples - 1) // 2))
        trust = trustworthiness(x_scaled, embedding, k=k_eff, metric=self.metric)

        return PipelineResult(
            embedding=embedding,
            labels=hdbscan_result.labels,
            probabilities=hdbscan_result.probabilities,
            persistence=hdbscan_result.cluster_persistence,
            significant=significant,
            trustworthiness=float(trust),
            n_clusters=hdbscan_result.n_clusters,
            best_params=search.best_params,
            search=search,
            noise_baseline=baseline,
            scaler=scaler,
            config=self._frozen_config(),
        )

    def _frozen_config(self) -> dict[str, Any]:
        return {
            "umap_kwargs": dict(self.umap_kwargs),
            "hdbscan_optuna_trials": self.hdbscan_optuna_trials,
            "mcs_range": list(self.mcs_range),
            "ms_range": list(self.ms_range),
            "metric": self.metric,
            "engine": self.engine,
            "noise_baseline_kwargs": dict(self.noise_baseline_kwargs),
            "skip_noise_baseline": self.skip_noise_baseline,
            "random_state": self.random_state,
            "hdbscan_objective": self.hdbscan_objective,
        }
