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

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from starfold.clustering import (
    Engine,
    OptunaSearchResult,
    TrialObjective,
    search_hdbscan,
)
from starfold.credibility import CredibilityReport, compute_credibility
from starfold.diagnostics import (
    auto_mcs_upper,
    diagnose_fit,
    validate_input_matrix,
    warn_fit_flags,
)
from starfold.embedding import _fit_umap_with_model
from starfold.hierarchy import HierarchicalStructure, extract_hierarchy
from starfold.noise_baseline import NoiseBaselineResult, compute_noise_baseline
from starfold.trustworthiness import continuity, trustworthiness
from starfold.uncertainty import (
    UncertaintyAwareFit,
    UncertaintyPropagation,
    build_replica_augmented_matrix,
    consensus_from_augmented_labels,
    propagate_uncertainty,
)

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike, NDArray

    from starfold.merge import MergeSuggestion
    from starfold.silhouette import SilhouetteResult
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
        :math:`T(k = n_\text{neighbors})` of the embedding. Measures
        whether the projection invented false neighbours.
    continuity
        :math:`C(k = n_\text{neighbors})` of the embedding, the dual of
        trustworthiness. Measures whether the projection tore true
        neighbours apart. Reporting both catches failure modes that one
        alone would miss: a trustworthy but discontinuous embedding has
        collapsed true neighbours, while a continuous but untrustworthy
        embedding has injected spurious ones.
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
    credibility
        Global :class:`CredibilityReport` comparing this run's
        ``n_clusters``, best Optuna objective, and maximum cluster
        persistence against the matching distributions on noise
        realisations. ``None`` iff the noise baseline was skipped.
    hierarchy
        :class:`HierarchicalStructure` view of HDBSCAN's condensed tree
        for this fit. Reports ``available=False`` on GPU backends that
        do not expose the tree; otherwise exposes merge-lambda,
        sibling, and sub-cluster helpers.
    scaler
        Fitted :class:`StandardScaler` so the exact transform can be
        applied to new data.
    umap_model
        The trained UMAP reducer from the fit. Retained so
        :meth:`propagate_uncertainty` can project perturbed samples
        through the same manifold without refitting. ``None`` on the
        cuml backend until input-uncertainty propagation ships there.
    flags
        Human-readable diagnostic warnings produced by
        :func:`starfold.diagnostics.diagnose_fit` (degenerate fit,
        high outlier fraction, low trustworthiness, noise-consistent
        clustering, unavailable hierarchy, Optuna plateau). Empty list
        iff no flags were raised.
    config
        Frozen record of pipeline arguments, useful for saving.
    """

    embedding: NDArray[np.floating[Any]]
    labels: NDArray[np.intp]
    probabilities: NDArray[np.floating[Any]]
    persistence: NDArray[np.floating[Any]]
    significant: NDArray[np.bool_] | None
    trustworthiness: float
    continuity: float
    n_clusters: int
    best_params: dict[str, Any]
    search: OptunaSearchResult
    noise_baseline: NoiseBaselineResult | None
    credibility: CredibilityReport | None
    hierarchy: HierarchicalStructure
    scaler: StandardScaler
    umap_model: Any = None
    flags: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a short printable metrics table.

        When :attr:`flags` is non-empty the diagnostic warnings are
        printed at the top of the report so a reader cannot miss them
        (degenerate fit, low trustworthiness, ...). See
        :func:`starfold.diagnostics.diagnose_fit` for the individual
        checks.
        """
        n = int(self.labels.shape[0])
        n_outliers = int(np.sum(self.labels < 0))
        objective = str(self.config.get("hdbscan_objective", "persistence_sum"))
        persistence_sum = float(self.persistence.sum()) if self.persistence.size else 0.0
        persistence_median = float(np.median(self.persistence)) if self.persistence.size else 0.0
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
        lines: list[str] = []
        if self.flags:
            lines.append("!! diagnostic flags")
            lines.append("-" * 32)
            lines.extend(f"- {flag}" for flag in self.flags)
            lines.append("")
        lines += [
            "starfold pipeline result",
            "-" * 32,
            f"n_samples        {n}",
            f"n_clusters       {self.n_clusters}",
            f"n_outliers       {n_outliers}  ({n_outliers / n:.1%})",
            f"trustworthiness  {self.trustworthiness:.4f}",
            f"continuity       {self.continuity:.4f}",
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
        if self.credibility is not None:
            verdict = "PASS" if self.credibility.passes else "FAIL"
            n_cred = int(self.credibility.per_cluster_significant.sum())
            n_total = int(self.credibility.per_cluster_significant.size)
            lines.append(
                f"credibility      {verdict} at alpha={self.credibility.alpha}"
                f"  p(n_clusters)={self.credibility.n_clusters_pvalue:.4f}"
                f"  p({self.credibility.objective_name})={self.credibility.objective_pvalue:.4f}"
                f"  p(max_persistence)={self.credibility.max_persistence_pvalue:.4f}"
            )
            lines.append(f"per-cluster      {n_cred}/{n_total} clusters credible")
        lines.append("persistence      " + np.array2string(self.persistence, precision=3))
        return "\n".join(lines)

    def save(self, directory: Path | str) -> Path:
        """Save the result to ``directory`` and return the path."""
        from starfold.io import save_pipeline_result  # noqa: PLC0415

        return save_pipeline_result(self, directory)

    def suggest_merges(
        self,
        *,
        cohesion_threshold: float = 0.9,
        gap_threshold: float = 2.0,
        sort_by: str = "cohesion_ratio",
    ) -> list[MergeSuggestion]:
        """Rank cluster pairs by whether they should be merged.

        Delegates to :func:`starfold.merge.suggest_merges` with this
        result's :attr:`hierarchy` and :attr:`embedding`. See that
        function for the signals (hierarchical cohesion ratio +
        embedding geometric gap) and the meaning of the thresholds.

        Parameters
        ----------
        cohesion_threshold, gap_threshold, sort_by
            Forwarded to :func:`starfold.merge.suggest_merges`.

        Returns
        -------
        list of MergeSuggestion
            One entry per unordered pair ``(i, j)`` of flat clusters,
            sorted by ``sort_by``.

        Raises
        ------
        RuntimeError
            When :attr:`hierarchy` is unavailable (cuml fits).
        """
        from starfold.merge import suggest_merges as _suggest_merges  # noqa: PLC0415

        return _suggest_merges(
            self.hierarchy,
            self.embedding,
            cohesion_threshold=cohesion_threshold,
            gap_threshold=gap_threshold,
            sort_by=sort_by,  # type: ignore[arg-type]
        )

    def silhouette(
        self,
        *,
        chunk_size: int = 512,
        metric: str = "euclidean",
    ) -> SilhouetteResult:
        """Chunked silhouette coefficients on this run's embedding.

        Delegates to :func:`starfold.silhouette.chunked_silhouette` on
        ``self.embedding`` and ``self.labels``. Kept as an opt-in
        method (rather than eagerly computed in :meth:`fit`) because
        the silhouette is :math:`O(N^2)` work: worth it on demand to
        decide whether two clusters should be merged, not worth it on
        every fit. See :mod:`starfold.silhouette` for memory and
        correctness notes.

        Parameters
        ----------
        chunk_size
            Number of rows per distance block. Controls peak memory.
        metric
            Distance metric. Defaults to the embedding-space
            ``euclidean``.

        Returns
        -------
        SilhouetteResult
            Overall, per-sample, and per-cluster silhouette scores.
        """
        from starfold.silhouette import chunked_silhouette  # noqa: PLC0415

        return chunked_silhouette(
            self.embedding,
            self.labels,
            metric=metric,
            chunk_size=chunk_size,
        )

    def propagate_uncertainty(
        self,
        X: ArrayLike,
        sigma: float | NDArray[np.floating[Any]],
        *,
        n_draws: int = 100,
        random_state: int | None = None,
        n_jobs: int = 1,
    ) -> UncertaintyPropagation:
        """Monte-Carlo propagate per-feature uncertainties through the fit.

        Each of ``n_draws`` perturbed copies of ``X`` is projected
        through the fitted :class:`StandardScaler` -> UMAP ->
        ``hdbscan.approximate_predict`` pipeline (no refit), and the
        per-sample cluster-label frequencies become a membership matrix.
        See :mod:`starfold.uncertainty` for the sampling geometry and
        limitations (currently CPU backend only).

        Parameters
        ----------
        X
            The original (pre-scaling) training matrix, shape
            ``(n_samples, n_features)``.
        sigma
            1-sigma uncertainties in the original feature units. Scalar,
            per-feature (length ``n_features``), or per-sample-per-feature.
        n_draws
            Monte Carlo draws. 100 is a reasonable default for the
            "is this sample near a boundary?" question; raise for tighter
            Bernoulli confidence intervals on the membership fractions.
        random_state
            Seed for reproducibility.
        n_jobs
            Parallel draws. ``-1`` uses every core.

        Returns
        -------
        UncertaintyPropagation
            Per-sample membership matrix (including an outlier column),
            consensus labels, and instability scores.
        """
        if self.umap_model is None:
            msg = (
                "propagate_uncertainty requires a CPU-backend fit; the "
                "UMAP model was not retained on this result. Refit with "
                "engine='cpu'."
            )
            raise NotImplementedError(msg)
        if self.search.model is None:
            msg = (
                "propagate_uncertainty requires a CPU HDBSCAN model with "
                "prediction_data; the current result has no such model."
            )
            raise NotImplementedError(msg)
        return propagate_uncertainty(
            np.asarray(X, dtype=np.float64),
            sigma,
            scaler=self.scaler,
            umap_model=self.umap_model,
            hdbscan_model=self.search.model,
            n_clusters=int(self.n_clusters),
            n_draws=n_draws,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def refit_subcluster(
        self,
        X: ArrayLike,
        cluster_id: int,
        *,
        overrides: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Rerun the full pipeline on the rows of ``X`` inside ``cluster_id``.

        This is the paper's "second run" pattern: fit the pipeline on
        the full sample, pick a cluster of interest, then refit on
        just that cluster to expose sub-structure. Unlike
        :meth:`HierarchicalStructure.subcluster_on` -- which only
        reruns HDBSCAN on the existing 2-D embedding -- this method
        refits UMAP, the Optuna search, the noise baseline, and the
        trustworthiness score on a freshly scaled subset, which is the
        scientifically meaningful operation.

        Parameters
        ----------
        X
            The *raw* (pre-scaling) training matrix that produced this
            result. Only rows with ``self.labels == cluster_id`` are
            used for the refit; the ``StandardScaler`` inside the new
            pipeline is re-fit on that subset.
        cluster_id
            The cluster to focus on. ``-1`` (the outlier class) is
            rejected to avoid accidentally fitting on "everything that
            did not fit anywhere".
        overrides
            Optional per-call overrides for
            :class:`UnsupervisedPipeline` kwargs, e.g.
            ``{"hdbscan_optuna_trials": 50,
            "skip_noise_baseline": True}``. Anything omitted is
            inherited from this result's :attr:`config`.

        Returns
        -------
        PipelineResult
            A fresh result on the subset.

        Raises
        ------
        ValueError
            If ``cluster_id`` is ``-1``, out of range, or matches no
            rows of ``self.labels``.
        """
        if cluster_id < 0:
            msg = (
                "cluster_id must be non-negative; refitting on the "
                "outlier class (-1) is not supported -- filter the "
                "outliers manually if you have a reason to cluster them."
            )
            raise ValueError(msg)
        if cluster_id >= self.n_clusters:
            msg = (
                f"cluster_id {cluster_id} is out of range for a result "
                f"with n_clusters={self.n_clusters}."
            )
            raise ValueError(msg)
        x = np.asarray(X, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] != self.labels.shape[0]:
            msg = (
                "X must be a 2-D array with the same n_samples the "
                "result was fitted on "
                f"(got shape {x.shape}, expected "
                f"({self.labels.shape[0]}, n_features))."
            )
            raise ValueError(msg)
        mask = self.labels == cluster_id
        if not np.any(mask):
            msg = f"cluster_id {cluster_id} contains no samples in self.labels."
            raise ValueError(msg)

        kwargs: dict[str, Any] = {
            "umap_kwargs": dict(self.config.get("umap_kwargs", {}) or {}),
            "hdbscan_optuna_trials": int(self.config.get("hdbscan_optuna_trials", 100)),
            "ms_range": tuple(self.config.get("ms_range", (1, 50))),
            "cluster_selection_methods": tuple(
                self.config.get("cluster_selection_methods", ("eom", "leaf"))
            ),
            "cluster_selection_epsilon_range": tuple(
                self.config.get("cluster_selection_epsilon_range", (0.0, 0.5))
            ),
            "alpha_range": tuple(self.config.get("alpha_range", (0.7, 1.5))),
            "metric": self.config.get("metric", "euclidean"),
            "engine": self.config.get("engine", "auto"),
            "noise_baseline_kwargs": dict(self.config.get("noise_baseline_kwargs", {}) or {}),
            "skip_noise_baseline": bool(self.config.get("skip_noise_baseline", False)),
            "random_state": self.config.get("random_state"),
            "hdbscan_objective": self.config.get("hdbscan_objective", "persistence_sum"),
        }
        # Inherit mcs_range only if the parent pipeline used an explicit
        # tuple; otherwise leave it at None so the subset gets its own
        # data-size-aware auto bound.
        if not self.config.get("mcs_range_was_auto", False):
            stored = self.config.get("mcs_range")
            if stored is not None:
                kwargs["mcs_range"] = tuple(stored)
        if overrides:
            kwargs.update(overrides)
        return UnsupervisedPipeline(**kwargs).fit(x[mask])

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
            study,
            metric="persistence_sum",
            ax=axes[0, 3],
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
        from starfold.trustworthiness import (  # noqa: PLC0415
            continuity_curve,
            trustworthiness_curve,
        )

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
                min_cluster_size=int(self.best_params["min_cluster_size"]),
                min_samples=int(self.best_params["min_samples"]),
                cluster_selection_method=str(
                    self.best_params.get("cluster_selection_method", "eom")
                ),
                cluster_selection_epsilon=float(
                    self.best_params.get("cluster_selection_epsilon", 0.0)
                ),
                alpha=float(self.best_params.get("alpha", 1.0)),
                n_subsamples=n_subsamples,
                subsample_fraction=subsample_fraction,
                engine=self.config.get("engine", "auto"),
                random_state=random_state,
            )

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])

        ax_conf = fig.add_subplot(gs[0, 0])
        plot_membership_confidence(
            self.embedding,
            self.labels,
            self.probabilities,
            ax=ax_conf,
        )
        ax_conf.set_title("(a) HDBSCAN membership-probability map")

        ax_imp = fig.add_subplot(gs[0, 1])
        plot_optuna_param_importance(self.search.study, ax=ax_imp)
        ax_imp.set_title("(b) parameter importance (fANOVA)")

        ax_trust = fig.add_subplot(gs[0, 2])
        scores = trustworthiness_curve(x_scaled, self.embedding, k_values=k_values)
        cont_scores = continuity_curve(x_scaled, self.embedding, k_values=k_values)
        plot_trustworthiness_curve(
            scores,
            continuity_scores=cont_scores,
            ax=ax_trust,
            threshold=0.9,
        )
        ax_trust.set_title("(c) trustworthiness T(k) and continuity C(k)")

        ax_sb1 = fig.add_subplot(gs[1, 0])
        ax_sb2 = fig.add_subplot(gs[1, 1])
        ax_sb3 = fig.add_subplot(gs[1, 2])
        plot_subsample_stability(
            stability,
            self.persistence,
            axes=[ax_sb1, ax_sb2, ax_sb3],
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
    mcs_range : tuple of int, optional
        Search range for ``min_cluster_size``. ``None`` (the default)
        lets the pipeline pick a data-size-aware upper bound via
        :func:`starfold.diagnostics.auto_mcs_upper`, so a 1M-sample
        embedding searches up to 5 000 while a 1k-sample run searches
        up to 50. An explicit tuple is always honoured.
    ms_range : tuple of int, default ``(1, 50)``
        Search range for HDBSCAN's ``min_samples`` forwarded to
        :func:`search_hdbscan`.
    cluster_selection_methods : tuple of str, default ``("eom", "leaf")``
        Optuna axis over HDBSCAN's cluster-selection method. ``eom``
        favours fewer, larger clusters (the paper default); ``leaf``
        exposes finer-grained structure. Including both lets Optuna
        pick the granularity that best matches the data. Pass a
        single-element tuple (e.g. ``("eom",)``) to pin the selection
        method.
    cluster_selection_epsilon_range : tuple of float, default ``(0.0, 0.5)``
        Range for the ``cluster_selection_epsilon`` Optuna axis. A
        positive epsilon merges HDBSCAN leaves whose birth lambdas are
        closer together than ``1/epsilon``, which is how the library
        recommends asking for a coarser clustering without raising
        ``min_cluster_size``. ``(0.0, 0.0)`` disables the axis.
    alpha_range : tuple of float, default ``(0.7, 1.5)``
        Range for HDBSCAN's ``alpha`` Optuna axis (log-uniform). Alpha
        reshapes the mutual-reachability metric: values below 1 spread
        density, values above 1 compress it. Exposing this axis lets
        Optuna find the density-weighting that maximises persistence
        on the data at hand.
    metric : str, default ``"euclidean"``
        HDBSCAN metric, threaded through to every trial.
    engine : {"auto", "cpu", "cuml"}, default ``"auto"``
        Backend selector. ``"auto"`` prefers the RAPIDS :mod:`cuml` GPU
        implementations when importable and falls back to CPU otherwise.
        The selector is honoured by UMAP on the main fit, by
        :func:`starfold.noise_baseline.compute_noise_baseline` (unless
        overridden in ``noise_baseline_kwargs``), and by the subsample
        stability helper. Note that each Optuna trial's HDBSCAN fit
        runs on CPU regardless of ``engine``: Optuna's TPE search and
        the per-trial persistence bookkeeping expect the :mod:`hdbscan`
        reference implementation, and the final HDBSCAN refit at the
        chosen hyperparameters is the step that can dispatch to cuml.
        Input-uncertainty propagation is likewise CPU-only for now.
    noise_baseline_kwargs : dict, optional
        Forwarded to :func:`compute_noise_baseline`. Pass
        ``{"n_realisations": 0}`` or set ``skip_noise_baseline=True``
        to bypass the (expensive) baseline. A nested ``"umap_kwargs"``
        entry, if present, overrides the main pipeline's UMAP config
        for noise fits only -- useful to run the baseline at
        ``n_epochs=500`` while the main fit uses the paper's 10 000
        (structureless Gaussian noise converges in UMAP well below
        the number of epochs real data needs).
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
        mcs_range: tuple[int, int] | None = None,
        ms_range: tuple[int, int] = (1, 50),
        cluster_selection_methods: tuple[str, ...] = ("eom", "leaf"),
        cluster_selection_epsilon_range: tuple[float, float] = (0.0, 0.5),
        alpha_range: tuple[float, float] = (0.7, 1.5),
        metric: str = "euclidean",
        engine: Engine = "auto",
        noise_baseline_kwargs: dict[str, Any] | None = None,
        skip_noise_baseline: bool = False,
        random_state: int | None = None,
        hdbscan_objective: TrialObjective = "persistence_sum",
    ) -> None:
        self.umap_kwargs: dict[str, Any] = dict(umap_kwargs or {})
        self.hdbscan_optuna_trials = int(hdbscan_optuna_trials)
        self.mcs_range: tuple[int, int] | None = (
            None if mcs_range is None else (int(mcs_range[0]), int(mcs_range[1]))
        )
        self.ms_range: tuple[int, int] = (int(ms_range[0]), int(ms_range[1]))
        self.cluster_selection_methods: tuple[str, ...] = tuple(
            str(m) for m in cluster_selection_methods
        )
        self.cluster_selection_epsilon_range: tuple[float, float] = (
            float(cluster_selection_epsilon_range[0]),
            float(cluster_selection_epsilon_range[1]),
        )
        self.alpha_range: tuple[float, float] = (
            float(alpha_range[0]),
            float(alpha_range[1]),
        )
        self.metric = metric
        self.engine: Engine = engine
        self.noise_baseline_kwargs: dict[str, Any] = dict(noise_baseline_kwargs or {})
        self.skip_noise_baseline = bool(skip_noise_baseline)
        self.random_state = random_state
        self.hdbscan_objective: TrialObjective = hdbscan_objective

    def _resolve_mcs_range(self, n_samples: int) -> tuple[int, int]:
        """Return the ``min_cluster_size`` search range to use for ``n_samples``.

        An explicit range is honoured; ``None`` triggers the
        data-size-aware upper bound from
        :func:`starfold.diagnostics.auto_mcs_upper`.
        """
        if self.mcs_range is not None:
            return self.mcs_range
        return (5, auto_mcs_upper(n_samples))

    def fit(self, X: ArrayLike) -> PipelineResult:
        """Run the full pipeline on ``X`` and return a :class:`PipelineResult`.

        The input matrix is hard-checked for NaN/inf and minimum sample
        count (see :func:`starfold.diagnostics.validate_input_matrix`)
        and the returned result's :attr:`PipelineResult.flags` is
        populated with any diagnostic issues found
        (:func:`starfold.diagnostics.diagnose_fit`). Flags are also
        emitted through :mod:`warnings` so logs capture them.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix. Standardised internally with
            :class:`StandardScaler`. Must be finite and have more
            samples than UMAP's ``n_neighbors``.

        Returns
        -------
        PipelineResult
            Every artefact of the run, including diagnostic
            :attr:`PipelineResult.flags`.
        """
        n_neighbors = int(self.umap_kwargs.get("n_neighbors", 15))
        x = validate_input_matrix(X, n_neighbors=n_neighbors)
        n_samples, n_features = x.shape

        scaler = StandardScaler().fit(x)
        x_scaled = scaler.transform(x)

        umap_kwargs = dict(self.umap_kwargs)
        umap_kwargs.setdefault("engine", self.engine)
        embedding, umap_reducer = _fit_umap_with_model(
            x_scaled, random_state=self.random_state, **umap_kwargs
        )

        mcs_range = self._resolve_mcs_range(n_samples)
        search = search_hdbscan(
            embedding,
            n_trials=self.hdbscan_optuna_trials,
            mcs_range=mcs_range,
            ms_range=self.ms_range,
            cluster_selection_methods=self.cluster_selection_methods,
            cluster_selection_epsilon_range=self.cluster_selection_epsilon_range,
            alpha_range=self.alpha_range,
            metric=self.metric,
            random_state=self.random_state,
            engine=self.engine,
            objective=self.hdbscan_objective,
        )
        hdbscan_result = search.hdbscan_result

        baseline: NoiseBaselineResult | None = None
        significant: NDArray[np.bool_] | None = None
        credibility: CredibilityReport | None = None
        if not self.skip_noise_baseline:
            baseline_kwargs = dict(self.noise_baseline_kwargs)
            baseline_kwargs.setdefault("random_state", self.random_state)
            baseline_kwargs.setdefault("engine", self.engine)
            baseline_kwargs.setdefault("objective", self.hdbscan_objective)
            baseline_kwargs.setdefault("mcs_range", mcs_range)
            baseline_kwargs.setdefault("ms_range", self.ms_range)
            baseline_kwargs.setdefault("cluster_selection_methods", self.cluster_selection_methods)
            baseline_kwargs.setdefault(
                "cluster_selection_epsilon_range",
                self.cluster_selection_epsilon_range,
            )
            baseline_kwargs.setdefault("alpha_range", self.alpha_range)
            baseline_umap_kwargs = baseline_kwargs.pop("umap_kwargs", self.umap_kwargs)
            baseline = compute_noise_baseline(
                n_samples=n_samples,
                n_features=n_features,
                umap_kwargs=baseline_umap_kwargs,
                **baseline_kwargs,
            )
            significant = hdbscan_result.cluster_persistence > baseline.threshold
            persistence = hdbscan_result.cluster_persistence
            max_persistence = float(persistence.max()) if persistence.size else 0.0
            try:
                best_objective = float(search.study.best_value)
            except ValueError:
                best_objective = 0.0
            credibility = compute_credibility(
                n_clusters=int(hdbscan_result.n_clusters),
                best_objective=best_objective,
                max_persistence=max_persistence,
                baseline=baseline,
                cluster_persistence=hdbscan_result.cluster_persistence,
            )

        k = int(self.umap_kwargs.get("n_neighbors", 15))
        k_eff = min(k, max(1, (n_samples - 1) // 2))
        trust = trustworthiness(x_scaled, embedding, k=k_eff, metric=self.metric)
        cont = continuity(x_scaled, embedding, k=k_eff, metric=self.metric)

        hierarchy = extract_hierarchy(search.model, hdbscan_result.labels)

        draft = PipelineResult(
            embedding=embedding,
            labels=hdbscan_result.labels,
            probabilities=hdbscan_result.probabilities,
            persistence=hdbscan_result.cluster_persistence,
            significant=significant,
            trustworthiness=float(trust),
            continuity=float(cont),
            n_clusters=hdbscan_result.n_clusters,
            best_params=search.best_params,
            search=search,
            noise_baseline=baseline,
            credibility=credibility,
            hierarchy=hierarchy,
            scaler=scaler,
            umap_model=umap_reducer,
            config=self._frozen_config(mcs_range),
        )
        flags = diagnose_fit(draft)
        warn_fit_flags(flags)
        return replace(draft, flags=flags)

    def fit_with_uncertainty(
        self,
        X: ArrayLike,
        sigma: float | NDArray[np.floating[Any]],
        *,
        n_replicas: int = 10,
    ) -> UncertaintyAwareFit:
        """Fit the pipeline on an uncertainty-augmented Monte Carlo matrix.

        This is the uncertainty-aware counterpart to :meth:`fit` +
        :meth:`PipelineResult.propagate_uncertainty`. Instead of
        asking "given a clean fit, how noisy are assignments?", this
        method asks "what clustering is stable when each sample is
        allowed to be anywhere in its error cloud?" by feeding the full
        pipeline (scaler → UMAP → Optuna+HDBSCAN → noise baseline →
        trustworthiness) an augmented matrix of shape
        ``(n_samples * (1 + n_replicas), n_features)`` -- the clean
        original samples stacked on top of ``n_replicas`` Gaussian
        perturbations at the user's ``sigma``. UMAP therefore sees the
        uncertainty cloud around every point, and HDBSCAN's density
        estimate is natively robust to the error bars.

        The returned :class:`~starfold.uncertainty.UncertaintyAwareFit`
        bundles the full augmented :class:`PipelineResult` with a
        per-original-sample :class:`UncertaintyPropagation` so the same
        plots and helpers (``plot_uncertainty_map``,
        :meth:`UncertaintyPropagation.confident_labels`) apply.

        Memory cost scales with ``1 + n_replicas``: a 100k-by-8
        features matrix at ``n_replicas=10`` is UMAP on ~1.1M rows,
        which is significant. Keep ``n_replicas`` small (the default
        10 is a practical sweet spot) and lean on ``engine='cuml'`` or
        ``umap_kwargs={"low_memory": True}`` if peak RSS is a concern.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Clean feature matrix in the original units.
        sigma
            Per-feature 1-sigma uncertainties. Accepts the same
            broadcasting as :meth:`PipelineResult.propagate_uncertainty`
            (scalar, ``(n_features,)``, or ``(n_samples, n_features)``).
        n_replicas : int, default 10
            Number of Gaussian-perturbed replicas to stack on the
            augmented matrix. ``0`` degenerates to the clean
            unaugmented fit (useful as a sanity check).

        Returns
        -------
        UncertaintyAwareFit
            Augmented :class:`PipelineResult`, per-sample
            :class:`UncertaintyPropagation`, group ids, and sigma
            summary.

        Raises
        ------
        ValueError
            If ``n_replicas < 0`` or ``sigma`` fails broadcasting.
        """
        n_neighbors = int(self.umap_kwargs.get("n_neighbors", 15))
        x = validate_input_matrix(X, n_neighbors=n_neighbors)
        if n_replicas < 0:
            msg = f"n_replicas must be >= 0 (got {n_replicas})."
            raise ValueError(msg)
        x_aug, group_ids = build_replica_augmented_matrix(
            x,
            sigma,
            n_replicas=n_replicas,
            random_state=self.random_state,
        )
        # The augmented matrix has (1 + n_replicas) * n_samples rows --
        # validate_input_matrix is called indirectly by ``fit`` but
        # this call also checks here so the error is raised before
        # noise generation on pathological inputs like tiny n_neighbors.
        sigma_arr = np.asarray(sigma, dtype=np.float64)
        sigma_mean = float(np.mean(sigma_arr)) if sigma_arr.size else 0.0
        sigma_max = float(np.max(sigma_arr)) if sigma_arr.size else 0.0
        augmented_result = self.fit(x_aug)
        propagation = consensus_from_augmented_labels(
            augmented_result.labels,
            group_ids,
            n_clusters=augmented_result.n_clusters,
        )
        return UncertaintyAwareFit(
            augmented_result=augmented_result,
            propagation=propagation,
            group_ids=group_ids,
            n_replicas=int(n_replicas),
            sigma_summary=(sigma_mean, sigma_max),
        )

    def _frozen_config(self, mcs_range: tuple[int, int]) -> dict[str, Any]:
        return {
            "umap_kwargs": dict(self.umap_kwargs),
            "hdbscan_optuna_trials": self.hdbscan_optuna_trials,
            "mcs_range": list(mcs_range),
            "mcs_range_was_auto": self.mcs_range is None,
            "ms_range": list(self.ms_range),
            "cluster_selection_methods": list(self.cluster_selection_methods),
            "cluster_selection_epsilon_range": list(self.cluster_selection_epsilon_range),
            "alpha_range": list(self.alpha_range),
            "metric": self.metric,
            "engine": self.engine,
            "noise_baseline_kwargs": dict(self.noise_baseline_kwargs),
            "skip_noise_baseline": self.skip_noise_baseline,
            "random_state": self.random_state,
            "hdbscan_objective": self.hdbscan_objective,
        }
