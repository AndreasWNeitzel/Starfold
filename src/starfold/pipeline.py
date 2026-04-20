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

from starfold.clustering import Engine, OptunaSearchResult, search_hdbscan
from starfold.embedding import run_umap
from starfold.noise_baseline import NoiseBaselineResult, compute_noise_baseline
from starfold.trustworthiness import trustworthiness

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import ArrayLike, NDArray

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
        lines = [
            "starfold pipeline result",
            "-" * 32,
            f"n_samples        {n}",
            f"n_clusters       {self.n_clusters}",
            f"n_outliers       {n_outliers}  ({n_outliers / n:.1%})",
            f"trustworthiness  {self.trustworthiness:.4f}",
            f"best_params      {self.best_params}",
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
        HDBSCAN backend.
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

        embedding = run_umap(x_scaled, random_state=self.random_state, **self.umap_kwargs)

        search = search_hdbscan(
            embedding,
            n_trials=self.hdbscan_optuna_trials,
            mcs_range=self.mcs_range,
            ms_range=self.ms_range,
            metric=self.metric,
            random_state=self.random_state,
            engine=self.engine,
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
        }
