"""starfold.

UMAP manifold learning followed by Optuna-tuned HDBSCAN clustering,
with a trustworthiness score and a statistical noise baseline, for
any numerical feature matrix.

The package uses ``X`` for feature matrices and ``X_high`` / ``X_low``
for paired high- and low-dimensional embeddings, following the
conventions of the scientific-Python literature this tool builds on.

The top-level namespace exposes the symbols a new user needs to go
from ``X`` to a validated clustering with a single import: the
pipeline classes, the four analytic building blocks (embedding,
trustworthiness, clustering, noise baseline), the science extensions
(credibility, uncertainty, merge, silhouette), a compact set of
publication-ready plots, on-disk I/O, and the data-size-aware
diagnostic helpers.

Advanced objects -- result dataclasses returned by the building
blocks, dashboard-panel plot primitives, and the engine type alias --
stay in their submodules (e.g. ``starfold.clustering.HDBSCANResult``,
``starfold.plotting.plot_credibility``) so the top-level surface
remains small and easy to learn.
"""

from __future__ import annotations

from starfold._engine import cuml_is_importable
from starfold.clustering import run_hdbscan, search_hdbscan
from starfold.credibility import compute_credibility
from starfold.diagnostics import (
    auto_mcs_upper,
    recommend_budget,
    validate_input_matrix,
)
from starfold.embedding import run_pca, run_tsne, run_umap
from starfold.io import load_pipeline_result, save_pipeline_result
from starfold.merge import suggest_merges
from starfold.noise_baseline import compute_noise_baseline
from starfold.pipeline import PipelineResult, UnsupervisedPipeline
from starfold.plotting import (
    plot_condensed_tree,
    plot_embedding,
    plot_embedding_comparison,
    plot_optuna_history,
    plot_optuna_param_importance,
    plot_trustworthiness_curve,
    plot_uncertainty_map,
)
from starfold.silhouette import chunked_silhouette
from starfold.trustworthiness import (
    continuity,
    continuity_curve,
    trustworthiness,
    trustworthiness_curve,
)
from starfold.uncertainty import propagate_uncertainty

__all__ = [
    "PipelineResult",
    "UnsupervisedPipeline",
    "auto_mcs_upper",
    "chunked_silhouette",
    "compute_credibility",
    "compute_noise_baseline",
    "continuity",
    "continuity_curve",
    "cuml_is_importable",
    "load_pipeline_result",
    "plot_condensed_tree",
    "plot_embedding",
    "plot_embedding_comparison",
    "plot_optuna_history",
    "plot_optuna_param_importance",
    "plot_trustworthiness_curve",
    "plot_uncertainty_map",
    "propagate_uncertainty",
    "recommend_budget",
    "run_hdbscan",
    "run_pca",
    "run_tsne",
    "run_umap",
    "save_pipeline_result",
    "search_hdbscan",
    "suggest_merges",
    "trustworthiness",
    "trustworthiness_curve",
    "validate_input_matrix",
]

__version__ = "0.0.1"
