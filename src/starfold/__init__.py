"""starfold.

UMAP manifold learning followed by Optuna-tuned HDBSCAN clustering,
with a trustworthiness score and a statistical noise baseline, for
any numerical feature matrix.

The package uses ``X`` for feature matrices and ``X_high`` / ``X_low``
for paired high- and low-dimensional embeddings, following the
conventions of the scientific-Python literature this tool builds on.
"""

from __future__ import annotations

from starfold.clustering import (
    HDBSCANResult,
    OptunaSearchResult,
    run_hdbscan,
    search_hdbscan,
)
from starfold.embedding import run_pca, run_tsne, run_umap
from starfold.noise_baseline import NoiseBaselineResult, compute_noise_baseline
from starfold.trustworthiness import trustworthiness, trustworthiness_curve

__all__ = [
    "HDBSCANResult",
    "NoiseBaselineResult",
    "OptunaSearchResult",
    "compute_noise_baseline",
    "run_hdbscan",
    "run_pca",
    "run_tsne",
    "run_umap",
    "search_hdbscan",
    "trustworthiness",
    "trustworthiness_curve",
]

__version__ = "0.0.1"
