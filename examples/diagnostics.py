"""Generate the diagnostic figures shown in README.md.

Produces, under ``docs/figures/``:

* ``embedding_comparison.png`` -- PCA / t-SNE / UMAP panels on a 5-D
  synthetic blob dataset, coloured by ground-truth label.
* ``trustworthiness_curve.png`` -- T(k) vs k for the same dataset's
  UMAP embedding.
* ``optuna_history.png`` -- running-best Optuna objective.
* ``optuna_importance.png`` -- fANOVA parameter importances.
* ``persistence_vs_baseline.png`` -- per-cluster persistence bars
  against a small noise baseline.
* ``pipeline_embedding.png`` -- final embedding coloured by HDBSCAN
  labels, outliers in grey.

Run:

    python examples/diagnostics.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    import numpy as np

from starfold import (
    UnsupervisedPipeline,
    compute_noise_baseline,
    plot_embedding,
    plot_embedding_comparison,
    plot_optuna_history,
    plot_optuna_param_importance,
    plot_persistence_vs_baseline,
    plot_trustworthiness_curve,
    run_pca,
    run_tsne,
    run_umap,
    trustworthiness_curve,
)

FIG_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"
SEED = 0


def _make_dataset() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_blobs(
        n_samples=600,
        n_features=5,
        centers=4,
        cluster_std=1.5,
        random_state=SEED,
    )
    return StandardScaler().fit_transform(X), y


def _save(fig: plt.Figure, name: str) -> Path:
    path = FIG_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_embedding_comparison(X: np.ndarray, y: np.ndarray) -> None:
    emb_pca = run_pca(X, random_state=SEED)
    emb_tsne = run_tsne(X, n_iter=1000, random_state=SEED)
    emb_umap = run_umap(X, n_epochs=2000, random_state=SEED)
    fig, _ = plot_embedding_comparison(
        {"PCA": emb_pca, "t-SNE": emb_tsne, "UMAP": emb_umap},
        y,
        figsize=(15.0, 4.5),
    )
    fig.suptitle(
        "Three ways to project a 5-D, 4-blob dataset to 2-D. "
        "Colour = ground-truth label.",
        y=1.02,
    )
    _save(fig, "embedding_comparison.png")


def generate_trustworthiness_curve(X: np.ndarray) -> None:
    emb_umap = run_umap(X, n_epochs=2000, random_state=SEED)
    scores = trustworthiness_curve(X, emb_umap, k_values=(5, 10, 15, 25, 50, 80))
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    plot_trustworthiness_curve(scores, ax=ax, threshold=0.9)
    ax.set_title("UMAP trustworthiness as a function of k")
    _save(fig, "trustworthiness_curve.png")


def generate_pipeline_figures(X: np.ndarray, y: np.ndarray) -> None:
    pipeline = UnsupervisedPipeline(
        umap_kwargs={"n_epochs": 2000, "n_neighbors": 30},
        hdbscan_optuna_trials=40,
        mcs_range=(20, 120),
        ms_range=(5, 25),
        engine="cpu",
        skip_noise_baseline=True,
        random_state=SEED,
    )
    result = pipeline.fit(X)

    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    plot_embedding(result.embedding, result.labels, ax=ax)
    ax.set_title(
        f"UMAP + Optuna-HDBSCAN -> {result.n_clusters} clusters "
        f"(T = {result.trustworthiness:.3f})"
    )
    _save(fig, "pipeline_embedding.png")

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    plot_optuna_history(result.search.study, ax=ax)
    ax.set_title("Optuna TPE: running-best persistence sum")
    _save(fig, "optuna_history.png")

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    plot_optuna_param_importance(result.search.study, ax=ax)
    ax.set_title("HDBSCAN parameter importance (fANOVA)")
    _save(fig, "optuna_importance.png")

    baseline = compute_noise_baseline(
        n_samples=X.shape[0],
        n_features=X.shape[1],
        umap_kwargs={"n_epochs": 2000, "n_neighbors": 30},
        n_realisations=15,
        per_realisation_trials=15,
        mcs_range=(20, 60),
        ms_range=(5, 15),
        random_state=SEED,
        cache_dir=FIG_DIR.parent / "_noise_cache",
    )
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    plot_persistence_vs_baseline(
        result.persistence,
        baseline=baseline.threshold,
        per_realisation_max=baseline.per_realisation_max,
        ax=ax,
    )
    ax.set_title(
        f"Per-cluster persistence vs 99.7-percentile noise baseline "
        f"({baseline.threshold:.3f})"
    )
    _save(fig, "persistence_vs_baseline.png")

    # Silence unused-variable warning for y when this function grows.
    del y


def main() -> None:
    X, y = _make_dataset()
    generate_embedding_comparison(X, y)
    generate_trustworthiness_curve(X)
    generate_pipeline_figures(X, y)
    print(f"figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
