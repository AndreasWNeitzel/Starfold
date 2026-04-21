"""Generate the diagnostic figures shown in README.md.

The dataset is the closed chain of eight Hopf-linked tori built by
``torus_chain.make_torus_chain``. Using the tutorial's dataset here
means the README's at-a-glance figures match what a new user sees
when running the quickstart notebook.

Produces, under ``docs/figures/``:

* ``torus_chain_3d.png`` -- the 3D chain in two views, coloured by
  ground-truth torus index.
* ``embedding_comparison.png`` -- PCA / t-SNE / UMAP panels on the
  chain's (x, y, z) coordinates, coloured by ground-truth label.
* ``trustworthiness_curve.png`` -- T(k) vs k for the UMAP embedding.
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
from sklearn.preprocessing import StandardScaler
from torus_chain import make_torus_chain

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
    trustworthiness_curve,
)

if TYPE_CHECKING:
    import numpy as np

FIG_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"


def _make_dataset() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_torus_chain(
        n_links=8,
        points_per_link=1200,
        big_radius=4.0,
        major_even=2.0,
        major_odd=2.5,
        minor_radius=0.15,
        solid=True,
        noise_std=0.02,
    )
    return StandardScaler().fit_transform(X), y, X


def _save(fig: plt.Figure, name: str) -> Path:
    path = FIG_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _scatter_chain(ax: plt.Axes, X: np.ndarray, labels: np.ndarray, title: str) -> None:
    cmap = plt.get_cmap("tab10")
    for k in range(8):
        mask = labels == k
        ax.scatter(
            X[mask, 0], X[mask, 1], X[mask, 2],
            s=2.0, color=cmap(k), alpha=0.55, label=f"link {k}",
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.set_box_aspect((1, 1, 0.5))


def generate_chain_3d(X_raw: np.ndarray, y: np.ndarray) -> None:
    fig = plt.figure(figsize=(13, 5.5))
    ax1 = fig.add_subplot(121, projection="3d")
    _scatter_chain(ax1, X_raw, y, "perspective")
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.view_init(elev=80, azim=-60)
    _scatter_chain(ax2, X_raw, y, "top-down")
    fig.legend(
        *ax1.get_legend_handles_labels(),
        loc="center right", bbox_to_anchor=(1.00, 0.5), fontsize=9,
    )
    fig.suptitle(
        "Hopf chain of 8 interlocked tori in 3D -- the starfold quickstart dataset.",
        y=1.02,
    )
    _save(fig, "torus_chain_3d.png")


def generate_embedding_comparison(
    X: np.ndarray, y: np.ndarray, emb_umap: np.ndarray,
) -> None:
    emb_pca = run_pca(X)
    emb_tsne = run_tsne(X, n_iter=1000)
    fig, _ = plot_embedding_comparison(
        {"PCA": emb_pca, "t-SNE": emb_tsne, "UMAP": emb_umap},
        y,
        figsize=(15.0, 4.5),
    )
    fig.suptitle(
        "Three ways to project the Hopf chain to 2-D. Colour = ground-truth torus.",
        y=1.02,
    )
    _save(fig, "embedding_comparison.png")


def generate_trustworthiness_curve(X: np.ndarray, emb_umap: np.ndarray) -> None:
    scores = trustworthiness_curve(X, emb_umap, k_values=(5, 10, 15, 25, 50, 100))
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    plot_trustworthiness_curve(scores, ax=ax, threshold=0.9)
    ax.set_title("UMAP trustworthiness as a function of k")
    _save(fig, "trustworthiness_curve.png")


def generate_pipeline_figures(X: np.ndarray, y: np.ndarray):
    pipeline = UnsupervisedPipeline(
        umap_kwargs={"n_epochs": 5000, "n_neighbors": 50, "min_dist": 0.0},
        hdbscan_optuna_trials=80,
        mcs_range=(400, 2000),
        ms_range=(1, 30),
        engine="cpu",
        skip_noise_baseline=True,
        hdbscan_objective="combined_geom",
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
        umap_kwargs={"n_epochs": 5000, "n_neighbors": 50, "min_dist": 0.0},
        n_realisations=15,
        per_realisation_trials=15,
        mcs_range=(400, 2000),
        ms_range=(1, 30),
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

    del y
    return result


def generate_hdbscan_tuning_dashboard(result) -> None:
    """Eight-panel HDBSCAN tuning dashboard saved as a single PNG."""
    fig = result.plot_tuning_dashboard()
    _save(fig, "hdbscan_tuning_dashboard.png")


def generate_quality_dashboard(X: np.ndarray, result) -> None:
    """Six-panel pipeline-quality dashboard saved as a single PNG."""
    fig = result.plot_quality_dashboard(
        X,
        n_subsamples=60,
        subsample_fraction=0.8,
        k_values=(5, 10, 15, 25, 50, 100),
    )
    _save(fig, "pipeline_quality_dashboard.png")


def main() -> None:
    X, y, X_raw = _make_dataset()
    generate_chain_3d(X_raw, y)
    result = generate_pipeline_figures(X, y)
    generate_embedding_comparison(X, y, result.embedding)
    generate_trustworthiness_curve(X, result.embedding)
    generate_hdbscan_tuning_dashboard(result)
    generate_quality_dashboard(X, result)
    print(f"figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
