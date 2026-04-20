"""Tests for the embedding wrappers.

Each wrapper is exercised on small, seeded inputs to verify:

* output shape is ``(n_samples, n_components)``,
* identical seeds produce identical embeddings,
* different seeds produce different embeddings (UMAP, t-SNE),
* PCA's first component aligns with the high-variance axis.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.manifold import TSNE as _TSNE

from starfold.embedding import run_pca, run_tsne, run_umap


@pytest.fixture(name="X")
def _fixture_x() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(120, 6))


# ---------------------------------------------------------------------- PCA


def test_pca_shape(X: np.ndarray) -> None:
    emb = run_pca(X, n_components=3)
    assert emb.shape == (X.shape[0], 3)


def test_pca_reproducible(X: np.ndarray) -> None:
    emb_a = run_pca(X, n_components=2, random_state=42)
    emb_b = run_pca(X, n_components=2, random_state=42)
    np.testing.assert_array_equal(emb_a, emb_b)


def test_pca_recovers_high_variance_axis() -> None:
    """Column 0 of the PCA embedding must correlate >0.99 with the high-variance axis."""
    rng = np.random.default_rng(1)
    n = 2000
    high_var_axis = rng.normal(size=n) * 10.0
    low_var_axis = rng.normal(size=n) * 0.1
    X = np.column_stack([low_var_axis, high_var_axis])
    emb = run_pca(X, n_components=2)
    correlation = np.corrcoef(emb[:, 0], high_var_axis)[0, 1]
    assert abs(correlation) > 0.99


# --------------------------------------------------------------------- UMAP


def test_umap_shape(X: np.ndarray) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emb = run_umap(X, n_epochs=50, random_state=0)
    assert emb.shape == (X.shape[0], 2)


def test_umap_reproducible(X: np.ndarray) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emb_a = run_umap(X, n_epochs=100, random_state=7)
        emb_b = run_umap(X, n_epochs=100, random_state=7)
    np.testing.assert_array_equal(emb_a, emb_b)


def test_umap_different_seeds_differ(X: np.ndarray) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emb_a = run_umap(X, n_epochs=100, random_state=0)
        emb_b = run_umap(X, n_epochs=100, random_state=1)
    assert not np.allclose(emb_a, emb_b)


def test_umap_honours_n_components(X: np.ndarray) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emb = run_umap(X, n_components=3, n_epochs=50, random_state=0)
    assert emb.shape == (X.shape[0], 3)


# --------------------------------------------------------------------- tSNE


def test_tsne_shape(X: np.ndarray) -> None:
    emb = run_tsne(X, n_iter=250, perplexity=20, random_state=0)
    assert emb.shape == (X.shape[0], 2)


def test_tsne_reproducible(X: np.ndarray) -> None:
    emb_a = run_tsne(X, n_iter=250, perplexity=20, random_state=5)
    emb_b = run_tsne(X, n_iter=250, perplexity=20, random_state=5)
    np.testing.assert_array_equal(emb_a, emb_b)


def test_tsne_seed_is_honoured(X: np.ndarray) -> None:
    """``random_state`` must reach sklearn's TSNE estimator (no silent override)."""
    reducer = _TSNE(n_components=2, perplexity=20, max_iter=250, random_state=123)
    direct = reducer.fit_transform(X)
    wrapped = run_tsne(X, n_iter=250, perplexity=20, random_state=123)
    np.testing.assert_array_equal(direct, wrapped)


# --------------------------------------------------------------- validation


def test_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match="2-D"):
        run_pca(np.zeros(10), n_components=2)
    with pytest.raises(ValueError, match="2-D"):
        run_tsne(np.zeros(10))
    with pytest.raises(ValueError, match="2-D"):
        run_umap(np.zeros(10))
