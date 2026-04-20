"""Shared pytest fixtures.

Deterministic, low-dimensional Gaussian-mixture data for the
clustering / pipeline / noise-baseline tests. Fixtures are scoped
``session`` so the same data is reused across tests.
"""

from __future__ import annotations

import numpy as np
import pytest


def _gaussian_mixture(
    seed: int,
    *,
    centres: np.ndarray,
    n_per: int,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    blobs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for k, c in enumerate(centres):
        blobs.append(rng.normal(loc=c, scale=scale, size=(n_per, c.shape[0])))
        labels.append(np.full(n_per, k, dtype=np.intp))
    X = np.vstack(blobs)
    y = np.concatenate(labels)
    perm = rng.permutation(X.shape[0])
    return X[perm], y[perm]


@pytest.fixture(scope="session")
def gmm_three_clusters_2d() -> tuple[np.ndarray, np.ndarray]:
    """Three well-separated 2-D Gaussians, 120 points each, seed 0."""
    centres = np.array([[-8.0, 0.0], [0.0, 0.0], [8.0, 0.0]])
    return _gaussian_mixture(seed=0, centres=centres, n_per=120, scale=0.7)


@pytest.fixture(scope="session")
def gmm_four_clusters_3d() -> tuple[np.ndarray, np.ndarray]:
    """Four 3-D Gaussians at the vertices of a tetrahedron, 90 points each, seed 1."""
    centres = np.array(
        [
            [5.0, 5.0, 5.0],
            [-5.0, -5.0, 5.0],
            [-5.0, 5.0, -5.0],
            [5.0, -5.0, -5.0],
        ]
    )
    return _gaussian_mixture(seed=1, centres=centres, n_per=90, scale=0.8)
