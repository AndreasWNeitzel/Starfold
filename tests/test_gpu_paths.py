from __future__ import annotations

import os

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from starfold import _engine
from starfold.backend import knn_graph, neighbor_adjacency
from starfold.consensus import coassociation
from starfold.objective import modularity, transfer_labels
from starfold.prep import scale

needs_cuml = pytest.mark.skipif(
    os.environ.get("STARFOLD_TEST_CUML") != "1",
    reason="set STARFOLD_TEST_CUML=1 to probe and use cuML",
)


def _require_cuml() -> None:
    available, reason = _engine.cuml_status()
    if not available:
        pytest.skip(reason or "cuML unavailable")


def _reference_vote(labels_fit: np.ndarray, neighbours: np.ndarray) -> np.ndarray:
    """Count-table majority vote; oracle for the smallest-label tie rule."""
    votes = labels_fit[neighbours] + 1
    counts = np.zeros((neighbours.shape[0], int(labels_fit.max()) + 2), dtype=np.int32)
    np.add.at(
        counts, (np.repeat(np.arange(neighbours.shape[0]), neighbours.shape[1]), votes.ravel()), 1
    )
    return counts.argmax(axis=1) - 1


def test_transfer_vote_matches_the_count_table_including_ties() -> None:
    rng = np.random.default_rng(0)
    x_fit = rng.normal(size=(400, 3))
    x_query = rng.normal(size=(150, 3))
    # Few labels and an even k force many tied votes.
    labels_fit = rng.integers(-1, 3, size=400).astype(np.int32)
    k = 6
    scaler = StandardScaler().fit(x_fit)
    index = NearestNeighbors(n_neighbors=k, algorithm="brute").fit(
        scaler.transform(x_fit).astype(np.float32)
    )
    neighbours = index.kneighbors(
        scaler.transform(x_query).astype(np.float32), return_distance=False
    )
    expected = _reference_vote(labels_fit, neighbours)
    got = transfer_labels(x_fit, labels_fit, x_query, scaling="standard", k=k)
    np.testing.assert_array_equal(got, expected)


@needs_cuml
def test_transfer_labels_gpu_matches_cpu() -> None:
    _require_cuml()
    rng = np.random.default_rng(1)
    x_fit = rng.normal(size=(3000, 5))
    x_query = rng.normal(size=(800, 5))
    labels_fit = rng.integers(-1, 40, size=3000).astype(np.int32)
    for scaling in ("standard", "robust"):
        cpu = transfer_labels(x_fit, labels_fit, x_query, scaling=scaling, engine="cpu")
        gpu = transfer_labels(x_fit, labels_fit, x_query, scaling=scaling, engine="cuml")
        assert gpu.dtype == np.int32
        np.testing.assert_array_equal(gpu, cpu)


@needs_cuml
def test_modularity_gpu_matches_cpu(single_gaussian: np.ndarray) -> None:
    _require_cuml()
    x = scale(single_gaussian, "standard")
    adjacency = neighbor_adjacency(knn_graph(x, 16, engine="cpu", seed=0), 15)
    labels = (np.arange(x.shape[0]) % 5 - 1).astype(np.int32)
    cpu = modularity(labels, adjacency, engine="cpu")
    gpu = modularity(labels, adjacency, engine="cuml")
    assert gpu == pytest.approx(cpu, abs=1e-12)
    noise = np.full(x.shape[0], -1, dtype=np.int32)
    assert modularity(noise, adjacency, engine="cuml") == 0.0


@needs_cuml
def test_coassociation_gpu_matches_cpu() -> None:
    _require_cuml()
    rng = np.random.default_rng(2)
    world_labels = rng.integers(-2, 6, size=(32, 700)).astype(np.int16)
    world_labels[:, :5] = -1  # rows never assigned sit at distance one
    rows = np.sort(rng.choice(700, size=300, replace=False)).astype(np.intp)
    cpu = coassociation(world_labels, rows, engine="cpu")
    gpu = coassociation(world_labels, rows, engine="cuml")
    assert gpu.dtype == np.float64
    np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-12)
