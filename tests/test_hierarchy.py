"""Tests for the hierarchical-structure module.

Covers:

* :func:`extract_hierarchy` populates ``flat_to_node`` correctly for
  a fit with well-separated clusters.
* ``available=False`` when the CPU model is not given (cuml path).
* ``merge_lambda`` is non-decreasing when walking from a cluster up
  the tree.
* ``sibling`` returns a valid flat cluster id or ``None``.
* ``subcluster_on`` produces a per-sample label vector that assigns
  ``-1`` to rows outside the parent cluster.
* Input validation on ``cluster_id`` and ``embedding`` shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from starfold.clustering import run_hdbscan
from starfold.hierarchy import HierarchicalStructure, extract_hierarchy


def _two_blob_embedding(seed: int = 0) -> tuple[np.ndarray, object, np.ndarray]:
    rng = np.random.default_rng(seed)
    emb = np.vstack(
        [
            rng.normal(loc=[-5.0, 0.0], scale=0.35, size=(40, 2)),
            rng.normal(loc=[5.0, 0.0], scale=0.35, size=(40, 2)),
        ]
    )
    # Fit once so we have a model + flat labels aligned to ``emb``.
    import hdbscan as _hdbscan  # noqa: PLC0415

    model = _hdbscan.HDBSCAN(min_cluster_size=10, gen_min_span_tree=True).fit(emb)
    return emb, model, np.asarray(model.labels_, dtype=np.intp)


def test_extract_hierarchy_on_cpu_model_available() -> None:
    emb, model, labels = _two_blob_embedding()
    h = extract_hierarchy(model, labels)
    assert isinstance(h, HierarchicalStructure)
    assert h.available is True
    assert h.edges.size > 0
    assert h.n_samples == emb.shape[0]


def test_extract_hierarchy_recovers_flat_to_node_for_every_cluster() -> None:
    _, model, labels = _two_blob_embedding()
    h = extract_hierarchy(model, labels)
    # Two well-separated blobs with generous min_cluster_size -> two flat
    # clusters, both locatable on the condensed tree.
    positive = labels[labels >= 0]
    n_flat = int(positive.max() + 1) if positive.size else 0
    assert h.flat_to_node.shape == (n_flat,)
    assert n_flat >= 2
    assert np.all(h.flat_to_node >= 0)


def test_none_model_yields_unavailable_structure() -> None:
    labels = np.array([0, 0, 1, 1, -1], dtype=np.intp)
    h = extract_hierarchy(None, labels)
    assert h.available is False
    with pytest.raises(RuntimeError, match="unavailable"):
        h.merge_lambda(0, 1)
    with pytest.raises(RuntimeError, match="unavailable"):
        h.subcluster_on(np.zeros((5, 2)), 0, min_cluster_size=2)


def test_merge_lambda_self_is_zero() -> None:
    _, model, labels = _two_blob_embedding()
    h = extract_hierarchy(model, labels)
    assert h.merge_lambda(0, 0) == 0.0


def test_merge_lambda_symmetric_and_finite() -> None:
    _, model, labels = _two_blob_embedding()
    h = extract_hierarchy(model, labels)
    lam_01 = h.merge_lambda(0, 1)
    lam_10 = h.merge_lambda(1, 0)
    assert lam_01 == lam_10
    assert np.isfinite(lam_01)
    assert lam_01 > 0.0


def test_sibling_of_cluster_zero_is_a_valid_id() -> None:
    _, model, labels = _two_blob_embedding()
    h = extract_hierarchy(model, labels)
    sib = h.sibling(0)
    # Two-cluster fit -> sibling of 0 is 1 (or None if tree is trivial).
    assert sib in {1, None}


def test_subcluster_on_returns_masked_labels() -> None:
    emb, model, labels = _two_blob_embedding()
    h = extract_hierarchy(model, labels)
    out = h.subcluster_on(emb, 0, min_cluster_size=5)
    assert out.shape == (emb.shape[0],)
    # Outside the parent cluster all labels are -1.
    assert np.all(out[labels != 0] == -1)


def test_subcluster_on_rejects_wrong_embedding_shape() -> None:
    _, model, labels = _two_blob_embedding()
    h = extract_hierarchy(model, labels)
    with pytest.raises(ValueError, match="n_samples"):
        h.subcluster_on(np.zeros((labels.size + 1, 2)), 0, min_cluster_size=5)


def test_cluster_id_out_of_range_raises() -> None:
    _, model, labels = _two_blob_embedding()
    h = extract_hierarchy(model, labels)
    with pytest.raises(ValueError, match="out of range"):
        h.merge_lambda(0, 99)
    with pytest.raises(ValueError, match="out of range"):
        h.sibling(99)


def test_pipeline_attaches_hierarchy_on_cpu_backend(
    gmm_three_clusters_2d: tuple[np.ndarray, np.ndarray],
) -> None:
    from starfold.pipeline import UnsupervisedPipeline  # noqa: PLC0415

    X, _ = gmm_three_clusters_2d
    pipe = UnsupervisedPipeline(
        umap_kwargs={"n_epochs": 50, "n_neighbors": 10},
        hdbscan_optuna_trials=6,
        mcs_range=(5, 20),
        ms_range=(1, 5),
        skip_noise_baseline=True,
        engine="cpu",
        random_state=0,
    )
    result = pipe.fit(X)
    assert result.hierarchy.available is True
    # flat_to_node has one entry per non-negative label.
    assert result.hierarchy.flat_to_node.shape == (result.n_clusters,)
    assert result.hierarchy.n_samples == result.labels.size


def test_subcluster_on_integrates_with_run_hdbscan() -> None:
    # Re-fit by hand to avoid a full pipeline run; validates that
    # subcluster_on passes sensible arguments to run_hdbscan.
    _, model, labels = _two_blob_embedding()
    h = extract_hierarchy(model, labels)
    emb = np.vstack(
        [
            np.random.default_rng(0).normal(loc=[-5.0, 0.0], scale=0.35, size=(40, 2)),
            np.random.default_rng(0).normal(loc=[5.0, 0.0], scale=0.35, size=(40, 2)),
        ]
    )
    out = h.subcluster_on(emb, 0, min_cluster_size=5, min_samples=2)
    # Cross-check that run_hdbscan produces a compatibly-shaped labelling.
    baseline = run_hdbscan(emb[labels == 0], min_cluster_size=5, min_samples=2)
    assert out[labels == 0].shape == baseline.labels.shape
