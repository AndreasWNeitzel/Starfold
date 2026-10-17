from __future__ import annotations

import os
import sys
import warnings
from types import ModuleType

import numpy as np
import pytest

from starfold import _engine, backend
from starfold.backend import (
    Fit,
    Params,
    _slice_knn,
    compare_backends,
    default_params,
    fit_hdbscan,
    fit_pipeline,
    fit_umap,
    knn_graph,
    neighbor_adjacency,
    rescale_params,
)
from starfold.prep import scale

CUML_TESTS_REQUESTED = os.environ.get("STARFOLD_TEST_CUML") == "1"
gpu = pytest.mark.gpu
needs_cuml = pytest.mark.skipif(
    not CUML_TESTS_REQUESTED, reason="set STARFOLD_TEST_CUML=1 to probe and use cuML"
)


def _two_blob_embedding() -> np.ndarray:
    rng = np.random.default_rng(3)
    left = rng.normal(loc=(-5.0, 0.0), scale=0.3, size=(50, 2))
    right = rng.normal(loc=(5.0, 0.0), scale=0.3, size=(50, 2))
    return np.concatenate([left, right]).astype(np.float32)


def test_default_params_are_the_library_defaults():
    assert default_params() == Params(15, 0.1, 2, 5, 5, "eom")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("n_neighbors", 1, "n_neighbors"),
        ("min_dist", -0.1, "min_dist"),
        ("min_dist", 1.5, "min_dist"),
        ("min_dist", float("nan"), "min_dist"),
        ("n_components", 0, "n_components"),
        ("min_cluster_size", 1, "min_cluster_size"),
        ("min_samples", 0, "min_samples"),
        ("min_samples", 6, "min_samples"),
        ("selection_method", "excess", "selection_method"),
    ],
)
def test_params_validation(field, value, message):
    fields = default_params().to_dict()
    fields[field] = value
    with pytest.raises(ValueError, match=message):
        Params(**fields)  # type: ignore[arg-type]


def test_params_dict_round_trip():
    params = Params(30, 0.0, 5, 20, 7, "leaf")
    as_dict = params.to_dict()
    assert as_dict == {
        "n_neighbors": 30,
        "min_dist": 0.0,
        "n_components": 5,
        "min_cluster_size": 20,
        "min_samples": 7,
        "selection_method": "leaf",
    }
    assert Params.from_dict(as_dict) == params
    as_dict["n_neighbors"] = 30.0
    as_dict["min_samples"] = np.int64(7)
    assert Params.from_dict(as_dict) == params


@pytest.mark.parametrize(
    ("field", "value"),
    [("n_neighbors", 15.5), ("n_components", True), ("min_dist", "0.1"), ("min_samples", None)],
)
def test_params_from_dict_rejects_bad_values(field, value):
    fields = default_params().to_dict()
    fields[field] = value
    with pytest.raises(ValueError, match="expected"):
        Params.from_dict(fields)


def test_params_from_dict_rejects_missing_field():
    fields = default_params().to_dict()
    del fields["min_dist"]
    with pytest.raises(ValueError, match="min_dist"):
        Params.from_dict(fields)


def test_knn_graph_convention_and_exactness(single_gaussian):
    x = scale(single_gaussian, "standard")
    indices, distances = knn_graph(x, 6, engine="cpu", seed=0)
    assert indices.shape == distances.shape == (200, 6)
    assert indices.dtype == np.int32
    assert distances.dtype == np.float32
    np.testing.assert_array_equal(indices[:, 0], np.arange(200))
    np.testing.assert_array_equal(distances[:, 0], 0.0)
    assert np.all(np.diff(distances, axis=1) >= 0)
    x64 = x.astype(np.float64)
    full = np.sqrt(((x64[:, None, :] - x64[None, :, :]) ** 2).sum(axis=-1))
    expected = np.argsort(full, axis=1, kind="stable")[:, :6]
    np.testing.assert_array_equal(indices[:, 1:], expected[:, 1:])
    np.testing.assert_allclose(distances, np.take_along_axis(full, expected, axis=1), atol=1e-4)


def test_knn_graph_caps_k_and_handles_duplicates(single_gaussian):
    x = scale(single_gaussian[:60], "standard")
    x[:5] = x[0]
    indices, distances = knn_graph(x, 100, engine="cpu", seed=0)
    assert indices.shape == (60, 60)
    np.testing.assert_array_equal(indices[:, 0], np.arange(60))
    np.testing.assert_array_equal(distances[:, 0], 0.0)
    assert set(indices[0, :5].tolist()) == {0, 1, 2, 3, 4}
    assert np.all(np.diff(distances, axis=1) >= 0)


def test_knn_graph_rejects_bad_input(single_gaussian):
    x = scale(single_gaussian, "standard")
    with pytest.raises(ValueError, match="k must be"):
        knn_graph(x, 0, engine="cpu", seed=0)
    with pytest.raises(ValueError, match="two-dimensional"):
        knn_graph(x[:, 0], 5, engine="cpu", seed=0)


@pytest.mark.parametrize("engine", ["auto", "gpu", "CPU"])
def test_stage_functions_reject_an_unresolved_engine(single_gaussian, engine):
    x = scale(single_gaussian, "standard")
    params = default_params()
    with pytest.raises(ValueError, match="engine must be"):
        knn_graph(x, 5, engine=engine, seed=0)
    with pytest.raises(ValueError, match="engine must be"):
        fit_umap(x, params, seed=0, engine=engine)
    with pytest.raises(ValueError, match="engine must be"):
        fit_hdbscan(_two_blob_embedding(), params, engine=engine)
    with pytest.raises(ValueError, match="engine must be"):
        fit_pipeline(single_gaussian, params, seed=0, engine=engine, scaling="standard")


def test_neighbor_adjacency_on_hand_built_graph():
    indices = np.array([[0, 1, 2], [1, 0, 3], [2, 0, 3], [3, 2, 1]], dtype=np.int32)
    distances = np.zeros_like(indices, dtype=np.float32)
    adjacency = neighbor_adjacency((indices, distances), k=2)
    dense = adjacency.toarray()
    assert adjacency.dtype == np.float32
    assert adjacency.format == "csr"
    expected = np.array(
        [[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(dense, expected)
    one = neighbor_adjacency((indices, distances), k=1).toarray()
    expected_one = np.array(
        [[0, 1, 1, 0], [1, 0, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(one, expected_one)
    wide = neighbor_adjacency((indices, distances), k=15).toarray()
    np.testing.assert_array_equal(wide, expected)


def test_neighbor_adjacency_ignores_self_and_missing_entries():
    indices = np.array([[0, 0, -1], [1, 1, 0], [2, 2, 2]], dtype=np.int32)
    distances = np.zeros_like(indices, dtype=np.float32)
    dense = neighbor_adjacency((indices, distances), k=2).toarray()
    expected = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float32)
    np.testing.assert_array_equal(dense, expected)


def test_slice_knn_copies_leading_columns(single_gaussian):
    x = scale(single_gaussian, "standard")
    graph = knn_graph(x, 12, engine="cpu", seed=0)
    indices, distances = _slice_knn(graph, 5, 200)
    assert indices.shape == distances.shape == (200, 5)
    assert indices.dtype == np.int32
    assert distances.dtype == np.float32
    assert indices.flags.c_contiguous
    np.testing.assert_array_equal(indices, graph[0][:, :5])
    np.testing.assert_array_equal(distances, graph[1][:, :5])
    indices[0, 0] = -1
    assert graph[0][0, 0] == 0
    same = _slice_knn(graph, 12, 200)
    assert not np.shares_memory(same[0], graph[0])


def test_fit_umap_rejects_unusable_graph_before_fitting(single_gaussian):
    x = scale(single_gaussian, "standard")
    graph = knn_graph(x, 8, engine="cpu", seed=0)
    params = Params(10, 0.1, 2, 5, 5, "eom")
    with pytest.raises(ValueError, match="fewer than n_neighbors"):
        fit_umap(x, params, seed=0, engine="cpu", knn=graph)
    with pytest.raises(ValueError, match="rows but the data has"):
        fit_umap(x[:100], params, seed=0, engine="cpu", knn=knn_graph(x, 12, engine="cpu", seed=0))
    with pytest.raises(ValueError, match="share a 2-D shape"):
        fit_umap(x, params, seed=0, engine="cpu", knn=(graph[0], graph[1][:, :4]))


def test_fit_hdbscan_cpu_on_hand_built_embedding():
    embedding = _two_blob_embedding()
    labels, strengths = fit_hdbscan(embedding, Params(15, 0.1, 2, 10, 5, "eom"), engine="cpu")
    assert labels.dtype == np.int32
    assert strengths.dtype == np.float32
    assert labels.shape == strengths.shape == (100,)
    assert np.all((strengths >= 0) & (strengths <= 1))
    assigned = labels >= 0
    assert assigned.mean() > 0.9
    assert len(set(labels[assigned].tolist())) == 2
    assert len(set(labels[:50][labels[:50] >= 0].tolist())) == 1
    assert len(set(labels[50:][labels[50:] >= 0].tolist())) == 1


def test_fit_hdbscan_leaf_and_noise_only():
    embedding = _two_blob_embedding()
    labels, _ = fit_hdbscan(embedding, Params(15, 0.1, 2, 10, 5, "leaf"), engine="cpu")
    assert labels.max() >= 1
    labels, strengths = fit_hdbscan(embedding, Params(15, 0.1, 2, 80, 80, "eom"), engine="cpu")
    assert np.all(labels == -1)
    assert np.all(strengths == 0)


def test_fit_dataclass_holds_arrays():
    fit = Fit(
        labels=np.zeros(3, dtype=np.int32),
        strengths=np.ones(3, dtype=np.float32),
        embedding=np.zeros((3, 2), dtype=np.float32),
        n_groups=1,
        seconds=0.0,
    )
    assert fit.n_groups == 1
    assert fit is not Fit(**{f: getattr(fit, f) for f in fit.__dataclass_fields__})


def test_compare_backends_requires_cuml(monkeypatch, two_blobs):
    monkeypatch.setattr(_engine, "_cuml_probe", lambda: (False, "ImportError: no cuml"))
    with pytest.raises(ImportError, match="no cuml"):
        compare_backends(two_blobs, default_params(), n_seeds=1)


def _stub_backends(monkeypatch, labels_by_engine):
    """Replace both fits by fixed label vectors and skip the cuML probe."""

    def fit(x_rows, params, *, seed, engine, scaling, knn=None):
        del params, seed, scaling, knn
        labels = labels_by_engine[engine]
        assert labels.shape == (x_rows.shape[0],)
        return Fit(
            labels=labels,
            strengths=np.ones(labels.shape[0], dtype=np.float32),
            embedding=np.zeros((labels.shape[0], 2), dtype=np.float32),
            n_groups=int(np.unique(labels[labels >= 0]).size),
            seconds=0.0,
        )

    monkeypatch.setattr(backend, "resolve_engine", lambda engine: engine)
    monkeypatch.setattr(backend, "release_backend_caches", lambda: None)
    monkeypatch.setattr(backend, "fit_pipeline", fit)


@pytest.mark.parametrize(
    ("cpu_assigns", "cuml_assigns", "expected"),
    [(True, True, 1.0), (False, False, 1.0), (True, False, 0.0), (False, True, 0.0)],
)
def test_compare_backends_normalised_when_a_backend_assigns_nothing(
    monkeypatch, two_blobs, cpu_assigns, cuml_assigns, expected
):
    n = two_blobs.shape[0]
    blocks = np.repeat(np.arange(2, dtype=np.int32), n // 2)
    noise = np.full(n, -1, dtype=np.int32)
    _stub_backends(
        monkeypatch,
        {"cpu": blocks if cpu_assigns else noise, "cuml": blocks if cuml_assigns else noise},
    )
    agreement = compare_backends(two_blobs, default_params(), n_seeds=2)
    assert agreement.normalised == (expected, expected)
    assert agreement.mean_normalised == expected
    if cpu_assigns != cuml_assigns:
        assert agreement.mean_normalised < 0.8


@gpu
@needs_cuml
def test_cuml_knn_matches_cpu(single_gaussian):
    available, reason = _engine.cuml_status()
    if not available:
        pytest.skip(reason or "cuML unavailable")
    x = scale(single_gaussian, "standard")
    cpu = knn_graph(x, 8, engine="cpu", seed=0)
    gpu_graph = knn_graph(x, 8, engine="cuml", seed=0)
    np.testing.assert_array_equal(gpu_graph[0][:, 0], np.arange(200))
    np.testing.assert_allclose(gpu_graph[1], cpu[1], atol=1e-3)


@gpu
@needs_cuml
def test_cuml_hdbscan_and_backend_agreement(two_blobs):
    available, reason = _engine.cuml_status()
    if not available:
        pytest.skip(reason or "cuML unavailable")
    labels, strengths = fit_hdbscan(
        _two_blob_embedding(), Params(15, 0.1, 2, 10, 5, "eom"), engine="cuml"
    )
    assert labels.dtype == np.int32
    assert strengths.dtype == np.float32
    agreement = compare_backends(two_blobs, Params(15, 0.1, 2, 10, 5, "eom"), n_seeds=1)
    assert len(agreement.agreement) == 1
    assert 0.0 <= agreement.mean_agreement <= 1.0
    assert agreement.agreement[0] == pytest.approx(agreement.ari[0] * agreement.coverage[0])
    assert 0.0 <= agreement.coverage[0] <= 1.0
    reachable = min(agreement.assigned_cpu[0], agreement.assigned_cuml[0])
    assert agreement.coverage[0] <= reachable + 1e-9
    assert agreement.normalised[0] == pytest.approx(agreement.agreement[0] / reachable)


@pytest.mark.parametrize(
    ("from_rows", "to_rows", "expected_size", "expected_samples"),
    [
        (800, 1000, 25, 10),
        (1000, 1000, 20, 8),
        (1000, 800, 16, 6),
        (100, 4, 2, 1),
    ],
)
def test_rescale_params_keeps_the_mass_fraction(
    from_rows, to_rows, expected_size, expected_samples
):
    params = Params(15, 0.1, 3, 20, 8, "leaf")
    scaled = rescale_params(params, from_rows, to_rows)
    assert scaled.min_cluster_size == expected_size
    assert scaled.min_samples == expected_samples
    assert (scaled.n_neighbors, scaled.min_dist, scaled.n_components) == (15, 0.1, 3)
    assert scaled.selection_method == "leaf"
    assert scaled.min_samples <= scaled.min_cluster_size


def test_rescale_params_identity_and_bounds():
    params = Params(15, 0.1, 2, 5, 5, "eom")
    assert rescale_params(params, 500, 500) is params
    huge = rescale_params(Params(15, 0.1, 2, 900, 900, "eom"), 100, 500_000)
    assert huge.min_samples == 1023
    assert huge.min_cluster_size >= huge.min_samples
    with pytest.raises(ValueError, match="row counts"):
        rescale_params(params, 0, 10)


def test_knn_graph_dispatches_to_seeded_descent_above_row_limit(monkeypatch):
    seen = []

    def fail_brute(*_args, **_kwargs):
        pytest.fail("large CPU inputs must use seeded descent")

    def descent(x, k, *, seed, n_jobs):
        seen.append((x.shape[0], k, seed, n_jobs))
        indices = np.repeat(np.arange(x.shape[0], dtype=np.int32)[:, None], k, axis=1)
        distances = np.zeros(indices.shape, dtype=np.float32)
        return indices, distances

    monkeypatch.setattr(backend, "_brute_knn", fail_brute)
    monkeypatch.setattr(backend, "_descent_knn", descent)
    x = np.zeros((backend._BRUTE_FORCE_MAX_ROWS + 1, 2), dtype=np.float32)  # noqa: SLF001

    indices, distances = knn_graph(x, 2, engine="cpu", seed=17, n_jobs=3)

    assert seen == [(x.shape[0], 2, 17, 3)]
    assert indices.shape == distances.shape == (x.shape[0], 2)
    np.testing.assert_array_equal(indices[:, 0], np.arange(x.shape[0]))


@pytest.mark.parametrize(
    ("labels", "expected_groups"),
    [([-1] * 6, 0), ([0] * 6, 1)],
)
def test_fit_pipeline_counts_zero_and_one_groups(monkeypatch, labels, expected_groups):
    x = np.ones((6, 2), dtype=np.float64)
    params = Params(2, 0.1, 2, 2, 1, "eom")
    labels = np.asarray(labels, dtype=np.int32)
    monkeypatch.setattr(backend, "scale", lambda rows, _scaling: rows.astype(np.float32))

    def fake_knn(rows, k, *, engine, seed):
        del engine, seed
        indices = np.repeat(np.arange(rows.shape[0], dtype=np.int32)[:, None], k, axis=1)
        return indices, np.zeros(indices.shape, dtype=np.float32)

    monkeypatch.setattr(backend, "knn_graph", fake_knn)
    monkeypatch.setattr(
        backend,
        "fit_umap",
        lambda rows, _params, **_kwargs: np.zeros((rows.shape[0], 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        backend,
        "fit_hdbscan",
        lambda embedding, _params, **_kwargs: (
            labels.copy(),
            np.ones(embedding.shape[0], dtype=np.float32),
        ),
    )

    fit = fit_pipeline(x, params, seed=0, engine="cpu", scaling="standard")

    assert fit.n_groups == expected_groups
    np.testing.assert_array_equal(fit.labels, labels)


def test_fit_umap_errors_when_umap_ignores_precomputed_graph(monkeypatch):
    class FakeReducer:
        def __init__(self, **_kwargs):
            pass

        def fit_transform(self, rows):
            warnings.warn(
                "precomputed_knn has a different number of samples than the data",
                UserWarning,
                stacklevel=2,
            )
            return np.zeros((rows.shape[0], 2), dtype=np.float32)

    fake_umap = ModuleType("umap")
    fake_umap.UMAP = FakeReducer
    monkeypatch.setitem(sys.modules, "umap", fake_umap)
    x = np.zeros((6, 2), dtype=np.float32)
    params = Params(2, 0.1, 2, 2, 1, "eom")
    indices = np.repeat(np.arange(6, dtype=np.int32)[:, None], 2, axis=1)
    graph = (indices, np.zeros(indices.shape, dtype=np.float32))

    with pytest.raises(UserWarning, match="precomputed_knn has a different number of samples"):
        fit_umap(x, params, seed=0, engine="cpu", knn=graph)
