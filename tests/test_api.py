from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest

import starfold
from starfold import _engine, api
from starfold.api import Config, Space, _null_stage, _prepare, recover
from starfold.backend import Params, default_params, rescale_params
from starfold.search import search as run_search


class _StageReachedError(Exception):
    """Ends ``recover`` once the stage under test has been reached."""


def test_config_defaults_are_valid_and_frozen() -> None:
    config = Config()
    assert config.n_explore == 16
    assert config.null_families == ()
    with pytest.raises(AttributeError):
        config.n_trials = 3  # type: ignore[misc]
    assert "terminator" not in {f.name for f in fields(Config)}


def test_public_surface_has_no_worlds_wrapper() -> None:
    assert "worlds" not in starfold.__all__
    assert "worlds" not in api.__all__
    assert not hasattr(api, "worlds")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_explore": 12}, "power of two"),
        ({"n_trials": 16}, "exceed n_explore"),
        ({"n_resamples": 1}, "at least 2"),
        ({"resample_fraction": 1.0}, "between 0 and 1"),
        ({"n_worlds": 1}, "between 2 and 32767"),
        ({"n_null_max": 5}, "cannot reach alpha"),
        ({"alpha": 0.0}, "alpha"),
        ({"null_families": ("elliptical_gaussian",)}, "null_families"),
        ({"null_procedure": "search"}, "null_procedure"),
        ({"scaling": "none"}, "scaling"),
        ({"consensus_check_rows": 100_000}, "fit consensus_budget_bytes"),
    ],
)
def test_config_rejects_invalid_values(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Config(**kwargs)


@pytest.mark.parametrize("space", [Space(n_neighbors=(20, 100)), Space(min_cluster_size=(10, 50))])
def test_config_accepts_bounds_that_exclude_the_default(space: Space) -> None:
    """User bounds override the library default; trial 0 becomes the default clipped into them."""
    Config(space=space)


def test_prepare_rejects_nonfinite_rows_and_aligns_errors() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 3))
    x[5, 1] = np.nan
    x[9, 0] = np.inf
    errors = np.abs(rng.normal(size=(80, 3)))
    kept_x, kept, kept_err, n_input = _prepare(x, errors)
    assert n_input == 80
    assert kept_x.shape == (78, 3)
    assert 5 not in kept
    assert 9 not in kept
    assert kept_err is not None
    assert kept_err.shape == (78, 3)
    with pytest.raises(ValueError, match="same shape"):
        _prepare(x, errors[:, :2])
    with pytest.raises(ValueError, match="non-negative"):
        _prepare(x, -errors)


def test_recover_rejects_two_measurement_models() -> None:
    x = np.random.default_rng(0).normal(size=(60, 3))
    with pytest.raises(ValueError, match="not both"):
        recover(x, X_err=np.ones_like(x), measurement_sampler=lambda r, _i, _g: r)


def test_recover_checks_cuml_before_any_stage_when_backends_are_compared(monkeypatch) -> None:
    def search(*_args, **_kwargs):
        pytest.fail("the search must not start")

    monkeypatch.setattr(_engine, "_cuml_probe", lambda: (False, "ImportError: no cuml"))
    monkeypatch.setattr(api, "_search", search)
    x = np.random.default_rng(0).normal(size=(60, 3))
    with pytest.raises(ImportError, match="no cuml"):
        recover(x, config=Config(check_backends=True), engine="cpu")


@pytest.mark.parametrize(("world_fraction", "min_cluster_size"), [(0.5, 10), (0.8, 16)])
def test_worlds_refit_the_selection_rescaled_to_the_world_rows(
    monkeypatch, world_fraction: float, min_cluster_size: int
) -> None:
    n = 100
    selected = Params(15, 0.1, 2, 10, 5, "eom")
    seen: dict[str, Params] = {}

    def search(*_args, **_kwargs):
        return SimpleNamespace(selected=selected, plan=SimpleNamespace(m=n // 2), n_fits=0)

    def fit(x, *_args, **_kwargs):
        return SimpleNamespace(labels=np.zeros(x.shape[0], dtype=np.int32))

    def worlds(_x, params, *_args, **_kwargs):
        seen["params"] = params
        raise _StageReachedError

    monkeypatch.setattr(api, "_search", search)
    monkeypatch.setattr(api, "fit_pipeline", fit)
    monkeypatch.setattr(api, "run_worlds", worlds)
    x = np.random.default_rng(0).normal(size=(n, 3))
    config = Config(resample_fraction=0.5, world_fraction=world_fraction)
    with pytest.raises(_StageReachedError):
        recover(x, config=config, engine="cpu")
    assert seen["params"] == rescale_params(selected, n // 2, round(world_fraction * n))
    assert seen["params"].min_cluster_size == min_cluster_size


def test_worlds_report_a_fraction_that_draws_no_rows(monkeypatch) -> None:
    selected = Params(15, 0.1, 2, 10, 5, "eom")

    def search(*_args, **_kwargs):
        return SimpleNamespace(selected=selected, plan=SimpleNamespace(m=48), n_fits=0)

    def fit(x, *_args, **_kwargs):
        return SimpleNamespace(labels=np.zeros(x.shape[0], dtype=np.int32))

    monkeypatch.setattr(api, "_search", search)
    monkeypatch.setattr(api, "fit_pipeline", fit)
    x = np.random.default_rng(0).normal(size=(60, 3))
    with pytest.raises(ValueError, match="draws no rows"):
        recover(x, config=Config(world_fraction=0.008), engine="cpu")


@pytest.mark.parametrize(
    ("engine", "m", "replay_jobs", "null_jobs"),
    [("cpu", 800, 1, 4), ("cpu", 20_001, 4, 1), ("cuml", 800, 4, 1)],
)
def test_null_realisations_run_concurrently_only_where_replays_are_unchanged(
    monkeypatch, engine: str, m: int, replay_jobs: int, null_jobs: int
) -> None:
    seen: dict[str, int] = {}

    def search(_x, *, n_jobs, **_kwargs):
        seen["replay"] = n_jobs
        return SimpleNamespace(exploration_max=0.1, n_fits=1)

    def null_test(x, _observed, statistic, *, n_jobs, **_kwargs):
        seen["null"] = n_jobs
        return statistic(x)

    monkeypatch.setattr(api, "_search", search)
    monkeypatch.setattr("starfold.nulls.null_test", null_test)
    searched = SimpleNamespace(plan=SimpleNamespace(m=m), exploration_max=0.5)
    x = np.zeros((10, 2))
    _null_stage(x, searched, config=Config(), random_state=0, engine=engine, n_jobs=4)
    assert seen == {"replay": replay_jobs, "null": null_jobs}


@pytest.mark.slow
def test_recover_end_to_end_small(two_blobs) -> None:
    config = Config(
        n_trials=3,
        n_explore=2,
        n_resamples=2,
        patience=1,
        n_worlds=3,
        null_families=(),
        n_rescore=1,
        n_complete_pruned=0,
    )
    result = recover(two_blobs, config=config, random_state=0, engine="cpu")
    assert result.labels.shape == (two_blobs.shape[0],)
    assert result.default_params == default_params()
    assert result.search.trials[0].phase == "default"
    assert result.worlds.world_labels.shape == (3, two_blobs.shape[0])
    assert "null_not_run" in result.flags
    assert result.provenance["engine_resolved"] == "cpu"
    assert result.provenance["total_fits"] > 0
    sampled = result.worlds.n_sampled > 0
    total = result.worlds.support[sampled].sum(axis=1) + result.worlds.unresolved[sampled]
    assert np.allclose(total, 1.0, atol=1e-4)


@pytest.mark.slow
def test_null_test_evidence_does_not_depend_on_n_jobs(two_blobs) -> None:
    """Concurrent single-threaded replays, a short final batch included, match the serial run."""
    x = two_blobs[::2]
    config = Config(
        n_trials=3,
        n_explore=2,
        n_resamples=2,
        alpha=0.5,
        n_null_max=3,
        null_exceedances=1,
    )
    searched = run_search(x, config=config, random_state=0, engine="cpu")
    serial = api.null_test(x, searched, config=config, random_state=0, engine="cpu", n_jobs=1)
    concurrent = api.null_test(x, searched, config=config, random_state=0, engine="cpu", n_jobs=2)
    assert serial.skipped_reason is None
    assert sum(serial.n_realisations) > 0
    assert replace(concurrent, seconds=0.0) == replace(serial, seconds=0.0)
