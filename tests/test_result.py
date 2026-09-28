import hashlib
import json
import sys
import types

import numpy as np
import pytest

from starfold.backend import BackendAgreement, Params, default_params, rescale_params
from starfold.consensus import ConsensusCheck
from starfold.nulls import NullEvidence
from starfold.prep import make_plan
from starfold.result import (
    FLAG_VOCABULARY,
    Result,
    derive_flags,
    load,
    matrix_fingerprint,
    peak_rss_mb,
    save,
)
from starfold.search import Candidate, SearchResult, TrialRecord
from starfold.worlds import WorldEvidence


def _params() -> Params:
    return Params(
        n_neighbors=20,
        min_dist=0.0,
        n_components=3,
        min_cluster_size=8,
        min_samples=4,
        selection_method="leaf",
    )


def _trial(number: int, params: Params, score: float, phase: str) -> TrialRecord:
    return TrialRecord(
        number=number,
        params=params,
        state="complete",
        score=score,
        raw_score=score,
        null_score=0.0,
        partial_score=None,
        stability=score,
        coverage=0.9,
        modularity=0.5,
        fit_counts=(2, 2, 2),
        fit_noise=(0.1, 0.1, 0.1),
        seconds=0.5,
        error=None,
        phase=phase,  # type: ignore[arg-type]
    )


def _search(n: int, *, stop_reason: str = "patience") -> SearchResult:
    plan = make_plan(n, n_resamples=3, fraction=0.8, seed=0)
    default = default_params()
    chosen = _params()
    trials = (
        _trial(0, default, 0.4, "default"),
        _trial(1, chosen, 0.7, "exploration"),
    )
    candidates = (
        Candidate(chosen, 1, 0.7, 0.65, (0.6, 0.7, 0.65), (2, 2, 2), (0.1, 0.2), 0.5),
        Candidate(default, 0, 0.4, 0.42, (0.4, 0.44, 0.42), (2, 2, 2), (0.3, 0.3), 0.12),
    )
    return SearchResult(
        selected=chosen,
        trials=trials,
        candidates=candidates,
        score_selected=0.65,
        score_default=0.42,
        gap_selected=0.5,
        gap_default=0.12,
        paired_differences=(0.2, 0.26, 0.23),
        beats_default_under_objective=True,
        selected_is_default=False,
        exploration_max=0.7,
        stop_reason=stop_reason,  # type: ignore[arg-type]
        n_completed=2,
        n_pruned=0,
        n_completed_after_prune=0,
        n_fits=12,
        seconds=3.0,
        plan=plan,
    )


def _worlds(n: int, labels: np.ndarray, n_worlds: int = 4) -> WorldEvidence:
    rng = np.random.default_rng(1)
    groups = int(labels.max()) + 1
    world_labels = np.full((n_worlds, n), -2, dtype=np.int16)
    for w in range(n_worlds):
        rows = np.sort(rng.choice(n, size=int(0.8 * n), replace=False))
        world_labels[w, rows] = labels[rows]
    n_sampled = (world_labels != -2).sum(axis=0).astype(np.uint16)
    support = np.zeros((n, groups), dtype=np.float32)
    unresolved = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if n_sampled[i] == 0:
            support[i] = np.nan
            unresolved[i] = np.nan
        elif labels[i] >= 0:
            support[i, labels[i]] = 1.0
        else:
            unresolved[i] = 1.0
    own = np.where(labels >= 0, 1.0, np.nan).astype(np.float32)
    return WorldEvidence(
        world_labels=world_labels,
        n_sampled=n_sampled,
        support=support,
        unresolved=unresolved,
        own_support=own,
        unresolved_mass=float(np.nanmean(unresolved)),
        group_recurrence=np.ones(groups, dtype=np.float32),
        group_recurrence_band=np.ones((groups, 2), dtype=np.float32),
        count_values=np.full(n_worlds, groups, dtype=np.int16),
        count_interval=(groups, groups),
        count_reference=groups,
        property_median=np.zeros((groups, 3)),
        property_band=np.zeros((groups, 3, 2)),
        property_propagated=False,
        world_agreement=np.ones(n_worlds, dtype=np.float32),
        world_seeds=tuple(range(n_worlds)),
        world_errors=(),
        n_fits=n_worlds,
        seconds=1.0,
    )


def _null(*, supported: bool = True, skipped: str | None = None) -> NullEvidence:
    return NullEvidence(
        families=("gaussian_copula",),
        statistic_name="exploration_design_max",
        observed=0.7,
        values=((0.1, 0.2),) if skipped is None else ((),),
        n_realisations=(2,) if skipped is None else (0,),
        exceedances=(0,),
        p_values=(1 / 3,) if skipped is None else (1.0,),
        p_intervals=((0.0, 0.84),) if skipped is None else ((0.0, 1.0),),
        p_value=0.04 if supported and skipped is None else 1.0,
        alpha=0.05,
        structure_supported=supported and skipped is None,
        stopped_early=(False,),
        skipped_reason=skipped,
        n_fits=0 if skipped else 8,
        seconds=0.1,
    )


def _result(n: int = 60) -> Result:
    rng = np.random.default_rng(0)
    labels = np.where(np.arange(n) % 3 == 2, -1, np.arange(n) % 2).astype(np.int32)
    x = rng.normal(size=(n, 3))
    search = _search(n)
    worlds = _worlds(n, labels)
    consensus = ConsensusCheck(
        labels=labels.copy(),
        rows=np.arange(n, dtype=np.intp),
        agreement=0.95,
        subsampled=False,
        n_rows=n,
        bytes_used=16 * n * n,
        within_association=0.9,
        between_association=0.1,
        contrast=0.8,
    )
    null = _null()
    flags = derive_flags(
        labels=labels,
        search=search,
        worlds=worlds,
        consensus=consensus,
        null=null,
        backends=None,
        rows_rejected=0,
        n_jobs_forced=False,
    )
    provenance = {
        "n_input_rows": n,
        "n_kept_rows": n,
        "n_jobs_forced_to_one_on_cuml": False,
        "input_fingerprint": matrix_fingerprint(x),
        "stage_seconds": {"total": 1.0},
    }
    return Result(
        labels=labels,
        strengths=np.full(n, 0.9, dtype=np.float32),
        default_labels=np.zeros(n, dtype=np.int32),
        kept_rows=np.arange(n, dtype=np.intp),
        params=_params(),
        deployed_params=rescale_params(_params(), 48, 60),
        default_params=default_params(),
        search=search,
        worlds=worlds,
        consensus=consensus,
        null=null,
        backends=None,
        flags=flags,
        provenance=provenance,
    )


def test_flag_vocabulary_is_frozen_and_ordered() -> None:
    result = _result()
    assert set(result.flags) <= set(FLAG_VOCABULARY)
    assert list(result.flags) == [f for f in FLAG_VOCABULARY if f in result.flags]
    assert "search_stopped_early" in result.flags
    assert "no_structure_supported" not in result.flags
    assert result.null.structure_supported is True
    assert not hasattr(result, "p_value")
    assert not hasattr(result, "structure_supported")
    assert result.n_groups == 2


def test_flags_react_to_evidence() -> None:
    result = _result()
    flags = derive_flags(
        labels=np.full(result.labels.size, -1, dtype=np.int32),
        search=_search(result.labels.size, stop_reason="budget"),
        worlds=result.worlds,
        consensus=ConsensusCheck(
            labels=result.consensus.labels,
            rows=result.consensus.rows,
            agreement=0.2,
            subsampled=True,
            n_rows=10,
            bytes_used=1600,
            within_association=0.4,
            between_association=0.3,
            contrast=0.1,
        ),
        null=_null(supported=False),
        backends=BackendAgreement(
            (0.5,), 0.5, (0.8,), (0.625,), (0.7,), (0.65,), (0.769,), 0.769, (2,), (3,)
        ),
        rows_rejected=3,
        n_jobs_forced=True,
    )
    for expected in (
        "search_budget_exhausted",
        "no_groups_in_reference",
        "consensus_subsampled",
        "consensus_disagrees",
        "no_structure_supported",
        "backend_disagreement",
        "rows_rejected_nonfinite",
        "n_jobs_forced_to_one_on_cuml",
    ):
        assert expected in flags
    skipped = derive_flags(
        labels=result.labels,
        search=result.search,
        worlds=result.worlds,
        consensus=result.consensus,
        null=_null(skipped="null_not_run"),
        backends=None,
        rows_rejected=0,
        n_jobs_forced=False,
    )
    assert "null_not_run" in skipped
    assert "no_structure_supported" not in skipped


def test_save_and_load_round_trip(tmp_path) -> None:
    result = _result()
    directory = save(result, tmp_path / "run")
    assert (directory / "manifest.json").exists()
    loaded = load(directory)
    assert np.array_equal(loaded.labels, result.labels)
    assert np.array_equal(loaded.worlds.world_labels, result.worlds.world_labels)
    assert np.allclose(loaded.worlds.support, result.worlds.support, equal_nan=True)
    assert loaded.params == result.params
    assert loaded.search.selected == result.search.selected
    assert loaded.search.trials == result.search.trials
    assert loaded.search.candidates == result.search.candidates
    assert loaded.search.plan.seeds == result.search.plan.seeds
    assert all(
        np.array_equal(a, b)
        for a, b in zip(loaded.search.plan.indices, result.search.plan.indices, strict=True)
    )
    assert loaded.null == result.null
    assert loaded.flags == result.flags
    assert loaded.provenance["input_fingerprint"] == result.provenance["input_fingerprint"]


def test_load_detects_tampering(tmp_path) -> None:
    result = _result()
    directory = save(result, tmp_path / "run")
    payload = json.loads((directory / "result.json").read_text())
    payload["search"]["score_selected"] = 0.99
    (directory / "result.json").write_text(json.dumps(payload, indent=1, sort_keys=True))
    with pytest.raises(ValueError, match="Checksum mismatch"):
        load(directory)


def test_load_reads_absent_association_fields_as_nan(tmp_path) -> None:
    result = _result()
    directory = save(result, tmp_path / "run")
    payload = json.loads((directory / "result.json").read_text())
    for key in ("within_association", "between_association", "contrast"):
        del payload["consensus"][key]
    (directory / "result.json").write_text(json.dumps(payload, indent=1, sort_keys=True))
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["result.json"] = hashlib.sha256((directory / "result.json").read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=1))
    loaded = load(directory)
    assert np.isnan(loaded.consensus.contrast)
    assert np.isnan(loaded.consensus.within_association)
    assert loaded.consensus.agreement == result.consensus.agreement


def test_load_rejects_broken_support_identity(tmp_path) -> None:
    result = _result()
    directory = save(result, tmp_path / "run")
    with np.load(directory / "arrays.npz", allow_pickle=False) as arrays:
        content = {name: arrays[name] for name in arrays.files}
    content["support"] = content["support"] * 0.5
    np.savez_compressed(directory / "arrays.npz", **content)
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["arrays.npz"] = hashlib.sha256((directory / "arrays.npz").read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=1))
    with pytest.raises(ValueError, match="Support identity"):
        load(directory)


def test_pruned_trial_components_round_trip_as_nan(tmp_path) -> None:
    result = _result()
    pruned = TrialRecord(
        number=2,
        params=_params(),
        state="pruned",
        score=0.3,
        raw_score=0.3,
        null_score=float("nan"),
        partial_score=0.3,
        stability=float("nan"),
        coverage=float("nan"),
        modularity=float("nan"),
        fit_counts=(2, 2),
        fit_noise=(0.1, 0.1),
        seconds=0.2,
        error=None,
        phase="convergence",
    )
    search = SearchResult(**{**result.search.__dict__, "trials": (*result.search.trials, pruned)})
    amended = Result(**{**result.__dict__, "search": search})
    loaded = load(save(amended, tmp_path / "run"))
    back = loaded.search.trials[-1]
    assert back.state == "pruned"
    assert back.partial_score == 0.3
    assert np.isnan(back.stability)
    assert np.isnan(back.coverage)
    assert np.isnan(back.modularity)


def test_peak_rss_is_nan_without_resource_and_bytes_on_macos(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "resource", None)
    assert np.isnan(peak_rss_mb())
    usage = types.SimpleNamespace(ru_maxrss=3 * 1024**2)
    fake = types.SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda _who: usage)
    monkeypatch.setitem(sys.modules, "resource", fake)
    monkeypatch.setattr(sys, "platform", "darwin")
    assert peak_rss_mb() == 3.0
    monkeypatch.setattr(sys, "platform", "linux")
    assert peak_rss_mb() == 3.0 * 1024


def test_matrix_fingerprint_depends_on_values_and_shape() -> None:
    x = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert matrix_fingerprint(x) == matrix_fingerprint(x.copy())
    assert matrix_fingerprint(x) != matrix_fingerprint(x.reshape(4, 3))
    assert matrix_fingerprint(x) != matrix_fingerprint(x + 1.0)


def test_default_within_noise_is_silent_when_the_default_is_selected() -> None:
    result = _result()
    search = SearchResult(
        **{
            **result.search.__dict__,
            "selected": default_params(),
            "selected_is_default": True,
            "score_selected": 0.42,
            "gap_selected": 0.12,
            "paired_differences": (0.0, 0.0, 0.0),
        }
    )
    flags = derive_flags(
        labels=result.labels,
        search=search,
        worlds=result.worlds,
        consensus=result.consensus,
        null=result.null,
        backends=None,
        rows_rejected=0,
        n_jobs_forced=False,
    )
    assert "selected_is_default" in flags
    assert "default_within_noise" not in flags
