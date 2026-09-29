"""Result contract, flag derivation, provenance, and checksummed persistence.

A result is persisted as one directory holding ``result.json`` (scalars,
strings, the trial table, the null values, provenance, flags),
``arrays.npz`` (every array, written and read with ``allow_pickle=False``),
and ``manifest.json`` (SHA-256 of both files). ``load`` verifies the
checksums, rebuilds every dataclass, re-derives the flags from the arrays
instead of trusting the stored list, and checks the support identity
``sum_g support[i, g] + unresolved[i] = 1`` on every sampled row.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from starfold._engine import library_versions
from starfold.backend import BackendAgreement, Params
from starfold.consensus import ConsensusCheck
from starfold.nulls import NullEvidence
from starfold.prep import ResamplePlan
from starfold.search import Candidate, SearchResult, TrialRecord
from starfold.worlds import WorldEvidence

if TYPE_CHECKING:
    from numpy.typing import NDArray

FLAG_VOCABULARY: tuple[str, ...] = (
    "search_stopped_early",
    "search_budget_exhausted",
    "selected_is_default",
    "default_within_noise",
    "no_groups_in_reference",
    "high_unresolved_mass",
    "count_unstable",
    "world_fit_failures",
    "consensus_subsampled",
    "consensus_disagrees",
    "null_not_run",
    "observed_statistic_zero",
    "no_structure_supported",
    "backend_disagreement",
    "rows_rejected_nonfinite",
    "n_jobs_forced_to_one_on_cuml",
)

_SUPPORT_IDENTITY_TOLERANCE = 1e-4


@dataclass(frozen=True)
class Result:
    """Complete outcome of :func:`starfold.recover`.

    Attributes
    ----------
    labels : ndarray of int32, shape (n_kept,)
        Emitted partition, one full-data fit of ``params``, with -1 for noise.
    strengths : ndarray of float32, shape (n_kept,)
        HDBSCAN within-fit strengths of that fit, not posteriors.
    default_labels : ndarray of int32, shape (n_kept,)
        One full-data fit of the library defaults, the external comparator.
    kept_rows : ndarray of intp, shape (n_kept,)
        Row indices of the input matrix that were finite and used.
    params : Params
        Configuration selected by the search, validated on resample rows.
    deployed_params : Params
        Configuration that produced ``labels``: ``params`` with the minimum
        cluster mass rescaled from the resample row count to the full row
        count, equal to ``params`` when the two counts agree.
    default_params : Params
        The library defaults, fitted at full size as the external comparator.
    search : SearchResult
        Trial table, candidates, fresh scores and the beats-default statement.
    worlds : WorldEvidence
        Recurrence support, unresolved mass, group recurrence, count and
        property stability intervals.
    consensus : ConsensusCheck
        Bounded label-invariant co-association check on the reference.
    null : NullEvidence
        Matched null calibration.
    backends : BackendAgreement or None
        Backend agreement when ``compare_backends`` ran.
    flags : tuple of str
        Machine-readable flags from :data:`FLAG_VOCABULARY`.
    provenance : dict
        Engines, versions, configuration, seeds, budgets, fit counts, wall
        time per stage, peak memory and the input fingerprint.
    """

    labels: NDArray[np.int32]
    strengths: NDArray[np.float32]
    default_labels: NDArray[np.int32]
    kept_rows: NDArray[np.intp]
    params: Params
    deployed_params: Params
    default_params: Params
    search: SearchResult
    worlds: WorldEvidence
    consensus: ConsensusCheck
    null: NullEvidence
    backends: BackendAgreement | None
    flags: tuple[str, ...]
    provenance: dict[str, Any]

    @property
    def n_groups(self) -> int:
        """Number of groups in the emitted partition."""
        return int(np.unique(self.labels[self.labels >= 0]).size)

    @property
    def support(self) -> NDArray[np.float32]:
        """Per-row recurrence support over reference groups, shape (n_kept, G)."""
        return self.worlds.support

    @property
    def unresolved_mass(self) -> float:
        """Mean unresolved mass over sampled rows."""
        return float(self.worlds.unresolved_mass)


def matrix_fingerprint(x: NDArray[np.floating[Any]]) -> str:
    """SHA-256 of the C-contiguous float64 bytes and the shape of a matrix."""
    array = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def peak_rss_mb() -> float:
    """Peak resident set size of this process in megabytes, NaN where unknown.

    ``ru_maxrss`` is in kilobytes on Linux and in bytes on macOS. The
    ``resource`` module does not exist on Windows.
    """
    try:
        import resource  # noqa: PLC0415 - absent on Windows
    except ImportError:
        return float("nan")
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak / 1024.0**2 if sys.platform == "darwin" else peak / 1024.0


def derive_flags(
    *,
    labels: NDArray[np.int32],
    search: SearchResult,
    worlds: WorldEvidence,
    consensus: ConsensusCheck,
    null: NullEvidence,
    backends: BackendAgreement | None,
    rows_rejected: int,
    n_jobs_forced: bool,
) -> tuple[str, ...]:
    """Derive the flag tuple from arrays and evidence, never from stored text.

    Returns
    -------
    tuple of str
        Flags in :data:`FLAG_VOCABULARY` order.
    """
    differences = np.asarray(search.paired_differences, dtype=np.float64)
    within_noise = False
    if differences.size >= 2 and not search.selected_is_default:
        # The paired differences are zero by construction when the default is
        # itself selected, so the comparison is only informative otherwise.
        noise = 2.0 * float(np.std(differences, ddof=1)) / np.sqrt(differences.size)
        within_noise = abs(search.score_selected - search.score_default) <= noise
    counts = np.asarray(worlds.count_values, dtype=np.int64)
    count_unstable = bool(counts.size) and float(np.mean(counts == worlds.count_reference)) < 0.5
    raised: dict[str, bool] = {
        "search_stopped_early": search.stop_reason == "patience",
        "search_budget_exhausted": search.stop_reason == "budget",
        "selected_is_default": bool(search.selected_is_default),
        "default_within_noise": within_noise,
        "no_groups_in_reference": not bool(np.any(labels >= 0)),
        "high_unresolved_mass": worlds.unresolved_mass > 0.5,
        "count_unstable": count_unstable,
        "world_fit_failures": any(bool(error) for error in worlds.world_errors),
        "consensus_subsampled": bool(consensus.subsampled),
        "consensus_disagrees": consensus.agreement < 0.5,
        "null_not_run": null.skipped_reason == "null_not_run",
        "observed_statistic_zero": null.skipped_reason == "observed_statistic_zero",
        "no_structure_supported": null.skipped_reason is None and not null.structure_supported,
        "backend_disagreement": backends is not None and backends.mean_normalised < 0.8,
        "rows_rejected_nonfinite": rows_rejected > 0,
        "n_jobs_forced_to_one_on_cuml": bool(n_jobs_forced),
    }
    return tuple(flag for flag in FLAG_VOCABULARY if raised[flag])


def build_provenance(
    *,
    x: NDArray[np.float64],
    config: Any,  # noqa: ANN401 - any frozen configuration dataclass
    random_state: int,
    engine_requested: str,
    engine_resolved: str,
    n_jobs: int,
    n_jobs_forced: bool,
    n_input_rows: int,
    stage_seconds: dict[str, float],
    stage_fits: dict[str, int],
) -> dict[str, Any]:
    """Assemble the provenance record of one run, as listed in ``Result.provenance``."""
    return {
        "starfold_version": _package_version(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libraries": library_versions(),
        "engine_requested": engine_requested,
        "engine_resolved": engine_resolved,
        "n_jobs": int(n_jobs),
        "n_jobs_forced_to_one_on_cuml": bool(n_jobs_forced),
        "random_state": int(random_state),
        "config": _jsonable(config),
        "n_input_rows": int(n_input_rows),
        "n_kept_rows": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "input_fingerprint": matrix_fingerprint(x),
        "stage_seconds": {k: float(v) for k, v in stage_seconds.items()},
        "stage_fits": {k: int(v) for k, v in stage_fits.items()},
        "total_fits": int(sum(stage_fits.values())),
        "peak_rss_mb": peak_rss_mb(),
    }


def _package_version() -> str:
    """Return ``starfold.__version__``."""
    from starfold import __version__  # noqa: PLC0415 - avoid an import cycle

    return str(__version__)


def _jsonable(value: Any) -> Any:  # noqa: ANN401, PLR0911 - recursive normalisation
    """Convert dataclasses, tuples, numpy scalars and arrays to JSON types."""
    if hasattr(value, "__dataclass_fields__"):
        return {name: _jsonable(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _search_payload(search: SearchResult) -> dict[str, Any]:
    """Scalar and table part of a search result (the resample rows go to the arrays)."""
    payload = {
        name: _jsonable(getattr(search, name))
        for name in search.__dataclass_fields__
        if name != "plan"
    }
    payload["plan"] = {
        "n": int(search.plan.n),
        "fraction": float(search.plan.fraction),
        "seed": int(search.plan.seed),
        "seeds": [int(s) for s in search.plan.seeds],
    }
    return payload


def _world_scalars(worlds: WorldEvidence) -> dict[str, Any]:
    """Scalar part of world evidence."""
    return {
        "unresolved_mass": float(worlds.unresolved_mass),
        "count_interval": [int(worlds.count_interval[0]), int(worlds.count_interval[1])],
        "count_reference": int(worlds.count_reference),
        "property_propagated": bool(worlds.property_propagated),
        "world_seeds": [int(s) for s in worlds.world_seeds],
        "world_errors": list(worlds.world_errors),
        "n_fits": int(worlds.n_fits),
        "seconds": float(worlds.seconds),
    }


def _consensus_scalars(consensus: ConsensusCheck) -> dict[str, Any]:
    """Scalar part of the consensus check."""
    return {
        "agreement": float(consensus.agreement),
        "subsampled": bool(consensus.subsampled),
        "n_rows": int(consensus.n_rows),
        "bytes_used": int(consensus.bytes_used),
        "within_association": _jsonable(consensus.within_association),
        "between_association": _jsonable(consensus.between_association),
        "contrast": _jsonable(consensus.contrast),
    }


def _arrays(result: Result) -> dict[str, NDArray[Any]]:
    """Every array of a result, keyed by its persisted name."""
    w = result.worlds
    return {
        "labels": result.labels,
        "strengths": result.strengths,
        "default_labels": result.default_labels,
        "kept_rows": result.kept_rows,
        "plan_indices": np.stack(result.search.plan.indices).astype(np.int64),
        "world_labels": w.world_labels,
        "n_sampled": w.n_sampled,
        "support": w.support,
        "unresolved": w.unresolved,
        "own_support": w.own_support,
        "group_recurrence": w.group_recurrence,
        "group_recurrence_band": w.group_recurrence_band,
        "count_values": w.count_values,
        "property_median": w.property_median,
        "property_band": w.property_band,
        "world_agreement": w.world_agreement,
        "consensus_labels": result.consensus.labels,
        "consensus_rows": result.consensus.rows,
    }


def _sha256(path: Path) -> str:
    """SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(result: Result, path: str | Path) -> Path:
    """Persist a result as a checksummed directory.

    Parameters
    ----------
    result : Result
        Result to persist.
    path : str or Path
        Target directory, created if absent. Existing files are overwritten.

    Returns
    -------
    Path
        The directory written.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": "starfold-result-2",
        "params": result.params.to_dict(),
        "deployed_params": result.deployed_params.to_dict(),
        "default_params": result.default_params.to_dict(),
        "search": _search_payload(result.search),
        "worlds": _world_scalars(result.worlds),
        "consensus": _consensus_scalars(result.consensus),
        "null": _jsonable(result.null),
        "backends": _jsonable(result.backends),
        "flags": list(result.flags),
        "provenance": _jsonable(result.provenance),
    }
    json_path = directory / "result.json"
    npz_path = directory / "arrays.npz"
    json_path.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    np.savez_compressed(npz_path, allow_pickle=False, **_arrays(result))
    manifest = {"result.json": _sha256(json_path), "arrays.npz": _sha256(npz_path)}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return directory


def _float_or_nan(value: object) -> float:
    """Read a missing component back as NaN, which JSON cannot hold."""
    return float("nan") if value is None else float(value)  # type: ignore[arg-type]


def _trial(record: dict[str, Any]) -> TrialRecord:
    """Rebuild a trial record."""
    return TrialRecord(
        number=int(record["number"]),
        params=Params.from_dict(record["params"]),
        state=record["state"],
        score=float(record["score"]),
        raw_score=_float_or_nan(record["raw_score"]),
        null_score=_float_or_nan(record["null_score"]),
        partial_score=None if record["partial_score"] is None else float(record["partial_score"]),
        stability=_float_or_nan(record["stability"]),
        coverage=_float_or_nan(record["coverage"]),
        modularity=_float_or_nan(record["modularity"]),
        fit_counts=tuple(int(v) for v in record["fit_counts"]),
        fit_noise=tuple(float(v) for v in record["fit_noise"]),
        seconds=float(record["seconds"]),
        error=record["error"],
        phase=record["phase"],
    )


def _candidate(record: dict[str, Any]) -> Candidate:
    """Rebuild a candidate."""
    return Candidate(
        params=Params.from_dict(record["params"]),
        trial_number=int(record["trial_number"]),
        score=float(record["score"]),
        fresh_score=float(record["fresh_score"]),
        fresh_pair_scores=tuple(float(v) for v in record["fresh_pair_scores"]),
        fit_counts=tuple(int(v) for v in record["fit_counts"]),
        null_scores=tuple(float(v) for v in record["null_scores"]),
        gap=float(record["gap"]),
    )


def _search(payload: dict[str, Any], plan_indices: NDArray[np.int64]) -> SearchResult:
    """Rebuild a search result."""
    plan_payload = payload["plan"]
    plan = ResamplePlan(
        n=int(plan_payload["n"]),
        fraction=float(plan_payload["fraction"]),
        seed=int(plan_payload["seed"]),
        indices=tuple(np.asarray(row, dtype=np.intp) for row in plan_indices),
        seeds=tuple(int(s) for s in plan_payload["seeds"]),
    )
    return SearchResult(
        selected=Params.from_dict(payload["selected"]),
        trials=tuple(_trial(t) for t in payload["trials"]),
        candidates=tuple(_candidate(c) for c in payload["candidates"]),
        score_selected=float(payload["score_selected"]),
        score_default=float(payload["score_default"]),
        gap_selected=float(payload["gap_selected"]),
        gap_default=float(payload["gap_default"]),
        paired_differences=tuple(float(v) for v in payload["paired_differences"]),
        beats_default_under_objective=bool(payload["beats_default_under_objective"]),
        selected_is_default=bool(payload["selected_is_default"]),
        exploration_max=float(payload["exploration_max"]),
        stop_reason=payload["stop_reason"],
        n_completed=int(payload["n_completed"]),
        n_pruned=int(payload["n_pruned"]),
        n_completed_after_prune=int(payload["n_completed_after_prune"]),
        n_fits=int(payload["n_fits"]),
        seconds=float(payload["seconds"]),
        plan=plan,
    )


def _worlds(scalars: dict[str, Any], arrays: Any) -> WorldEvidence:  # noqa: ANN401 - npz file
    """Rebuild world evidence."""
    return WorldEvidence(
        world_labels=arrays["world_labels"].astype(np.int16),
        n_sampled=arrays["n_sampled"].astype(np.uint16),
        support=arrays["support"].astype(np.float32),
        unresolved=arrays["unresolved"].astype(np.float32),
        own_support=arrays["own_support"].astype(np.float32),
        unresolved_mass=float(scalars["unresolved_mass"]),
        group_recurrence=arrays["group_recurrence"].astype(np.float32),
        group_recurrence_band=arrays["group_recurrence_band"].astype(np.float32),
        count_values=arrays["count_values"].astype(np.int16),
        count_interval=(int(scalars["count_interval"][0]), int(scalars["count_interval"][1])),
        count_reference=int(scalars["count_reference"]),
        property_median=arrays["property_median"].astype(np.float64),
        property_band=arrays["property_band"].astype(np.float64),
        property_propagated=bool(scalars["property_propagated"]),
        world_agreement=arrays["world_agreement"].astype(np.float32),
        world_seeds=tuple(int(s) for s in scalars["world_seeds"]),
        world_errors=tuple(str(e) for e in scalars["world_errors"]),
        n_fits=int(scalars["n_fits"]),
        seconds=float(scalars["seconds"]),
    )


def _null(payload: dict[str, Any]) -> NullEvidence:
    """Rebuild null evidence."""
    return NullEvidence(
        families=tuple(str(f) for f in payload["families"]),
        statistic_name=str(payload["statistic_name"]),
        observed=float(payload["observed"]),
        values=tuple(tuple(float(v) for v in row) for row in payload["values"]),
        n_realisations=tuple(int(v) for v in payload["n_realisations"]),
        exceedances=tuple(int(v) for v in payload["exceedances"]),
        p_values=tuple(float(v) for v in payload["p_values"]),
        p_intervals=tuple((float(a), float(b)) for a, b in payload["p_intervals"]),
        p_value=float(payload["p_value"]),
        alpha=float(payload["alpha"]),
        structure_supported=bool(payload["structure_supported"]),
        stopped_early=tuple(bool(v) for v in payload["stopped_early"]),
        skipped_reason=payload["skipped_reason"],
        n_fits=int(payload["n_fits"]),
        seconds=float(payload["seconds"]),
    )


def _backends(payload: dict[str, Any] | None) -> BackendAgreement | None:
    """Rebuild backend agreement."""
    if payload is None:
        return None
    n_seeds = len(payload["agreement"])
    unknown = [float("nan")] * n_seeds
    return BackendAgreement(
        agreement=tuple(float(v) for v in payload["agreement"]),
        mean_agreement=float(payload["mean_agreement"]),
        ari=tuple(float(v) for v in payload.get("ari", unknown)),
        coverage=tuple(float(v) for v in payload.get("coverage", unknown)),
        assigned_cpu=tuple(float(v) for v in payload.get("assigned_cpu", unknown)),
        assigned_cuml=tuple(float(v) for v in payload.get("assigned_cuml", unknown)),
        normalised=tuple(float(v) for v in payload.get("normalised", unknown)),
        mean_normalised=_float_or_nan(payload.get("mean_normalised")),
        n_groups_cpu=tuple(int(v) for v in payload["n_groups_cpu"]),
        n_groups_cuml=tuple(int(v) for v in payload["n_groups_cuml"]),
    )


def _check_support_identity(worlds: WorldEvidence) -> None:
    """Raise if the support identity fails on any sampled row."""
    sampled = worlds.n_sampled > 0
    total = worlds.support[sampled].sum(axis=1) + worlds.unresolved[sampled]
    if sampled.any() and not np.allclose(total, 1.0, atol=_SUPPORT_IDENTITY_TOLERANCE):
        msg = "Support identity sum_g support + unresolved = 1 fails on a sampled row."
        raise ValueError(msg)


def load(path: str | Path) -> Result:
    """Load a persisted result, verifying checksums and re-deriving flags.

    Parameters
    ----------
    path : str or Path
        Directory written by :func:`save`.

    Returns
    -------
    Result
        The rebuilt result.

    Raises
    ------
    ValueError
        If a checksum does not match, the format is unknown, the support
        identity fails, or the stored flags differ from the re-derived ones.
    """
    directory = Path(path)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        actual = _sha256(directory / name)
        if actual != expected:
            msg = f"Checksum mismatch for {name}: expected {expected}, found {actual}."
            raise ValueError(msg)
    payload = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    if payload.get("format") != "starfold-result-2":
        msg = f"Unknown result format {payload.get('format')!r}."
        raise ValueError(msg)
    with np.load(directory / "arrays.npz", allow_pickle=False) as arrays:
        search = _search(payload["search"], arrays["plan_indices"])
        worlds = _worlds(payload["worlds"], arrays)
        consensus = ConsensusCheck(
            labels=arrays["consensus_labels"].astype(np.int32),
            rows=arrays["consensus_rows"].astype(np.intp),
            agreement=float(payload["consensus"]["agreement"]),
            subsampled=bool(payload["consensus"]["subsampled"]),
            n_rows=int(payload["consensus"]["n_rows"]),
            bytes_used=int(payload["consensus"]["bytes_used"]),
            within_association=_float_or_nan(payload["consensus"].get("within_association")),
            between_association=_float_or_nan(payload["consensus"].get("between_association")),
            contrast=_float_or_nan(payload["consensus"].get("contrast")),
        )
        labels = arrays["labels"].astype(np.int32)
        strengths = arrays["strengths"].astype(np.float32)
        default_labels = arrays["default_labels"].astype(np.int32)
        kept_rows = arrays["kept_rows"].astype(np.intp)
    _check_support_identity(worlds)
    null = _null(payload["null"])
    backends = _backends(payload["backends"])
    provenance = payload["provenance"]
    flags = derive_flags(
        labels=labels,
        search=search,
        worlds=worlds,
        consensus=consensus,
        null=null,
        backends=backends,
        rows_rejected=int(provenance["n_input_rows"]) - int(provenance["n_kept_rows"]),
        n_jobs_forced=bool(provenance["n_jobs_forced_to_one_on_cuml"]),
    )
    if list(flags) != list(payload["flags"]):
        msg = f"Stored flags {payload['flags']} differ from re-derived flags {list(flags)}."
        raise ValueError(msg)
    return Result(
        labels=labels,
        strengths=strengths,
        default_labels=default_labels,
        kept_rows=kept_rows,
        params=Params.from_dict(payload["params"]),
        deployed_params=Params.from_dict(payload["deployed_params"]),
        default_params=Params.from_dict(payload["default_params"]),
        search=search,
        worlds=worlds,
        consensus=consensus,
        null=null,
        backends=backends,
        flags=flags,
        provenance=provenance,
    )
