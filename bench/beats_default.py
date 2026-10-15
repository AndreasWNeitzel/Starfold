"""Compare the selected configuration with the library defaults on labelled data.

For each dataset the procedure runs with the labels hidden. Afterwards the
emitted partition, the consensus check, and the default configuration over
several seeds are scored against truth with the coverage-adjusted ARI (ARI on
jointly assigned rows times the assigned fraction, over rows with a truth
label) and the raw ARI. A recovered group counts as a mixture when its
majority truth class holds fewer than ``--mixture-threshold`` (default 0.8) of
its members. Spearman correlation between the trial score and the trial ARI
(one extra full-data fit per completed trial, bench only) measures whether the
objective tracks truth. The matched null is off by default; run
``bench.null_stage`` on the saved result, or pass ``--null gaussian_copula``.

Each record is written to ``<output>/<dataset>.json`` (default
``bench/results``) and the saved result to ``<results-dir>/<dataset>``
(default ``output/bench``).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

from bench import datasets
from starfold import Config, recover, save
from starfold._engine import derive_seed, require_resolved
from starfold.backend import default_params, fit_pipeline

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from starfold.result import Result


def coverage_adjusted_ari(truth: NDArray[np.int64], predicted: NDArray[np.int32]) -> float:
    """ARI on jointly assigned rows times the assigned fraction, over labelled rows."""
    labelled = truth >= 0
    assigned = labelled & (predicted >= 0)
    if assigned.sum() < 2 or labelled.sum() == 0:
        return 0.0
    ari = adjusted_rand_score(truth[assigned], predicted[assigned])
    return float(max(0.0, ari) * assigned.sum() / labelled.sum())


def raw_ari(truth: NDArray[np.int64], predicted: NDArray[np.int32]) -> float:
    """ARI over labelled rows with noise kept as its own label."""
    labelled = truth >= 0
    if labelled.sum() < 2:
        return 0.0
    return float(adjusted_rand_score(truth[labelled], predicted[labelled]))


def mixture_fraction(
    truth: NDArray[np.int64], predicted: NDArray[np.int32], threshold: float = 0.8
) -> tuple[float, int]:
    """Fraction of recovered groups whose majority truth class is below ``threshold``."""
    groups = np.unique(predicted[predicted >= 0])
    if groups.size == 0:
        return float("nan"), 0
    mixtures = 0
    for group in groups:
        members = truth[predicted == group]
        members = members[members >= 0]
        if members.size == 0:
            continue
        majority = np.bincount(members).max() / members.size
        mixtures += int(majority < threshold)
    return mixtures / groups.size, int(groups.size)


@dataclass(frozen=True)
class Scored:
    """Scores of one partition against truth."""

    coverage_ari: float
    raw_ari: float
    mixture_fraction: float
    n_groups: int
    assigned_fraction: float


def score(truth: NDArray[np.int64], predicted: NDArray[np.int32], threshold: float) -> Scored:
    """Score one partition."""
    mixture, groups = mixture_fraction(truth, predicted, threshold)
    return Scored(
        coverage_adjusted_ari(truth, predicted),
        raw_ari(truth, predicted),
        mixture,
        groups,
        float(np.mean(predicted >= 0)),
    )


def trial_spearman(
    result: Result,
    x: NDArray[np.float64],
    truth: NDArray[np.int64],
    *,
    engine: str,
    scaling: str,
    random_state: int,
) -> tuple[float, int, list[dict[str, object]]]:
    """Correlations between the trial scores and the full-data trial ARI.

    Every complete trial is fitted once on all rows so that its gap, its data
    score and its null score can be related to the recovery it would have
    emitted. The per-trial rows are returned so the correlation can be
    recomputed for any variant of the score without refitting. ``engine``
    must be resolved (``"cpu"`` or ``"cuml"``).
    """
    engine = require_resolved(engine)
    rows: list[dict[str, object]] = []
    for trial in result.search.trials:
        if trial.state != "complete":
            continue
        fit = fit_pipeline(
            x,
            trial.params,
            seed=derive_seed(random_state, 4),
            engine=engine,
            scaling=scaling,  # type: ignore[arg-type]
        )
        rows.append(
            {
                "number": trial.number,
                "gap": trial.score,
                "raw_score": trial.raw_score,
                "null_score": trial.null_score,
                "stability": trial.stability,
                "coverage": trial.coverage,
                "modularity": trial.modularity,
                "coverage_ari": coverage_adjusted_ari(truth, fit.labels),
                "n_groups": fit.n_groups,
                "params": trial.params.to_dict(),
            }
        )
    if len(rows) < 3:
        return float("nan"), len(rows), rows
    gaps = [float(r["gap"]) for r in rows]  # type: ignore[arg-type]
    aris = [float(r["coverage_ari"]) for r in rows]  # type: ignore[arg-type]
    return float(spearmanr(gaps, aris).statistic), len(rows), rows


def run_dataset(
    name: str,
    *,
    config: Config,
    engine: str,
    n_jobs: int,
    random_state: int,
    default_seeds: int,
    threshold: float,
    output: Path,
    results_dir: Path,
    spearman: bool,
) -> dict[str, object]:
    """Run the procedure on one dataset and score it afterwards.

    The bench's own fits use the engine ``recover`` resolved.
    """
    dataset = datasets.load(name)
    x = dataset.X
    truth = dataset.truth
    started = time.perf_counter()
    result = recover(x, config=config, random_state=random_state, engine=engine, n_jobs=n_jobs)  # type: ignore[arg-type]
    elapsed = time.perf_counter() - started
    save(result, results_dir / name)
    kept_truth = truth[result.kept_rows]
    resolved = require_resolved(str(result.provenance["engine_resolved"]))
    defaults = []
    for seed_index in range(default_seeds):
        fit = fit_pipeline(
            x[result.kept_rows],
            default_params(),
            seed=derive_seed(random_state, 4, 1, seed_index),
            engine=resolved,
            scaling=config.scaling,
        )
        defaults.append(asdict(score(kept_truth, fit.labels, threshold)))
    rho, n_scored, trial_rows = (
        trial_spearman(
            result,
            x[result.kept_rows],
            kept_truth,
            engine=resolved,
            scaling=config.scaling,
            random_state=random_state,
        )
        if spearman
        else (float("nan"), 0, [])
    )
    consensus_truth = kept_truth[result.consensus.rows]
    record: dict[str, object] = {
        "dataset": name,
        "n_rows": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "n_classes": dataset.n_classes,
        "selected": asdict(score(kept_truth, result.labels, threshold)),
        "default_single": asdict(score(kept_truth, result.default_labels, threshold)),
        "default_seeds": defaults,
        "default_median_coverage_ari": float(np.median([d["coverage_ari"] for d in defaults])),
        "consensus": asdict(score(consensus_truth, result.consensus.labels, threshold)),
        "consensus_subsampled": result.consensus.subsampled,
        "params": result.params.to_dict(),
        "score_selected": result.search.score_selected,
        "score_default": result.search.score_default,
        "beats_default_under_objective": result.search.beats_default_under_objective,
        "selected_is_default": result.search.selected_is_default,
        "stop_reason": result.search.stop_reason,
        "n_completed": result.search.n_completed,
        "n_pruned": result.search.n_pruned,
        "spearman_score_ari": rho,
        "spearman_n_trials": n_scored,
        "trial_diagnostics": trial_rows,
        "count_interval": list(result.worlds.count_interval),
        "unresolved_mass": result.worlds.unresolved_mass,
        "p_value": result.null.p_value,
        "structure_supported": result.null.structure_supported,
        "null_realisations": list(result.null.n_realisations),
        "flags": list(result.flags),
        "mixture_threshold": threshold,
        "seconds": elapsed,
        "stage_seconds": result.provenance["stage_seconds"],
        "total_fits": result.provenance["total_fits"],
        "peak_rss_mb": result.provenance["peak_rss_mb"],
        "engine": result.provenance["engine_resolved"],
    }
    (output / f"{name}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", help=f"names among {sorted(datasets.LOADERS)}")
    parser.add_argument("--output", type=Path, default=Path("bench/results"))
    parser.add_argument("--results-dir", type=Path, default=Path("output/bench"))
    parser.add_argument("--engine", default="auto", choices=["auto", "cpu", "cuml"])
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-trials", type=int, default=48)
    parser.add_argument("--n-explore", type=int, default=16)
    parser.add_argument("--n-worlds", type=int, default=32)
    parser.add_argument("--null", nargs="*", default=[], help="null families; none by default")
    parser.add_argument("--n-null-max", type=int, default=19)
    parser.add_argument("--default-seeds", type=int, default=8)
    parser.add_argument("--mixture-threshold", type=float, default=0.8)
    parser.add_argument("--no-spearman", action="store_true")
    args = parser.parse_args()
    config = Config(
        n_trials=args.n_trials,
        n_explore=args.n_explore,
        n_worlds=args.n_worlds,
        null_families=tuple(args.null),
        n_null_max=args.n_null_max,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        record = run_dataset(
            name,
            config=config,
            engine=args.engine,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            default_seeds=args.default_seeds,
            threshold=args.mixture_threshold,
            output=args.output,
            results_dir=args.results_dir,
            spearman=not args.no_spearman,
        )
        selected = record["selected"]
        assert isinstance(selected, dict)  # noqa: S101 - structural
        print(
            f"{name:12s} selected {selected['coverage_ari']:.3f} "
            f"default-median {record['default_median_coverage_ari']:.3f} "
            f"K {selected['n_groups']} p {record['p_value']:.3f} "
            f"rho {record['spearman_score_ari']:.2f} "
            f"{record['seconds']:.0f}s"
        )


if __name__ == "__main__":
    main()
