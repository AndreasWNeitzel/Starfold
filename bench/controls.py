"""Negative control and count neutrality on the synthetic regimes.

The generated matrices are method validation, not data; every record carries
``synthetic: true``. For each regime in ``bench.synthetic`` one ``recover``
runs with the labels hidden, and three checks are read from it:

1. Negative control. The single anisotropic population must not come out as
   supported structure under the Gaussian copula null, which no real dataset
   can test. The matched null runs for the cases in ``--null-cases`` only.
   Recovery, the count stability interval and the emitted count are reported
   for every regime.
2. Searched partitions. Every complete trial of that same search gets one
   full-data fit whose coverage-adjusted ARI and group count are recorded:
   the Spearman correlation between trial score and ARI, and the count of the
   best-scoring trial against the true count.
3. Exact partitions. With refit agreement fixed at one (the same label vector
   on every resample) the score reduces to coverage and input-space
   modularity. The true partition is scored beside every merge of two groups
   and every split of one group along its first principal component. No fit
   runs here; ``--exact-only`` skips checks 1 and 2.

The engine is resolved once, before any fit. The records are written to
``<output>/controls.json`` (default ``bench/results``) and the saved results
to ``<results-dir>/<case>`` (default ``output/bench``).
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from bench import synthetic
from bench.beats_default import score, trial_spearman
from starfold import Config, recover, save
from starfold._engine import require_resolved, resolve_engine
from starfold.backend import Fit, knn_graph, neighbor_adjacency
from starfold.objective import score_fits
from starfold.prep import make_plan, scale

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from starfold.prep import ResamplePlan


def _fits_from_labels(labels: NDArray[np.int64], plan: ResamplePlan) -> list[Fit]:
    """Fits that reproduce a fixed label vector on every resample."""
    fits = []
    for rows in plan.indices:
        rows_labels = labels[rows].astype(np.int32)
        fits.append(
            Fit(
                labels=rows_labels,
                strengths=np.ones(rows.size, dtype=np.float32),
                embedding=np.zeros((rows.size, 2), dtype=np.float32),
                n_groups=int(np.unique(rows_labels[rows_labels >= 0]).size),
                seconds=0.0,
            )
        )
    return fits


def merges(truth: NDArray[np.int64]) -> list[tuple[str, NDArray[np.int64]]]:
    """Every merge of two groups."""
    groups = np.unique(truth[truth >= 0])
    merged = []
    for a, b in itertools.combinations(groups.tolist(), 2):
        labels = truth.copy()
        labels[labels == b] = a
        merged.append((f"merge_{a}_{b}", labels))
    return merged


def splits(x: NDArray[np.float64], truth: NDArray[np.int64]) -> list[tuple[str, NDArray[np.int64]]]:
    """Every split of one group along its first principal component."""
    groups = np.unique(truth[truth >= 0])
    split = []
    next_label = int(groups.max()) + 1
    for g in groups.tolist():
        members = np.flatnonzero(truth == g)
        if members.size < 4:
            continue
        centred = x[members] - x[members].mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        projection = centred @ vt[0]
        labels = truth.copy()
        labels[members[projection > np.median(projection)]] = next_label
        split.append((f"split_{g}", labels))
    return split


def exact_partition_ranking(
    x: NDArray[np.float64], truth: NDArray[np.int64], *, seed: int, engine: str
) -> dict[str, object]:
    """Score truth, merges and splits with agreement fixed at one.

    ``engine`` must be resolved (``"cpu"`` or ``"cuml"``).
    """
    resolved = require_resolved(engine)
    plan = make_plan(x.shape[0], n_resamples=3, fraction=0.8, seed=seed)
    adjacency = []
    for rows, fit_seed in zip(plan.indices, plan.seeds, strict=True):
        graph = knn_graph(scale(x[rows], "standard"), 16, engine=resolved, seed=fit_seed)
        adjacency.append(neighbor_adjacency(graph, 15))
    scores: dict[str, float] = {}
    for name, labels in [("truth", truth), *merges(truth), *splits(x, truth)]:
        scores[name] = score_fits(
            _fits_from_labels(labels, plan),
            plan,
            adjacency,
            x=x,
            scaling="standard",
            engine=resolved,
        ).score
    merged = [v for k, v in scores.items() if k.startswith("merge")]
    split = [v for k, v in scores.items() if k.startswith("split")]
    return {
        "scores": scores,
        "truth_score": scores["truth"],
        "max_merge": max(merged) if merged else None,
        "max_split": max(split) if split else None,
        "truth_ranks_first": all(scores["truth"] > v for k, v in scores.items() if k != "truth"),
    }


def searched_partition_ranking(
    trial_rows: list[dict[str, object]], rho: float
) -> dict[str, object]:
    """Relate the trial scores of one search to the recovery each trial would emit."""
    scores = [float(r["gap"]) for r in trial_rows]  # type: ignore[arg-type]
    aris = [float(r["coverage_ari"]) for r in trial_rows]  # type: ignore[arg-type]
    counts = [int(r["n_groups"]) for r in trial_rows]  # type: ignore[call-overload]
    best = int(np.argmax(scores)) if scores else -1
    return {
        "n_trials_scored": len(scores),
        "spearman_score_ari": rho,
        "best_trial_count": counts[best] if best >= 0 else None,
        "best_trial_ari": aris[best] if best >= 0 else None,
        "counts": counts,
        "scores": scores,
        "aris": aris,
    }


def run_case(
    case: synthetic.SyntheticCase,
    *,
    config: Config,
    engine: str,
    n_jobs: int,
    seed: int,
    threshold: float,
    results_dir: Path,
) -> dict[str, object]:
    """Run ``recover`` on one regime and read all three checks from it."""
    resolved = require_resolved(engine)
    result = recover(case.X, config=config, random_state=seed, engine=resolved, n_jobs=n_jobs)
    save(result, results_dir / case.name)
    kept = result.kept_rows
    truth = case.truth[kept]
    rho, _, trial_rows = trial_spearman(
        result, case.X[kept], truth, engine=resolved, scaling=config.scaling, random_state=seed
    )
    return {
        "case": case.name,
        "description": case.description,
        "synthetic": True,
        "n_rows": int(case.X.shape[0]),
        "n_features": int(case.X.shape[1]),
        "true_groups": int(np.unique(truth).size),
        "selected": asdict(score(truth, result.labels, threshold)),
        "default": asdict(score(truth, result.default_labels, threshold)),
        "count_interval": list(result.worlds.count_interval),
        "unresolved_mass": result.unresolved_mass,
        "consensus_contrast": result.consensus.contrast,
        "null_families": list(config.null_families),
        "p_value": result.null.p_value,
        "structure_supported": result.null.structure_supported,
        "null_realisations": list(result.null.n_realisations),
        "exceedances": list(result.null.exceedances),
        "flags": list(result.flags),
        "total_fits": result.provenance["total_fits"],
        "seconds": result.provenance["stage_seconds"]["total"],
        "engine": resolved,
        "searched": searched_partition_ranking(trial_rows, rho),
        "exact": exact_partition_ranking(case.X[kept], truth, seed=seed, engine=resolved),
    }


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("bench/results"))
    parser.add_argument("--results-dir", type=Path, default=Path("output/bench"))
    parser.add_argument("--engine", default="auto", choices=["auto", "cpu", "cuml"])
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-trials", type=int, default=48)
    parser.add_argument("--n-worlds", type=int, default=32)
    parser.add_argument(
        "--null-cases",
        nargs="*",
        default=["single_anisotropic_gaussian"],
        help="cases that also run the matched null test; the others skip it",
    )
    parser.add_argument("--n-null-max", type=int, default=19)
    parser.add_argument("--mixture-threshold", type=float, default=0.8)
    parser.add_argument("--exact-only", action="store_true", help="run check 3 only, no fits")
    args = parser.parse_args()
    engine = resolve_engine(args.engine)
    args.output.mkdir(parents=True, exist_ok=True)

    records = []
    for case in synthetic.all_cases(args.seed):
        if args.exact_only:
            record: dict[str, object] = {
                "case": case.name,
                "description": case.description,
                "synthetic": True,
                "engine": engine,
                "exact": exact_partition_ranking(case.X, case.truth, seed=args.seed, engine=engine),
            }
        else:
            config = Config(
                n_trials=args.n_trials,
                n_worlds=args.n_worlds,
                null_families=("gaussian_copula",) if case.name in args.null_cases else (),
                n_null_max=args.n_null_max,
            )
            args.results_dir.mkdir(parents=True, exist_ok=True)
            record = run_case(
                case,
                config=config,
                engine=engine,
                n_jobs=args.n_jobs,
                seed=args.seed,
                threshold=args.mixture_threshold,
                results_dir=args.results_dir,
            )
            selected = record["selected"]
            assert isinstance(selected, dict)  # noqa: S101 - structural
            print(
                f"{case.name:28s} truth K={record['true_groups']} "
                f"emitted K={selected['n_groups']} recovery {selected['coverage_ari']:.3f} "
                f"interval {record['count_interval']} p {record['p_value']:.3f} "
                f"supported {record['structure_supported']}"
            )
        exact = record["exact"]
        assert isinstance(exact, dict)  # noqa: S101 - structural
        print(
            f"{case.name:28s} exact truth {exact['truth_score']:.3f} "
            f"merge {exact['max_merge']} split {exact['max_split']} "
            f"first {exact['truth_ranks_first']}"
        )
        records.append(record)
    (args.output / "controls.json").write_text(json.dumps(records, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
