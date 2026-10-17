"""Run the matched null test on a saved search, separately from the search.

Replays the saved run's null procedure on up to ``n_null_max`` null matrices
per family with the saved seed, configuration and search space. The engine and
worker count default to the saved run's resolved engine and n_jobs.

The dataset is reloaded by name and cut to the saved run's kept rows; a
fingerprint mismatch with the saved provenance stops the run. The record is
written to ``<output>/<dataset>_null.json`` (default ``bench/results``).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from bench import datasets
from starfold import Config, Space, load, null_test, save
from starfold.result import derive_flags, matrix_fingerprint

_SPACE_FIELDS = frozenset(Space.__dataclass_fields__)


def config_from_provenance(record: dict[str, Any]) -> Config:
    """Rebuild the configuration a saved run used."""
    stored = dict(record)
    space = stored.pop("space", None)
    fields = {f for f in Config.__dataclass_fields__ if f != "space"}
    config = Config(**{k: v for k, v in stored.items() if k in fields})
    if space is None:
        return config
    unknown = set(space) - _SPACE_FIELDS
    if unknown:
        msg = (
            f"the saved search space has unknown bounds {sorted(unknown)}; the replay would differ."
        )
        raise ValueError(msg)
    bounds = {k: None if v is None else tuple(v) for k, v in space.items()}
    return replace(config, space=Space(**bounds))


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help=f"'apogee' or one of {sorted(datasets.LOADERS)}")
    parser.add_argument(
        "result",
        type=Path,
        help="directory written by starfold.save, for example output/bench/<dataset>",
    )
    parser.add_argument("--output", type=Path, default=Path("bench/results"))
    parser.add_argument(
        "--engine",
        choices=["auto", "cpu", "cuml"],
        help="default: the engine the saved run resolved to",
    )
    parser.add_argument("--n-jobs", type=int, help="default: the saved run's n_jobs")
    parser.add_argument("--families", nargs="+", default=["gaussian_copula"])
    parser.add_argument("--n-null-max", type=int, default=19)
    parser.add_argument(
        "--update-result",
        action="store_true",
        help="rewrite the saved result with this null evidence and recomputed flags",
    )
    args = parser.parse_args()

    result = load(args.result)
    x = datasets.rows(args.dataset)[result.kept_rows]
    fingerprint = matrix_fingerprint(x)
    if fingerprint != result.provenance["input_fingerprint"]:
        msg = (
            f"matrix fingerprint {fingerprint[:16]} does not match the saved run "
            f"{str(result.provenance['input_fingerprint'])[:16]}; the dataset changed."
        )
        raise SystemExit(msg)

    config = replace(
        config_from_provenance(result.provenance["config"]), n_null_max=args.n_null_max
    )
    evidence = null_test(
        x,
        result.search,
        config=config,
        families=tuple(args.families),
        random_state=int(result.provenance["random_state"]),
        engine=args.engine or str(result.provenance["engine_resolved"]),
        n_jobs=int(result.provenance["n_jobs"]) if args.n_jobs is None else args.n_jobs,
    )
    record = {
        "dataset": args.dataset,
        "result": str(args.result),
        "statistic_name": evidence.statistic_name,
        "observed": evidence.observed,
        "families": list(evidence.families),
        "n_realisations": list(evidence.n_realisations),
        "exceedances": list(evidence.exceedances),
        "p_values": list(evidence.p_values),
        "p_intervals": [list(interval) for interval in evidence.p_intervals],
        "p_value": evidence.p_value,
        "alpha": evidence.alpha,
        "structure_supported": evidence.structure_supported,
        "stopped_early": list(evidence.stopped_early),
        "null_values": [list(values) for values in evidence.values],
        "n_fits": evidence.n_fits,
        "seconds": evidence.seconds,
    }
    if args.update_result:
        provenance = result.provenance
        updated = replace(
            result,
            null=evidence,
            flags=derive_flags(
                labels=result.labels,
                search=result.search,
                worlds=result.worlds,
                consensus=result.consensus,
                null=evidence,
                backends=result.backends,
                rows_rejected=int(provenance["n_input_rows"]) - int(provenance["n_kept_rows"]),
                n_jobs_forced=bool(provenance["n_jobs_forced_to_one_on_cuml"]),
            ),
        )
        save(updated, args.result)
        print(f"rewrote {args.result} with flags {list(updated.flags)}")

    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / f"{args.dataset}_null.json"
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(
        f"{args.dataset:12s} p {evidence.p_value:.3f} supported {evidence.structure_supported} "
        f"realisations {list(evidence.n_realisations)} fits {evidence.n_fits} "
        f"{evidence.seconds:.0f}s -> {out}"
    )


if __name__ == "__main__":
    main()
