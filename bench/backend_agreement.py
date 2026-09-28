"""Backend agreement: does the cuML path emit the partition the CPU path does?

Labels differ by permutation and floating-point summation order, so the
backends are compared by partition agreement: the adjusted Rand index on
jointly assigned rows times joint coverage, the quantity the objective uses
for held-out prediction.

Each case refits the deployed configuration of a saved result, rescaled to
the fitted row count with ``rescale_params``. A dataset result is read from
``<result-root>/<case>``, where ``bench.beats_default`` saves it (the
``apogee`` result is saved there too). ``--result CASE=PATH`` overrides it.
``flag_would_fire`` marks normalised agreement below 0.8, the
``backend_disagreement`` threshold.

CPU UMAP dominates the runtime, so ``--max-rows`` subsamples each case; fitted
and available row counts are recorded. The records are written to
``<output>/backend_agreement.json`` (default ``bench/results``).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from bench import datasets
from starfold import load
from starfold.backend import compare_backends, rescale_params

DISAGREEMENT_FLAG = 0.8


def saved_results(cases: list[str], root: Path, overrides: list[str]) -> dict[str, Path]:
    """Saved result directory per case: a ``CASE=PATH`` override or ``root/case``."""
    explicit: dict[str, Path] = {}
    for item in overrides:
        case, sep, path = item.partition("=")
        if not sep or not case or not path:
            msg = f"--result takes CASE=PATH (got {item!r})."
            raise ValueError(msg)
        if case not in cases:
            msg = f"--result names {case!r}, which is not among the cases {cases}."
            raise ValueError(msg)
        explicit[case] = Path(path)
    return {case: explicit.get(case, root / case) for case in cases}


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("bench/results"))
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["digits", "rice", "landsat", "waveform", "apogee"],
        help="dataset names or 'apogee'; each needs a saved result",
    )
    parser.add_argument("--result-root", type=Path, default=Path("output/bench"))
    parser.add_argument(
        "--result",
        action="append",
        default=[],
        metavar="CASE=PATH",
        help="saved result of one case, overriding its default location; repeatable",
    )
    parser.add_argument("--max-rows", type=int, default=6000)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        saved_paths = saved_results(args.cases, args.result_root, args.result)
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)

    records = []
    for name in args.cases:
        saved = saved_paths[name]
        if not (saved / "result.json").exists():
            print(f"{name:10s} skipped, no saved result at {saved}")
            continue
        result = load(saved)
        full = datasets.rows(name)
        rng = np.random.default_rng(args.seed)
        x = full
        if full.shape[0] > args.max_rows:
            x = full[np.sort(rng.choice(full.shape[0], size=args.max_rows, replace=False))]
        params = rescale_params(result.deployed_params, result.labels.size, x.shape[0])
        started = time.perf_counter()
        agreement = compare_backends(x, params, n_seeds=args.n_seeds, random_state=args.seed)
        record = {
            "case": name,
            "n_rows_fitted": int(x.shape[0]),
            "n_rows_available": int(full.shape[0]),
            "params": asdict(params),
            "seconds": time.perf_counter() - started,
            "disagreement_flag_below": DISAGREEMENT_FLAG,
            "flag_would_fire": agreement.mean_normalised < DISAGREEMENT_FLAG,
            "assigned_fraction_in_saved_run": float(np.mean(result.labels >= 0)),
            **asdict(agreement),
        }
        records.append(record)
        print(
            f"{name:10s} n={record['n_rows_fitted']:5d} agreement "
            f"{agreement.mean_agreement:.3f} = ari {np.mean(agreement.ari):.3f} "
            f"x coverage {np.mean(agreement.coverage):.3f}, normalised "
            f"{agreement.mean_normalised:.3f}, assigned cpu "
            f"{np.mean(agreement.assigned_cpu):.2f} cuml {np.mean(agreement.assigned_cuml):.2f}, "
            f"counts cpu {agreement.n_groups_cpu} cuml {agreement.n_groups_cuml} "
            f"({record['seconds']:.0f} s)"
        )
    (args.output / "backend_agreement.json").write_text(
        json.dumps(records, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
