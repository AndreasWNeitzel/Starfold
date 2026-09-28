"""Summarise the benchmark records into ``summary.json``; no fit runs here.

Reads the per-dataset records of ``bench.beats_default`` (``<dataset>.json``),
the null records of ``bench.null_stage`` (``<dataset>_null.json``) and, when
present, ``injection_recovery.json``, ``backend_agreement.json`` and
``controls.json`` from ``--results`` (default ``bench/results``). Prints one
row per dataset with the assigned fraction beside every adjusted Rand index,
then the selector diagnostics of every record with per-trial rows, and writes
the compact summary that ``bench.figures`` draws from. A dataset counts as at
or above the default when its recovery is at most ``TOLERANCE`` below the
median recovery of the default configuration over seeds.

The selector diagnostics are post hoc and use the labels, so the procedure
never sees them. ``regret`` is the best recovery among the explored trials
minus the emitted recovery. ``ari_at_gap_argmax`` and ``ari_at_raw_argmax``
are the recoveries of the trials ranked first by the gap and by the data score
alone, and ``rho_raw`` is the Spearman correlation of the data score with
recovery (``spearman_score_ari`` is that of the gap). The control variant
replaces the unit coefficient on the null score by ``beta``, the least-squares
slope of the data score on the null score; without null scores ``beta`` is 0
and the control ranking is the raw one.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from bench import datasets

TOLERANCE = 0.02
SELECTOR_MEANS = (
    "regret",
    "best_trial_ari",
    "ari_at_gap_argmax",
    "ari_at_raw_argmax",
    "ari_at_control_argmax",
)


def _finite(value: Any) -> Any:
    """Replace a non-finite float by ``None`` so the summary is strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _read(path: Path) -> Any:
    """Parsed JSON of one record, or ``None`` when absent."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def selector(record: dict[str, Any]) -> dict[str, Any] | None:
    """Selector diagnostics from the per-trial rows, ``None`` with fewer than three."""
    rows = record.get("trial_diagnostics") or []
    if len(rows) < 3:
        return None
    gap = np.array([float(r["gap"]) for r in rows])
    raw = np.array([float(r["raw_score"]) for r in rows])
    null = np.array([float(r["null_score"]) for r in rows])
    ari = np.array([float(r["coverage_ari"]) for r in rows])
    finite = np.isfinite(null)
    beta = 0.0
    if finite.sum() > 2 and null[finite].std() > 0:
        covariance = np.cov(raw[finite], null[finite])
        beta = float(covariance[0, 1] / covariance[1, 1])
    control = raw - beta * np.where(finite, null, 0.0)
    return {
        "n_trials": len(rows),
        "rho_raw": _finite(float(spearmanr(raw, ari).statistic)),
        "rho_control": _finite(float(spearmanr(control, ari).statistic)),
        "beta": beta,
        "best_trial_ari": float(ari.max()),
        "regret": float(ari.max() - record["selected"]["coverage_ari"]),
        "ari_at_gap_argmax": float(ari[int(np.argmax(gap))]),
        "ari_at_raw_argmax": float(ari[int(np.argmax(raw))]),
        "ari_at_control_argmax": float(ari[int(np.argmax(control))]),
    }


def dataset_rows(results: Path) -> list[dict[str, Any]]:
    """One compact row per benchmark record present, development set first."""
    rows = []
    for split, names in (("development", datasets.DEVELOPMENT), ("sealed", datasets.SEALED)):
        for name in names:
            record = _read(results / f"{name}.json")
            if record is None:
                continue
            selected = record["selected"]
            rows.append(
                {
                    "dataset": name,
                    "split": split,
                    "n_rows": record["n_rows"],
                    "n_features": record["n_features"],
                    "n_classes": record["n_classes"],
                    "coverage_ari": selected["coverage_ari"],
                    "assigned_fraction": selected["assigned_fraction"],
                    "default_coverage_ari": record["default_median_coverage_ari"],
                    "gain": selected["coverage_ari"] - record["default_median_coverage_ari"],
                    "n_groups": selected["n_groups"],
                    "count_interval": record["count_interval"],
                    "unresolved_mass": record["unresolved_mass"],
                    "mixture_fraction": _finite(selected["mixture_fraction"]),
                    "spearman_score_ari": _finite(record["spearman_score_ari"]),
                    "total_fits": record["total_fits"],
                    "peak_rss_mb": _finite(record["peak_rss_mb"]),
                    "engine": record["engine"],
                    "selector": selector(record),
                }
            )
    return rows


def means(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Mean recovery, default recovery, gain and selector diagnostics per split and overall.

    The selector means run over the rows with per-trial records and are
    ``None`` when a split has none.
    """
    out: dict[str, dict[str, Any]] = {}
    for split in ("development", "sealed", "all"):
        chosen = [r for r in rows if split in {"all", r["split"]}]
        if not chosen:
            continue
        gains = [r["gain"] for r in chosen]
        out[split] = {
            "n": len(chosen),
            "coverage_ari": float(np.mean([r["coverage_ari"] for r in chosen])),
            "default_coverage_ari": float(np.mean([r["default_coverage_ari"] for r in chosen])),
            "gain": float(np.mean(gains)),
            "at_or_above_default": sum(g >= -TOLERANCE for g in gains),
            "selector": _selector_means([r["selector"] for r in chosen if r["selector"]]),
        }
    return out


def _selector_means(blocks: list[dict[str, Any]]) -> dict[str, float | int] | None:
    """Means of the selector diagnostics over the rows that have them."""
    if not blocks:
        return None
    return {"n": len(blocks)} | {
        key: float(np.mean([b[key] for b in blocks])) for key in SELECTOR_MEANS
    }


def null_rows(results: Path) -> list[dict[str, Any]]:
    """The matched-null record of every dataset that has one."""
    rows = []
    for name in (*datasets.LOADERS, "apogee"):
        record = _read(results / f"{name}_null.json")
        if record is not None:
            rows.append(
                {
                    "dataset": name,
                    "families": record["families"],
                    "p_value": record["p_value"],
                    "p_intervals": record["p_intervals"],
                    "n_realisations": record["n_realisations"],
                    "exceedances": record["exceedances"],
                    "structure_supported": record["structure_supported"],
                    "n_fits": record["n_fits"],
                }
            )
    return rows


def injection_rows(results: Path) -> list[dict[str, Any]]:
    """Retrieval, purity and completeness per contrast level."""
    keys = (
        "contrast",
        "n_rows",
        "k",
        "rows_per_group",
        "n_injected_rows",
        "n_groups_emitted",
        "count_interval",
        "retrieval_rate",
        "by_threshold",
    )
    return [
        {key: record[key] for key in keys}
        for record in _read(results / "injection_recovery.json") or []
    ]


def backend_rows(results: Path) -> list[dict[str, Any]]:
    """Partition agreement between the CPU and cuML paths per case."""
    keys = (
        "case",
        "n_rows_fitted",
        "n_rows_available",
        "mean_agreement",
        "mean_normalised",
        "flag_would_fire",
        "n_groups_cpu",
        "n_groups_cuml",
    )
    return [
        {key: _finite(record[key]) for key in keys}
        for record in _read(results / "backend_agreement.json") or []
    ]


def control_rows(results: Path) -> list[dict[str, Any]]:
    """Negative control and count neutrality per synthetic regime."""
    rows = []
    for record in _read(results / "controls.json") or []:
        row: dict[str, Any] = {"case": record["case"], "synthetic": True}
        if "selected" in record:
            searched = record["searched"]
            row |= {
                "true_groups": record["true_groups"],
                "n_groups": record["selected"]["n_groups"],
                "coverage_ari": record["selected"]["coverage_ari"],
                "count_interval": record["count_interval"],
                "p_value": record["p_value"],
                "structure_supported": record["structure_supported"],
                "spearman_score_ari": _finite(searched["spearman_score_ari"]),
                "best_trial_count": searched["best_trial_count"],
            }
        row["truth_ranks_first"] = record["exact"]["truth_ranks_first"]
        rows.append(row)
    return rows


def summarize(results: Path) -> dict[str, Any]:
    """Every summary table read from the records under ``results``."""
    rows = dataset_rows(results)
    return {
        "tolerance": TOLERANCE,
        "datasets": rows,
        "means": means(rows),
        "null": null_rows(results),
        "injection": injection_rows(results),
        "backends": backend_rows(results),
        "controls": control_rows(results),
    }


def _print(summary: dict[str, Any]) -> None:
    """Print the per-dataset table, the means and the selector diagnostics."""
    header = (
        f"{'dataset':10s} {'split':11s} {'rows':>6s} {'K*':>3s} {'sel ARI':>8s} "
        f"{'assign':>7s} {'def ARI':>8s} {'K':>4s} {'interval':>9s} {'unres':>6s} "
        f"{'mix':>5s} {'rho':>6s} {'fits':>5s}"
    )
    print(header)
    print("-" * len(header))
    for r in summary["datasets"]:
        interval = f"{r['count_interval'][0]}-{r['count_interval'][1]}"
        mixture = r["mixture_fraction"] if r["mixture_fraction"] is not None else float("nan")
        rho = r["spearman_score_ari"] if r["spearman_score_ari"] is not None else float("nan")
        print(
            f"{r['dataset']:10s} {r['split']:11s} {r['n_rows']:6d} {r['n_classes']:3d} "
            f"{r['coverage_ari']:8.3f} {r['assigned_fraction']:7.2f} "
            f"{r['default_coverage_ari']:8.3f} {r['n_groups']:4d} {interval:>9s} "
            f"{r['unresolved_mass']:6.2f} {mixture:5.2f} {rho:6.2f} {r['total_fits']:5d}"
        )
    for split, m in summary["means"].items():
        print(
            f"{split:11s} mean selected {m['coverage_ari']:.3f}, "
            f"mean default {m['default_coverage_ari']:.3f}, mean gain {m['gain']:+.3f}, "
            f"at or above the default {m['at_or_above_default']}/{m['n']}"
        )
    detail = [(r["dataset"], r["selector"]) for r in summary["datasets"] if r["selector"]]
    if not detail:
        return
    header = (
        f"\n{'dataset':10s} {'rho_raw':>8s} {'rho_ctrl':>9s} {'beta':>6s} {'best':>6s} "
        f"{'regret':>7s} {'gapmax':>7s} {'rawmax':>7s} {'ctrlmax':>8s}"
    )
    print(header)
    for name, d in detail:
        rho_raw = d["rho_raw"] if d["rho_raw"] is not None else float("nan")
        rho_control = d["rho_control"] if d["rho_control"] is not None else float("nan")
        print(
            f"{name:10s} {rho_raw:8.2f} {rho_control:9.2f} {d['beta']:6.2f} "
            f"{d['best_trial_ari']:6.3f} {d['regret']:7.3f} {d['ari_at_gap_argmax']:7.3f} "
            f"{d['ari_at_raw_argmax']:7.3f} {d['ari_at_control_argmax']:8.3f}"
        )
    for split, m in summary["means"].items():
        chosen = m["selector"]
        if chosen:
            print(
                f"{split:11s} over {chosen['n']} with trials: mean regret "
                f"{chosen['regret']:.3f}, best by gap {chosen['ari_at_gap_argmax']:.3f}, "
                f"by raw {chosen['ari_at_raw_argmax']:.3f}, "
                f"by control {chosen['ari_at_control_argmax']:.3f}"
            )


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("bench/results"))
    parser.add_argument("--output", type=Path, help="default: <results>/summary.json")
    args = parser.parse_args()
    summary = summarize(args.results)
    _print(summary)
    output = args.output or args.results / "summary.json"
    output.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
