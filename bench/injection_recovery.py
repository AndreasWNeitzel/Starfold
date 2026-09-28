"""Injection-recovery: retrieval of injected groups as a function of contrast.

Synthetic groups of declared size and density contrast are injected into a
real feature matrix, ``recover`` runs with the injection hidden, and
retrieval is reported per contrast level. This is the selection-function
construction of Hunt 2025 (arXiv:2506.18708) in the validation shape of
UPMASK (Krone-Martins and Moitinho 2013, arXiv:1309.4471). It exists because
recurrence cannot flag a dense random fluctuation that is a feature of the
data and the copula null answers only the partition-level question.

Contrast is defined here as the ratio of the injected group's density to the
local background density at the injection site: the site is a random
existing row, the radius is the distance to its ``k``-th nearest neighbour
in standard-scaled space (which contains about ``k`` background rows), and
``contrast * k`` rows are injected uniformly inside that ball. A group is
retrieved when some emitted group has Jaccard at least 0.5 with the injected
rows. Purity of the matched group and completeness of the injected rows are
reported against the support threshold. Injected rows are never persisted;
each record carries ``synthetic_rows_present`` and never enters a scientific
table. Each record also carries the subsample size ``n_rows`` and ``k``; the
records are written to ``<output>/injection_recovery.json`` (default
``bench/results``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.neighbors import NearestNeighbors

from bench import datasets
from starfold import Config, recover
from starfold.prep import scale

if TYPE_CHECKING:
    from numpy.typing import NDArray


def inject(
    x: NDArray[np.float64],
    *,
    n_groups: int,
    contrast: float,
    k: int,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Return the augmented matrix and an injection label (-1 for real rows)."""
    rng = np.random.default_rng(seed)
    scaled = scale(x, "standard").astype(np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    neighbours = NearestNeighbors(n_neighbors=k + 1).fit(scaled)
    sites = rng.choice(x.shape[0], size=n_groups, replace=False)
    distances, _ = neighbours.kneighbors(scaled[sites])
    blocks = []
    labels = []
    m = max(2, round(contrast * k))
    for g, site in enumerate(sites):
        radius = float(distances[g, -1])
        direction = rng.normal(size=(m, x.shape[1]))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        radii = radius * rng.uniform(size=m) ** (1.0 / x.shape[1])
        points_scaled = scaled[site] + direction * radii[:, None]
        blocks.append(points_scaled * std + mean)
        labels.append(np.full(m, g, dtype=np.int64))
    augmented = np.vstack([x, *blocks])
    injection = np.concatenate([np.full(x.shape[0], -1, dtype=np.int64), *labels])
    order = rng.permutation(augmented.shape[0])
    return augmented[order], injection[order]


MATCH_JACCARD = 0.5


def retrieval(
    injection: NDArray[np.int64],
    labels: NDArray[np.int32],
    own_support: NDArray[np.float32],
    thresholds: tuple[float, ...],
) -> dict[str, object]:
    """Match each injected group to an emitted group and report its reliability.

    Each injected group is matched to the emitted group of largest Jaccard
    overlap (Hennig 2007, doi:10.1016/j.csda.2006.11.025, the same matching the
    library uses for group recurrence) and counted as retrieved at Jaccard
    ``MATCH_JACCARD``. Purity and completeness are then reported inside that
    matched group as the recurrence-support threshold rises: how much of the
    group is the injected population, and how much of that population the
    group holds.
    """
    groups = np.unique(injection[injection >= 0])
    emitted_groups = np.unique(labels[labels >= 0])
    support = np.nan_to_num(own_support, nan=0.0)
    retrieved, jaccards, matched = [], [], []
    per_threshold: dict[float, list[tuple[float | None, float]]] = {tau: [] for tau in thresholds}
    for g in groups:
        members = injection == g
        best, best_emitted = 0.0, -1
        for emitted in emitted_groups:
            found = labels == emitted
            union = int(np.logical_or(members, found).sum())
            overlap = float(np.logical_and(members, found).sum() / union) if union else 0.0
            if overlap > best:
                best, best_emitted = overlap, int(emitted)
        retrieved.append(best >= MATCH_JACCARD)
        jaccards.append(best)
        matched.append(best_emitted)
        for tau in thresholds:
            kept = (
                (labels == best_emitted) & (support >= tau)
                if best_emitted >= 0
                else members & False
            )
            hit = int(np.logical_and(kept, members).sum())
            purity = float(hit / kept.sum()) if kept.sum() else None
            per_threshold[tau].append((purity, float(hit / members.sum())))
    by_threshold = []
    for tau in thresholds:
        purities = [p for p, _ in per_threshold[tau] if p is not None]
        completeness = [c for _, c in per_threshold[tau]]
        by_threshold.append(
            {
                "threshold": tau,
                "mean_purity_of_matched_group": float(np.mean(purities)) if purities else None,
                "mean_completeness_of_injected": float(np.mean(completeness))
                if completeness
                else None,
                "groups_with_a_match": len(purities),
            }
        )
    return {
        "retrieved": retrieved,
        "retrieval_rate": float(np.mean(retrieved)) if retrieved else None,
        "best_jaccard": jaccards,
        "matched_group": matched,
        "by_threshold": by_threshold,
    }


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("bench/results"))
    parser.add_argument("--engine", default="auto", choices=["auto", "cpu", "cuml"])
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-rows", type=int, default=2000, help="subsample of the example table")
    parser.add_argument("--n-groups", type=int, default=3)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--contrasts", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0, 16.0])
    parser.add_argument("--n-trials", type=int, default=32)
    parser.add_argument("--n-worlds", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    x_full, _ = datasets.example_table()
    rng = np.random.default_rng(args.seed)
    x = x_full[
        np.sort(rng.choice(x_full.shape[0], size=min(args.n_rows, x_full.shape[0]), replace=False))
    ]
    config = Config(n_trials=args.n_trials, n_explore=16, n_worlds=args.n_worlds, null_families=())
    thresholds = (0.0, 0.5, 0.8, 0.9)
    records = []
    for contrast in args.contrasts:
        augmented, injection = inject(
            x, n_groups=args.n_groups, contrast=contrast, k=args.k, seed=args.seed
        )
        result = recover(
            augmented, config=config, random_state=args.seed, engine=args.engine, n_jobs=args.n_jobs
        )  # type: ignore[arg-type]
        kept = result.kept_rows
        summary = retrieval(injection[kept], result.labels, result.worlds.own_support, thresholds)
        record = {
            "contrast": contrast,
            "n_rows": int(x.shape[0]),
            "k": args.k,
            "n_injected_rows": int((injection >= 0).sum()),
            "rows_per_group": int((injection == 0).sum()),
            "n_groups_emitted": result.n_groups,
            "count_interval": list(result.worlds.count_interval),
            "synthetic_rows_present": True,
            **summary,
        }
        records.append(record)
        print(
            f"contrast {contrast:5.1f} injected {record['n_injected_rows']} "
            f"retrieval {summary['retrieval_rate']} emitted {result.n_groups}"
        )
    (args.output / "injection_recovery.json").write_text(
        json.dumps(records, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
