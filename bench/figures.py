"""Draw the resolution and injection-recovery figures from the summary.

Reads ``summary.json`` written by ``bench.summarize`` (default
``bench/results/summary.json``) and writes ``bench_resolution`` and
``bench_injection`` as PNG and PDF into ``--output``
(default ``docs/figures``). No fit runs here.

Group sizes are drawn in units of ``sqrt(15 m)``. Modularity maximisation
cannot resolve a community whose number of internal edges is below about
``sqrt(2E)``, with ``E`` the number of edges of the graph (Fortunato and
Barthelemy 2007, doi:10.1073/pnas.0605965104). The modularity graph of a
resample is the symmetrised 15-nearest-neighbour graph on its ``m`` rows, so
``2E`` is at least ``15 m``, with equality when every neighbour relation is
mutual, and ``sqrt(15 m)`` is that bound in edges. It is a scale, not a
bound in rows: in the mutual case a group of ``s`` rows that keeps its
neighbours inside holds ``7.5 s`` edges, so in rows the bound is near
``2 sqrt(m / 15)``, 7.5 times smaller. The dashed line marks a group of
``sqrt(15 m)`` rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator, StrMethodFormatter

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

RESAMPLE_FRACTION = 0.8
NEIGHBORS = 15
COLUMN = 3.5
DEVELOPMENT_COLOUR = "#2c7fb8"
SEALED_COLOUR = "#d95f0e"
STYLE = {
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}
PRETTY = {
    "digits": "Digits",
    "rice": "Rice",
    "landsat": "Landsat",
    "pendigits": "PenDigits",
    "dry_bean": "Dry Bean",
    "waveform": "Waveform",
    "coil20": "COIL-20",
    "isolet": "Isolet",
    "gas_drift": "Gas drift",
    "letter": "Letter",
}
TICKS = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
LABEL_SIZE = 6.0


def modularity_scale(rows: NDArray[np.float64]) -> NDArray[np.float64]:
    """``sqrt(15 m)`` with ``m = 0.8 n``, the edge bound of a resample of ``n`` rows."""
    return np.sqrt(NEIGHBORS * RESAMPLE_FRACTION * rows)


def _log_ticks(ax: Axes, values: NDArray[np.float64]) -> None:
    """Label the log x axis at the round values that span the data."""
    low, high = float(np.min(values)) / 1.5, float(np.max(values)) * 1.5
    ax.set_xscale("log")
    ax.set_xlim(low, high)
    ax.xaxis.set_major_locator(FixedLocator([t for t in TICKS if low <= t <= high]))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())


def _annotate(ax: Axes, x: NDArray[np.float64], y: NDArray[np.float64], labels: list[str]) -> None:
    """Label points left to right, stepping each label until it overlaps no earlier one."""
    fig = ax.get_figure()
    assert fig is not None  # noqa: S101 - structural
    ax.get_ylim()  # settles autoscaling before transData is read
    anchors = ax.transData.transform(np.column_stack([x, y])) * 72.0 / fig.dpi
    height = 1.2 * LABEL_SIZE
    boxes: list[tuple[float, float, float, float]] = []
    for index in np.argsort(anchors[:, 0]):
        width = 0.55 * LABEL_SIZE * len(labels[index])
        for step in (0.0, -1.0, 1.0, -2.0, 2.0, -3.0, 3.0):
            dy = 2.0 + step * height
            left, bottom = anchors[index, 0] + 3.0, anchors[index, 1] + dy
            if all(
                left + width <= b_left
                or b_right <= left
                or bottom + height <= b_bottom
                or b_top <= bottom
                for b_left, b_bottom, b_right, b_top in boxes
            ):
                break
        boxes.append((left, bottom, left + width, bottom + height))
        ax.annotate(
            labels[index],
            (x[index], y[index]),
            xytext=(3.0, dy),
            textcoords="offset points",
            fontsize=LABEL_SIZE,
        )


def figure_resolution(summary: dict[str, Any]) -> Figure:
    """Recovery gain and score-recovery correlation against class size over ``sqrt(15 m)``."""
    rows = summary["datasets"]
    names = [r["dataset"] for r in rows]
    n = np.array([r["n_rows"] for r in rows], dtype=float)
    classes = np.array([r["n_classes"] for r in rows], dtype=float)
    ratio = RESAMPLE_FRACTION * n / classes / modularity_scale(n)
    gain = np.array([r["gain"] for r in rows], dtype=float)
    rho = np.array(
        [np.nan if r["spearman_score_ari"] is None else r["spearman_score_ari"] for r in rows]
    )
    sealed = np.array([r["split"] == "sealed" for r in rows])
    fig = Figure(figsize=(2.1 * COLUMN, COLUMN))
    axes = fig.subplots(1, 2)
    panels = (
        (axes[0], gain, "recovery minus default recovery"),
        (axes[1], rho, "Spearman of trial score against recovery"),
    )
    for ax, y, label in panels:
        ax.axhline(0, color="k", lw=0.8)
        ax.axvline(1.0, color="0.4", lw=0.8, ls="--", label=r"group of $\sqrt{15 m}$ rows")
        ax.scatter(ratio[~sealed], y[~sealed], s=30, color=DEVELOPMENT_COLOUR, label="development")
        ax.scatter(ratio[sealed], y[sealed], s=30, color=SEALED_COLOUR, marker="s", label="sealed")
        _log_ticks(ax, ratio)
        ax.set_xlabel(r"mean class size in a resample, over $\sqrt{15 m}$")
        ax.set_ylabel(label)
        ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    for ax, y, _ in panels:
        _annotate(ax, ratio, y, [PRETTY.get(name, name) for name in names])
    return fig


def figure_injection(summary: dict[str, Any]) -> Figure:
    """Retrieval, purity and completeness of injected groups against their size."""
    records = summary["injection"]
    size = np.array([r["rows_per_group"] for r in records], dtype=float)
    total = np.array([r["n_rows"] + r["n_injected_rows"] for r in records], dtype=float)
    ratio = RESAMPLE_FRACTION * size / modularity_scale(total)
    retrieval = np.array([r["retrieval_rate"] for r in records], dtype=float)
    fig = Figure(figsize=(2.1 * COLUMN, COLUMN))
    axes = fig.subplots(1, 2)
    axes[0].plot(ratio, retrieval, "o-", color=DEVELOPMENT_COLOUR, label="injected groups")
    axes[0].axvline(1.0, color="0.4", lw=0.8, ls="--", label=r"group of $\sqrt{15 m}$ rows")
    for x, y, s in zip(ratio, retrieval, size, strict=True):
        axes[0].annotate(
            f"{int(s)} rows", (x, y), fontsize=LABEL_SIZE, xytext=(3, 5), textcoords="offset points"
        )
    axes[0].set_ylim(-0.05, 1.15)
    _log_ticks(axes[0], ratio)
    axes[0].set_xlabel(r"injected group in a resample, over $\sqrt{15 m}$")
    axes[0].set_ylabel("fraction of injected groups retrieved")
    axes[0].legend(frameon=False, loc="center right")
    colours = mpl.colormaps["viridis"](np.linspace(0, 0.85, len(records)))
    for record, colour in zip(records, colours, strict=True):
        if not record["retrieval_rate"]:
            continue
        taus = [b["threshold"] for b in record["by_threshold"]]
        purity = [b["mean_purity_of_matched_group"] for b in record["by_threshold"]]
        complete = [b["mean_completeness_of_injected"] for b in record["by_threshold"]]
        axes[1].plot(
            taus, purity, "o-", color=colour, label=f"{record['rows_per_group']} rows per group"
        )
        axes[1].plot(taus, complete, "s--", color=colour, mfc="none", lw=0.8)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel("recurrence support threshold")
    axes[1].set_ylabel("solid: purity.  dashed: completeness")
    axes[1].legend(frameon=False, fontsize=6, loc="center left")
    fig.suptitle("groups injected into the real APOGEE table (method validation)")
    fig.tight_layout()
    return fig


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("bench/results/summary.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(STYLE):
        for name, draw in (
            ("bench_resolution", figure_resolution),
            ("bench_injection", figure_injection),
        ):
            fig = draw(summary)
            for suffix in ("png", "pdf"):
                fig.savefig(args.output / f"{name}.{suffix}")
            print(f"wrote {args.output / name}.png and .pdf")


if __name__ == "__main__":
    main()
