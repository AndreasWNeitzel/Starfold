# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
# ---

# %% [markdown]
# # Calibration
#
# UMAP draws a low-dimensional map of a table, HDBSCAN finds dense groups on that
# map, and with the right settings the two can find groups in almost any table.
# For this reason, the calibration test asks whether the table scores higher than
# copies of it that have no groups at all.
#
# This notebook shows what such a copy looks like and reads the tests recorded on
# the handwritten digits and on APOGEE DR19. It runs no fits.

# %% [markdown]
# ## What the p-value answers
#
# The statistic is the best score among the first 17 trials of the search: the
# library defaults and 16 settings spread evenly over the search range (a Sobol
# design). Each structureless copy goes through the same 17 trials with the same
# subsets and seeds. The p-value is the fraction of copies that score at least as
# high as the table, counting the table as one more draw.
#
# It does not tell you:
#
# - the probability that the groups are real;
# - how many groups there are, or whether any one group is real;
# - how your table compares with any kind of copy other than the family you
#   declared.

# %% [markdown]
# ## What it costs
#
# Every trial fits 3 subsets of the copy and 3 of a structureless copy of the copy,
# because the search scores each setting as its excess over such a copy. That is
# 102 fits per copy. The test draws up to 19 copies (1,938 fits), about six times
# the fits of the rest of a run. Given this cost, `recover` skips it by default and
# `starfold.null_test` runs it as a separate step.
#
# This notebook does not rerun the 1,938 fits that each recorded test below made.

# %%
import sys
from pathlib import Path

ROOT = next(
    (p for p in (Path.cwd(), *Path.cwd().parents) if (p / "pyproject.toml").is_file()),
    None,
)
if ROOT is None:
    msg = "Run this notebook from inside a clone of the starfold repository."
    raise FileNotFoundError(msg)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# %%
import json

import matplotlib.pyplot as plt
import numpy as np
from bench.datasets import example_table
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from scipy.stats import spearmanr

import starfold
from starfold.nulls import monte_carlo_pvalue

# %% tags=["parameters"]
RANDOM_STATE = 0
# "auto" uses cuML when it imports and sees a GPU, and warns when it falls back to the CPU.
ENGINE = "auto"
N_JOBS = 1
# Set to ROOT / "docs" / "figures" to rewrite the handbook PNGs.
FIGURES = None

# %%
RECORDS = ROOT / "bench" / "results"


def save_figure(fig, name):
    """Write a figure under FIGURES at 150 dpi; do nothing when FIGURES is None."""
    if FIGURES is None:
        return
    folder = Path(FIGURES)
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / name, dpi=150, bbox_inches="tight")


# %% [markdown]
# ## What a structureless copy is
#
# A copy invents nothing: it keeps every value of every column and changes only
# which row each value sits in. There are two families:
#
# - `gaussian_copula` (the default) reorders the values in each column such that
#   the columns keep one smooth, elliptical trend between them;
# - `independent_marginals` shuffles each column on its own, removing the trend as
#   well.
#
# A copy keeps anything visible in one column alone, such as two peaks, while
# removing structure that shows only when columns are combined.

# %% [markdown]
# In [Fe/H] against [Mg/Fe], the 9,242 real stars of the APOGEE table form two
# sequences, the high- and low-alpha discs. The cell below draws one copy of each
# family and checks that every column of each copy holds exactly the observed
# values.

# %%
x_apogee, names = example_table()
fe_h, mg_fe = names.index("fe_h"), names.index("mg_fe")
copies = {
    family: starfold.generate_null(x_apogee, family, seed=RANDOM_STATE)
    for family in ("gaussian_copula", "independent_marginals")
}

rho = spearmanr(x_apogee[:, fe_h], x_apogee[:, mg_fe]).statistic
print(f"{'data':22s} Spearman [Fe/H] vs [Mg/Fe] {rho:+.2f}")
for family, copy in copies.items():
    if not np.array_equal(np.sort(copy, axis=0), np.sort(x_apogee, axis=0)):
        msg = f"A column of the {family} copy does not hold the observed values."
        raise RuntimeError(msg)
    rho = spearmanr(copy[:, fe_h], copy[:, mg_fe]).statistic
    moved = np.mean(np.any(copy != x_apogee, axis=1))
    print(
        f"{family:22s} Spearman [Fe/H] vs [Mg/Fe] {rho:+.2f}, "
        f"sorted columns identical, {moved:.1%} of rows changed"
    )

# %%
panels = {
    "data": x_apogee,
    "copy: same values, no groups": copies["gaussian_copula"],
    "shuffled copy: same values, no trend": copies["independent_marginals"],
}
fe_edges = np.linspace(x_apogee[:, fe_h].min(), x_apogee[:, fe_h].max(), 81)
mg_edges = np.linspace(x_apogee[:, mg_fe].min(), x_apogee[:, mg_fe].max(), 81)
counts = {
    title: np.histogram2d(matrix[:, fe_h], matrix[:, mg_fe], bins=(fe_edges, mg_edges))[0]
    for title, matrix in panels.items()
}
norm = LogNorm(vmin=1, vmax=max(c.max() for c in counts.values()))

fig, axes = plt.subplot_mosaic(
    [["j0", "j0", "j1", "j1", "j2", "j2"], ["m0", "m0", "m0", "m1", "m1", "m1"]],
    figsize=(11, 7),
    height_ratios=(1.4, 1),
    layout="constrained",
)
for i, (title, count) in enumerate(counts.items()):
    ax = axes[f"j{i}"]
    mesh = ax.pcolormesh(
        fe_edges, mg_edges, np.ma.masked_equal(count.T, 0), cmap="viridis", norm=norm
    )
    ax.set(title=title, xlabel="[Fe/H] (dex)", ylabel="[Mg/Fe] (dex)")
fig.colorbar(mesh, ax=[axes["j0"], axes["j1"], axes["j2"]], label="stars per bin")

for ax, column, edges, label in (
    (axes["m0"], fe_h, fe_edges, "[Fe/H]"),
    (axes["m1"], mg_fe, mg_edges, "[Mg/Fe]"),
):
    ax.hist(x_apogee[:, column], bins=edges, color="C0", alpha=0.35, label="data")
    ax.hist(
        copies["gaussian_copula"][:, column],
        bins=edges,
        histtype="step",
        color="C1",
        linewidth=2,
        label="copy (gaussian_copula)",
    )
    ax.hist(
        copies["independent_marginals"][:, column],
        bins=edges,
        histtype="step",
        color="C2",
        linestyle="--",
        label="shuffled copy (independent_marginals)",
    )
    ax.set(title=f"{label} alone: identical", xlabel=f"{label} (dex)", ylabel="stars")
handles, labels = axes["m0"].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncol=3)
save_figure(fig, "apogee_surrogate.png")
plt.show()

# %% [markdown]
# ## Running the test
#
# `starfold.null_test` takes the rows the search used, the search itself, and the
# same `Config`, seed, engine and (above 20,000 rows per resample) `n_jobs`, to make
# every copy go through exactly the steps the data went through. If you searched
# a custom `Space`, pass that `Config` again. A run saved on cuML must be tested on
# a machine where cuML works. Run it once, on the result you report.

# %%
# result = starfold.recover(X, random_state=0)
# null = starfold.null_test(X[result.kept_rows], result.search, random_state=0)
# print(null.p_value, null.structure_supported)

# %% [markdown]
# ## Recorded results
#
# The two records below come from `python -m bench.null_stage digits <saved run>`
# and `python -m bench.null_stage apogee <saved run>` at the default budgets. In
# each record, `null_values` holds each copy's best score and `observed` the data's.
# `exceedances` counts the copies at or above the data; a tie counts against the
# data.

# %%
records = {
    name: json.loads((RECORDS / f"{name}_null.json").read_text()) for name in ("digits", "apogee")
}
for name, record in records.items():
    low, high = record["p_intervals"][0]
    print(
        f"{name:7s} observed {record['observed']:.3f}, {record['n_realisations'][0]} copies, "
        f"{record['exceedances'][0]} at or above the data, p {record['p_value']:.2f}, "
        f"fraction at or above {low:.3f} to {high:.3f} (95% Monte Carlo interval), "
        f"{record['n_fits']} fits, supported {record['structure_supported']}"
    )

# %%
titles = {"digits": "Digits (1,797 images)", "apogee": "APOGEE DR19 (9,242 stars)"}
for name, record in records.items():
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    values = np.asarray(record["null_values"][0])
    edges = np.histogram_bin_edges(np.append(values, record["observed"]), bins=12)
    ax.hist(values, bins=edges, color="C0", alpha=0.7, label="copies (gaussian_copula)")
    ax.axvline(record["observed"], color="black", linestyle="--", label="data")
    low, high = record["p_intervals"][0]
    ax.set(
        title=(
            f"{titles[name]}: p = {record['p_value']:.2f}, copies at or above the data "
            f"{low:.2f} to {high:.2f}"
        ),
        xlabel="best score of the exploration design",
        ylabel="copies",
    )
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    save_figure(fig, f"{name}_null.png")
    plt.show()

# %% [markdown]
# For the digits, no copy scored as high as the data, so p is at its floor of 0.05
# and the table's structure is supported against this family. For APOGEE, one copy
# scored as high as the data; with p = 0.10 above alpha, the test at 19 copies does
# not support structure for the whole table. That is not a finding of no groups. It
# means at least one copy that keeps the table's overall trend scored as high as
# the table.

# %% [markdown]
# ## Why 0.05 is the smallest p-value at 19 copies
#
# With h of L copies at or above the data, p = (h + 1) / (L + 1): the data count
# as one more draw (Phipson and Smyth 2010, doi:10.2202/1544-6115.1585). With all
# 19 copies below the data, p = 1 / 20 = 0.05, exactly the default alpha. Thus
# p = 0.05 means that no copy scored as high as the data, without saying by how
# much the data beat them.
#
# The test stops once 3 copies reach the data (Besag and Clifford 1991,
# doi:10.1093/biomet/78.2.301), after which p is at least 4 / 20 and cannot reach
# 0.05. The interval is the 95 percent Clopper-Pearson range of the fraction h / L
# (Clopper and Pearson 1934, doi:10.1093/biomet/26.4.404), not an interval on p,
# and measures Monte Carlo precision only.

# %%
print("copies   p for h = 0, 1, 2, 3 copies at or above the data")
for n_copies in (4, 9, 19, 99):
    row = "  ".join(f"{monte_carlo_pvalue(h, n_copies):.3f}" for h in range(4))
    print(f"{n_copies:6d}   {row}")

# %% [markdown]
# A lower floor needs more copies: `dataclasses.replace(config, n_null_max=99)`
# brings it to 0.01 at up to 10,098 fits. Neither `n_null_max` nor `alpha` changes
# the search, so the replay stays matched.

# %% [markdown]
# ## Attaching the test to a saved result
#
# A result from `recover` with the test off carries an empty null and the flag
# `null_not_run`. Replace the null with `dataclasses.replace`, and re-derive the
# flags with `starfold.result.derive_flags` in the same step: `starfold.load`
# re-derives the flags from the evidence and refuses a result whose stored flags
# differ. The provenance still describes the `recover` run, while the null evidence
# records its own families, fits and seconds.

# %%
# provenance = result.provenance
# calibrated = dataclasses.replace(
#     result,
#     null=null,
#     flags=derive_flags(
#         labels=result.labels,
#         search=result.search,
#         worlds=result.worlds,
#         consensus=result.consensus,
#         null=null,
#         backends=result.backends,
#         rows_rejected=int(provenance["n_input_rows"]) - int(provenance["n_kept_rows"]),
#         n_jobs_forced=bool(provenance["n_jobs_forced_to_one_on_cuml"]),
#     ),
# )
# starfold.save(calibrated, "runs/digits_null")
