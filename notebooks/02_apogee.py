# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # Worked example: 9,242 APOGEE DR19 stars
#
# This notebook runs Starfold on a real, unlabelled table, reads the evidence in physical
# units and ends with what the run supports and what it does not. It also redraws the APOGEE
# figures of the handbook.
#
# The table holds 9,242 Milky Way stars in eight columns: three abundances from the APOGEE
# DR19 ASPCAP pipeline and five orbital quantities integrated with galpy from Gaia DR3
# astrometry. Its sources, cuts and checksum are listed in the
# [provenance note](../bench/data/stellar_chemokinematics_apogee_dr19.md).

# %% [markdown]
# Treat the run as a check of the method, not as a finding about the Milky Way. The disc is
# known to hold two chemical sequences, so a method that finds no structure here, or fifty
# groups, has a problem. Finding the known split without being told about it passes a sanity
# check and nothing more.

# %% [markdown]
# At the default budgets the run takes under 4 minutes, 334 fits and 1.7 GB of host memory
# on a laptop GPU with RAPIDS cuML. The CPU path has not been timed at this size; for a quick
# CPU pass on a seeded subsample, set `N_ROWS = 3000`. The calibration test is off: it costs
# 102 fits per copy of the table for up to 19 copies (1,938 fits), about six times the rest of
# the run.
#
# The notebook needs a clone of the repository, because the table ships with the repository
# and not with the installed package. The first cell finds the clone.

# %%
import sys
from pathlib import Path

here = Path.cwd().resolve()
ROOT = next((p for p in (here, *here.parents) if (p / "pyproject.toml").is_file()), None)
if ROOT is None:
    msg = "Run this notebook from inside a clone of the starfold repository."
    raise FileNotFoundError(msg)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print(ROOT)

# %% [markdown]
# ## Parameters
#
# `ENGINE = "auto"` uses cuML when it imports and sees a GPU, and otherwise warns and runs on
# the CPU; `"cuml"` fails instead of falling back. `N_JOBS` sets worker processes on the CPU
# and is forced to 1 on cuML. Set `FIGURES = ROOT / "docs" / "figures"` to rewrite the
# handbook PNGs; with `None`, nothing is written.

# %% tags=["parameters"]
RANDOM_STATE = 0
ENGINE = "auto"
N_JOBS = 1
N_ROWS = None  # 3000 runs a seeded subsample, for a quick CPU pass
FIGURES = None

# %%
import matplotlib.pyplot as plt
import numpy as np
from bench.datasets import example_table
from matplotlib.lines import Line2D

import starfold
from starfold._engine import derive_seed
from starfold.backend import fit_pipeline
from starfold.consensus import coassociation
from starfold.prep import scale

if FIGURES is not None and N_ROWS is not None:
    msg = "The handbook figures use every star; set N_ROWS = None to write them."
    raise ValueError(msg)
RUN = ROOT / "notebooks" / "runs" / ("apogee" if N_ROWS is None else f"apogee_{N_ROWS}")
TAB10 = plt.get_cmap("tab10")


def save_figure(fig, name):
    """Write a figure under FIGURES as a 150 dpi PNG; do nothing when FIGURES is None."""
    if FIGURES is None:
        return
    path = Path(FIGURES) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


# %% [markdown]
# ## The table
#
# - `fe_h` is [Fe/H], the iron abundance relative to the Sun in dex: 0 is solar, -1 is a
#   tenth of solar.
# - `alpha_m` ([alpha/M]) and `mg_fe` ([Mg/Fe]) indicate how rich a star is in alpha elements
#   such as magnesium. Stars that formed early and quickly are richer in them.
# - `J_R` and `J_z` are orbital actions in kpc km/s: how far the orbit swings in and out,
#   and up and down through the disc.
# - `L_z` is the angular momentum about the Galaxy's axis, large and positive for stars that
#   rotate with the disc.
# - `ecc` is the orbital eccentricity, 0 for a circle. `E` is the orbital energy in
#   km^2/s^2, more negative for a more tightly bound star.

# %%
x, names = example_table()
if N_ROWS is not None:
    rng = np.random.default_rng(RANDOM_STATE)
    x = x[np.sort(rng.choice(x.shape[0], size=N_ROWS, replace=False))]

UNITS = {
    "fe_h": "dex",
    "alpha_m": "dex",
    "mg_fe": "dex",
    "J_R": "kpc km/s",
    "J_z": "kpc km/s",
    "L_z": "kpc km/s",
    "ecc": "",
    "E": "km^2/s^2",
}
LABELS = {
    "fe_h": "[Fe/H] (dex)",
    "alpha_m": r"[$\alpha$/M] (dex)",
    "mg_fe": "[Mg/Fe] (dex)",
    "J_R": r"$J_R$ (kpc km s$^{-1}$)",
    "J_z": r"$J_z$ (kpc km s$^{-1}$)",
    "L_z": r"$L_z$ (kpc km s$^{-1}$)",
    "ecc": "eccentricity",
    "E": r"$E$ (km$^2$ s$^{-2}$)",
}
SHORT = {
    "fe_h": "[Fe/H]",
    "alpha_m": r"[$\alpha$/M]",
    "mg_fe": "[Mg/Fe]",
    "J_R": r"$J_R$",
    "J_z": r"$J_z$",
    "L_z": r"$L_z$",
    "ecc": "ecc",
    "E": r"$E$",
}


def fmt(value):
    """Three significant figures, or whole numbers with separators from 1,000 up."""
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.3g}"


low, middle, high = np.percentile(x, [5, 50, 95], axis=0)
print(f"{x.shape[0]:,} stars")
print(f"{'column':8}{'median':>10}{'5 to 95 percent':>26}  unit")
for j, name in enumerate(names):
    print(f"{name:8}{fmt(middle[j]):>10}{fmt(low[j]):>13} to {fmt(high[j]):>9}  {UNITS[name]}")

# %% [markdown]
# ## The run
#
# UMAP draws a map of the stars in a few dimensions that keeps neighbours in the table close,
# and HDBSCAN marks the dense regions of that map as groups and leaves the stars between them
# unassigned. Their settings can turn the same table into a handful of groups or hundreds, so
# Starfold chooses them first.

# %% [markdown]
# `recover` tries up to 48 settings: the library defaults, 16 spread evenly over the space,
# and the rest where earlier scores were highest. Each setting is scored without labels on the
# same three 80 percent subsets. The winner is fitted once on all stars, and that fit is the
# partition.
# Starfold then refits it 32 times on fresh 80 percent subsets with new seeds to see which
# groups come back; these refits are the worlds in `result.worlds`.

# %%
CONFIG = starfold.Config()  # null_families is empty, so the calibration test is skipped
result = starfold.recover(x, config=CONFIG, random_state=RANDOM_STATE, engine=ENGINE, n_jobs=N_JOBS)
starfold.save(result, RUN)
kept = x[result.kept_rows]
engine = result.provenance["engine_resolved"]  # "cpu" or "cuml", as the stage functions need

provenance = result.provenance
print(
    f"engine {provenance['engine_resolved']}, {provenance['total_fits']} fits, "
    f"{provenance['stage_seconds']['total']:.0f} s, "
    f"peak host memory {provenance['peak_rss_mb']:,.0f} MB"
)
print("seconds per stage:", {k: round(v) for k, v in provenance["stage_seconds"].items()})
print("fits per stage:", provenance["stage_fits"])
print(f"saved to {RUN}")

# %% [markdown]
# ## Flags
#
# Flags are the result's verdict on itself, in a form a script can check. `null_not_run` is
# expected, because the calibration test was not run, and so is `consensus_subsampled` on all
# stars, because the co-association check below uses 4,096 of them.

# %%
MEANING = {
    "search_stopped_early": "the best score stopped improving before the trial budget ran out",
    "search_budget_exhausted": "the search used every trial; a larger n_trials may help",
    "selected_is_default": "the library defaults won the search",
    "default_within_noise": "the winner and the defaults score within the comparison's noise",
    "no_groups_in_reference": "the fit on all stars found no groups",
    "high_unresolved_mass": "the mean unresolved mass is above 0.5",
    "count_unstable": "fewer than half the refits find the partition's number of groups",
    "world_fit_failures": "at least one refit raised an error; see result.worlds.world_errors",
    "consensus_subsampled": "the co-association check ran on a seeded subset of stars",
    "consensus_disagrees": "the cut of the co-association matrix agrees with the partition "
    "below 0.5",
    "null_not_run": "no calibration test ran, so there is no p-value; run starfold.null_test",
    "observed_statistic_zero": "the observed statistic is zero, so the test was skipped",
    "no_structure_supported": "the calibration test ran and p is above alpha",
    "backend_disagreement": "CPU and GPU fits of the selected setting disagree",
    "rows_rejected_nonfinite": "rows with missing or infinite values were dropped",
    "n_jobs_forced_to_one_on_cuml": "cuML runs in one process, so n_jobs was set to 1",
}
for flag in result.flags:
    print(f"{flag}: {MEANING.get(flag, 'see the handbook')}")
if not result.flags:
    print("no flags raised")

# %% [markdown]
# ## Groups and how often they come back
#
# Group recurrence matches each group with its closest group in every refit, takes the stars
# the two share as a fraction of the stars in either (Jaccard) and averages over refits: 1
# means the group comes back with the same members every time. For the group count, the
# interval is the 5 to 95 percent range over refits: a stability interval, not a confidence
# interval. Unresolved mass scores each star in every refit that drew it: 1 if the refit left
# the star unassigned, otherwise the share of its refit group that the partition leaves
# unassigned. Those scores are averaged over the star's refits, then over stars.

# %%
worlds = result.worlds
groups = np.unique(result.labels[result.labels >= 0])
noise = result.labels < 0
if groups.size <= 10:
    colours = [TAB10(i) for i in range(groups.size)]
else:
    colours = [plt.get_cmap("viridis")(t) for t in np.linspace(0.0, 0.9, groups.size)]

print(f"{result.n_groups} groups; {noise.sum():,} of {noise.size:,} stars unassigned")
for i, g in enumerate(groups):
    low, high = worlds.group_recurrence_band[i]
    print(
        f"  group {g}: {np.sum(result.labels == g):,} stars, recurrence "
        f"{worlds.group_recurrence[i]:.3f} (5 to 95 percent over refits {low:.3f} to {high:.3f})"
    )
counts, frequency = np.unique(worlds.count_values, return_counts=True)
print(
    "groups per refit:",
    ", ".join(f"{c} in {f} refits" for c, f in zip(counts, frequency, strict=True)),
)
print(f"interval {worlds.count_interval[0]} to {worlds.count_interval[1]}")
print(f"unresolved mass {result.unresolved_mass:.3f}")

# %% [markdown]
# ## The setting the search chose
#
# Each setting is scored on how well a fit on one subset predicts the groups of stars it never
# saw, how many stars both fits assign, and how closely the groups follow the neighbours in
# the table itself. The search also scores the same setting on a structureless copy of the
# table and counts only the excess over the copy. For the fit on all stars, `min_cluster_size`
# and `min_samples` are scaled up from the 80 percent subsets to the full table.


# %%
def show(value):
    """Floats to three significant figures, anything else as text."""
    return f"{value:.3g}" if isinstance(value, float) else str(value)


settings = (result.params, result.deployed_params, result.default_params)
print(f"{'':18}{'on subsets':>12}{'all stars':>12}{'defaults':>12}")
for field in result.params.to_dict():
    print(f"{field:18}" + "".join(f"{show(getattr(s, field)):>12}" for s in settings))
search = result.search
print(
    f"{len(search.trials)} trials: {search.n_completed} complete, {search.n_pruned} pruned, "
    f"{search.n_completed_after_prune} completed after pruning; stop: {search.stop_reason}"
)
print(f"beats the defaults under the objective: {search.beats_default_under_objective}")

# %% [markdown]
# ## The map
#
# When the selected setting draws its map in two dimensions, the next cell refits it on all
# stars with the same seed and checks that the refit gives back exactly the same groups, so
# the picture is the map HDBSCAN cut. With more dimensions, the cell instead draws a separate
# two-dimensional refit for display only and says so in the axis label. This display refit
# adds one UMAP fit on the CPU. UMAP axes have no units; only which stars sit near which
# matters.

# %%
deployed = result.deployed_params
if deployed.n_components == 2:
    # seed id 4 is the seed recover uses for the fit on all stars
    fit = fit_pipeline(
        kept, deployed, seed=derive_seed(RANDOM_STATE, 4), engine=engine, scaling=CONFIG.scaling
    )
    np.testing.assert_array_equal(
        fit.labels, result.labels, err_msg="the refit did not reproduce the emitted groups"
    )
    layout = fit.embedding
    axis_note = "the map HDBSCAN cut"
else:
    import umap

    reducer = umap.UMAP(
        n_neighbors=deployed.n_neighbors,
        min_dist=deployed.min_dist,
        n_components=2,
        random_state=RANDOM_STATE,
    )
    layout = reducer.fit_transform(scale(kept, CONFIG.scaling))
    axis_note = f"display refit; groups cut in {deployed.n_components} dimensions"
print(f"layout: {axis_note}")

# %% [markdown]
# Opacity is each star's recurrence support: how consistently the star lands with the other
# members of its group across the 32 refits. Faint stars are the ones the refits place
# inconsistently.

# %%
own = np.clip(np.nan_to_num(worlds.own_support, nan=0.0), 0.05, 1.0)
fig, ax = plt.subplots(figsize=(6.5, 6), layout="constrained")
ax.scatter(layout[noise, 0], layout[noise, 1], s=2, color="0.75", lw=0)
handles = [
    Line2D([], [], ls="", marker="o", color="0.75", label=f"unassigned, {noise.sum():,} stars")
]
for i, (g, colour) in enumerate(zip(groups, colours, strict=True)):
    members = result.labels == g
    rgba = np.tile(colour, (members.sum(), 1))
    rgba[:, 3] = own[members]
    ax.scatter(layout[members, 0], layout[members, 1], s=3, c=rgba, lw=0)
    label = f"group {g}, {members.sum():,} stars, recurrence {worlds.group_recurrence[i]:.2f}"
    handles.append(Line2D([], [], ls="", marker="o", color=colour, label=label))
ax.set_xlabel(f"UMAP 1 ({axis_note})")
ax.set_ylabel("UMAP 2")
ax.set_title("Stars by group; opacity is recurrence support")
ax.legend(handles=handles, fontsize=8)
save_figure(fig, "apogee/01_layout.png")
plt.show()

# %% [markdown]
# ## What the regions of the map are
#
# The same map, coloured once by each column, shows what the regions are in measured
# quantities. Each colour scale is clipped to the column's 2nd to 98th percentile to keep a
# long tail from flattening the rest.

# %%
fig, axes = plt.subplots(2, 4, figsize=(13, 6), layout="constrained")
for ax, name, values in zip(axes.flat, names, kept.T, strict=True):
    low, high = np.percentile(values, [2, 98])
    points = ax.scatter(
        layout[:, 0], layout[:, 1], c=values, s=1, lw=0, cmap="viridis", vmin=low, vmax=high
    )
    fig.colorbar(points, ax=ax, label=LABELS[name])
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle(f"The map coloured by each column ({axis_note})")
save_figure(fig, "apogee/08_feature_layout.png")
plt.show()

# %% [markdown]
# ## The groups in physical units
#
# In every refit, each group is matched to its closest refit group, whose median is taken in
# each column. The table gives the median of those medians over the 32 refits, their 5 to 95
# percent range, and the median of all stars for comparison.

# %%
median, band = worlds.property_median, worlds.property_band
overall = np.median(kept, axis=0)
for i, g in enumerate(groups):
    print(f"group {g}: median over refits (5 to 95 percent), then all stars")
    for j, name in enumerate(names):
        print(
            f"  {name:8}{fmt(median[i, j]):>9} ({fmt(band[i, j, 0])} to {fmt(band[i, j, 1])})"
            f"{fmt(overall[j]):>11}  {UNITS[name]}"
        )

# %% [markdown]
# In the left panel, every star appears in the [Fe/H]-[Mg/Fe] plane with each group's median
# and its 5 to 95 percent range over refits. The right panel puts all eight columns on one
# axis as each group's median minus the table mean, in units of the table's standard
# deviation.


# %%
def spans(value, low, high):
    """Errorbar lengths below and above a value, never negative."""
    return np.clip([np.atleast_1d(value - low), np.atleast_1d(high - value)], 0.0, None)


fe, mg = names.index("fe_h"), names.index("mg_fe")
centre, spread = kept.mean(axis=0), kept.std(axis=0)
positions = np.arange(len(names))
offsets = np.linspace(-0.2, 0.2, groups.size) if groups.size > 1 else np.zeros(groups.size)
fig, (plane, panel) = plt.subplots(
    1, 2, figsize=(13, 5), width_ratios=(1, 1.5), layout="constrained"
)
plane.scatter(kept[noise, fe], kept[noise, mg], s=2, color="0.8", lw=0)
plane_handles = [Line2D([], [], ls="", marker="o", ms=4, color="0.8", label="unassigned stars")]
for i, (g, colour) in enumerate(zip(groups, colours, strict=True)):
    members = result.labels == g
    plane.scatter(kept[members, fe], kept[members, mg], s=2, color=colour, alpha=0.3, lw=0)
    marker = plane.errorbar(
        median[i, fe],
        median[i, mg],
        xerr=spans(median[i, fe], *band[i, fe]),
        yerr=spans(median[i, mg], *band[i, mg]),
        fmt="o",
        ms=8,
        mfc=colour,
        mec="k",
        ecolor="k",
        capsize=3,
        label=f"group {g} median, 5 to 95 percent over refits",
    )
    plane_handles.append(marker)
    z = (median[i] - centre) / spread
    z_band = (band[i] - centre[:, None]) / spread[:, None]
    panel.errorbar(
        positions + offsets[i],
        z,
        yerr=spans(z, z_band[:, 0], z_band[:, 1]),
        fmt="o",
        ms=6,
        color=colour,
        capsize=3,
        label=f"group {g}",
    )
plane.set_xlabel(LABELS["fe_h"])
plane.set_ylabel(LABELS["mg_fe"])
plane.set_title("Abundance plane")
plane.legend(handles=plane_handles, fontsize=8)
panel.axhline(0.0, color="0.5", lw=0.8, ls=":", label="table mean")
panel.set_xticks(positions, [SHORT[name] for name in names])
panel.set_xlabel("column")
panel.set_ylabel("group median minus table mean (standard deviations)")
panel.set_title("All columns, standardised")
panel.legend(fontsize=8)
save_figure(fig, "apogee/04_properties.png")
plt.show()

# %% [markdown]
# The bars show how far a group's median moves over the 32 refits. They are stability
# intervals, not confidence intervals, and contain no measurement error: the table has no
# per-star uncertainties.

# %% [markdown]
# ## Reading the groups
#
# The Milky Way disc has two chemical sequences in the [Fe/H]-[Mg/Fe] plane (Bensby et al.
# 2014, doi:10.1051/0004-6361/201322631; Hayden et al. 2015, doi:10.1088/0004-637X/808/2/132).
# The high-alpha sequence holds older, metal-poorer stars on hotter orbits, with lower `L_z`
# and larger `J_R`, `J_z` and eccentricity, while the low-alpha sequence reaches solar [Fe/H]
# and above on cooler, more circular orbits. Starfold was told none of this.

# %% [markdown]
# A group that is high in both alpha columns, low in [Fe/H] and `L_z`, and high in `J_R`,
# `J_z` and eccentricity sits on the high-alpha sequence, and a group with the reverse pattern
# sits on the low-alpha sequence. The right panel above makes that comparison for all eight
# columns at once.

# %% [markdown]
# ## Co-association: a check that ignores label numbers
#
# Take the refits that sampled and assigned both stars of a pair. Their co-association is the
# share of those refits that put the two in the same group. Pairs in the same group of the
# partition should sit near 1 and pairs in different groups near 0. A pair no refit assigned
# together counts as 0, and with 32 refits the largest value is 1 - 1/64. The contrast, the
# difference of the two means, does not depend on how each refit numbers its groups.

# %% [markdown]
# The check also cuts the co-association matrix into groups of its own and compares that cut
# with the partition (the agreement, flagged as `consensus_disagrees` below 0.5). However,
# the cut is a second clustering and can fail where the pair frequencies still separate the
# groups clearly, so read the contrast first. The check uses a seeded subset of 4,096 stars,
# as the pair matrices for all 9,242 stars would exceed the 512 MiB budget.

# %%
consensus = result.consensus
rows = consensus.rows
together = coassociation(worlds.world_labels, rows, engine=engine)
np.subtract(1.0, together, out=together)  # distance to co-association
reference = result.labels[rows]
pairs = np.logical_and.outer(reference >= 0, reference >= 0)
np.fill_diagonal(pairs, val=False)
same = np.equal.outer(reference, reference)
within, between = together[pairs & same], together[pairs & ~same]
del together, pairs, same

print(
    f"{consensus.n_rows:,} stars{' (seeded subset)' if consensus.subsampled else ''}: "
    f"within {consensus.within_association:.3f}, between {consensus.between_association:.3f}, "
    f"contrast {consensus.contrast:.3f}, cut agreement {consensus.agreement:.3f}"
)
fig, ax = plt.subplots(figsize=(6.5, 4.5), layout="constrained")
bins = np.linspace(0.0, 1.0, 41)
for values, colour, kind, mean in (
    (within, TAB10(0), "same group", consensus.within_association),
    (between, TAB10(3), "different groups", consensus.between_association),
):
    if values.size:
        ax.hist(
            values,
            bins=bins,
            density=True,
            alpha=0.6,
            color=colour,
            label=f"{kind} (mean {mean:.3f})",
        )
ax.set_yscale("log")
ax.set_xlabel("co-association: share of refits assigning both that grouped them together")
ax.set_ylabel("density of star pairs")
ax.set_title(f"Contrast {consensus.contrast:.3f} on {consensus.n_rows:,} stars")
ax.legend()
save_figure(fig, "apogee/07_consensus.png")
plt.show()

# %% [markdown]
# ## Score on the data against score on a copy
#
# Every trial was scored twice on the same subsets, once on the data and once on a copy of the
# table that keeps each column's values and a single smooth correlation between columns but
# has the groups removed. What a setting scores on the copy, it scores without any structure.
# The search ranks settings by the difference, and points on the diagonal gain nothing from
# the data.

# %% [markdown]
# The points are the search's own scores. For the final choice, the top three and the
# defaults are re-scored on fresh subsets against three fresh copies and the largest fresh gap
# is kept; the cell prints those gaps under the figure.

# %%
selected = next(t for t in search.trials if t.number == search.candidates[0].trial_number)
default = search.trials[0]
complete = [t for t in search.trials if t.state == "complete"]
PHASES = {
    "exploration": (TAB10(0), "Sobol design"),
    "convergence": (TAB10(1), "TPE search"),
    "completion": (TAB10(2), "completed after pruning"),
}
fig, ax = plt.subplots(figsize=(5.5, 5.5), layout="constrained")
for phase, (colour, label) in PHASES.items():
    scored = [t for t in complete if t.phase == phase]
    if scored:
        ax.scatter(
            [t.raw_score for t in scored],
            [t.null_score for t in scored],
            s=18,
            color=colour,
            label=label,
        )
ax.scatter(
    default.raw_score,
    default.null_score,
    marker="*",
    s=220,
    color="k",
    zorder=3,
    label="defaults (trial 0)",
)
ax.scatter(
    selected.raw_score,
    selected.null_score,
    s=260,
    facecolors="none",
    edgecolors="k",
    linewidths=1.5,
    zorder=4,
    label=f"selected (trial {selected.number})",
)
limit = 1.05 * np.nanmax([[t.raw_score, t.null_score] for t in complete])
ax.plot([0.0, limit], [0.0, limit], ls="--", color="0.5", label="no advantage")
ax.set(xlim=(0.0, limit), ylim=(0.0, limit), aspect="equal")
ax.set_xlabel("score on the data")
ax.set_ylabel("score on the structureless copy")
ax.legend(loc="upper left", fontsize=8)
save_figure(fig, "apogee_gap.png")
plt.show()
print(
    f"fresh gap over three fresh copies: selected {search.gap_selected:.3f}, "
    f"defaults {search.gap_default:.3f}"
)

# %% [markdown]
# ## The calibration test, as its own step
#
# The test replays the 17-trial exploration (the defaults and the 16 Sobol points) on copies
# of the table to ask how often a copy scores at least as high as the real table. Each copy
# costs 102 fits, and the test stops at 19 copies (1,938 fits) or as soon as 3 copies score at
# least as high as the data. With 19 copies the smallest possible p-value is 0.05. Run it
# once, on the result you report, with the same config, seed and engine:
#
# ```python
# null = starfold.null_test(
#     kept, result.search, config=CONFIG, random_state=RANDOM_STATE, engine=engine
# )
# print(null.p_value, null.p_intervals, null.structure_supported)
# ```
#
# Here `result.null.skipped_reason` is `"null_not_run"`, and its p-value of 1.0 is a
# placeholder, not a measurement. From the repository root,
# `python -m bench.null_stage apogee notebooks/runs/apogee` runs the test on this table and
# writes `bench/results/apogee_null.json`. The command reloads the full table and for this
# reason works on the all-stars run only. Notebook `03_calibration.py` shows how to attach the
# test to a saved result.

# %% [markdown]
# ## How to report it
#
# The reference run (all stars, `RANDOM_STATE = 0`, default budgets, laptop GPU) gave:
#
# | quantity | value |
# |---|---|
# | groups, stars in each, unassigned | 2 groups of 1,588 and 6,586 stars; 1,068 unassigned |
# | groups per refit, 5 to 95 percent over 32 refits | 2 to 2 |
# | group recurrence | 0.91 (0.89 to 0.94) and 0.92 (0.88 to 0.94) |
# | unresolved mass | 0.154 |
# | co-association within, between, contrast | 0.979, 0.003, 0.976 |
# | group medians of [Fe/H] and [Mg/Fe] (dex) | group 0: -0.42 and +0.27; group 1: -0.08 and +0.04 |
# | regrouping agreement | 0.0, all 4,096 subset stars cut as noise |
# | calibration p-value and interval | 0.10 (0.001 to 0.260), 1 of 19 copies at or above |

# %% [markdown]
# Name the evidence behind each number. The groups come from one fit of the selected setting
# on all stars, while recurrence, the count interval and the bands come from 32 refits on 80
# percent subsets with new seeds. The p-value comes from the separate calibration run in
# `bench/results/apogee_null.json`; it compares the best score of the 17-trial exploration on
# the data with the same exploration on structureless copies.

# %% [markdown]
# What the run supports:
#
# - How reproducible the split is. Recurrence and the count interval say how often the same
#   groups come back when a fifth of the stars and the seeds change.
# - How the groups differ in physical units (property table and figure) and how much those
#   differences move across refits.
# - A sanity check of the method, when the group medians fall on the two alpha sequences: a
#   known feature of the disc, recovered with no labels and no astronomy in the method.

# %% [markdown]
# What it does not support:
#
# - That the groups are distinct populations. Reproducible is not true, and the calibration
#   test does not show it either: p at or below 0.05 says only that no copy keeping each
#   column's values and the correlations between columns scored as high as the table, while p
#   above 0.05 says at least one did.
# - That the Milky Way holds this many populations. The count depends on the settings and on
#   what you call a population, and its interval is over refits, not a confidence interval.
# - A membership probability for each star. Support counts how often a star recurs with its
#   group, and HDBSCAN's strengths describe one fit. Neither is a posterior.
# - Anything about the Galaxy as a whole. These are APOGEE stars with SNR above 70 and Gaia
#   orbits inside a volume cut, and no selection function was modelled.
# - Anything about measurement error. With no per-star uncertainties in the table, the bands
#   reflect subsampling and seeds only; `X_err` adds measurement draws to every refit.
# - A citable result. The bundled table is for tutorials; for published work, rebuild it from
#   APOGEE and Gaia.
