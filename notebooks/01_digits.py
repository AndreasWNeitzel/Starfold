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
# # First run: handwritten digits
#
# This notebook runs Starfold once on 1,797 real images of handwritten digits
# and reads every part of the result: the flags, the groups, how often they
# come back, and the settings that produced them. It also redraws five of the
# Digits figures in the handbook; notebook 03 draws the calibration figure.
#
# From a clone, install with `uv sync --extra bench` (add `--extra gpu-cu13` or
# `--extra gpu-cu12` for cuML), then open this file in Jupyter with the kernel of
# the project's `.venv`; Jupytext reads it as a notebook.
#
# Starfold never sees which digit an image shows. The true digits stay in `y`
# and appear only in cells marked **post hoc**, to check the answer afterwards.
#
# On a laptop RTX 3060 with cuML the run takes about 80 seconds, 314 fits and
# 1.4 GB of host memory. No CPU timing is recorded; expect the CPU path to take
# several times longer, and set `N_JOBS = 3` to run fits in parallel. The
# display refit further down adds one UMAP fit on the CPU.

# %% [markdown]
# ## UMAP and HDBSCAN, briefly
#
# UMAP turns the 64 pixel columns into a few coordinates, keeping images that
# look alike close together. HDBSCAN then finds dense clumps in those
# coordinates: each clump is a group, and rows between clumps are noise
# (label -1).
#
# Their settings can turn the same table into a handful of groups or hundreds.
# Starfold chooses the settings without labels and then measures how
# reproducible the answer is.

# %%
import sys
from pathlib import Path

ROOT = next(
    (
        folder
        for folder in (Path.cwd(), *Path.cwd().parents)
        if (folder / "pyproject.toml").is_file()
    ),
    None,
)
if ROOT is None:
    msg = "Run this notebook from inside a clone of the starfold repository."
    raise FileNotFoundError(msg)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print(ROOT)

# %% [markdown]
# ## Parameters
#
# `ENGINE = "auto"` uses cuML when it imports and sees a GPU, and otherwise
# warns and runs on the CPU. Set `FIGURES = ROOT / "docs" / "figures"` to
# rewrite the handbook PNGs; with `None`, nothing is written.

# %% tags=["parameters"]
RANDOM_STATE = 0
ENGINE = "auto"
N_JOBS = 1
FIGURES = None

# %%
import matplotlib.pyplot as plt
import numpy as np
import umap
from bench.beats_default import coverage_adjusted_ari
from matplotlib import ticker
from matplotlib.lines import Line2D
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

import starfold
from starfold.result import FLAG_VOCABULARY

RUN_DIR = ROOT / "notebooks" / "runs" / "digits"
TAB10 = plt.get_cmap("tab10")
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
)


def save_figure(fig, name):
    """Write the figure to FIGURES / name at 150 dpi; do nothing when FIGURES is None."""
    if FIGURES is None:
        return
    folder = Path(FIGURES)
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / name, dpi=150, bbox_inches="tight")
    print(f"wrote {folder / name}")


# %% [markdown]
# ## The data
#
# Each row is one 8 by 8 image, flattened to 64 pixel values from 0 to 16.
# `y` holds the true digit of each image; it is kept aside.

# %%
X, y = load_digits(return_X_y=True)
print(f"{X.shape[0]} images, {X.shape[1]} pixel columns, values {X.min():.0f} to {X.max():.0f}")
print(f"true digits, kept aside: {np.unique(y).size} classes")

# %% [markdown]
# ## One call
#
# `recover` searches the settings, fits the chosen setting on all rows, refits
# it on 32 random 80 percent subsets, checks the result with a co-association
# matrix, and raises flags wherever the evidence is weak. The calibration test
# is off by default because it costs far more than the rest of the run; the
# section on it below says how to run it.

# %%
config = starfold.Config()
print("calibration test families:", config.null_families or "none, the test is off")
result = starfold.recover(X, config=config, random_state=RANDOM_STATE, engine=ENGINE, n_jobs=N_JOBS)

prov = result.provenance
print(f"engine: asked for {prov['engine_requested']}, used {prov['engine_resolved']}")
print(f"starfold {prov['starfold_version']}, Python {prov['python']}, {prov['platform']}")
print(f"rows kept: {prov['n_kept_rows']} of {prov['n_input_rows']}, columns: {prov['n_features']}")
print("seconds per stage:", {k: round(v, 1) for k, v in prov["stage_seconds"].items()})
print("fits per stage:", prov["stage_fits"], "total", prov["total_fits"])
print(f"peak host memory: {prov['peak_rss_mb']:.0f} MB")

# %% [markdown]
# ## Flags first
#
# Flags are the machine-readable warnings. Read them before any number.
# `null_not_run` is expected here, and one of the two `search_...` flags is
# always raised: it only says how the search ended.

# %%
FLAG_TEXT = {
    "search_stopped_early": "the search stopped once its best score stopped improving; "
    "this only says how the search ended",
    "search_budget_exhausted": "the search used every trial it was given; "
    "this only says how the search ended",
    "selected_is_default": "the library defaults won the fresh re-score",
    "default_within_noise": "the fresh scores of the chosen setting and the defaults differ "
    "by at most twice the standard error of their paired differences; the choice between "
    "them is not resolved",
    "no_groups_in_reference": "the fit on all rows put every row in noise",
    "high_unresolved_mass": "the unresolved mass is above 0.5: on average, most of a row's "
    "refit outcome is noise rather than its group",
    "count_unstable": "fewer than half of the refits found the same number of groups "
    "as the fit on all rows",
    "world_fit_failures": "at least one refit raised an error; see result.worlds.world_errors",
    "consensus_subsampled": "the co-association check ran on a seeded subset of rows "
    "to stay inside its memory budget",
    "consensus_disagrees": "the cut of the co-association matrix agrees with labels "
    "below 0.5; read the contrast before concluding anything",
    "null_not_run": "the calibration test did not run, so there is no p-value "
    "(expected here, see the calibration section)",
    "observed_statistic_zero": "the best score of the exploration design is zero, "
    "so there is nothing to calibrate",
    "no_structure_supported": "the calibration test ran and gave a p-value above alpha",
    "backend_disagreement": "CPU and cuML partitions agree below 0.8",
    "rows_rejected_nonfinite": "some input rows held NaN or infinity and were dropped; "
    "result.kept_rows lists the rows used",
    "n_jobs_forced_to_one_on_cuml": "n_jobs above 1 was asked for on cuML, which runs one job",
}
if set(FLAG_TEXT) != set(FLAG_VOCABULARY):
    msg = "FLAG_TEXT does not cover the flag vocabulary of this Starfold version."
    raise RuntimeError(msg)

print(f"{len(result.flags)} of {len(FLAG_VOCABULARY)} flags raised")
for flag in result.flags:
    print(f"- {flag}: {FLAG_TEXT[flag]}")

# %% [markdown]
# ## Groups, count and reproducibility
#
# `labels` has one entry per kept row; `result.kept_rows` maps it back to `X`.
# Each refit fits the chosen setting again on a random 80 percent of the rows.
#
# - **count interval**: the 5th to 95th percentile of the number of groups the
#   refits found, rounded outward to whole numbers. It is a stability range,
#   not a confidence interval.
# - **unresolved mass**: in each refit, a row counts 1 if it lands in noise,
#   and otherwise the share of its refit group that the full fit calls noise.
#   This is averaged over the refits that drew the row, then over rows.
# - **group recurrence**: for each group, the mean over refits of its best
#   Jaccard overlap with a refit group (shared rows over rows in either).
#   1 means the group comes back exactly.

# %%
w = result.worlds
labels = result.labels
sizes = np.bincount(labels[labels >= 0], minlength=result.n_groups)
print(f"groups in the fit on all rows: {result.n_groups}")
print(f"noise rows: {(labels < 0).sum()} of {labels.size}")
print(
    f"refits: {w.count_values.size}; groups found: {w.count_interval[0]} to {w.count_interval[1]}"
)
print(f"unresolved mass: {result.unresolved_mass:.3f}")
print("group recurrence, with its 5 to 95 percent band over refits:")
for group, (recurrence, (low, high), size) in enumerate(
    zip(w.group_recurrence, w.group_recurrence_band, sizes, strict=True)
):
    print(
        f"  group {group:2d}: {size:4d} rows, recurrence {recurrence:.2f} ({low:.2f} to {high:.2f})"
    )

# %% [markdown]
# ## The settings
#
# `params` is the setting the search chose on 80 percent subsets.
# `deployed_params` is the same setting with `min_cluster_size` and
# `min_samples` scaled up to all rows; it produced `labels`. `default_params`
# are the umap-learn and hdbscan defaults, fitted once on all rows as the
# comparison (`default_labels`).


# %%
def as_text(value):
    return f"{value:.3f}" if isinstance(value, float) else str(value)


columns = {
    "chosen": result.params,
    "deployed": result.deployed_params,
    "defaults": result.default_params,
}
print(f"{'setting':<20}" + "".join(f"{name:>12}" for name in columns))
for name in result.params.to_dict():
    row = "".join(f"{as_text(p.to_dict()[name]):>12}" for p in columns.values())
    print(f"{name:<20}{row}")

# %% [markdown]
# ## How the setting was chosen
#
# The search tries up to 48 settings: trial 0 is the library defaults,
# trials 1 to 16 a Sobol design that spreads over the space, and the rest come
# from TPE (tree-structured Parzen estimator), a sampler that learns from
# earlier trials.
#
# Each trial is fitted on 3 shared 80 percent subsets. A fit on one subset
# predicts the group of each row it never saw from that row's nearest
# neighbours, and the prediction is compared with the fit on another subset
# that did see the row. Two more terms count the share of rows both fits
# assign and how well the groups follow the neighbour graph of the table
# itself; a zero in any of the three makes the score zero.
#
# The same setting is also scored on a copy of the table with the groups
# removed: every column keeps exactly its values and the overall correlation
# between columns stays, but the rows form one smooth elliptical cloud. The
# trial score is the excess over the copy, the gap.
#
# The top 3 trials and the defaults are then scored again on fresh subsets and
# fresh copies, and the largest fresh gap wins. That table is the sensitivity
# report: close runners-up mean the choice among them is not sharp.

# %%
s = result.search
print(f"trials run: {len(s.trials)} of {config.n_trials}; stop reason: {s.stop_reason}")
print(
    f"complete {s.n_completed}, pruned {s.n_pruned}, "
    f"completed after pruning {s.n_completed_after_prune}"
)
print(
    f"search fits: {s.n_fits}; best score of trial 0 and the Sobol design: {s.exploration_max:.3f}"
)
print()
print(f"{'trial':>6}{'search gap':>12}{'fresh score':>13}{'fresh gap':>11}  groups per fresh fit")
for rank, candidate in enumerate(s.candidates):
    tags = ["chosen"] if rank == 0 else []
    tags += ["defaults"] if candidate.trial_number == 0 else []
    print(
        f"{candidate.trial_number:>6}{candidate.score:>12.3f}{candidate.fresh_score:>13.3f}"
        f"{candidate.gap:>11.3f}  {list(candidate.fit_counts)} {', '.join(tags)}"
    )
print()
print(f"fresh score: chosen {s.score_selected:.3f}, defaults {s.score_default:.3f}")
print(f"fresh gap: chosen {s.gap_selected:.3f}, defaults {s.gap_default:.3f}")
print("chosen minus defaults per subset pair:", [round(d, 3) for d in s.paired_differences])
print(f"beats the defaults under the objective: {s.beats_default_under_objective}")
print(f"chosen setting is the defaults: {s.selected_is_default}")

# %% [markdown]
# ## A picture of the groups
#
# The groups were found in the UMAP coordinates of the deployed setting, which
# can have up to 10 dimensions; the figure title gives the number. To draw
# them, this cell runs one more UMAP on the standardised pixels with the same
# `n_neighbors` and `min_dist` but two dimensions. It is a display refit only:
# distances in it are not the ones HDBSCAN saw.
#
# Left, post hoc: the true digits. Middle: Starfold's groups, with opacity set
# by each row's own support (floored at 0.1 so every row stays visible).
# Right: one fit of the library defaults with one seed.

# %%
x = X[result.kept_rows]
truth = y[result.kept_rows]
layout = umap.UMAP(
    n_neighbors=result.deployed_params.n_neighbors,
    min_dist=result.deployed_params.min_dist,
    n_components=2,
    random_state=0,
    n_jobs=1,
).fit_transform(StandardScaler().fit_transform(x))


def group_colours(n):
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    return [cmap(i % cmap.N) for i in range(n)]


def draw_partition(ax, partition, title, *, legend_title, opacity=None):
    """Scatter the display layout coloured by a partition, noise in grey."""
    noise = partition < 0
    ax.scatter(*layout[noise].T, s=3, color="0.8", lw=0)
    groups = np.unique(partition[~noise])
    colours = group_colours(groups.size)
    for colour, group in zip(colours, groups, strict=True):
        members = partition == group
        alpha = 1.0 if opacity is None else np.clip(np.nan_to_num(opacity[members]), 0.1, 1.0)
        ax.scatter(*layout[members].T, s=3, color=colour, alpha=alpha, lw=0)
    handles = (
        [Line2D([], [], ls="", marker="o", ms=4, color="0.8", label=f"noise ({noise.sum()})")]
        if noise.any()
        else []
    )
    if groups.size <= 20:
        handles += [
            Line2D([], [], ls="", marker="o", ms=4, color=c, label=str(g))
            for c, g in zip(colours, groups, strict=True)
        ]
    else:
        handles.append(
            Line2D([], [], ls="", marker="o", ms=4, color=colours[0], label="colours repeat")
        )
    ax.legend(
        handles=handles,
        title=legend_title,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=6,
        frameon=False,
        handletextpad=0.2,
        columnspacing=0.8,
    )
    ax.set_title(title)
    ax.set_xlabel("display axis 1")
    ax.set_ylabel("display axis 2")
    ax.set_xticks([])
    ax.set_yticks([])


n_default = np.unique(result.default_labels[result.default_labels >= 0]).size
fig, axes = plt.subplots(1, 3, figsize=(11, 4.8))
draw_partition(axes[0], truth, "true digits (post hoc)", legend_title="digit")
draw_partition(
    axes[1],
    labels,
    f"Starfold: {result.n_groups} groups, opacity = own support",
    legend_title="group",
    opacity=w.own_support,
)
draw_partition(
    axes[2],
    result.default_labels,
    f"library defaults: {n_default} groups (one fit, one seed)",
    legend_title="group",
)
fig.suptitle(
    "Display refit: 2-D UMAP with the deployed n_neighbors and min_dist; "
    f"the groups were found in {result.deployed_params.n_components} dimensions"
)
fig.tight_layout()
save_figure(fig, "digits_partitions.png")
plt.show()

# %% [markdown]
# ## Settings against groups
#
# Each point is one complete trial of the search: its score on the horizontal
# axis, the number of groups its fits found on the vertical axis. No new fit
# runs here; everything comes from `result.search.trials`. Pruned trials
# stopped after two fits and have no gap, so they are left out. The score
# never reads the group count; the dotted line at the true number of digits is
# post hoc.

# %%
complete = [t for t in s.trials if t.state == "complete"]
score = np.array([t.score for t in complete])
found = np.array([np.median(t.fit_counts) for t in complete])
phase = np.array([t.phase for t in complete])
number = np.array([t.number for t in complete])
shown = found > 0
chosen = s.candidates[0].trial_number

fig, ax = plt.subplots(figsize=(6.4, 4))
for i, (key, name) in enumerate(
    (
        ("exploration", "Sobol design"),
        ("convergence", "TPE"),
        ("completion", "completed after pruning"),
    )
):
    mask = shown & (phase == key)
    if mask.any():
        ax.scatter(score[mask], found[mask], s=24, color=TAB10(i), label=name)
mask = shown & (number == 0)
if mask.any():
    ax.scatter(
        score[mask], found[mask], marker="*", s=170, color="k", label="trial 0: library defaults"
    )
mask = shown & (number == chosen)
if mask.any():
    ax.scatter(
        score[mask],
        found[mask],
        s=280,
        facecolors="none",
        edgecolors="k",
        lw=1.2,
        label=f"chosen: trial {chosen}",
    )
ax.axhline(np.unique(y).size, ls=":", color="0.4", label="true digits (post hoc)")
ax.set_yscale("log")
ax.yaxis.set_major_locator(ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.yaxis.set_minor_formatter(ticker.NullFormatter())
ax.set_xlabel("trial score (gap over copy)")
ax.set_ylabel(f"groups found (median of {config.n_resamples} fits)")
hidden = f"; {(~shown).sum()} with no group not shown" if (~shown).any() else ""
ax.set_title(f"{shown.sum()} complete trials, {s.n_pruned} pruned{hidden}")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
fig.tight_layout()
save_figure(fig, "digits_settings_vs_groups.png")
plt.show()

# %% [markdown]
# ## Support per row
#
# A row's own support says how reliably the refits put it back with the
# members of its group, from 0 to 1. Its unresolved share is what went to
# noise instead. Keeping only high-support rows buys purity at the cost of
# completeness; the right panel, post hoc, shows how support relates to rows
# that match their group's majority digit.


# %%
def majority_digit(partition):
    """Post hoc: each group's majority true digit and that digit's share of the group."""
    groups = np.unique(partition[partition >= 0])
    counts = [np.bincount(truth[partition == g], minlength=10) for g in groups]
    return (
        groups,
        np.array([c.argmax() for c in counts]),
        np.array([c.max() / c.sum() for c in counts]),
    )


own = w.own_support
grouped = labels >= 0
groups, digit, share = majority_digit(labels)
matches = np.zeros(labels.size, dtype=bool)
matches[grouped] = truth[grouped] == digit[np.searchsorted(groups, labels[grouped])]
scored = grouped & np.isfinite(own)
bins = np.linspace(0.0, 1.0, 26)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
axes[0].hist(own[scored], bins=bins, color=TAB10(0))
axes[0].set_xlabel("own support")
axes[0].set_ylabel("rows")
axes[0].set_title("how reliably each grouped row comes back")
unresolved = w.unresolved[np.isfinite(w.unresolved)]
axes[1].hist(unresolved, bins=bins, color=TAB10(0), label="rows")
axes[1].axvline(result.unresolved_mass, color="k", ls="--", label="mean: unresolved mass")
axes[1].set_xlabel("unresolved share")
axes[1].set_ylabel("rows")
axes[1].set_title("share of refits that left each row unresolved")
axes[1].legend(frameon=False)
axes[2].hist(
    [own[scored & matches], own[scored & ~matches]],
    bins=bins,
    stacked=True,
    color=[TAB10(0), TAB10(1)],
    label=["matches its group's majority digit", "does not match"],
)
axes[2].set_xlabel("own support")
axes[2].set_ylabel("rows")
axes[2].set_title("support against the true digits (post hoc)")
axes[2].legend(frameon=False, loc="upper left")
fig.tight_layout()
save_figure(fig, "digits_support.png")
plt.show()

# %% [markdown]
# ## Recurrence per group
#
# Hennig (2007, doi:10.1016/j.csda.2006.11.025) reads a mean Jaccard of 0.85
# or more as a highly stable group and 0.5 or less as a dissolved one. Here
# they are marks to read against, not filters. The right panel, post hoc,
# asks whether groups that recur are also pure.

# %%
index = np.arange(w.group_recurrence.size)
lower = np.clip(w.group_recurrence - w.group_recurrence_band[:, 0], 0.0, None)
upper = np.clip(w.group_recurrence_band[:, 1] - w.group_recurrence, 0.0, None)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
axes[0].errorbar(
    index,
    w.group_recurrence,
    yerr=[lower, upper],
    fmt="o",
    ms=4,
    capsize=3,
    color="k",
    label="mean over refits, 5 to 95 percent band",
)
axes[0].axhline(0.85, ls=":", color=TAB10(0), label="0.85: highly stable (Hennig 2007)")
axes[0].axhline(0.5, ls="--", color=TAB10(1), label="0.5: dissolved (Hennig 2007)")
axes[0].set_ylim(0.0, 1.05)
axes[0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
axes[0].set_xlabel("group")
axes[0].set_ylabel("recurrence (best Jaccard match)")
axes[0].set_title("how often each group comes back")
axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
axes[1].bar(index, sizes, color=TAB10(0))
axes[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
axes[1].set_xlabel("group")
axes[1].set_ylabel("rows")
axes[1].set_title(f"group sizes ({(labels < 0).sum()} noise rows)")
points = axes[2].scatter(w.group_recurrence, share, c=sizes, cmap="viridis", s=30)
fig.colorbar(points, ax=axes[2], label="rows in the group")
axes[2].set_xlim(0.0, 1.05)
axes[2].set_ylim(0.0, 1.05)
axes[2].set_xlabel("recurrence")
axes[2].set_ylabel("share of the majority digit")
axes[2].set_title("recurrence against purity (post hoc)")
fig.tight_layout()
save_figure(fig, "digits_groups.png")
plt.show()

# %% [markdown]
# ## How many groups
#
# The number of groups each refit found, with the stability interval shaded.
# A narrow interval that holds the count of the fit on all rows means the
# count is reproducible under resampling. It does not mean the count is true.

# %%
values, freq = np.unique(w.count_values, return_counts=True)
low, high = w.count_interval
fig, ax = plt.subplots(figsize=(5.6, 3.8))
ax.axvspan(low - 0.5, high + 0.5, color=TAB10(0), alpha=0.2, lw=0, label="5 to 95 percent interval")
ax.bar(values, freq / w.count_values.size, width=0.8, color=TAB10(0), label="refits")
ax.set_xlim(min(values.min(), low) - 1, max(values.max(), high) + 1)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.set_xlabel("groups found in a refit")
ax.set_ylabel("share of refits")
ax.set_title(
    f"groups found in each of {w.count_values.size} refits\n"
    f"fit on all rows: {result.n_groups} groups; true digits (post hoc): {np.unique(y).size}"
)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
fig.tight_layout()
save_figure(fig, "digits_counts.png")
plt.show()

# %% [markdown]
# ## What else the refits recorded
#
# `support` holds, for every row and group, how often the refits put the row
# with that group; with `unresolved` it sums to 1 on every row a refit drew.
# `property_median` and `property_band` give each group's median value per
# column across refits, in original units. `strengths` are HDBSCAN's
# within-fit membership strengths, not probabilities.

# %%
sampled = w.n_sampled > 0
total = result.support[sampled].sum(axis=1) + w.unresolved[sampled]
failures = sum(bool(error) for error in w.world_errors)
print(f"refit fits: {w.n_fits}, failures: {failures}, seconds: {w.seconds:.1f}")
print(f"refits that drew each row: {w.n_sampled.min()} to {w.n_sampled.max()}")
print(
    f"support: shape {result.support.shape}; support + unresolved = 1 on drawn rows: "
    f"{np.allclose(total, 1.0, atol=1e-4)}"
)
agreement = w.world_agreement
print(
    f"agreement of each refit with labels (ARI times coverage): median "
    f"{np.median(agreement):.2f}, {agreement.min():.2f} to {agreement.max():.2f}"
)
print(
    f"property_median: shape {w.property_median.shape} (group by column); "
    f"measurement draws: {w.property_propagated}"
)
width = w.property_band[..., 1] - w.property_band[..., 0]
print(f"median width of the property bands: {np.nanmedian(width):.2f} pixel units")
print(f"strengths of grouped rows: median {np.median(result.strengths[grouped]):.2f}")
print("backend comparison:", "not run" if result.backends is None else result.backends)

# %% [markdown]
# ## The co-association check
#
# Over the refits, count how often each pair of rows lands in the same group.
# If the groups hold, pairs inside a group co-occur far more often than pairs
# across groups; the contrast is the difference and needs no label matching.
# The check also cuts that pair matrix into groups and compares the cut with
# `labels` (agreement). The cut can fail on matrices with many ties, which
# raises `consensus_disagrees` even when the contrast is high.

# %%
c = result.consensus
print(f"rows used: {c.n_rows} (subset: {c.subsampled}), memory: {c.bytes_used / 2**20:.0f} MiB")
print(f"within-group co-association:  {c.within_association:.3f}")
print(f"between-group co-association: {c.between_association:.3f}")
print(f"contrast:                     {c.contrast:.3f}")
print(f"agreement of the cut with labels (ARI times coverage): {c.agreement:.3f}")
print(f"groups in the cut: {np.unique(c.labels[c.labels >= 0]).size}")

# %% [markdown]
# ## The calibration test
#
# It did not run: `result.null.skipped_reason` is `null_not_run`, and
# `result.null.p_value` holds a placeholder of 1.0, not a result. The test asks whether
# structureless copies of the table, run through the same exploration, score
# as high as the table does. Each copy costs 102 fits and the test draws up to
# 19 copies (1,938 fits), about six times the rest of this run. Run it once,
# on the result you report, with the same configuration, seed, engine and
# `n_jobs`:
#
# ```python
# null = starfold.null_test(
#     X[result.kept_rows],
#     result.search,
#     config=config,
#     random_state=RANDOM_STATE,
#     engine=result.provenance["engine_resolved"],
#     n_jobs=N_JOBS,
# )
# print(null.p_value, null.structure_supported)
# ```
#
# Notebook 03 runs it on the run saved below and attaches it to the result.

# %%
print("skipped:", result.null.skipped_reason, "| families:", result.null.families)

# %% [markdown]
# ## Post hoc: against the true digits
#
# Only now do the true digits enter. The coverage-adjusted ARI is the adjusted
# Rand index on rows both sides assign, floored at 0, times the share of rows
# assigned, so calling most rows noise cannot score well. Below, the mean image
# of each group, titled with its majority digit.

# %%
print(f"coverage-adjusted ARI, Starfold: {coverage_adjusted_ari(truth, labels):.3f}")
print(f"coverage-adjusted ARI, defaults: {coverage_adjusted_ari(truth, result.default_labels):.3f}")

ncols = min(8, max(groups.size, 1))
nrows = max(1, -(-groups.size // ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(1.3 * ncols + 1, 1.5 * nrows + 0.6), squeeze=False)
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_visible(False)
image = None
for ax, group, majority, part in zip(axes.flat, groups, digit, share, strict=False):
    ax.set_visible(True)
    image = ax.imshow(
        x[labels == group].mean(axis=0).reshape(8, 8), cmap="cividis", vmin=0, vmax=16
    )
    ax.set_title(f"group {group}, {sizes[group]} rows\n{part:.0%} digit {majority}", fontsize=7)
if image is not None:
    fig.colorbar(image, ax=axes, label="mean pixel value", shrink=0.8)
fig.supxlabel("pixel column")
fig.supylabel("pixel row")
plt.show()

# %% [markdown]
# ## Save and reload
#
# `starfold.save` writes `result.json`, `arrays.npz` (read back with
# `allow_pickle=False`) and a manifest of SHA-256 checksums. `starfold.load`
# checks the checksums and derives the flags again from the arrays. The run
# goes to `notebooks/runs/digits`, which git ignores.

# %%
path = starfold.save(result, RUN_DIR)
loaded = starfold.load(path)
if not (np.array_equal(loaded.labels, result.labels) and loaded.flags == result.flags):
    msg = "The reloaded result differs from the one in memory."
    raise RuntimeError(msg)
print(f"saved to {path}; labels and flags match after reloading")

# %% [markdown]
# ## Next
#
# Notebook 02 runs Starfold on a real table with no labels at all, 9,242 APOGEE
# stars. Notebook 03 runs the calibration test on the Digits run saved above.
