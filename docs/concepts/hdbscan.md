# HDBSCAN: dense clumps and noise

This page explains how {term}`HDBSCAN` finds groups in a map, why it leaves some rows out as noise, and which of its settings Starfold searches.

```{eval-rst}
.. index::
   single: HDBSCAN
   single: noise
   single: group
   single: min_cluster_size
   single: min_samples
   single: selection method
   pair: HDBSCAN; strengths
```

## A toy picture

Look at a night photograph of a country from space. Cities glow as bright patches, villages as faint dots, and farmland is dark. To list the cities, you would not cut the picture into pieces of equal size. You would circle every patch that is dense enough and big enough to count, and leave the dark land uncircled.

HDBSCAN does this to the points of the {term}`embedding`. A {term}`group` is a dense patch of points. A row that lies in the thin land between patches belongs to no group and gets the label -1, called {term}`noise`.

## Why it leaves noise

Most clustering methods, such as k-means, put every row in some group. HDBSCAN does not, because a group is defined as a region denser than its surroundings. A row in a sparse region is not part of any dense region, so claiming it would be a guess.

Noise therefore means that no clump claimed the row. It does not mean the measurement was bad. A row can be noise for three reasons. It is an outlier, it sits on a thin bridge between two groups, or the settings demand a denser clump than the table has there.

The amount of noise tells you something about the table. A run with 30 noise rows out of 1,797 and a run with most rows in noise describe different tables. The {term}`flag` `high_unresolved_mass` warns when refits leave most rows unplaced.

HDBSCAN also reports a strength for each grouped row, from 0 to 1, where 1 is the core of a group. The strength describes one fit. It is not a probability that the row belongs to the group.

## The settings Starfold searches

The defaults below are those of the libraries, and trial 0 of every search. If you narrow a range so that it excludes a default, trial 0 uses the nearest bound instead.

| Setting | Default | What it does |
|---|---|---|
| {term}`min_cluster_size` | 5 | The smallest clump that may be called a group. Raise it and small groups disappear or merge into bigger ones. |
| {term}`min_samples` | 5 | How many neighbours a row looks at when HDBSCAN estimates the density around it. A larger value makes the clustering more conservative: more rows become noise and groups keep to denser regions. |
| selection method | `eom` | Which clumps HDBSCAN keeps when clumps sit inside clumps. `eom` keeps the clumps that stay dense over the widest range of density levels (the persistent ones), which tend to be larger. `leaf` keeps the innermost, finest clumps, which gives more and smaller groups. |

Starfold searches `min_cluster_size` from 5 up to a bound that depends on your table (at most 1,023), on a log scale. It searches `min_samples` as a fraction of `min_cluster_size`, from 0.05 to 1.0.

Both numbers count rows, and a fit on a subset has fewer rows than the full table. So before the final fit, Starfold scales both numbers up by the ratio of the row counts. The result stores both versions: `params` (the setting as searched) and `deployed_params` (the setting used for the final labels).

## Where HDBSCAN misleads

- **It finds clumps in any map.** Given a smooth cloud, it still finds dense patches. {doc}`copies-and-gaps` says how Starfold guards against this.
- **The count depends on the settings and the seed.** In the recorded run on the scikit-learn digits dataset (seed 0), one fit of the library defaults on all rows found 51 groups, while the chosen setting found 10. Another seed gives another count for the defaults. The count is not a finding: see {doc}`refits`. The recorded runs quoted on this page ship in the repository under `notebooks/runs/digits` and `notebooks/runs/apogee`. Notebooks 01 and 02 write them, and `starfold.load` reads them.
- **Noise depends on `min_samples`.** Compare noise shares only between runs with the same settings.
- **Strengths are not posteriors.** They rank rows within one fit.

## Learn more

- [How HDBSCAN works](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html): an illustrated walk through the algorithm, one stage per figure.
- [Parameter selection for HDBSCAN](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html): how `min_cluster_size` and `min_samples` change labels and noise.
- [HDBSCAN frequently asked questions](https://hdbscan.readthedocs.io/en/latest/faq.html): answers by symptom, such as too much noise or one large cluster.

Next: {doc}`optuna`.
