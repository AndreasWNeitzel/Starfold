# UMAP: a map of your table

This page explains what {term}`UMAP` does to your table, which of its settings Starfold searches, and why the map it draws distorts distances.

```{eval-rst}
.. index::
   single: UMAP
   single: embedding
   single: n_neighbors
   single: min_dist
   single: n_components
   single: nearest neighbours
   pair: UMAP; distortion
```

## A toy picture

Take a city of 1,000 people and write down, for each person, their age, height, income and 61 other numbers. Each person is a point in a 64-dimensional space that you cannot draw. You want a flat sheet of paper where people with similar numbers stand close together.

UMAP makes that sheet. First it finds, for every person, the few people most like them (their {term}`nearest neighbours`). It links each person to those neighbours. Then it places all points on the sheet and moves them until linked people are close and unlinked people are far. The result is the {term}`embedding`: one row of coordinates per row of your table. The sheet can have more than two axes. Starfold searches between 2 and 10.

## Why a map distorts distances

A 64-dimensional cloud does not fit on a flat sheet without damage, as a globe does not fit on a flat atlas page without damage. UMAP keeps the neighbour links and gives up the rest.

- **Short distances are kept better than long ones.** Two rows that were neighbours stay close. The gap between two islands on the map does not say how far apart the groups are in your table.
- **Sizes change.** UMAP spreads dense regions and squeezes sparse ones, so the area of an island is not the number of rows or the spread of their values.
- **Axes have no units.** "Display axis 1" is not a column. Only which rows sit near which counts.
- **Different seeds draw different maps.** A {term}`seed` is the number that starts the random choices of a fit. The same table and settings with another seed can give a rotated, stretched or split layout.

These are the reasons Starfold does not trust the map on its own. It checks the groups in three other ways. It predicts groups for rows left out of a fit (held-out rows). It measures how well the groups follow the neighbour links of the original scaled columns, with no map involved. And it refits on random subsets of the rows. The figure you see in the tutorials is a separate two-dimensional display fit. Read the neighbourhoods on it, not the distances. You do not need to set any of these options: Starfold searches them for you.

## The settings Starfold searches

Starfold searches three UMAP settings. The defaults are those of `umap-learn` and are always trial 0 of the search (the first setting it tries; see {doc}`optuna`).

| Setting | Default | Range searched | What it does |
|---|---|---|---|
| {term}`n_neighbors` | 15 | 5 to 100 (log scale) | How many similar rows each row looks at. Small values keep fine detail and can tear a smooth region into pieces. Large values emphasise the broad layout and can merge small groups. |
| {term}`min_dist` | 0.1 | 0.0 to 0.5 | How tightly the map packs neighbours. Low values make dense clumps that are simple for HDBSCAN to find. High values spread rows out. It changes the look of the map, not the neighbour links. |
| {term}`n_components` | 2 | 2 to 10 | The number of map coordinates. More axes keep more of the table, and give HDBSCAN more room. |

Starfold also rescales each column before UMAP (the {term}`scaling` setting, `standard` by default), so that a column measured in large units does not dominate.

## Where UMAP misleads

- **A clean island is not a real group.** A small `n_neighbors` can tear a smooth region into pieces that look like islands. This is why Starfold subtracts a score on a {term}`structureless copy`: see {doc}`copies-and-gaps`.
- **Distance between islands.** Do not read it as similarity between groups.
- **Low `min_dist` exaggerates.** Packing rows tightly can make a loose region look like a clump.
- **One picture is one seed.** A claim about groups needs refits: see {doc}`refits`.

## Learn more

- [How UMAP works](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html): the route from neighbour graphs to a map, with the mathematics left out.
- [Basic UMAP parameters](https://umap-learn.readthedocs.io/en/latest/parameters.html): what `n_neighbors`, `min_dist` and `n_components` do, with paired figures.
- [Using UMAP for clustering](https://umap-learn.readthedocs.io/en/latest/clustering.html): why clustering a UMAP embedding works and where it misleads.

Next: {doc}`hdbscan`.
