# Modularity

This page explains modularity, the third term of the {term}`score`, and shows how to read its value.

```{eval-rst}
.. index::
   single: modularity
   pair: metric; modularity
   single: graph
   single: community
   single: Newman modularity
   single: neighbour graph
```

## The question it answers

Do the {term}`group`s follow the neighbourhoods in your own table, before any map is drawn?

## A picture

A graph is a set of dots joined by lines. Here each dot is a {term}`row` of your table, and each row is joined to up to 15 {term}`nearest neighbours` (all other rows on a subset of 16 rows or fewer), measured in your {term}`column`s after {term}`scaling`. The search is exact up to 20,000 rows per subset and approximate above that. The map that UMAP draws is not used, so UMAP cannot invent a separation that this term would reward. Starfold drops the direction of the lines: two rows are joined if either one lists the other. Every line counts the same.

A community is a set of dots that are joined to each other more than chance would join them. Modularity scores a labelling of the dots by counting the lines that stay inside the groups. It compares that count with the count expected if the same dots were joined at random, each keeping its number of lines.

High picture: two dense clumps of dots, each joined almost only to itself, with a few lines between them. Labelling the two clumps as two groups puts nearly all lines inside groups, far more than chance. Low picture: one cloud where every dot is joined to its neighbours all around. Cutting it into pieces puts many lines across the cuts, and the value falls toward zero.

## Range and reference values

| Quantity | Value |
|---|---|
| Range | below 1 in theory; negative values are possible (Starfold clips them to 0) |
| Perfect result | approaches 1 as the groups grow in number and isolation, never reaches it |
| Pure chance | about 0 |

Rows in {term}`noise` keep their lines in the graph but belong to no group. Their lines count in the total, never as lines inside a group, so their effect on the value depends on the graph. One group that holds every row gives exactly 0. A graph with no lines gives 0.

## How to read a value

Newman and Girvan (2004) write that, for the networks they studied, modularity values in practice fall between about 0.3 and 0.7, and that higher values are rare. That is a description of observed values in networks with clear communities, not a cut-off, and it was not derived for graphs built from nearest neighbours. Starfold does not gate on it. Because neighbours sit close together, a nearest-neighbour graph can give a positive value even to a cut through a smooth cloud. The bands below are only a guide and are [no-source: internal-reasoning].

The decision question is: do the groups of this setting follow the neighbourhoods of my table, or do they cut through them? Compare the value with the other trials of the same run, and with the score on a structureless copy in {doc}`gap`.

| Value | Reading | Decision |
|---|---|---|
| Below 0.3 | The groups cut through many neighbourhoods | Prefer another trial, or read the groups as coarse slices of a smooth cloud |
| 0.3 to 0.6 | Partial agreement with the neighbourhoods | Check which rows sit near the borders, for example with {doc}`support` |
| Above 0.6 | Most lines stay inside groups | Check the resolution caveat below, then go on to {doc}`recurrence` |

## What misleads

1. Modularity has a resolution limit: it can merge small communities into one (Fortunato and Barthelemy 2007). A high value does not show that the number of groups is right.
2. The graph uses only the 15 nearest rows, so it sees local structure. A table with a few large, loosely connected groups can score lower than a table cut into many tight pieces.
3. Noise rows lower the value. A setting that calls many rows noise can raise its other terms while modularity drops, and a drop to zero makes the {term}`score` zero.

## Where Starfold prints it

Each trial stores `modularity` in `result.search.trials`. It is the mean over the fits of the trial of the modularity of each fit on its own subset graph, each clipped at 0.

The notebooks do not print this term alone. The recorded Digits run (1,797 images, `random_state=0`) stores it in `bench/results/digits.json`, under `trial_diagnostics`, written by `bench/beats_default.py` from these attributes:

```python
for number in (0, 13, 33):
    trial = result.search.trials[number]
    print(number, f"{trial.modularity:.3f}")
```

You see (values read from the recorded file, on the search's three subsets):

```text
0 0.645
13 0.230
33 0.788
```

Trial 13 found 3 groups in a fit on all rows and covered every tested row, but its modularity was low. Trial 33 is the setting Starfold selected.

:::{dropdown} Formula
On subset `r`, the graph links each row to its 15 nearest rows in the scaled columns, symmetrised, unweighted, with no self loops. With `E` the number of lines, `e_c` the lines inside group `c`, and `vol_c` the summed number of lines of the rows in `c`,

```text
Q_r = sum over groups c >= 0 of [ e_c / E - (vol_c / 2E)^2 ]
q = mean over fits r of max(0, Q_r)
```

Noise rows (label -1) keep their lines in `E` and form no group. The source is `modularity` in `src/starfold/objective.py` and `neighbor_adjacency` in `src/starfold/backend.py`.
:::

## Learn more

- Newman 2006, modularity and community structure in networks: [doi:10.1073/pnas.0601602103](https://doi.org/10.1073/pnas.0601602103).
- Newman and Girvan 2004, the original definition and the typical range of 0.3 to 0.7 in their networks: [doi:10.1103/PhysRevE.69.026113](https://doi.org/10.1103/PhysRevE.69.026113).
- Fortunato and Barthelemy 2007, the resolution limit: [doi:10.1073/pnas.0605965104](https://doi.org/10.1073/pnas.0605965104).
- Neighbouring terms: {doc}`prediction-agreement`, {doc}`coverage`, {doc}`score`.
