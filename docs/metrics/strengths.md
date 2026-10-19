# HDBSCAN strengths

This page explains strengths, the per-row number that HDBSCAN reports for one fit, and why it is not a probability.

```{eval-rst}
.. index::
   single: strengths
```
```{eval-rst}
.. index::
   single: probabilities_
```
```{eval-rst}
.. index::
   single: HDBSCAN strength
```
```{eval-rst}
.. index::
   single: posterior
```
```{eval-rst}
.. index::
   pair: metric; strength
```

## The question it answers

How close to the core of its group does one fit place a row?

{term}`HDBSCAN` gives every row of a single fit a number from 0 to 1. A row in the dense core of a group has a strength near 1. A row at the edge of a group has a lower one. The number describes that one fit, so it is a within-fit strength. It is not a {term}`posterior`, the probability of membership given the data and a stated model.

## A picture

Picture a group as a dense cloud of points on the {term}`embedding`.

```text
core of the cloud   : strength near 1.0
middle of the cloud : strength between the two
edge of the cloud   : strength near 0
```

The recorded runs quoted on this page ship in the repository under `notebooks/runs/digits` and `notebooks/runs/apogee`. Notebooks 01 and 02 write them, and `starfold.load` reads them. In the recorded run on the scikit-learn digits dataset, most grouped rows sit in cores, so the median strength is 1.00. A table whose groups are diffuse has lower strengths.

## Range and reference values

- Range: 0 to 1.
- Perfect result: 1, the row sits at the core of its group in the fit.
- Pure chance: not applicable. Nothing in the number compares the fit with a structureless table.

The hdbscan package sets the strength of a {term}`noise` row to 0 (see the upstream docs linked below).

## How to read a value

Starfold sets no cut on strengths. The bands are a rule of thumb from this project, not from a published source.

| Strength | Reading | Decision |
|---|---|---|
| Near 1 | Core row of its group in this fit | Use it to pick typical members of the group |
| Between 0 and 1 | Edge row of its group in this fit | Expect its membership to move with small changes |
| 0 | Noise in this fit | Treat it as unassigned |

For a claim that a row is reproducibly in a group, use {doc}`support` instead. Support is measured over refits. A strength is measured once.

## What misleads

- A strength of 1 is not a probability of 1. It is a position in one fit, which depends on the embedding and on the settings.
- A high strength does not imply a recurrent group. In the recorded digits run the median strength over all grouped rows is 1.00, yet five of the ten groups recur at 0.73 or less (see {doc}`recurrence`).
- Strengths are not comparable across fits. Two fits with different settings can give the same row different strengths.
- A strength is not a measurement uncertainty. Nothing in it reflects the errors of your columns.

## Where Starfold prints it

The attribute is `result.strengths`, one float32 value per kept row, taken from the `probabilities_` attribute of the HDBSCAN fit of the selected setting on all rows. The name in the code follows HDBSCAN, and the values are within-fit strengths, not posteriors. The notebook `notebooks/01_digits.ipynb` printed the recorded line below.

```python
grouped = labels >= 0
print(f"strengths of grouped rows: median {np.median(result.strengths[grouped]):.2f}")
```

```text
strengths of grouped rows: median 1.00
```

:::{dropdown} Formula
Starfold does not compute the strengths. HDBSCAN builds a hierarchy of clusters over the mutual-reachability graph. In the hdbscan package, used on the CPU backend, the strength of a row in a selected cluster is the row's $\lambda$ value divided by the largest $\lambda$ in that cluster. That gives a number between 0 and 1. The $\lambda$ value is the inverse of the distance scale at which the row leaves the cluster. See Campello, Moulavi and Sander (2013) for the hierarchy and the hdbscan documentation for the exact membership rule.
:::

## Learn more

- HDBSCAN (Campello, Moulavi and Sander 2013): [doi:10.1007/978-3-642-37456-2_14](https://doi.org/10.1007/978-3-642-37456-2_14).
- Soft membership in the hdbscan package: [soft clustering](https://hdbscan.readthedocs.io/en/latest/soft_clustering.html) and [how HDBSCAN works](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html).
- Glossary: {term}`HDBSCAN`, {term}`posterior`, {term}`noise`. Related pages: {doc}`support`, {doc}`recurrence`.
