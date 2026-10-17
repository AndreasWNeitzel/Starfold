# Support and own support

This page explains support, the per-row number that says how often a row keeps company with the same rows across refits.

```{eval-rst}
.. index::
   single: support
```
```{eval-rst}
.. index::
   single: own support
```
```{eval-rst}
.. index::
   single: own_support
```
```{eval-rst}
.. index::
   pair: metric; support
```
```{eval-rst}
.. index::
   single: refit
```

## The question it answers

When you fit again on a random 80 percent of the rows, does a row land with the same rows every time?

Each {term}`refit` puts a row either in noise or in a refit group. That refit group is made of rows that the full fit splits among its own groups, or calls noise. Take the share of the row's refit group that sits in a given full-fit group. {term}`Support` of the row for that group is the average of this share over the refits that drew the row.

{term}`Own support` is the support for the row's own group in the full fit. It is undefined (NaN) for rows that the full fit calls noise.

## A picture

Take one row r that the full fit puts in group 0. Four refits drew it.

```text
refit 1: r's refit group is 10 rows, all in full-fit group 0   -> 1.0 for group 0
refit 2: r's refit group is 10 rows, 8 in group 0, 2 in noise  -> 0.8 for group 0, 0.2 unresolved
refit 3: r is noise in the refit                               -> 1.0 unresolved
refit 4: r's refit group is 10 rows, 5 in group 0, 5 in group 1 -> 0.5 for group 0, 0.5 for group 1
```

Averaging the four refits gives support 0.575 for group 0, 0.125 for group 1 and 0.3 unresolved (toy numbers). The three sum to 1. Own support is 0.575.

The left panel below shows own support for every grouped row of the recorded Digits run.

![Left: own support of every grouped Digits row. Middle: the unresolved share of every row. Right: own support split by a post hoc check against the true digits.](../figures/digits_support.png)

Look at the left panel. Most rows sit between 0.8 and 1.0, with a tail toward 0. Rows in the tail keep changing company. Rows at 1.0 only ever share a refit cluster with rows of their own full-fit group.

## Range and reference values

- Range: 0 to 1 per row. For each sampled row, the support over all groups plus the {term}`unresolved mass` of the row sums to 1.
- Perfect result: own support 1. Every refit cluster that holds the row has only rows of its full-fit group. That cluster can still be just part of the group.
- Pure chance: not applicable.

## How to read a value

Starfold sets no cut on support. Support measures how often a row stays with its group across refits. It is not a purity and not a probability. The bands below are a reading aid, labelled `[no-source: internal-reasoning]`.

| Own support | Reading | Decision |
|---|---|---|
| 0.9 or more | The row keeps the same company | Keep it, knowing that this does not make it correct |
| Between 0.5 and 0.9 | The row moves at the edges of its group | Keep it, and report that it is an edge row |
| Below 0.5 | The row often lands elsewhere | Treat it as unassigned for any claim about the group |

## What misleads

- High support is not membership probability. It is how often a row recurs with its group, not the chance that the row belongs there given a model.
- A row that only one or two refits drew has a coarse average. Check `result.worlds.n_sampled`. A row that no refit drew has NaN support.
- A failed refit counts as all noise, so it lowers support. Check `result.worlds.world_errors`.
- Keeping only high-support rows loses rows and need not gain purity. In the recorded Digits run, 81 percent of the rows with own support of 0.9 or more match their group's majority digit (a post hoc check). For all 1,767 grouped rows the share is 83 percent. On a table with a diffuse background, the loss of rows can be large.

## Where Starfold prints it

The attributes are `result.support` (shape rows by groups, same array as `result.worlds.support`), `result.worlds.own_support` (one value per row) and `result.worlds.n_sampled` (refits that drew each row). The notebook `notebooks/01_digits.py` printed the recorded check below. These lines run inside the notebook, where `w`, `result` and `np` are already defined.

```python
sampled = w.n_sampled > 0
total = result.support[sampled].sum(axis=1) + w.unresolved[sampled]
print(f"refits that drew each row: {w.n_sampled.min()} to {w.n_sampled.max()}")
print(
    f"support: shape {result.support.shape}; support + unresolved = 1 on drawn rows: "
    f"{np.allclose(total, 1.0, atol=1e-4)}"
)
```

You see (recorded Digits run, 32 refits):

```text
refits that drew each row: 16 to 31
support: shape (1797, 10); support + unresolved = 1 on drawn rows: True
```

A per-row own support value from the recorded run is not measured here. The figure above shows the distribution.

:::{dropdown} Formula
Let $W_i$ be the refits that drew row $i$. In refit $w$, row $i$ sits in refit cluster $k$. Let $s_{kg}$ be the share of the rows of $k$ that the full fit puts in group $g$ and $s_{k0}$ the share it calls noise. Then

$$
\text{support}[i,g] = \frac{1}{|W_i|}\sum_{w \in W_i} s_{k(w,i)\,g},
\qquad
\text{unresolved}[i] = \frac{1}{|W_i|}\sum_{w \in W_i} u_{w,i},
$$

where $u_{w,i} = 1$ if refit $w$ calls row $i$ noise and $u_{w,i} = s_{k0}$ otherwise. Then $\sum_g \text{support}[i,g] + \text{unresolved}[i] = 1$, and $\text{own\_support}[i] = \text{support}[i, \text{label}_i]$ for rows with $\text{label}_i \ge 0$. Rows that the full fit calls noise have NaN own support.
:::

## Learn more

- Hennig (2007) on cluster-wise stability under resampling: [doi:10.1016/j.csda.2006.11.025](https://doi.org/10.1016/j.csda.2006.11.025).
- HDBSCAN soft membership, a different quantity from support: [soft clustering in hdbscan](https://hdbscan.readthedocs.io/en/latest/soft_clustering.html).
- Glossary: {term}`Support`, {term}`Own support`, {term}`refit`. Related pages: {doc}`unresolved`, {doc}`strengths`.
