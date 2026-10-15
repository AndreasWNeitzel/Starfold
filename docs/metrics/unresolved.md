# Unresolved mass

This page explains unresolved mass, the share of refit outcomes that place a row in no group of the full fit.

```{eval-rst}
.. index::
   single: unresolved mass
```
```{eval-rst}
.. index::
   single: unresolved
```
```{eval-rst}
.. index::
   single: high_unresolved_mass
```
```{eval-rst}
.. index::
   pair: metric; unresolved mass
```
```{eval-rst}
.. index::
   single: flag
```

## The question it answers

How much of the table do the refits fail to place in a group that the full fit also has?

A refit draws a row when the row is in its random 80 percent subset. For one {term}`refit`, give a row 1 if the refit called it {term}`noise`. Otherwise give it the share of its refit group that the full fit calls noise. The unresolved share of the row is the average of that number over the refits that drew it. {term}`Unresolved mass` is the mean of the unresolved share over all rows that at least one refit drew.

## A picture

Take a row r with four refits (toy numbers).

```text
refit 1: r in a refit group that the full fit never calls noise   -> 0.0
refit 2: r in a refit group, 2 of its 10 rows are noise in the full fit -> 0.2
refit 3: r is noise in the refit                                  -> 1.0
refit 4: r in a refit group that the full fit never calls noise   -> 0.0
```

The unresolved share of r is (0 + 0.2 + 1 + 0) / 4 = 0.3. Averaging such shares over all rows gives the unresolved mass.

The middle panel below shows the unresolved share of every row of the recorded Digits run. The dashed line is the mean.

![Middle panel: histogram of the unresolved share of every Digits row, with the mean (the unresolved mass) as a dashed line.](../figures/digits_support.png)

Look at the tall bar near 0 and the dashed line next to it. Nearly all rows are placed by almost every refit, so the mass is small.

## Range and reference values

- Range: 0 to 1.
- Perfect result: 0. Every refit agrees with the full fit on every row.
- Pure chance: not applicable.

## How to read a value

The 0.5 mark is the threshold of the `high_unresolved_mass` {term}`flag` in the code. The other bands are `[no-source: internal-reasoning]`.

| Unresolved mass | Reading | Decision |
|---|---|---|
| Below 0.1 | Refits place almost every row | Read the other numbers as usual |
| Between 0.1 and 0.5 | A noticeable share of outcomes is noise | Check which groups lose rows with {doc}`support` |
| Above 0.5 | The groups describe a minority of the rows | Describe the groups as a minority of the table (flag `high_unresolved_mass`) |

## What misleads

- A low value does not make the groups true. It says the refits agree on where noise is not.
- The value counts the full fit's own noise rows. A table with a large diffuse background can have a high value although the groups are stable. The full fit's noise share is in `(result.labels < 0).mean()`.
- Refits that fail count as all noise, so failures raise the mass. Check `result.worlds.world_errors`.
- A mean hides structure. One group can lose many rows while the mean stays low.

## Where Starfold prints it

The attribute is `result.unresolved_mass` (same value as `result.worlds.unresolved_mass`). The per-row values are in `result.worlds.unresolved`. The notebook `notebooks/01_digits.py` printed the recorded line below.

```python
print(f"unresolved mass: {result.unresolved_mass:.3f}")
```

You see (recorded Digits run, 32 refits):

```text
unresolved mass: 0.044
```

The recorded APOGEE run (`notebooks/02_apogee.py`) gave 0.154, which the notebook reads as about 15 percent of a star's refit outcomes being "no group".

:::{dropdown} Formula
For row $i$ with $W_i$ the refits that drew it, let $u_{w,i} = 1$ if refit $w$ gives row $i$ noise. Otherwise $u_{w,i} = s_{k0}$, the share of its refit cluster $k$ that the full fit calls noise. Then

$$
\text{unresolved}[i] = \frac{1}{|W_i|}\sum_{w\in W_i} u_{w,i},
\qquad
\text{unresolved mass} = \frac{1}{|S|}\sum_{i \in S} \text{unresolved}[i],
$$

where $S$ is the set of rows drawn by at least one refit. For every $i \in S$, $\sum_g \text{support}[i,g] + \text{unresolved}[i] = 1$.
:::

## Learn more

- The flags page, {doc}`../reference/flags`, defines `high_unresolved_mass` and its 0.5 threshold.
- HDBSCAN noise and `min_cluster_size`: [parameter selection in hdbscan](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html).
- Glossary: {term}`Unresolved mass`, {term}`noise`, {term}`flag`. Related pages: {doc}`support`, {doc}`recurrence`.
