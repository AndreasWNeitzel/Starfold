# Score

This page explains the score that Starfold gives every {term}`trial`, and shows how to read it.

```{eval-rst}
.. index::
   single: score
   pair: metric; score
   single: geometric mean
   single: raw_score
   single: fresh score
   single: objective
```

## The question it answers

How well does one {term}`setting` do on three separate tests at once: its groups repeat on unseen rows, they cover the rows, and they follow the neighbourhoods of your table?

## A picture

The {term}`score` joins three numbers, each from 0 to 1: {doc}`prediction agreement <prediction-agreement>` (a), {doc}`coverage` (c) and {doc}`modularity` (q). It takes the cube root of their product. This is the geometric mean, and it is zero when any one of the three is zero. No threshold is needed, and no setting gets a pass for being strong in two terms and absent in the third.

Toy picture (toy numbers, not Starfold output):

| a | c | q | Geometric mean | Plain average |
|---|---|---|---|---|
| 0.9 | 0.9 | 0.9 | 0.9 | 0.9 |
| 0.9 | 0.9 | 0.1 | 0.433 | 0.633 |
| 0.9 | 0.9 | 0 | 0 | 0.6 |

As the last two rows show, a setting that is weak in one term loses most of its score, and a setting that fails one term completely scores zero. This covers a fit that puts every row in noise, a fit that puts every row in one group (modularity 0), and a fit that crashed.

The score never reads the number of groups. Two settings with 3 and 40 groups compete only on the three terms.

## Range and reference values

| Quantity | Value |
|---|---|
| Range | 0 to 1 |
| Perfect result | 1 (all three terms equal 1; modularity below 1 in practice, so 1 is not reached) |
| Pure chance | not applicable (a structureless copy of the same table gives the reference, see {doc}`gap`) |

A score on the data alone has no scale of its own. The recorded runs quoted on this page ship in the repository under `notebooks/runs/digits` and `notebooks/runs/apogee`. Notebooks 01 and 02 write them, and `starfold.load` reads them. In the recorded digits run, the library defaults scored 0.739 on the data and 0.201 on a structureless copy of the table, both on the search's three subsets.

## How to read a value

The bands are a rule of thumb from this project, not from a published source. Compare scores only between settings on the same table. Starfold ranks by the {doc}`gap`, not by the score. When a table says "score", check whether it means the score on the data (`raw_score`), the score on a copy (`null_score`) or a fresh score.

Use the value to find which of the three terms holds this setting back, and check whether the score on the data beats the score on a structureless copy. A score alone never settles the second part.

| Value | Reading | Decision |
|---|---|---|
| Below 0.4 | At least one term is weak | Open the trial in `trial_diagnostics` or `result.search.trials` and find the weak term |
| 0.4 to 0.7 | Mixed | Compare with the score of the same setting on a structureless copy (`null_score`), which is the {doc}`gap` |
| Above 0.7 | All three terms are fair to strong | Still read the gap: in the recorded digits run the defaults scored 0.739 on the data and 0.201 on a copy, and a smooth cloud can score well |

## What misleads

1. A high score on the data can be a high score on anything. Some settings chop a smooth cloud into pieces that repeat and look tidy. The {doc}`gap` compares with a structureless copy for this reason.
2. The score is label-free. It does not measure agreement with any truth. The recorded digits run has 34 complete trials. Across them, the coverage-adjusted ARI of each full-data fit against the true digits has a Spearman correlation of 0.361 with the score on the data (p = 0.04). With the gap it is 0.159 (p = 0.37). Treat the score as a ranking rule with a stated design, not as a measure of correctness.
3. A stable partition is not a true partition. The score cannot tell you that the number of groups is right, because it never reads it.

## Where Starfold prints it

| Where | What it holds |
|---|---|
| `result.search.trials[i].raw_score` | score of trial `i` on the data, on the search's three subsets |
| `result.search.candidates[i].fresh_score` | score of a top trial on fresh subsets |
| `result.search.score_selected`, `result.search.score_default` | fresh scores of the chosen setting and the defaults |

The first notebook prints the fresh scores:

```python
print(f"fresh score: chosen {s.score_selected:.3f}, defaults {s.score_default:.3f}")
```

The line below comes from `notebooks/01_digits.ipynb`:

```text
fresh score: chosen 0.828, defaults 0.789
```

The terms of one trial are in `bench/results/digits.json`, under `trial_diagnostics`. Trial 13 had agreement 0.691, coverage 1.0 and modularity 0.230. Its score on the data was 0.542, well below the plain average of 0.640, because the low modularity pulls the geometric mean down.

:::{dropdown} Formula
With `a`, `c` the means of agreement and coverage over the 6 ordered pairs of subsets, and `q` the mean over subsets of the clipped modularity,

```text
score = ( max(0, a) * max(0, c) * max(0, q) ) ^ (1/3)
```

The score is 0 when fewer than two fits are present or when any fit raised an error. The source is `combine` and `score_fits` in `src/starfold/objective.py`. The idea follows von Luxburg 2010 on the preference of stability for few groups, which modularity counters.
:::

## Learn more

- von Luxburg 2010, why stability alone prefers the coarsest partition: [arXiv:1007.1075](https://arxiv.org/abs/1007.1075).
- Newman 2006, the modularity term: [doi:10.1073/pnas.0601602103](https://doi.org/10.1073/pnas.0601602103).
- Hubert and Arabie 1985, the adjusted Rand index inside agreement: [doi:10.1007/BF01908075](https://doi.org/10.1007/BF01908075).
- Next: {doc}`gap`.
