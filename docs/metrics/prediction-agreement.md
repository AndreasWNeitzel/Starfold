# Prediction agreement

This page explains prediction agreement, the first of the three terms in the {term}`score`, and shows how to read its value.

```{eval-rst}
.. index::
   single: prediction agreement
   single: adjusted Rand index
   pair: metric; prediction agreement
   single: stability (TrialRecord attribute)
   single: held-out rows
   single: ARI
```

:::{note}
**Four meanings of agreement.** Starfold uses the word for four numbers. This page covers only the first.

- Prediction agreement (this page): the ARI of labels predicted for unseen rows against the fit on those rows. It is a term of the {term}`score`. It carries no coverage factor, because coverage is a separate term.
- Refit agreement: for each refit, the ARI against the full-data labels times joint coverage. See {doc}`recurrence`.
- Consensus agreement: the ARI of the co-association cut against the full-data labels times coverage. See {doc}`coassociation`.
- Agreement with known classes: a check after the run against labels you hold. The search never sees it.
:::

## The question it answers

If you fit the groups on one random part of your table, do the same groups come out when you predict the rows that part never saw?

## A picture

Starfold draws three random subsets of your {term}`row`s, each holding 80 percent of them. It fits UMAP and HDBSCAN on each subset. For every ordered pair of subsets (A, B), it takes the rows that are in B but not in A. Six pairs exist with three subsets.

For each such row, Starfold looks at the 15 closest rows that A did see ({term}`nearest neighbours`, measured in your {term}`column`s after {term}`scaling`, found exactly). It gives the row the most common {term}`group` label among them. Noise counts as a label, so a row whose neighbours are mostly {term}`noise` is predicted as noise. The prediction is then compared with the label that the fit on B gave the same row.

The comparison uses the {term}`adjusted Rand index` (ARI). The index looks at every pair of rows and asks whether the two labellings treat the pair alike (together in both, or apart in both). Group numbers do not matter, only who sits with whom. Here are three toy labellings of the same six rows, compared with the labelling `0 0 0 1 1 1`:

| Other labelling | Reading | ARI |
|---|---|---|
| `5 5 5 7 7 7` | the same groups under other names | 1 |
| `0 0 1 1 1 1` | one row moved to the other group | 0.324 |
| `0 1 0 1 0 1` | every group split across the other | -0.111 |

The three values come from the ARI formula applied to six rows; they are toy values, not Starfold output. The third is negative because the labelling puts pairs together or apart less often than chance would match them. A four-row example, `0 0 1 1` against `0 1 0 1`, gives exactly -0.5 by hand. The contingency table is all ones, so the pair count is 0, the expected count is 2 x 2 / 6, and the maximum is 2.

High picture: a setting that finds the same ten digits in every subset gives labels that predict unseen rows well. Low picture: a setting that tears a smooth region into many small clumps gives clumps that do not repeat in another subset, so the prediction and the other fit disagree.

## Range and reference values

| Quantity | Value |
|---|---|
| Range in Starfold | 0 to 1 (negative ARI is clipped to 0) |
| Perfect result | 1 (prediction and fit give identical groupings) |
| Pure chance | about 0 |

Starfold averages the value over the six ordered pairs. It sets the value of a pair to 0 when fewer than two rows are assigned on both sides. It does the same when either side holds fewer than two distinct labels on those rows.

## How to read a value

The bands below are [no-source: internal-reasoning]. Starfold has no gate on this term: a low value lowers the {term}`score`, nothing more.

The decision question is: does this setting deserve the trust its groups would need, or should I look at another trial? Read the value of the trial you care about in `result.search.trials`.

| Value | Reading | Decision |
|---|---|---|
| Below 0.5 | Groups barely repeat on unseen rows | Prefer another trial; this setting's groups change with the rows drawn |
| 0.5 to 0.8 | Groups repeat in part | Keep the trial, then read {doc}`recurrence` and {doc}`support` before using any group |
| Above 0.8 | Groups repeat on unseen rows | Keep the trial, then check the group count and {doc}`coverage` and {doc}`modularity`, since a few large groups repeat even without structure |

## What misleads

1. A coarse partition agrees well by default. Two or three large groups come back in every subset, so agreement alone favours the coarsest partition (von Luxburg 2010). In the recorded Digits run, trial 37 found 3 groups in a fit on all rows and reached an agreement of 0.908, for a table with 10 true digits. Modularity and coverage exist to counter this.
2. The ARI counts only rows that both sides assigned to a group. A setting that calls most rows noise can keep a high ARI on the few rows left. The {doc}`coverage` term reports how many rows that is.
3. Agreement measures whether the partition repeats across subsets of the same table. A stable partition is not automatically a true partition.

## Where Starfold prints it

Each trial stores its agreement as `stability` in `result.search.trials` (`stability` is the code name for prediction agreement, averaged over the six ordered pairs). The re-scored candidates in `result.search.candidates` keep the per-pair values combined with coverage and modularity in `fresh_pair_scores`.

The notebooks do not print this term alone. The recorded Digits run (1,797 images, 64 pixel columns, `random_state=0`) stores it in `bench/results/digits.json`, under `trial_diagnostics`, written by `bench/beats_default.py` from these attributes:

```python
for number in (0, 33):
    trial = result.search.trials[number]
    print(number, round(trial.stability, 3))
```

You see (values read from the recorded file, `stability` on the search's three subsets):

```text
0 0.708
33 0.872
```

Trial 0 is the library defaults. Trial 33 is the setting Starfold selected. Both values come from the search plan, not from the fresh re-score.

:::{dropdown} Formula
With `J` the rows of the held-out set of the pair (r, s) that both the prediction `P` and the fit `L_s` assign to a group,

```text
a_rs = max(0, ARI(P[J], L_s[J]))        (0 if |J| < 2 or a side has fewer than 2 labels on J)
agreement = mean of a_rs over the 6 ordered pairs r != s
```

`P` is the most frequent label among the 15 nearest rows of subset `r` (Euclidean, in the scaling fitted on `r`). Noise votes count, and ties go to the smallest label. The source is `pair_stability` and `transfer_labels` in `src/starfold/objective.py`.
:::

## Learn more

- Hubert and Arabie 1985, the paper that defines the adjusted Rand index: [doi:10.1007/BF01908075](https://doi.org/10.1007/BF01908075).
- Lange, Roth, Braun and Buhmann 2004, validating clusters by predicting held-out rows: [doi:10.1162/089976604773717621](https://doi.org/10.1162/089976604773717621).
- von Luxburg 2010, why stability alone prefers few groups: [arXiv:1007.1075](https://arxiv.org/abs/1007.1075).
- scikit-learn, [clustering metrics, including the adjusted Rand index](https://scikit-learn.org/stable/modules/clustering.html).
- Next term of the score: {doc}`coverage`. Combined value: {doc}`score`.
