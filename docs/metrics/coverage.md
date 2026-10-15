# Coverage

This page explains coverage, the second term of the {term}`score`, and shows how to read its value.

```{eval-rst}
.. index::
   single: coverage
   pair: metric; coverage
   single: joint coverage
   single: assigned rows
   single: noise share
```

## The question it answers

Of the unseen rows that Starfold tested, what share did both fits place in a {term}`group` instead of in {term}`noise`?

## A picture

Take the test in {doc}`prediction-agreement`. A fit on subset A predicts a label for every row of subset B that A never saw. The fit on B has its own label for each of those rows. Coverage is the share of these rows that have a group label on both sides: the prediction and the fit on B.

High picture (toy numbers): 100 unseen rows, 98 of them in a group on both sides. Coverage is 0.98. Low picture: the same 100 rows, but the setting leaves 45 of them in noise on one side or the other. Coverage is 0.55, and the agreement of the remaining rows tells you little about the whole table.

Without this term, a setting could call almost every row noise, keep a handful of easy rows, and score a perfect agreement on them.

## Range and reference values

| Quantity | Value |
|---|---|
| Range | 0 to 1 |
| Perfect result | 1 (every unseen row is in a group on both sides) |
| Pure chance | not applicable |

Coverage is 0 for a pair with no unseen rows. Starfold averages it over the six ordered pairs of subsets.

## How to read a value

The bands are [no-source: internal-reasoning]. Starfold has no gate: a low coverage lowers the {term}`score` and a zero makes it zero.

The decision question is: how much of my table do the groups of this setting describe? Answer it before you report the groups as a description of the whole table.

| Value | Reading | Decision |
|---|---|---|
| Below 0.6 | Most tested rows fall in noise on at least one side | Say in any report that the groups describe a minority of the table, or prefer another trial |
| 0.6 to 0.9 | A large minority of rows are left out | Look at which rows are noise (see {doc}`unresolved`) before you use the groups |
| Above 0.9 | Nearly every tested row is in a group | Check {doc}`modularity` to rule out one giant group |

## What misleads

1. A setting that lumps every row into one or two giant groups has coverage near 1. In the recorded Digits run, trial 13 reached a coverage of 1.0 and found 3 groups in a fit on all rows, for 10 true digits. Its modularity was 0.230, which is what exposed it.
2. Coverage counts rows, not quality. A row in a wrong group counts the same as a row in a right one.
3. Coverage says nothing about the rows in your table outside the tested sets. It uses only the rows that one subset held and another did not.

## Where Starfold prints it

Each trial stores `coverage` in `result.search.trials`. The re-scored candidates keep the per-pair values in `pair_coverage` inside the trial score and combine them into `fresh_pair_scores`.

The notebooks do not print this term alone. The recorded Digits run (1,797 images, `random_state=0`) stores it in `bench/results/digits.json`, under `trial_diagnostics`, written by `bench/beats_default.py` from these attributes:

```python
for number in (0, 6, 33):
    trial = result.search.trials[number]
    print(number, round(trial.coverage, 3))
```

You see (values read from the recorded file, on the search's three subsets):

```text
0 0.884
6 0.553
33 0.975
```

Trial 0 is the library defaults. Its group count here (42) comes from one fit on all rows at the recorded seed, with the trial's own settings and no rescaling. Another default-seed fit in the same run found 51 groups, so the count of a trial varies with the seed. The emitted fit of the selected setting rescales some settings to the full table. That is why trial 33 shows 14 groups in the trial table and 10 in the emitted fit. Trial 6 found 77 groups in such a single fit and left almost half of the tested rows out. Trial 33 is the setting Starfold selected.

:::{dropdown} Formula
With `O` the held-out rows of the ordered pair (r, s) and `J` the rows of `O` that both the prediction and the fit on `s` assign to a group,

```text
c_rs = |J| / |O|        (0 if O is empty)
coverage = mean of c_rs over the 6 ordered pairs r != s
```

The source is `pair_stability` in `src/starfold/objective.py`, which returns agreement and coverage together.
:::

## Learn more

- Lange, Roth, Braun and Buhmann 2004, the prediction-based validation that coverage completes: [doi:10.1162/089976604773717621](https://doi.org/10.1162/089976604773717621).
- HDBSCAN documentation, [why points end up as noise and how the parameters change it](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html).
- Neighbouring terms: {doc}`prediction-agreement`, {doc}`modularity`, {doc}`score`.
