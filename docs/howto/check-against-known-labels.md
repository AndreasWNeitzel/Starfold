# Check a result against known labels

This guide scores a Starfold partition against classes you already know, after the run, and explains how to read the score.

```{eval-rst}
.. index::
   single: post hoc check
   single: known labels
   single: adjusted Rand index
   single: coverage-adjusted ARI
   pair: labels; post hoc
```

## Prerequisites

- A result from `starfold.recover`.
- One known class per row of your table, for example the digit shown in each image. Rows without a known class can carry a negative label.
- The labels must have stayed out of the run. Starfold never reads them. This check comes after the groups exist, and nothing you learn from it may change the groups.

## Steps

1. Keep the labels aside when you call `recover`. Do not pass them to any argument.

   ```python
   import starfold
   from sklearn.datasets import load_digits

   X, y = load_digits(return_X_y=True)
   result = starfold.recover(X, random_state=0)
   ```

2. Align the labels with the rows Starfold kept.

   ```python
   truth = y[result.kept_rows]
   ```

3. Compute the coverage-adjusted {term}`adjusted Rand index` for Starfold and for the library defaults. The function is in the repository's `bench` package, so run it from a clone.

   ```python
   from bench.beats_default import coverage_adjusted_ari

   print(f"coverage-adjusted ARI, Starfold: {coverage_adjusted_ari(truth, result.labels):.3f}")
   print(f"coverage-adjusted ARI, defaults: {coverage_adjusted_ari(truth, result.default_labels):.3f}")
   ```

4. If you work outside a clone, compute the same quantity with scikit-learn.

   ```python
   from sklearn.metrics import adjusted_rand_score

   assigned = result.labels >= 0
   ari = adjusted_rand_score(truth[assigned], result.labels[assigned])
   print(max(0.0, ari) * assigned.mean())
   ```

   This form assumes every row has a known class. The `bench` function also skips rows whose label is negative.

## Expected result

For step 3, the recorded Digits run prints the lines below. Step 4 prints the Starfold value without rounding, because every digit has a known class.

You see:

```text
coverage-adjusted ARI, Starfold: 0.740
coverage-adjusted ARI, defaults: 0.594
```

This number is the {term}`agreement with known classes`, a {term}`post hoc` check that the search never saw. It is the {term}`adjusted Rand index` on the rows that were assigned to a group, times the share of rows assigned ({term}`coverage`). The factor keeps a fit that labels a few easy rows and calls the rest {term}`noise` from scoring well. In this run, 30 of 1,797 rows are noise.

## How to read it

- A score near 1 means the groups match the classes. A score near 0 means they match no better than chance.
- The score compares one partition with one set of classes. A low value does not show that Starfold is wrong. Classes can be groups of a kind that density does not separate, and the table may not contain enough information to separate them.
- A high value on one table does not validate the method on another. The benchmark in {doc}`../limits` covers this.
- Do not raise or lower the budgets, the space or the scaling until this score improves. That turns a check into a fit to the labels, and the score no longer says anything about new data.
- Do not compare group counts with class counts. A stable partition is not a true partition, and the group count is not a finding ({doc}`../limits`).

## Variations

- **Per-group purity.** The digits notebook ({doc}`plot-your-result`) marks each group by the share of its majority class and plots it against recurrence. The plot is in the cell after the heading "Post hoc: against the true digits".
- **Rows with no known class.** Give them a negative label. The `bench` function leaves them out of both the ARI and the coverage share.

## Related pages

- {doc}`../metrics/prediction-agreement` (a different quantity: a score term, not agreement with known classes)
- {doc}`../metrics/coverage`
- {doc}`plot-your-result`
