# Pass measurement errors to recover

This guide shows how to give Starfold the uncertainty of every measured value, so that each {term}`refit` sees a different plausible version of your table.

```{eval-rst}
.. index::
   single: X_err
   single: measurement errors
   single: measurement_sampler
   pair: refit; measurement draw
   single: uncertainties
```

## Prerequisites

- A prepared table `X` ({doc}`prepare-table`).
- A second array `X_err` with the standard deviation of each value of `X`, in the same units as `X`. You need measured uncertainties. An invented error bar produces an invented spread.

## Steps

1. Check that `X_err` has the shape of `X`.

   ```python
   assert X_err.shape == X.shape
   ```

2. Pass it to `recover`.

   ```python
   import starfold

   result = starfold.recover(X, X_err=X_err, random_state=0)
   ```

   `X_err` must be finite and non-negative on every row that `recover` keeps. Rows dropped for a NaN in `X` are dropped from `X_err` as well, so a NaN in `X_err` on such a row does no harm. A wrong shape, a NaN or a negative value on a kept row raises a `ValueError`.

3. Confirm that the refits used the errors.

   ```python
   print(result.worlds.property_propagated)
   ```

## What X_err adds to each refit

Each refit takes a random subset of the rows, then replaces every value by `X + X_err * N(0, 1)`, where each value gets its own standard normal draw. The scaler and the fit then run on the redrawn values.

The errors enter the {term}`refit` stage only. They do not change the search for the {term}`setting`, the fit on all rows that gives `result.labels`, or the calibration test. The spread of `count_interval`, of the group recurrence bands and of `property_band` therefore includes your measurement errors, while the partition itself does not move.

## Expected result

`result.worlds.property_propagated` is `True`. The recorded runs passed no errors, so their notebooks print `measurement draws: False`. No recorded run uses real measured uncertainties, and the effect on the bands is not measured here.

## Variations

- **Correlated errors.** If the errors of your columns are correlated, or are not Gaussian, pass a function instead of `X_err`. Give one or the other, never both, or `recover` raises a `ValueError`.

  ```python
  # X_kept: the finite rows of X; err_kept: the matching rows of your errors
  def sampler(rows, indices, rng):
      # rows: the sampled rows; indices: their positions among the kept rows
      return rows + err_kept[indices] * rng.standard_normal(rows.shape)

  result = starfold.recover(X_kept, measurement_sampler=sampler, random_state=0)
  ```

  The function receives a copy of the sampled rows, the positions of those rows in the matrix of kept rows, and a NumPy random generator. It returns an array of the same shape with finite values. With `n_jobs` above 1 it must be picklable ({doc}`run-parallel`). Use the generator it gives you, so the draws follow `random_state`.

- **No errors.** Leave both arguments out. The refits then differ only in which rows they drew.

## Related pages

- {doc}`reproduce-a-run`
- {doc}`../reference/api`
- {doc}`../metrics/recurrence`
