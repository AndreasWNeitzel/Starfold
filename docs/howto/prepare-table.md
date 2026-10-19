# Prepare your table for recover

This guide turns a table of measurements into the array `starfold.recover` accepts, and shows what happens to rows with missing values.

```{eval-rst}
.. index::
   single: table preparation
   single: missing values
   single: kept rows
   pair: input; finite values
   single: columns; choice of
   single: units
```

## Prerequisites

- Starfold is installed ({doc}`../start/install`).
- Your data are in a NumPy array or a pandas table. One {term}`row` holds one object (a star, a cell, an image). One {term}`column` holds one measured number.

## Steps

1. Turn the table into a two-dimensional float array.

   ```python
   import numpy as np

   X = df[["fe_h", "mg_fe", "J_R", "J_z"]].to_numpy(dtype=float)
   print(X.shape)  # (number of rows, number of columns)
   ```

   `recover` needs at least 2 columns and at least 50 finite rows. Otherwise it raises a `ValueError` that states the counts.

2. Count the rows that hold a missing value (NaN) or an infinite value.

   ```python
   finite = np.isfinite(X).all(axis=1)
   print(f"{finite.sum()} of {X.shape[0]} rows are finite")
   ```

3. Decide what to do with missing measurements. Starfold drops every row that has a NaN or an infinity in any column. It never fills a gap with a mean, a median, zero or any other number. A filled value is one nobody measured, and many copies of it make a dense clump that Starfold then reports as a {term}`group`.

   - If a few rows are incomplete, pass the table as it is. The dropped rows are reported (step 5).
   - If one column is missing for most rows, drop that column before the call. Keeping it would drop most of your rows.
   - If both fixes lose too many rows, the measurements you need are absent. Report the loss and do not invent them.

4. Choose the columns. Every column gets the same weight after {term}`scaling`, so the columns you pick define what "similar" means.

   - Remove columns that measure the same quantity twice. The quantity would count twice.
   - Remove columns that are irrelevant to your question. They blur real groups.
   - Remove categorical columns. A category coded 0, 1, 2 is not a distance. Run within each category instead.
   - Remove duplicate rows. A row listed twice counts twice.

5. Run `recover` and confirm which rows it kept.

   ```python
   import starfold

   result = starfold.recover(X, random_state=0)
   kept = X[result.kept_rows]  # lines up with result.labels
   prov = result.provenance
   print(f"rows kept: {prov['n_kept_rows']} of {prov['n_input_rows']}, columns: {prov['n_features']}")
   ```

## Expected result

`result.labels` has one entry per kept row, with -1 for {term}`noise`. `result.kept_rows` holds the positions of the kept rows in your input. If any row was dropped, `result.flags` contains `rows_rejected_nonfinite` (see {doc}`../reference/flags`).

For the scikit-learn digits dataset, which has no missing values, the recorded run prints:

```text
rows kept: 1797 of 1797, columns: 64
```

If your table has missing values, the two numbers differ and the output depends on your table.

## Units

Starfold scales each column inside every resample, so columns in different units (dex, kpc km/s, km/s squared) need no conversion to a common unit. The per-group medians in `result.worlds.property_median` are reported in the original units of each column.

Scaling does not remove a skewed or heavy-tailed column. If a few extreme values squash the rest of a column, see {doc}`choose-scaling`.

## Variations

- If you have per-value measurement uncertainties, pass them as well: {doc}`measurement-errors`. They must have the same shape as `X`.
- If `recover` raises a `ValueError` about the rows or columns, compare your shape with the limits in step 1. More symptoms are in {doc}`../troubleshooting`.

## Related pages

- {doc}`choose-scaling`
- {doc}`measurement-errors`
- {doc}`../reference/api` (the `recover` signature)
- {doc}`../reference/flags` (`rows_rejected_nonfinite`)
