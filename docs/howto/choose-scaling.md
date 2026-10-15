# Choose between standard and robust scaling

This guide shows how to set the {term}`scaling` option and when to change it from the default.

```{eval-rst}
.. index::
   single: scaling
   pair: Config; scaling
   single: standard scaling
   single: robust scaling
   single: outliers
```

## Prerequisites

- A prepared table ({doc}`prepare-table`).
- You know which columns hold a few extreme values, from a histogram or from `np.percentile`.

## What the two options do

Starfold rescales every column so that columns in different units can be compared. It fits the scaler on the rows of one resample or refit only, never on rows outside it.

| Option | Centres on | Divides by | Use when |
|---|---|---|---|
| `"standard"` (default) | the mean | the standard deviation | columns have no extreme values |
| `"robust"` | the median | the interquartile range | a few extreme values squash the rest of a column |

The two options are the scikit-learn `StandardScaler` and `RobustScaler`. A constant column stays at its centred value.

## Steps

1. Look at the spread of each column. A column whose 99th percentile lies far from its 75th percentile has a long tail.

   ```python
   import numpy as np

   for j in range(X.shape[1]):
       q = np.percentile(X[:, j], [25, 75, 99])
       print(j, q)
   ```

2. Keep the default if no column has a long tail.

   ```python
   import starfold

   result = starfold.recover(X, random_state=0)  # scaling="standard"
   ```

3. Switch to `"robust"` if a few extreme values set the standard deviation of a column.

   ```python
   config = starfold.Config(scaling="robust")
   result = starfold.recover(X, config=config, random_state=0)
   ```

4. Reuse the same `config` in every later call that has to match the run. {doc}`run-calibration-test` takes it as `config=config`, and {doc}`compare-backends` takes `scaling=config.scaling`.

## Expected result

`result.provenance["config"]["scaling"]` records the option you used. Any other value raises `ValueError: scaling must be 'standard' or 'robust'` when you build the `Config`.

The output of a run depends on your table. No recorded run compares the two options, so the effect on the groups is not measured here.

## Do not choose by the group count

Change the scaling for a reason you can state about the columns. Do not try both and keep the option that gives more groups or fewer. The group count is not a finding, and choosing by it biases the result. See {doc}`../limits`.

## Related pages

- {doc}`prepare-table`
- {doc}`../reference/config` (every `Config` field)
- {doc}`../concepts/index`
