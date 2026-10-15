# Compare the CPU and cuML partitions

This guide fits one setting on both backends with `starfold.compare_backends` and shows what the {term}`backend agreement` numbers mean. This is a fifth sense of the word, separate from the four listed under {term}`agreement`.

```{eval-rst}
.. index::
   single: compare_backends
   single: backend agreement
   pair: CPU; cuML
   single: backend_disagreement
   single: normalised agreement
```

## Prerequisites

- A working cuML install ({doc}`use-gpu`). Without it, `compare_backends` raises `ImportError`.
- A result from `recover`, and the table it used.
- The CPU fits are slower than the GPU fits. The recorded Digits comparison took 19.7 s.

## Steps

1. Take the finite rows and the settings that produced `result.labels`.

   ```python
   import starfold

   Xk = X[result.kept_rows]
   params = result.deployed_params
   ```

2. Compare. Each of 3 seeds fits the same settings on both backends on all rows.

   ```python
   agreement = starfold.compare_backends(
       Xk, params, scaling="standard", n_seeds=3, random_state=0
   )
   ```

   Pass the `scaling` of your run. The arguments shown are the defaults.

3. Read the three numbers per seed.

   ```python
   print(agreement.mean_normalised)
   print(agreement.n_groups_cpu, agreement.n_groups_cuml)
   ```

## What backend agreement means

Backend agreement compares partitions, never label numbers, because the backends add floating-point numbers in a different order. For each seed:

- `ari` is the {term}`adjusted Rand index` on the rows both backends assigned to a group.
- `coverage` is the share of rows both assigned ({term}`coverage`).
- `agreement` is `ari` times `coverage`.
- `normalised` divides `agreement` by the smaller of the two assigned shares, so a setting that leaves most rows as {term}`noise` does not look like disagreement.

`Config(check_backends=True)` runs the same comparison inside `recover` and raises the {term}`flag` `backend_disagreement` when `mean_normalised` is below 0.8. That threshold is a convention of Starfold, not a statistical test. A value above it does not show that the two backends give the same groups, only that they agree on most rows they both group.

## Expected result

Recorded comparisons, from `bench/results/backend_agreement.json` (3 seeds each):

| Table | Rows fitted | Mean normalised backend agreement | Groups CPU, then cuML per seed |
|---|---:|---:|---|
| Digits | 1,797 | 0.850 | 14, 12, 13 then 10, 10, 10 |
| Rice | 3,810 | 0.917 | 3, 3, 3 then 3, 3, 2 |
| Landsat | 4,000 of 6,430 | 0.933 | 2, 2, 3 then 3, 3, 3 |
| Waveform | 4,000 of 5,000 | 0.583 | 5, 9, 5 then 13, 7, 14 |
| APOGEE | 4,000 of 9,242 | 0.935 | 4, 2, 3 then 3, 3, 3 |

Four of the five tables score about 0.85 or higher. Waveform, at 0.583, is the one that would raise `backend_disagreement`. The group counts differ between backends and between seeds even where the backend agreement is high, which is one more reason never to compare counts. Tables above 4,000 rows were fitted on a seeded 4,000-row subset, so the Landsat, Waveform and APOGEE rows describe the subset.

## What to do with a low value

If `backend_disagreement` fires, the answer depends on the backend. Report the partitions from both, and do not choose the one with the count you prefer. See {doc}`../reference/flags`.

## Variations

- **More seeds.** Raise `n_seeds` for a tighter view. Each seed adds 2 fits.
- **Another setting.** Pass any `Params`, for example `starfold.default_params()`.

## Related pages

- {doc}`use-gpu`
- {doc}`reproduce-a-run`
- {doc}`../metrics/prediction-agreement`
- {doc}`../metrics/coverage`
