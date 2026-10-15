# Set the budgets and run on a laptop

This guide shows how to set the number of trials, resamples and refits, how many fits each choice costs, and how to shrink a run to laptop size.

```{eval-rst}
.. index::
   single: budget
   pair: Config; n_trials
   pair: Config; n_worlds
   pair: Config; n_resamples
   single: fits
   single: run time
```

## Prerequisites

- A prepared table ({doc}`prepare-table`).
- A fit here means one run of UMAP followed by HDBSCAN on one set of rows.

## The budgets

| Field | Default | Allowed | What it controls |
|---|---|---|---|
| `n_trials` | 48 | above `n_explore` | most trials ({term}`trial`) the search may run |
| `n_explore` | 16 | power of two, at least 2 | trials of the {term}`Sobol design`, never pruned |
| `n_resamples` | 3 | at least 2 | resamples every trial is scored on |
| `resample_fraction` | 0.8 | strictly between 0 and 1 | share of rows in each resample |
| `n_worlds` | 32 | 2 to 32767 | number of refits ({term}`refit`) of the selected setting |
| `world_fraction` | 0.8 | strictly between 0 and 1 | share of rows in each refit |

A value outside its range raises `ValueError` when you build the `Config`.

## What a run costs in fits

The cost of a run is counted in fits. A complete trial fits each of its `n_resamples` resamples once on your data. With the default {term}`structureless copy` (`gap_family="gaussian_copula"`) it fits the same number again on the copy. A pruned trial stops after 2 fits. The search then re-scores the best trials on new resamples. After the search come 2 full-data fits and `n_worlds` refits.

Recorded runs, from each run's provenance, with the calibration test off:

| Table | Search fits | Reference fits | Refit fits | Total |
|---|---:|---:|---:|---:|
| Digits, 1,797 rows | 280 | 2 | 32 | 314 |
| APOGEE, 9,242 rows | 300 | 2 | 32 | 334 |

The Digits run takes 80.4 s with cuML on a laptop RTX 3060 under WSL 2 when the run has the GPU to itself. The recorded output shows 182.0 s because notebook 02 ran at the same time. The recorded APOGEE time of 242 s does not say whether it was a solo run, and no solo time is recorded. No CPU time is recorded, so the CPU time of the same runs is not measured. The {doc}`calibration test <run-calibration-test>` costs about six times the rest of a run.

## Steps to a laptop-sized run

1. Test your pipeline on a random sample of rows, so a mistake costs seconds.

   ```python
   import numpy as np

   rng = np.random.default_rng(0)
   X_small = X[np.sort(rng.choice(X.shape[0], size=2000, replace=False))]
   ```

2. Shrink the budgets. Keep `n_explore` a power of two and `n_trials` above it.

   ```python
   import starfold

   config = starfold.Config(n_trials=24, n_explore=8, n_worlds=16)
   result = starfold.recover(X_small, config=config, random_state=0)
   ```

3. Read `result.provenance["stage_fits"]` and `result.provenance["stage_seconds"]` to see where the time went.

   ```python
   print(result.provenance["stage_fits"], result.provenance["total_fits"])
   print(result.provenance["stage_seconds"])
   ```

4. Read the flags. With fewer trials the search is more likely to stop at its budget, which raises `search_budget_exhausted`. It means the score was still improving. It does not invalidate the result.

5. Run the full budget on the whole table for the result you report.

## Expected result

The fit counts follow your budgets, and the times depend on your machine. No recorded run uses the reduced budgets above, so no output is shown for them. A run with `n_trials=24`, `n_explore=8` and `n_worlds=16` searches fewer settings and measures recurrence over fewer refits. Its {term}`count interval` and recurrence bands are based on 16 refits instead of 32.

## Variations

- **`n_resamples`.** Each trial costs `n_resamples` fits on the data, so the cost scales with it. The {term}`prediction agreement` is scored on ordered pairs of resamples, so 3 resamples give 6 pairs.
- **`resample_fraction`.** The held-out rows of a pair are the rows of one resample that the other did not see. A fraction near 1 leaves few such rows.
- **Calibration cost.** Each copy replays `n_explore + 1` trials on `n_resamples` fits, for the data and for the copy. At the defaults that is 17 x 3 x 2 = 102 fits per copy. With `n_explore=8` it is 9 x 3 x 2 = 54, derived from the same structure. The {doc}`calibration guide <run-calibration-test>` has the rest.
- **Cluster.** For a sweep over seeds, see {doc}`run-on-a-cluster`.

## Related pages

- {doc}`use-gpu`
- {doc}`run-parallel`
- {doc}`../reference/config`
