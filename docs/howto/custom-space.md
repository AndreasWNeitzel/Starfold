# Narrow the search space

This guide shows how to set the ranges of the six {term}`setting` axes that {term}`Optuna` searches, and explains why the library defaults are always {term}`trial` 0.

```{eval-rst}
.. index::
   single: Space
   pair: search; space
   single: trial 0
   single: library defaults
   single: min_cluster_size
   single: n_neighbors
```

## Prerequisites

- A prepared table ({doc}`prepare-table`).
- A reason to narrow a range. Examples: you know that groups smaller than 30 rows are not of interest, or a setting produced hundreds of tiny groups.

## The fields of Space

`starfold.Space` holds inclusive bounds. The defaults are the values you get with `Space()`.

| Field | Default bounds | Scale | What narrowing does |
|---|---|---|---|
| `n_neighbors` | (5, 100) | log | A higher floor removes fine local detail. The upper bound may not exceed 100 and is capped at the resample size minus 1. |
| `min_dist` | (0.0, 0.5) | linear | A higher floor packs similar rows less tightly in the {term}`embedding`. |
| `n_components` | (2, 10) | linear, integer | Fewer dimensions give a simpler map. |
| `min_cluster_size` | `None`, then (5, min(max(50, n // 20), 1023)) | log | A higher floor forbids small groups. A lower ceiling forbids large ones. |
| `min_samples_fraction` | (0.05, 1.0) | log | The ratio `min_samples / min_cluster_size`. A higher floor calls more rows {term}`noise`. |
| `selection_method` | ("eom", "leaf") | categorical | `"eom"` picks the most persistent clumps, `"leaf"` the finest ones. One entry fixes the choice. |

The 1,023 cap on `min_cluster_size` is the cuML limit on `min_samples`. It keeps the CPU and GPU searches the same. Here `n` is the number of kept rows.

## Steps

1. Build the space. Change only the fields you have a reason to change.

   ```python
   import starfold

   space = starfold.Space(min_cluster_size=(30, 200), selection_method=("eom",))
   ```

2. Put the space in a `Config` and run.

   ```python
   config = starfold.Config(space=space)
   result = starfold.recover(X, config=config, random_state=0)
   ```

3. Read what trial 0 became. The search moves the library default into your bounds.

   ```python
   trial0 = next(t for t in result.search.trials if t.number == 0)
   print(trial0.params)
   ```

   With the bounds above, trial 0 takes `min_cluster_size=30` (the nearest bound to the default 5). It keeps the default fraction `min_samples / min_cluster_size` of 1, so `min_samples` becomes 30 as well.

4. Keep the same `config`, including the space, for every call that has to match the run. {doc}`run-calibration-test` replays the search, and a different space gives a different replay.

## Expected result

`result.params` and `result.deployed_params` lie inside your bounds, except that `deployed_params` rescales `min_cluster_size` and `min_samples` from the resample size to the full row count. For the recorded Digits run the two tables differ in exactly this way (`min_cluster_size` 11 and `min_samples` 9 in `params`, 14 and 11 in `deployed_params`).

A bound that makes a range empty, or that reaches zero on a log axis, raises `ValueError` when you build the `Config`. An `n_neighbors` bound above 100 raises the same error.

## Why the defaults are trial 0

Trial 0 of every search is the library default `(n_neighbors=15, min_dist=0.1, n_components=2, min_cluster_size=5, min_samples=5, selection_method="eom")`, clipped into your space. The search therefore always scores the answer you would get without Starfold. At the end the search re-scores the best three trials and trial 0 on new resamples (a {term}`fresh re-score`). If trial 0 wins, `result.flags` contains `selected_is_default`. If the winner and trial 0 differ by less than the noise, it contains `default_within_noise`. Both flags are listed in {doc}`../reference/flags`.

In the recorded Digits run, trial 0 had a fresh gap of 0.565 and the chosen trial 33 had 0.661.

## Variations

- To forbid small groups, raise the floor of `min_cluster_size` and keep the ceiling of the data rule. For 2,000 kept rows the rule gives 100, so `Space(min_cluster_size=(30, 100))` does it.
- To search wider than the defaults, you cannot raise `n_neighbors` past 100, because the search caches one neighbour graph of that width.

## Related pages

- {doc}`set-budgets`
- {doc}`../reference/config`
- {doc}`../metrics/score`
