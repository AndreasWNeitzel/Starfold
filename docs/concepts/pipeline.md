# How Starfold chains the tools

This page follows one call of `starfold.recover` through its six stages, and says what each stage produces and which page explains it.

```{eval-rst}
.. index::
   single: pipeline
   single: recover
   single: full fit
   single: deployed settings
   pair: pipeline; stages
```

## The idea

UMAP, HDBSCAN and Optuna each do one job: {doc}`umap` draws a map, {doc}`hdbscan` finds clumps in it, and {doc}`optuna` searches the settings of both. Starfold wraps them in a score that never reads labels, a discount for what a structureless table would score, and refits that measure how much the answer depends on the rows you happened to have.

```text
your table -> search -> score -> choice -> full fit -> refits -> (calibration test)
                                              |           |
                                           labels      evidence, flags
```

## The six stages

1. **Search.** Optuna tries up to 48 {term}`setting`s: the library defaults first, then 16 spread evenly, then guided guesses. Each {term}`trial` is fitted on three random 80 percent subsets. It produces a table of trials, stored in `result.search.trials`. Read {doc}`optuna`.

2. **Score.** Each trial gets a {term}`score` on your table from {term}`prediction agreement`, {term}`coverage` and {term}`modularity`. Starfold fits the same setting on a {term}`structureless copy` of your table (your table with its cross-column structure destroyed) and scores that fit the same way. The score on your table minus the score on the copy is the trial's {term}`gap`. By default the search tries to make the gap large, so the number it ranks trials by is the gap, not the score. It produces one gap per trial. Read {doc}`copies-and-gaps` and {doc}`../metrics/gap`.

3. **Choice.** The top three trials and the library defaults are scored again on new subsets and new copies (a {term}`fresh re-score`), because the best of many noisy gaps is partly luck. The largest fresh gap wins, and ties go to the defaults. It produces one chosen setting, `result.params`, and a table of the candidates with their fresh scores and fresh gaps. That table is the sensitivity report.

4. **Full fit.** The chosen setting is fitted once on all rows. Because the count-based settings `min_cluster_size` and `min_samples` depend on how many rows a fit sees, Starfold rescales them to the full table (`result.deployed_params`). It produces `result.labels`, one group number per kept row, with -1 for {term}`noise`. The groups and the noise rows together are the partition.

5. **Refits.** The chosen setting is fitted again on 32 random 80 percent subsets. It produces {term}`support` per row, {term}`recurrence` per group, the {term}`count interval` and the {term}`unresolved mass`. A bounded {term}`co-association` check on the full fit follows: it counts how often each pair of rows lands in the same group across refits. Read {doc}`refits`.

6. **Calibration test.** An optional test replays the first 17 trials (the defaults and the 16-trial Sobol design) on up to 19 structureless copies. It compares the best gap of each copy with the best gap of your table. The option `null_procedure="full_search"` replays the whole search instead. It produces a {term}`p-value`. It is off by default, because it costs about six times the fits of the rest of a run (four to nine times the wall time in the recorded runs, which ship in the repository under `notebooks/runs/digits` and `notebooks/runs/apogee`). Run it once, on the result you report. Read {doc}`copies-and-gaps`.

Every stage ends in a {term}`flag` when its evidence is weak. `result.flags` collects them.

## Which page explains which output

| Output | Where it is stored | Page |
|---|---|---|
| Labels and noise | `result.labels` | {doc}`hdbscan` |
| Chosen and default settings | `result.params`, `result.default_params` | {doc}`optuna` |
| Gap | `result.search.trials` | {doc}`../metrics/gap` |
| Support, recurrence, count interval | `result.support`, `result.worlds` | {doc}`refits` |
| p-value | `result.null` | {doc}`copies-and-gaps` |
| Flags | `result.flags` | {doc}`../reference/flags` |

The {doc}`../metrics/index` pages give one page per printed number.

## Where the chain breaks

- The search can pick a setting that scores well because of the sample it happened to draw. The fresh re-score reduces this. It does not remove it, because the winner is still the largest of up to four noisy fresh gaps.
- A full fit is one fit. The refits show how far to trust it.
- The calibration test runs only when you ask, so a default result has no p-value and carries the flag `null_not_run`.

Next: {doc}`copies-and-gaps`.
