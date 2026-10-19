# Troubleshooting

This page lists 15 things you may see in a result or a run, with the likely causes, the check to run, and the fix. Find the heading that matches what you see.

```{eval-rst}
.. index::
   single: troubleshooting
```
```{eval-rst}
.. index::
   single: symptoms
```
```{eval-rst}
.. index::
   single: error messages
```

Each answer follows one order: causes, check, fix, where to read more. All checks use an object named `result`, the return value of `starfold.recover`. Causes come from the code and from the recorded runs. These ship in the repository under `notebooks/runs/digits` and `notebooks/runs/apogee`. Notebooks 01 and 02 write them, and `starfold.load` reads them. Where an effect was not measured, the answer says so.

## Almost every row is noise

Causes. The {term}`min_cluster_size` range may sit above the size of your groups. A large `min_samples` fraction makes HDBSCAN call more rows {term}`noise`. Columns that do not separate anything, or are badly scaled, leave no dense clumps. If every row is noise, the flag `no_groups_in_reference` is set.

Check. Print the noise share and the flags.

```python
import numpy as np
print(np.mean(result.labels < 0), result.flags)
```

Then read `fit_noise` of each trial in `result.search.trials`. A noise share of 0.12 (1,068 of 9,242 rows) is what the recorded run on the APOGEE DR19 example table gave, so some noise is normal for a table with a diffuse background.

Fix. Work through these in order.

1. Drop columns that do not separate your rows, and check the scaling ({doc}`howto/prepare-table`, {doc}`howto/choose-scaling`).
2. If your groups are small, lower the upper bound of `min_cluster_size` in `Space`. `Space` is the set of ranges that the search may propose ({doc}`howto/custom-space`). For example, `starfold.Config(space=starfold.Space(min_cluster_size=(5, 30)))` searches only for groups of 5 to 30 rows or more. Pick the upper bound from the smallest group you expect, not from this example.
3. Run again and check that `np.mean(result.labels < 0)` falls.

Read {doc}`metrics/unresolved` to separate noise in the full fit from noise across refits.

## The count interval is wide

Causes. The interval is the 5th to 95th percentile of the number of groups that the {term}`refits <refit>` found, so a wide interval means the table supports several readings. Groups near the small-group scale merge and split from one refit to the next ({doc}`limits`). Irrelevant columns blur the groups. The recorded run on the scikit-learn digits dataset gave 10 groups with an interval of 3 to 16.

Check. Print `result.worlds.count_interval`, `result.worlds.count_reference` and `np.bincount(result.worlds.count_values)`.

Fix. Quote the interval and not `n_groups`. Drop irrelevant columns, or add rows. The group count is not a finding either way ({doc}`metrics/count-interval`).

## Every group has low recurrence

Causes. The table may hold no groups that come back: check the calibration p-value. A group that recurs below 0.85 but above 0.5 is a separate case ({doc}`the recurrence page <metrics/recurrence>`). A refit that raised an error counts as all noise in that refit, which lowers recurrence. Large `X_err` values perturb every refit, so an inflated or invented error bar produces a low recurrence.

Check. Print `result.worlds.group_recurrence`, `result.worlds.group_recurrence_band` and `result.worlds.world_errors`. Check `result.flags` for `world_fit_failures`, `count_unstable` and `no_structure_supported`.

Fix. Remove a failing input (the error text is in `world_errors`). Check that `X_err` holds real standard deviations ({doc}`howto/measurement-errors`). Recurrence is a rate over refits, not the probability that a group is real ({doc}`metrics/recurrence`).

## Consensus agreement is 0.0 but contrast is high

Causes. The `consensus_disagrees` flag reads the {term}`consensus agreement` of a regrouping, not the refits themselves. The regrouping cuts the {term}`co-association` distance with HDBSCAN. It returns only noise when `min_cluster_size` exceeds the rows used. In the recorded APOGEE run the cut returned only noise (consensus agreement 0.0) while within was 0.979, between 0.003 and contrast 0.976. The cause of that all-noise cut is not isolated. That run also used a seeded subset of 4,096 of 9,242 stars (flag `consensus_subsampled`).

Check. Print `result.consensus.within_association`, `between_association`, `contrast` and `agreement` (the consensus agreement). Which of the four agreements you need is explained under {ref}`Which agreement do I read <which-agreement>`.

Fix. Trust the contrast, which needs no cut. Whether more refits (`n_worlds`) help the cut is not measured. See {doc}`metrics/coassociation`.

(which-agreement)=
## Which agreement do I read?

Causes. Four different numbers carry the word agreement. They compare different pairs of groupings, and none of them is the probability that a group is real. A fifth, {term}`backend agreement`, compares CPU with GPU fits ({doc}`howto/compare-backends`).

| Name | What it compares | Where it is printed | Recorded value |
|---|---|---|---|
| {term}`Prediction agreement` | Fits on two subsets, on rows one of them never saw | `result.search.trials[i].stability` | 0.908 (Digits trial 37, 3 groups) |
| {term}`Refit agreement` | Each refit with the full-data labels, as the adjusted Rand index times coverage | `result.worlds.world_agreement` | Digits median 0.79 (0.13 to 0.89) |
| {term}`Consensus agreement` | The co-association cut with the full-data labels, as the adjusted Rand index times coverage | `result.consensus.agreement` | Digits 0.135, APOGEE 0.000 |
| {term}`Agreement with known classes` | Your labels with classes you already know, after the run | your own code ({doc}`howto/check-against-known-labels`) | Digits 0.740 (defaults 0.594) |

Check. Decide the question first.

- Do the groups come back on resampled rows: read {term}`recurrence` per group, then the refit agreement as a one-number summary.
- Did the search choose well: prediction agreement is one term inside the {term}`score`. Do not report it alone, because a coarse partition with few groups scores high on it.
- Does the co-association matrix support the partition: read the contrast first. Low consensus agreement with a high contrast points to the cut and not to the groups (the recorded APOGEE case).
- Do the groups match classes: use the agreement with known classes after the run, and never tune the search on it.

Fix. Write the full name every time. Do not write agreement alone in a paper. See {doc}`metrics/index`.

## The p-value is 0.10 at 19 copies

Causes. The p-value is `(h + 1) / (L + 1)`. Here `L` is the number of {term}`structureless copies <structureless copy>` drawn, and `h` is the number of copies whose statistic is at least as high as your table's (an exceedance). With `L = 19` the smallest p is 1/20 = 0.05, so 0.10 means exactly one copy matched your table, and p moves in steps of 0.05. A larger `L` allows smaller values. In the recorded APOGEE run one copy scored 0.366 against the table's 0.362. The test replays only trials 0 to 16, and its power against the statistic of the full search is not measured. The default `gaussian_copula` family keeps the correlations between columns.

Check. Print `result.null.exceedances`, `n_realisations`, `p_intervals` and `statistic_name`. The interval for APOGEE ran from 0.001 to 0.260.

Fix. Report the p-value with its interval and do not round it to a verdict. A larger `n_null_max` gives finer steps and costs up to 102 fits per copy. The 0.05 floor and the stopping rule (3 exceedances) hold at the default 19 copies only. A second family (`independent_marginals`) answers a different and easier question. See {doc}`howto/run-calibration-test` and {doc}`metrics/calibration-p`.

## The run says it fell back to the CPU

Causes. `engine="auto"` warns with a `RuntimeWarning` when cuML cannot be imported or no CUDA device is visible. The reason is in the message.

Check. Read the warning text, then `result.provenance["engine_requested"]`, `["engine_resolved"]` and `["libraries"]["cuml"]`.

Fix. Install the GPU extra that matches your driver (`gpu-cu12` or `gpu-cu13`; Linux or WSL 2 only). Pass `engine="cuml"` to get an `ImportError` with the reason instead of a fallback, or `engine="cpu"` to silence the warning. See {doc}`howto/use-gpu`.

## The default within noise flag is set

Causes. The selected setting is not the library default, and the fresh scores of the two settings are close. They differ by no more than twice the standard error of their paired differences (6 differences at `n_resamples=3`).

Check. Print `result.search.paired_differences`, `score_selected` and `score_default`.

Fix. Do not claim that the search improved on the defaults. The result is still one valid partition. See {doc}`metrics/default-comparison` and {doc}`reference/flags`.

## The run is much slower than expected

Causes. `null_families` is set, which adds up to 1,938 fits per family, about six times the rest of a run. The CPU is running instead of the GPU. The table is large, since every fit sees 0.8 of the rows.

Check. Print `result.provenance["stage_seconds"]`, `["stage_fits"]` and `["engine_resolved"]`. Recorded runs with the test off used 314 fits on the scikit-learn digits dataset and 334 fits on the APOGEE DR19 example table.

Fix. Keep the test off for sweeps and run `null_test` once on the result you report. Lower `n_trials`, `n_worlds` or `n_resamples` ({doc}`howto/set-budgets`). Use several processes on the CPU ({doc}`howto/run-parallel`).

## I get a ValueError

Causes. `recover` checks the table and `Config` checks its own fields when you build it.

| Message starts with | Cause |
|---|---|
| `X must keep at least 50 finite rows` | Fewer than 50 rows have only finite values. |
| `X must have at least 2 columns` | The table has one column. |
| `X must be a numeric array` | The table cannot be converted to floats, for example because it holds non-numeric text. |
| `X_err must have the same shape as X` | Wrong shape. `X_err` must also be finite and non-negative on kept rows. |
| `Give X_err or measurement_sampler, not both` | Both were passed. |
| `n_explore must be a power of two` | For example 12 or 20. |
| `n_trials must exceed n_explore` | At the default `n_explore` of 16, `n_trials` must be at least 17. |
| `n_null_max cannot reach alpha` | Fewer than `ceil(1 / alpha) - 1` copies. |
| `consensus_check_rows squared times 16 bytes` | The subset does not fit `consensus_budget_bytes`. |
| `n_neighbors bounds ... reach above 100` | A `Space` bound above the cached graph width. |

Fix. The message names the field. See {doc}`howto/prepare-table`.

## `load` refuses my file

Causes. A checksum in `manifest.json` does not match the file. The format tag is not `"starfold-result-2"`. The support identity fails on a sampled row. The stored flags differ from the flags derived from the arrays.

Check. Read the message. It names the manifest or the file with a checksum problem, the unknown format, the support identity, or both flag lists.

Fix. Save the result again from the run that produced it. Do not edit `result.json` by hand. See {doc}`reference/artifacts` and {doc}`howto/save-and-load`.

## There are hundreds of tiny groups

Causes. A smooth table cut into small pieces, or classes near the small-group scale. On the benchmark, Gas drift gave 390 groups for 6 classes and Letter 284 for 26, both with low recovery (0.038 and 0.203).

Check. Print `np.bincount(result.labels[result.labels >= 0])` and `result.worlds.count_interval`. Check `no_structure_supported`.

Fix. Raise the lower bound of `min_cluster_size` in `Space` and drop irrelevant columns. Do not treat the count as a finding ({doc}`limits`, {doc}`howto/custom-space`).

## A group I expect is missing or merged

Causes. Groups below about `sqrt(15 m)` rows merge into a neighbour, with `m` the rows per resample (155 at 2,000 rows, 346 at 10,000 rows). In an injection test on 2,000 rows of the APOGEE DR19 example table, groups of 30 and 60 rows were retrieved 0 of 3 times. Groups of 120, 240 and 480 rows were retrieved 3 of 3 times.

Check. Compare the expected group size with `sqrt(15 * 0.8 * n)`.

Fix. Add rows, drop irrelevant columns, or expect the group to merge. See {doc}`limits`.

## CPU and GPU give different labels

Causes. The backends differ in floating-point order, so labels and counts differ without the partitions being wrong. On the recorded digits fits the group counts were 14, 12, 13 on the CPU and 10, 10, 10 on the GPU.

Check. Run `starfold.compare_backends` and read `mean_normalised`. Below 0.8 the flag `backend_disagreement` is raised. The recorded Waveform value was 0.58.

Fix. Compare partitions and evidence, never label numbers or counts. Report both when they differ. See {doc}`howto/compare-backends`.

## The same seed gives a different result

Causes. A different `Config`, `Space`, `random_state`, engine or library version. Above 20,000 rows per resample the CPU neighbour graph is approximate and depends on the thread count, so `n_jobs` must match. CPU and cuML runs differ.

Check. Compare `result.provenance["config"]`, `["libraries"]`, `["engine_resolved"]`, `["n_jobs"]` and `["input_fingerprint"]` between the two runs.

Fix. Match them, and keep the `Config` in one module that every job imports. See {doc}`howto/reproduce-a-run`.
