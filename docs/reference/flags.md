# Flags

This page lists all 16 {term}`flags <flag>` that `result.flags` can hold, the exact condition that raises each one, what it means and what to do.

```{eval-rst}
.. index::
   single: flag
```
```{eval-rst}
.. index::
   single: flags; vocabulary
```
```{eval-rst}
.. index::
   single: FLAG_VOCABULARY
```
```{eval-rst}
.. index::
   pair: flag; fault
```

A flag is a short name for one event or weakness in a run. Starfold derives flags from the stored arrays by `starfold.result.derive_flags`, in the order of `starfold.result.FLAG_VOCABULARY`. `load` derives them again and refuses a file whose stored flags differ. Read the flags first, then the numbers they point to.

"Fault" in the last column means the flag says the run or the table is in trouble, so a claim that depends on it needs repair. "No" means the flag reports a fact or a limit on the claim, not a failure.

| Flag | Raised when | Fault |
|---|---|---|
| `search_stopped_early` | `search.stop_reason == "patience"` | No |
| `search_budget_exhausted` | `search.stop_reason == "budget"` | No |
| `selected_is_default` | `search.selected_is_default` is True | No |
| `default_within_noise` | The winner is not the default, and `abs(score_selected - score_default) <= 2 * std(paired_differences, ddof=1) / sqrt(len(paired_differences))`, with at least 2 differences | No |
| `no_groups_in_reference` | No entry of `labels` is 0 or above | Yes |
| `high_unresolved_mass` | `worlds.unresolved_mass > 0.5` | No |
| `count_unstable` | Fewer than half of `worlds.count_values` equal `worlds.count_reference` | Yes: do not quote a single group count |
| `world_fit_failures` | Any entry of `worlds.world_errors` is a non-empty string | Yes |
| `consensus_subsampled` | `consensus.subsampled` is True | No |
| `consensus_disagrees` | `consensus.agreement < 0.5` | Only if the contrast is low too |
| `null_not_run` | `null.skipped_reason == "null_not_run"` (no family declared) | No |
| `observed_statistic_zero` | `null.skipped_reason == "observed_statistic_zero"` (the observed statistic is not above 0) | No, a finding |
| `no_structure_supported` | The test ran (`skipped_reason` is `None`) and `null.structure_supported` is False, so `p_value > alpha` | No, a finding |
| `backend_disagreement` | `backends` is not `None` and `backends.mean_normalised < 0.8` | Yes |
| `rows_rejected_nonfinite` | Input rows minus kept rows is above 0 | No |
| `n_jobs_forced_to_one_on_cuml` | The engine resolved to cuML and `n_jobs` was not 1 | No |

## What each flag means and what to do

`search_stopped_early`: the best score improved by less than `min_improvement` (0.01) over the last `patience` (16) trials, checked from trial `n_explore + 1 + patience` (33 at the defaults). The search converged before `n_trials`. Do nothing.

`search_budget_exhausted`: all `n_trials` ran and the patience rule never stopped the search. The score may still have been rising. Raise `n_trials` if the score matters ({doc}`../howto/set-budgets`).

`selected_is_default`: the library defaults won the fresh re-score, with ties going to the defaults. No searched setting did better. Report this as the outcome. The defaults are a valid answer.

`default_within_noise`: the two fresh scores differ by no more than twice the standard error of their paired differences (6 differences at 3 resamples). The flag compares the fresh scores of the two settings. The search picks the setting with the larger fresh {term}`gap`, which is the fresh score minus the mean score on the copies. Each setting's copy score differs, so the two criteria need not rank the settings the same way. The flag is never raised when the defaults were selected, because the differences are then all 0. Do not claim that the search improved on the defaults. See {doc}`../metrics/default-comparison`.

`no_groups_in_reference`: every row of the emitted partition is noise. Check the columns and the scaling ({doc}`../howto/prepare-table`, {doc}`../howto/choose-scaling`), and lower the `min_cluster_size` range ({doc}`../howto/custom-space`). See {doc}`../troubleshooting`.

`high_unresolved_mass`: rows on average placed more than half their {term}`refit` mass on noise or on rows the emitted partition calls noise. The groups describe a minority of the table. Describe them that way ({doc}`../metrics/unresolved`).

`count_unstable`: fewer than half the refits found the same number of groups as the emitted partition. Quote `worlds.count_interval` and not `n_groups`. The count is not a finding in any case ({doc}`../metrics/count-interval`).

`world_fit_failures`: a refit raised an exception. Its rows count as all noise in that refit, which lowers recurrence. Read `worlds.world_errors` for the text.

`consensus_subsampled`: the matrix of `16 * n**2` bytes did not fit `consensus_budget_bytes` (above 5,792 rows at 512 MiB), so the check used the recorded subset of `consensus_check_rows` rows. Read `consensus.rows`.

`consensus_disagrees`: the regrouping from co-association matches the emitted partition with agreement below 0.5. The agreement is 0 when the cut returns only noise or when `min_cluster_size` exceeds the rows used. Read `consensus.contrast`, which needs no cut. A high contrast with a low agreement points to the cut, not to the groups ({doc}`../metrics/coassociation`).

`null_not_run`: no family was declared, so there is no p-value (`p_value` is 1.0). Run `starfold.null_test` once on the result you report ({doc}`../howto/run-calibration-test`). A separate `null_test` returns its own evidence and leaves this flag on the result.

`observed_statistic_zero`: the largest search {term}`gap` among the opening trials (trials 0 to 16) is 0 or below, so the test was skipped. The search found no setting that beats a structureless copy. Check columns and scaling. Report that no structure was found.

`no_structure_supported`: the test ran and `p_value` exceeds `alpha`. Structureless copies scored as high as your table in too many draws. Report the groups as reproducible, not as shown to exceed the copy family ({doc}`../metrics/calibration-p`).

`backend_disagreement`: CPU and cuML partitions of the same setting agree below 0.8 after normalising. The answer depends on the backend. Compare the partitions and report both ({doc}`../howto/compare-backends`). It needs `check_backends=True`.

`rows_rejected_nonfinite`: some rows held NaN or infinity and were dropped, never filled. Use `result.kept_rows` to align labels with your table ({doc}`../howto/prepare-table`).

`n_jobs_forced_to_one_on_cuml`: you asked for several CPU workers and the engine is cuML, which runs one fit at a time. Nothing to fix ({doc}`../howto/run-parallel`).
