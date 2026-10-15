# Result fields

This page lists every field of `Result` and of the objects it holds, with the type, the meaning and the page that explains the value. It teaches nothing. Start from {doc}`../metrics/index` to learn how to read a number.

```{eval-rst}
.. index::
   single: Result
```
```{eval-rst}
.. index::
   single: result fields
```
```{eval-rst}
.. index::
   single: provenance
```
```{eval-rst}
.. index::
   single: kept_rows
```

Shapes use these symbols: `n_kept` for the rows that were finite and used, `G` for the number of groups in the emitted partition, `W` for the number of {term}`refits <refit>`, and `d` for the number of columns. Group ids of the emitted partition run from 0, and -1 marks {term}`noise`.

## `Result`

`recover` returns one `Result`. It is immutable.

| Field | Type | Meaning | Explained on |
|---|---|---|---|
| `labels` | `int32`, `(n_kept,)` | The emitted {term}`partition`: one full-data fit of `params`, -1 for noise. | {doc}`../concepts/pipeline` |
| `strengths` | `float32`, `(n_kept,)` | HDBSCAN within-fit strengths of that fit. They are not probabilities. | {doc}`../metrics/strengths` |
| `default_labels` | `int32`, `(n_kept,)` | One full-data fit of the library defaults, kept as the external comparator. | {doc}`../metrics/default-comparison` |
| `kept_rows` | `intp`, `(n_kept,)` | Indices into the input of the rows that were finite and used. `X[result.kept_rows]` lines up with `labels`. | {doc}`../howto/prepare-table` |
| `params` | `Params` | The {term}`setting` the search selected, validated on resample rows. | {doc}`../concepts/optuna` |
| `deployed_params` | `Params` | The setting that produced `labels`: `params` with `min_cluster_size` and `min_samples` rescaled from the resample row count to the full row count. Equal to `params` when the counts agree. | {doc}`../concepts/pipeline` |
| `default_params` | `Params` | The library defaults, `Params(15, 0.1, 2, 5, 5, "eom")`. | {doc}`../metrics/default-comparison` |
| `search` | `SearchResult` | Trial table, candidates, fresh scores and the comparison with the defaults. | see below |
| `worlds` | `WorldEvidence` | Recurrence evidence over the refits. | see below |
| `consensus` | `ConsensusCheck` | The co-association check on the emitted partition. | see below |
| `null` | `NullEvidence` | The calibration test. Before it runs, `skipped_reason` is `"null_not_run"`. | see below |
| `backends` | `BackendAgreement` or `None` | CPU against cuML agreement. `None` unless `recover` ran with `Config(check_backends=True)`. A standalone `compare_backends` call returns its own `BackendAgreement` and leaves `result.backends` as `None`. | see below |
| `flags` | `tuple[str, ...]` | The raised flags, in vocabulary order. | {doc}`flags` |
| `provenance` | `dict` | Engines, versions, configuration, seeds, budgets, fit counts, time per stage, peak memory and the input fingerprint. | see below |

Three properties read from the parts: `n_groups` (number of groups in `labels`), `support` (the `worlds.support` matrix) and `unresolved_mass` (`worlds.unresolved_mass`).

## `SearchResult` (`result.search`)

| Field | Type | Meaning | Explained on |
|---|---|---|---|
| `selected` | `Params` | The chosen setting (equal to `Result.params`). | {doc}`../concepts/optuna` |
| `trials` | `tuple[TrialRecord, ...]` | Every trial, in trial order. | {doc}`../metrics/score` |
| `candidates` | `tuple[Candidate, ...]` | The re-scored configurations, best fresh gap first. Empty in exploration-only mode. | {doc}`../metrics/gap` |
| `score_selected`, `score_default` | `float` | Fresh scores of the selected setting and the defaults. | {doc}`../metrics/default-comparison` |
| `gap_selected`, `gap_default` | `float` | Fresh scores minus the mean score on the fresh copies. Equal to the scores when no copy was scored. | {doc}`../metrics/gap` |
| `paired_differences` | `tuple[float, ...]` | Selected minus default fresh pair score, one per ordered pair of resamples (6 at `n_resamples=3`). | {doc}`../metrics/default-comparison` |
| `beats_default_under_objective` | `bool` | `gap_selected > gap_default`. It does not test the size of the difference. | {doc}`../metrics/default-comparison` |
| `selected_is_default` | `bool` | Whether trial 0 won. | {doc}`flags` |
| `exploration_max` | `float` | Largest score among trials 0 to `n_explore`. The statistic of the calibration test. | {doc}`../metrics/calibration-p` |
| `stop_reason` | `str` | `"patience"`, `"budget"` or `"exploration_only"`. | {doc}`flags` |
| `n_completed`, `n_pruned`, `n_completed_after_prune` | `int` | Counts of complete trials, pruned trials, and pruned trials finished at the end. | {doc}`config` |
| `n_fits`, `seconds` | `int`, `float` | Fits and wall time of the search. | {doc}`config` |
| `plan` | `ResamplePlan` | The resample indices and seeds of the main plan. | {doc}`../concepts/pipeline` |

Each `TrialRecord` holds `number`, `params`, `state` (`"complete"`, `"pruned"` or `"failed"`), `score` (the search {term}`gap` the sampler maximises, which is `raw_score` minus `null_score`; `raw_score` is the {term}`score` on your table and `null_score` the score on the {term}`structureless copy`), `partial_score`, `stability` (the {term}`prediction agreement`), `coverage`, `modularity`, `fit_counts` (groups per fit), `fit_noise` (noise share per fit), `seconds`, `error` and `phase` (`"default"`, `"exploration"`, `"convergence"` or `"completion"`). See {doc}`../metrics/score`, {doc}`../metrics/prediction-agreement`, {doc}`../metrics/coverage` and {doc}`../metrics/modularity`.

Each `Candidate` holds `params`, `trial_number`, `score` (search score), `fresh_score`, `fresh_pair_scores`, `fit_counts`, `null_scores` and `gap`.

## `WorldEvidence` (`result.worlds`)

| Field | Type | Meaning | Explained on |
|---|---|---|---|
| `world_labels` | `int16`, `(W, n_kept)` | Per refit: -2 not sampled, -1 noise, otherwise the refit's cluster id. | {doc}`../concepts/refits` |
| `n_sampled` | `uint16`, `(n_kept,)` | Refits that sampled each row. | {doc}`../metrics/support` |
| `support` | `float32`, `(n_kept, G)` | Recurrence support of each row for each group. NaN where no refit sampled the row. | {doc}`../metrics/support` |
| `unresolved` | `float32`, `(n_kept,)` | Mass a row put on noise. `support.sum(1) + unresolved` is 1 on every sampled row. | {doc}`../metrics/unresolved` |
| `own_support` | `float32`, `(n_kept,)` | `support[i, labels[i]]`, NaN for noise rows. | {doc}`../metrics/support` |
| `unresolved_mass` | `float` | Mean of `unresolved` over sampled rows. | {doc}`../metrics/unresolved` |
| `group_recurrence` | `float32`, `(G,)` | Mean over refits of the best Jaccard match of each group. | {doc}`../metrics/recurrence` |
| `group_recurrence_band` | `float32`, `(G, 2)` | 5th and 95th percentiles of the per-refit Jaccard match. | {doc}`../metrics/recurrence` |
| `count_values` | `int16`, `(W,)` | Number of clusters in each refit. | {doc}`../metrics/count-interval` |
| `count_interval` | `tuple[int, int]` | Floor of the 5th and ceiling of the 95th percentile of `count_values`. | {doc}`../metrics/count-interval` |
| `count_reference` | `int` | Number of groups in `labels`. | {doc}`../metrics/count-interval` |
| `property_median` | `float64`, `(G, d)` | Median over refits of the per-column median of the matched refit cluster, in original units. | {doc}`../metrics/properties` |
| `property_band` | `float64`, `(G, d, 2)` | 5th and 95th percentiles of the same per-refit medians. | {doc}`../metrics/properties` |
| `property_propagated` | `bool` | Whether refits carried measurement draws. | {doc}`../howto/measurement-errors` |
| `world_agreement` | `float32`, `(W,)` | Per refit, the {term}`refit agreement`: adjusted Rand index (floored at 0) times joint coverage against the emitted partition. | {term}`refit agreement` |
| `world_seeds` | `tuple[int, ...]` | UMAP seed of each refit. | {doc}`../howto/reproduce-a-run` |
| `world_errors` | `tuple[str, ...]` | Exception text of each refit whose fit raised, empty string otherwise. | {doc}`flags` |
| `n_fits`, `seconds` | `int`, `float` | Refit fits attempted and wall time. | {doc}`config` |

## `ConsensusCheck` (`result.consensus`)

| Field | Type | Meaning | Explained on |
|---|---|---|---|
| `labels` | `int32` | Consensus partition over `rows`, -1 for noise. | {doc}`../metrics/coassociation` |
| `rows` | `intp` | Sorted row indices the check used. | {doc}`../metrics/coassociation` |
| `agreement` | `float` | The {term}`consensus agreement`: adjusted Rand index (floored at 0) times joint coverage between `labels` and the emitted partition on `rows`. | {doc}`../metrics/coassociation` |
| `subsampled` | `bool` | Whether the byte budget forced a seeded subset. | {doc}`../metrics/coassociation` |
| `n_rows`, `bytes_used` | `int` | Rows used and `16 * n_rows**2`. | {doc}`config` |
| `within_association`, `between_association` | `float` | Mean co-association of pairs in the same group and in different groups. | {doc}`../metrics/coassociation` |
| `contrast` | `float` | `within_association - between_association`. NaN when the cut is skipped. | {doc}`../metrics/coassociation` |

## `NullEvidence` (`result.null`)

| Field | Type | Meaning | Explained on |
|---|---|---|---|
| `families` | `tuple[str, ...]` | Families run, in order. | {doc}`../metrics/calibration-p` |
| `statistic_name` | `str` | `"exploration_design_max"` or `"full_search_max"`. | {doc}`../metrics/calibration-p` |
| `observed` | `float` | The statistic on your table: the largest search gap among the replayed trials (trials 0 to 16 by default). | {doc}`../metrics/calibration-p` |
| `values` | `tuple[tuple[float, ...], ...]` | The statistic on each copy, per family. | {doc}`../metrics/calibration-p` |
| `n_realisations`, `exceedances` | `tuple[int, ...]` | Copies drawn, and copies at or above `observed`, per family. | {doc}`../metrics/calibration-p` |
| `p_values` | `tuple[float, ...]` | `(exceedances + 1) / (n_realisations + 1)` per family. | {doc}`../metrics/calibration-p` |
| `p_intervals` | `tuple[tuple[float, float], ...]` | Nominal 95 percent Clopper-Pearson interval on the exceedance share. It describes Monte Carlo precision only. | {doc}`../metrics/calibration-p` |
| `p_value` | `float` | The largest `p_values` entry. 1.0 when skipped. | {doc}`../metrics/calibration-p` |
| `alpha` | `float` | The declared level. | {doc}`../metrics/calibration-p` |
| `structure_supported` | `bool` | `p_value <= alpha`. | {doc}`../metrics/calibration-p` |
| `stopped_early` | `tuple[bool, ...]` | Whether the stopping rule ended a family before `n_null_max`. | {doc}`../metrics/calibration-p` |
| `skipped_reason` | `str` or `None` | `"null_not_run"`, `"observed_statistic_zero"` or `None`. | {doc}`flags` |
| `n_fits`, `seconds` | `int`, `float` | Fits and wall time of the stage. | {doc}`config` |

## `BackendAgreement` (`result.backends`)

| Field | Type | Meaning | Explained on |
|---|---|---|---|
| `agreement`, `mean_agreement` | `tuple[float, ...]`, `float` | The {term}`backend agreement`, per seed and mean: adjusted Rand index times joint coverage between the CPU and cuML fits. | {doc}`../howto/compare-backends` |
| `ari`, `coverage` | `tuple[float, ...]` | The two factors of `agreement`, per seed. | {doc}`../howto/compare-backends` |
| `assigned_cpu`, `assigned_cuml` | `tuple[float, ...]` | Share of rows each backend put in a group, per seed. | {doc}`../howto/compare-backends` |
| `normalised`, `mean_normalised` | `tuple[float, ...]`, `float` | `agreement` divided by the smaller assigned share. `mean_normalised` is what `backend_disagreement` reads. | {doc}`../howto/compare-backends` |
| `n_groups_cpu`, `n_groups_cuml` | `tuple[int, ...]` | Group count per seed on each backend. | {doc}`../howto/compare-backends` |

## `provenance` keys

`result.provenance` is a dictionary with these keys, all JSON types.

| Key | Holds |
|---|---|
| `starfold_version`, `python`, `platform` | Versions and operating system. |
| `libraries` | Installed versions of `numpy`, `scipy`, `scikit-learn`, `umap-learn`, `hdbscan`, `optuna`, `joblib` and `cuml`, or `"absent"`. |
| `engine_requested`, `engine_resolved` | The {term}`engine` you asked for and the one used. |
| `n_jobs`, `n_jobs_forced_to_one_on_cuml` | Worker processes used and whether cuML forced 1. |
| `random_state` | The base {term}`seed`. |
| `config` | The full `Config`, with `Space`, as nested JSON. |
| `n_input_rows`, `n_kept_rows`, `n_features` | Rows given, rows kept and columns. |
| `input_fingerprint` | SHA-256 of the kept matrix as C-ordered float64 bytes plus its shape. |
| `stage_seconds`, `stage_fits`, `total_fits` | Wall time per stage (`search`, `reference`, `worlds`, `consensus`, `null`, `backends`, `total`), fits per stage (the same names except `consensus` and `total`) and the fit total. |
| `peak_rss_mb` | Peak resident memory of the process in MB, NaN where unknown (saved as `null`). |

{doc}`../howto/reproduce-a-run` shows how to use these keys.
