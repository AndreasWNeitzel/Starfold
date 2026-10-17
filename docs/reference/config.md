# Configuration fields

This page lists every field of `Config` and `Space` with its type, default, meaning and effect on cost. It teaches nothing: {doc}`../howto/set-budgets` and {doc}`../howto/custom-space` show how to use the fields.

```{eval-rst}
.. index::
   single: Config
```
```{eval-rst}
.. index::
   single: Space
```
```{eval-rst}
.. index::
   single: budget
```
```{eval-rst}
.. index::
   pair: setting; search space
```

A fit is one run of {term}`UMAP` followed by {term}`HDBSCAN`. Costs are counted in fits because wall time depends on the machine. Recorded runs with the default `Config` and the calibration test off used 314 fits on Digits (1,797 rows), 334 on APOGEE (9,242 rows) and 258 on Letter (20,000 rows). These numbers come from each run's provenance and are recorded in the benchmark results. Wall time per fit is not measured here.

`Config(...)` raises `ValueError` in four cases: a numeric range is empty or reaches zero on a log axis, `n_explore` is not a power of two, `n_null_max` cannot reach `alpha`, or a budget is outside the range in the tables below. The message names the field. `Config` does not catch every unusable `Space`. An empty `selection_method` tuple, or `min_dist` bounds outside 0 to 1, pass construction and fail later in the search.

## `Space`

`Space` holds the ranges of the six {term}`setting` values that the search may propose. The five numeric fields are tuples of bounds, inclusive at both ends. `selection_method` is a tuple of allowed choices. {term}`Trial <trial>` 0 is always the library default, with each numeric value moved to the nearest bound when a bound excludes it. Its selection method is `"eom"` when offered, otherwise the first choice you list.

| Field | Type and default | What it controls | Cost effect |
|---|---|---|---|
| `n_neighbors` | `tuple[int, int]`, `(5, 100)` | Bounds of {term}`n_neighbors`, searched on a log scale. The upper bound may not exceed 100 and is capped at the rows per resample minus 1. | None on the neighbour graph, which is built once per resample at width 100. Effect on UMAP time: not measured. |
| `min_dist` | `tuple[float, float]`, `(0.0, 0.5)` | Bounds of {term}`min_dist`, searched on a linear scale. | Not measured. |
| `n_components` | `tuple[int, int]`, `(2, 10)` | Bounds of {term}`n_components`, searched on a linear integer scale. | Not measured. |
| `min_cluster_size` | `tuple[int, int] or None`, `None` | Bounds of {term}`min_cluster_size`, searched on a log scale. `None` means `[5, min(max(50, n // 20), 1023)]` with `n` the number of rows. | Not measured. |
| `min_samples_fraction` | `tuple[float, float]`, `(0.05, 1.0)` | Bounds of `min_samples / min_cluster_size` (see {term}`min_samples`), searched on a log scale. The ratio cannot exceed 1, because `min_samples` is capped at `min_cluster_size`. A lower bound above 1 is not rejected and cannot be honoured. | Not measured. |
| `selection_method` | `tuple[str, ...]`, `("eom", "leaf")` | The allowed values of the {term}`selection method`. | None. |

## `Config`: search

These fields set how the search spends its trials. A complete trial costs `n_resamples` fits, or twice that when a structureless copy is scored beside it. A pruned trial stops after 2 fits.

| Field | Type and default | What it controls | Cost effect |
|---|---|---|---|
| `space` | `Space`, `Space()` | The ranges above. | See `Space`. |
| `scaling` | `"standard"` or `"robust"`, `"standard"` | How each column is {term}`scaling` inside every resample and refit. `"robust"` uses the median and interquartile range. | None. |
| `n_trials` | `int`, `48` | Maximum number of {term}`trials <trial>`. Must exceed `n_explore`. | About one complete trial per unit, at most `n_resamples` fits (times 2 with a copy). |
| `n_explore` | `int`, `16` | Number of {term}`Sobol design` trials after trial 0. Must be a power of two of at least 2. Design trials are never pruned. | Same per trial as above. The calibration test replays these `n_explore + 1` trials. |
| `n_resamples` | `int`, `3` | Number of random subsets every trial is scored on. At least 2. | Fits per trial grow in step with it, and the held-out pairs grow as `n_resamples * (n_resamples - 1)`. |
| `resample_fraction` | `float`, `0.8` | Fraction of rows in each subset, strictly between 0 and 1. | Fewer rows per fit: not measured. |
| `patience` | `int`, `16` | The search stops when the best score improved by less than `min_improvement` over this many trials. At least 1. | Raising it allows more trials. |
| `min_improvement` | `float`, `0.01` | The smallest improvement that counts. Non-negative. | Raising it stops the search sooner. |
| `prune` | `bool`, `True` | Whether convergence trials may be stopped after 2 fits by {term}`pruning`. | Saves the third fit of each pruned trial. |
| `n_complete_pruned` | `int`, `3` | Number of pruned trials finished at the end when their partial score beats the best complete trial. Non-negative. | At most this many extra trials. |
| `n_rescore` | `int`, `3` | Number of top complete trials that get a {term}`fresh re-score`. The defaults are added when they are not among them. At least 1. | At most `n_rescore + 1` configurations, each scored on a new plan and on `n_null_rescore` new copies. |
| `n_null_rescore` | `int`, `3` | Number of fresh structureless copies scored at the re-score. With 0, no copy is scored at all and the score is not a {term}`gap`. Non-negative. | Each copy adds `n_rescore + 1` scorings of `n_resamples` fits. |
| `gap_family` | `str or None`, `"gaussian_copula"` | The family of {term}`structureless copy` that defines the gap. One of `"gaussian_copula"`, `"independent_marginals"` or `None`. With `None` the search scores raw data, not a gap. | Scoring the copy doubles the fits of every complete trial. |

## `Config`: refits, consensus and calibration

These fields set the {term}`refit` count, the {term}`consensus check` and the optional calibration test.

| Field | Type and default | What it controls | Cost effect |
|---|---|---|---|
| `n_worlds` | `int`, `32` | Number of {term}`refits <refit>` (also called worlds). Between 2 and 32,767. | One fit per refit. |
| `world_fraction` | `float`, `0.8` | Fraction of rows each refit draws without replacement, strictly between 0 and 1. | Not measured. |
| `null_families` | `tuple[str, ...]`, `()` | Families of {term}`structureless copy` for the calibration test inside `recover`. Empty means the test does not run (`null_not_run`). | See `n_null_max`. |
| `null_procedure` | `"exploration"` or `"full_search"`, `"exploration"` | What each copy replays: the `n_explore + 1` opening trials, or the whole search. | `"exploration"`: 102 fits per copy at the defaults. `"full_search"`: higher, not measured. |
| `n_null_max` | `int`, `19` | Largest number of copies per family. Must be at least `ceil(1 / alpha) - 1`. | Up to 1,938 fits per family at 19 copies, about six times the rest of a run. Each extra family adds as many. |
| `null_exceedances` | `int`, `3` | The family stops once this many copies reach the table's value. At least 1. | Stops the test sooner when the table is not distinguishable. |
| `alpha` | `float`, `0.05` | The {term}`alpha` level for `structure_supported`. Strictly between 0 and 1. | Smaller `alpha` forces a larger `n_null_max`. |
| `consensus_budget_bytes` | `int`, `536870912` (512 MiB) | Memory the co-association matrices may use. | The dense check needs `16 * n**2` bytes, so it fits up to 5,792 rows at the default. |
| `consensus_check_rows` | `int`, `4096` | Rows in the recorded subset used above the budget. At least 2, and `16 * rows**2` must fit the budget. | Memory of the check: 268,435,456 bytes at the default. |
| `check_backends` | `bool`, `False` | Whether `recover` also runs `compare_backends`. Needs cuML. | `2 * n_seeds` extra fits (6 at the default 3 seeds). |

## `recover` arguments

| Argument | Type and default | What it controls |
|---|---|---|
| `X` | array-like, rows by columns | The table. At least 50 finite rows and 2 columns. |
| `X_err` | array-like or `None` | Per-value standard deviations, same shape as `X`. Exclusive with `measurement_sampler`. |
| `measurement_sampler` | callable or `None` | A measurement model `sampler(rows, indices, rng)`. Exclusive with `X_err`. |
| `config` | `Config` or `None` | Budgets and rules. `None` means `Config()`. |
| `random_state` | `int`, `0` | The {term}`seed` from which every stage derives its own seed. |
| `engine` | `"auto"`, `"cpu"` or `"cuml"`, `"auto"` | The {term}`engine`. `"auto"` uses cuML when it imports with a device and otherwise warns and uses the CPU. `"cuml"` raises `ImportError` instead of falling back. |
| `n_jobs` | `int`, `1` | CPU worker processes. Forced to 1 on cuML (flag `n_jobs_forced_to_one_on_cuml`). |
