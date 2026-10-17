# Run the calibration test

This guide runs the matched Monte Carlo test and shows how to read its p-value. The test asks whether {term}`structureless copy` tables of your table score as high as your table does.

```{eval-rst}
.. index::
   single: calibration test
   single: null_test
   pair: p-value; calibration
   single: structure_supported
   single: n_null_max
   single: alpha
   single: generator family
```

## Prerequisites

- A result from `starfold.recover` with the test off (the default), its table `X`, and the `config`, `random_state` and `engine` of that run.
- Time. One copy costs 102 fits at the default budgets (17 trials x 3 resamples x 2 tables, the copy needing a copy of its own). Up to 19 copies cost up to 1,938 fits, about six times the rest of a run.

## What the test answers

Each copy is drawn from the declared family and goes through the first 17 trials of the same search. The {term}`p-value` is the share of copies that reach a gap at least as high as your table's (your table counts as one more draw). If your table is a draw from the declared family, p is valid: the chance that p falls at or below `alpha` is at most `alpha`. It is not spread evenly over its possible values, because the test stops at the third copy that reaches your table. It holds only for the declared family of copies and the procedure that was simulated. It is not the probability that the groups are real, and it says nothing about how many groups exist. See {term}`null hypothesis`.

## Steps

1. Keep the rows `recover` kept. `null_test` refuses a matrix that has a non-finite value.

   ```python
   import starfold

   Xk = X[result.kept_rows]
   ```

2. Choose the family of copies. The default is `("gaussian_copula",)`.

   | Family | Keeps | Destroys | Tests against |
   |---|---|---|---|
   | `gaussian_copula` | every column's values and a fitted, shrunk Gaussian dependence between columns | structure that needs several columns together | one population whose columns keep their shapes and are linked only by a Gaussian dependence |
   | `independent_marginals` | every column's values | all dependence between columns | columns with no relation |

   Neither family removes multimodality within a single column: a column with two peaks keeps both peaks in every copy. A {term}`Gaussian copula` copy can score almost as high as your table when your groups show up only as a correlation between columns. `independent_marginals` is easier to beat. With several families, `null.p_value` is the largest of their p-values. Every value in a copy is a value you measured, only reordered.

3. Choose `alpha` and `n_null_max` in the `Config`, the same `Config` as the run. The test needs `n_null_max` of at least `ceil(1 / alpha) - 1` (19 at `alpha=0.05`), or building the `Config` raises `ValueError`. `null_test` has no arguments for these two, so change them with `dataclasses.replace`.

   ```python
   import dataclasses

   config = starfold.Config()  # the config of the run you calibrate
   config = dataclasses.replace(config, n_null_max=99)  # optional, lowers the floor to p = 0.01
   ```

   With 99 copies the cost rises to up to 10,098 fits. Neither field changes the search, so the replay stays matched.

4. Run the test.

   ```python
   null = starfold.null_test(
       Xk,
       result.search,
       config=config,
       families=("gaussian_copula",),
       random_state=result.provenance["random_state"],
       engine=result.provenance["engine_resolved"],
   )
   print(null.p_value, null.p_intervals, null.structure_supported)
   ```

   The config (with its `Space`), `random_state` and `engine` must equal those of the run, so the replay uses the same procedure and seeds. The call reads the last two from the run's record. `null_test` defaults to `engine="auto"` and `random_state=0`, which do not match a run on another engine or seed. Above 20,000 rows per resample, `n_jobs` must match too ({doc}`run-parallel`).

5. Read the result. `null.observed` is the largest search {term}`gap` (score on your table minus score on a copy) among the 17 opening trials: the defaults and the 16-trial {term}`Sobol design`. For Digits, 0.710 is the search gap of trial 7. It is not the best score of the full search. `null.n_realisations` counts the copies drawn and `null.exceedances` the copies at or above your table. With `h` exceedances in `L` copies, `p = (h + 1) / (L + 1)`.

   | Copies `L` | p for h = 0 | h = 1 | h = 2 | h = 3 |
   |---:|---:|---:|---:|---:|
   | 4 | 0.200 | 0.400 | 0.600 | 0.800 |
   | 9 | 0.100 | 0.200 | 0.300 | 0.400 |
   | 19 | 0.050 | 0.100 | 0.150 | 0.200 |
   | 99 | 0.010 | 0.020 | 0.030 | 0.040 |

   At any `n_null_max`, the test stops once 3 copies reach your table. At the default 19 copies its p is then at least 4 / 20 = 0.2, above any usual `alpha`, so more copies would only refine a p that is already too high. The 0.05 floor holds at 19 copies only. At 99 copies the floor is 0.01, and if the third copy to reach your table is the 99th, p = 0.04. `structure_supported` is `p_value <= alpha`. It is a statement about copies, never a probability that groups are real. `null.p_intervals` is a 95 percent Clopper-Pearson interval on the share of copies at or above your table. It measures the Monte Carlo precision of 19 copies and nothing else.

6. Attach the test to the saved result, so the flags are re-derived. A result from `recover` with the test off carries `null_not_run` and a placeholder p of 1.0.

   ```python
   from starfold.result import derive_flags

   prov = result.provenance
   calibrated = dataclasses.replace(
       result,
       null=null,
       flags=derive_flags(
           labels=result.labels,
           search=result.search,
           worlds=result.worlds,
           consensus=result.consensus,
           null=null,
           backends=result.backends,
           rows_rejected=int(prov["n_input_rows"]) - int(prov["n_kept_rows"]),
           n_jobs_forced=bool(prov["n_jobs_forced_to_one_on_cuml"]),
       ),
   )
   starfold.save(calibrated, "runs/digits_null")
   ```

   `starfold.load` re-derives the flags from the evidence and refuses a result whose stored flags differ, so the flags must be replaced in the same step.

## Expected result

The recorded 19-copy runs on two tables, read from `bench/results/digits_null.json` and `bench/results/apogee_null.json`:

You see:

```text
digits  observed 0.710, 19 copies, 0 at or above the data, p 0.05, fraction at or above 0.000 to 0.176 (95% Monte Carlo interval), 1938 fits, supported True
apogee  observed 0.362, 19 copies, 1 at or above the data, p 0.10, fraction at or above 0.001 to 0.260 (95% Monte Carlo interval), 1938 fits, supported False
```

The Digits table beat all 19 copies, and p sits on its floor of 0.05, which equals `alpha`, so the structure is supported at that level. The APOGEE table was matched by one copy (0.366 against 0.362), p is 0.10, and `result.flags` gains `no_structure_supported`. The decision this supports: for APOGEE, report the groups as reproducible across refits and do not claim that they exceed a correlated single population. The Digits record took 791 s for 1,938 fits (the engine is not stored in the record).

## Variations

- **Inside `recover`.** Pass `Config(null_families=("gaussian_copula",))`. The test then runs as part of the call and the result has the flags already.
- **A full replay.** `Config(null_procedure="full_search")` replays the whole search for every copy instead of the 17 opening trials, and the statistic changes with it: it is the selected setting's fresh score, not a gap. The power of the default test against the full-search statistic is not measured. The cost is much higher and is not measured here.
- **A zero observed statistic.** If the recorded statistic (the gap, or the fresh score under `full_search`) is zero or below, the test is skipped and `result.flags` contains `observed_statistic_zero`.
- **Sweeps.** Keep the test off while you sweep seeds or tables, and run it once on the result you report ({doc}`run-on-a-cluster`).

## Related pages

- {doc}`../metrics/calibration-p`
- {doc}`../reference/flags` (`null_not_run`, `no_structure_supported`, `observed_statistic_zero`)
- {doc}`../limits`
- {doc}`save-and-load`
