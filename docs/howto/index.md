# How-to guides

Each guide finishes one task. Pick the task, follow the numbered steps, and check the expected result. If you have not run Starfold yet, start with {doc}`../start/first-run`. If you want to know why a step works, follow the links to {doc}`../concepts/index` and {doc}`../metrics/index`.

```{eval-rst}
.. index::
   single: how-to guides
```

## Prepare

- {doc}`prepare-table`: Prepare your table for `recover` (shape, finite values, missing measurements, column choice, units).
- {doc}`choose-scaling`: Choose between standard and robust scaling.
- {doc}`custom-space`: Narrow the search space, and see why the library defaults are always trial 0.
- {doc}`measurement-errors`: Pass measurement errors so every refit redraws your values.

## Run

- {doc}`set-budgets`: Set the search and refit budgets, and run a laptop-sized fit.
- {doc}`use-gpu`: Run on a GPU with cuML, or fall back to the CPU.
- {doc}`run-parallel`: Run fits in parallel with `n_jobs`.
- {doc}`reproduce-a-run`: Reproduce a run from its seed and its record.

## Tune

- {doc}`custom-space`: Narrow the ranges Optuna searches.
- {doc}`set-budgets`: Trade run time against how well the search explores.

## Check

- {doc}`run-calibration-test`: Run the calibration test and read its p-value.
- {doc}`compare-backends`: Compare the CPU and cuML partitions.
- {doc}`check-against-known-labels`: Check a result against known labels after the fact.
- {doc}`plot-your-result`: Redraw the plots of the digits notebook for your own result.
- {doc}`report-your-result`: Choose what you may write in a paper about a result, and what you may not.

## Save

- {doc}`save-and-load`: Save a result to disk and load it back with checksums checked.

## Scale up

- {doc}`run-parallel`: Use several CPU processes.
- {doc}`use-gpu`: Use one GPU.
- {doc}`run-on-a-cluster`: Run a seed sweep on a SLURM cluster, then calibrate the result you report.

```{toctree}
:hidden:
:maxdepth: 1

prepare-table
choose-scaling
custom-space
measurement-errors
set-budgets
use-gpu
run-parallel
reproduce-a-run
run-calibration-test
save-and-load
compare-backends
check-against-known-labels
run-on-a-cluster
plot-your-result
report-your-result
```
