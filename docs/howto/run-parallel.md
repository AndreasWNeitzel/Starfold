# Run fits in parallel with n_jobs

This guide shows how to spread Starfold's fits over several CPU processes with `n_jobs`, and where the option does nothing.

```{eval-rst}
.. index::
   single: n_jobs
   single: parallel
   single: worker processes
   pair: cuML; n_jobs
   single: seed
```

## Prerequisites

- A CPU run (`engine="cpu"`, or `"auto"` on a machine without a usable GPU).
- A machine with more cores than you use now. `n_jobs` counts worker processes, so it should not exceed your physical cores.

## Steps

1. Set `n_jobs` in the call.

   ```python
   import starfold

   result = starfold.recover(X, engine="cpu", n_jobs=3, random_state=0)
   ```

2. Read what it used.

   ```python
   prov = result.provenance
   print(prov["n_jobs"], prov["n_jobs_forced_to_one_on_cuml"])
   ```

## What n_jobs spreads

| Stage | Parallel over | Limit |
|---|---|---|
| search | the fits of one trial | a complete trial has `n_resamples` fits, 3 by default, so more than 3 workers do not speed up a trial |
| refits | the `n_worlds` refits ({term}`refit`) | `n_worlds` |
| calibration test | null copies | only up to 20,000 rows per resample |

Each worker is a separate single-threaded process, so throughput comes from processes and not from threads. Memory grows with the number of workers. The peak memory of multi-process runs is not measured here, and multi-process CPU runs at scale are not validated.

## Expected result

The output equals that of the serial run for the refits, and for the calibration copies up to 20,000 rows per resample. The docstrings state both, and the test suite checks the calibration case. Above 20,000 rows per resample, the neighbour graph comes from an approximate search whose result depends on the thread count, so a run reproduces only with the same `n_jobs`. Give the same `n_jobs` to {doc}`run-calibration-test` in that case.

No recorded run uses `n_jobs` above 1, so the speed-up is not measured here.

## On cuML

On cuML, `n_jobs` above 1 is forced to 1, and `result.flags` contains `n_jobs_forced_to_one_on_cuml`. The GPU runs one fit at a time. To use several CPU cores, pass `engine="cpu"`.

## Variations

- A `measurement_sampler` must be picklable when `n_jobs` is above 1 ({doc}`measurement-errors`).
- `n_jobs` below 1 raises a `ValueError`.
- To run many seeds or tables at once, use a job array instead: {doc}`run-on-a-cluster`.

## Related pages

- {doc}`use-gpu`
- {doc}`reproduce-a-run`
- {doc}`../reference/flags` (`n_jobs_forced_to_one_on_cuml`)
