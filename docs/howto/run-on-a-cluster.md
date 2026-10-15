# Run a seed sweep on a SLURM cluster

This guide runs Starfold as a job array with one seed per task, then runs the calibration test once, as a second job, on the result you report.

```{eval-rst}
.. index::
   single: SLURM
   single: cluster
   single: job array
   pair: two-stage; calibration
   single: seed sweep
```

## Prerequisites

- Access to a cluster that uses SLURM, the job scheduler of most university clusters, with GPU nodes if you want cuML ({doc}`use-gpu`).
- A clone of the repository or an installed `starfold` on the cluster, and your table on a path the compute nodes can read.
- A single Python module that holds your `Config`, so both stages import the same one.

## Why two stages

The calibration test multiplies the cost of a run by up to 19 copies. During a sweep over seeds, column sets or tables, keep it off and save every result. Afterwards run `null_test` once, on the result you report. The second job must use an equal `Config` (including any custom `Space`) and the same `random_state` and `engine`. Above 20,000 rows per resample it also needs the same `n_jobs`. Then its replay matches the search that produced the result ({doc}`run-calibration-test`).

The SLURM directives below are a template. Adapt them to your cluster, because they were not run for this manual.

## Steps

1. Put the `Config` in one module, for example `sweep_config.py`.

   ```python
   import starfold

   CONFIG = starfold.Config()  # add your Space, scaling and budgets here
   ```

2. Write the first-stage script `run.py`. It fits one seed and saves the result.

   ```python
   import argparse

   import numpy as np

   import starfold
   from sweep_config import CONFIG

   parser = argparse.ArgumentParser()
   parser.add_argument("--seed", type=int, required=True)
   parser.add_argument("--out", required=True)
   args = parser.parse_args()

   X = np.load("my_table.npy")  # your table
   result = starfold.recover(X, config=CONFIG, random_state=args.seed, engine="cuml")
   starfold.save(result, args.out)
   ```

3. Write the job script `sweep.sh`. SLURM starts one task per array index, and the index becomes the seed.

   ```bash
   #!/bin/bash
   #SBATCH --job-name=starfold
   #SBATCH --gres=gpu:1
   #SBATCH --cpus-per-task=4
   #SBATCH --time=HH:MM:SS           # set for your table and cluster
   #SBATCH --array=0-7               # one task per seed (or per table)
   export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"   # node-local numba cache
   uv run python run.py --seed "$SLURM_ARRAY_TASK_ID" --out "runs/seed_${SLURM_ARRAY_TASK_ID}"
   ```

   The `NUMBA_CACHE_DIR` line avoids numba cache errors when the home directory is read-only or shared. On cuML `n_jobs` stays at 1 ({doc}`run-parallel`), so `--cpus-per-task` does not speed up the fits.

4. Submit the array.

   ```console
   sbatch sweep.sh
   ```

   SLURM prints the job id of the array.

5. Choose the result you report. Read its flags and recurrence with {doc}`save-and-load`, and compare seeds to see how much the groups depend on the draw.

6. Write the second-stage script `calibrate.py`. It loads the chosen result and runs the test with the recorded seed and engine.

   ```python
   import dataclasses
   import sys

   import numpy as np

   import starfold
   from starfold.result import derive_flags
   from sweep_config import CONFIG

   directory = sys.argv[1]
   result = starfold.load(directory)
   prov = result.provenance
   X = np.load("my_table.npy")
   null = starfold.null_test(
       X[result.kept_rows],
       result.search,
       config=CONFIG,
       families=("gaussian_copula",),
       random_state=prov["random_state"],
       engine=prov["engine_resolved"],
       n_jobs=prov["n_jobs"],
   )
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
   starfold.save(calibrated, directory + "_null")
   print(null.p_value, null.structure_supported)
   ```

7. Submit it as a job that waits for the array to finish.

   ```console
   sbatch --dependency=afterok:<array job id> calibrate.sh
   ```

   Here `calibrate.sh` has the same `#SBATCH` header as `sweep.sh`, without `--array`, and runs `uv run python calibrate.py runs/seed_0` (name the directory of your chosen seed).

## Expected result

Stage one leaves one saved directory per seed, `runs/seed_0` to `runs/seed_7`. Stage two leaves `runs/seed_0_null` with the calibration test attached and the flags re-derived. No cluster run is recorded for this manual, so no timings are shown. The repository's own recorded 19-copy runs on Digits and APOGEE cost 1,938 fits each ({doc}`run-calibration-test`).

## Variations

- **The repository's tables.** For the benchmark tables, the repository has a script for each stage. `python -m bench.beats_default <table> --engine cuml --random-state <seed> --results-dir output/bench` runs and saves a result. `python -m bench.null_stage <table> output/bench/<table> --output output/bench --update-result` runs the test on that result and rewrites it with the flags re-derived. The second script rebuilds the `Config` from the saved provenance and uses the default `Space`, so it matches only default-space runs. Without `--output`, the script writes `bench/results/<table>_null.json`, which holds the repository's recorded record and is overwritten.
- **A sweep over tables.** Use one array index per table instead of one per seed.

## Related pages

- {doc}`run-calibration-test`
- {doc}`set-budgets`
- {doc}`reproduce-a-run`
- {doc}`../reference/artifacts`
