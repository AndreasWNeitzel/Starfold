# Reproduce a run

This guide shows how to fix the {term}`seed`, read the record Starfold keeps of a run, and repeat the run so that you can compare the two results.

```{eval-rst}
.. index::
   single: reproducibility
   single: random_state
   single: provenance
   pair: seed; random_state
   single: input fingerprint
```

## Prerequisites

- A result from `recover`, or the directory saved with {doc}`save-and-load`.
- The same table, as the same array of numbers.

## What fixes a run

`random_state` is the base seed of every random stage: the resamples, the structureless copies, the Sobol design, the Optuna sampler, the UMAP fits and the refits. Each fit, refit and null copy gets its own seed, derived from `random_state` and its place in the procedure, never from the worker process that runs it. The default is `random_state=0`.

`result.provenance` records what else a repeat needs:

| Key | What it holds |
|---|---|
| `random_state` | the base seed |
| `config` | every field of the `Config`, including the `Space` |
| `engine_requested`, `engine_resolved` | what you asked for and what ran (`cpu` or `cuml`) |
| `n_jobs` | the worker count used |
| `libraries` | versions of numpy, scipy, scikit-learn, umap-learn, hdbscan, optuna, joblib and cuml |
| `starfold_version`, `python`, `platform` | package, interpreter and operating system |
| `input_fingerprint` | SHA-256 of the kept rows and their shape |
| `stage_seconds`, `stage_fits`, `peak_rss_mb` | cost of the run |

## Steps

1. Set `random_state` in the original run, and keep the saved result.

   ```python
   import starfold

   result = starfold.recover(X, random_state=0)
   starfold.save(result, "runs/digits")
   ```

2. Later, load the record and read it.

   ```python
   result = starfold.load("runs/digits")
   prov = result.provenance
   print(prov["random_state"], prov["engine_resolved"], prov["libraries"]["cuml"])
   ```

3. Check that your table is the same one.

   ```python
   from starfold.result import matrix_fingerprint

   same = matrix_fingerprint(X[result.kept_rows]) == prov["input_fingerprint"]
   print(same)
   ```

   If this prints `False`, the values or their order differ, and no seed can bring the result back.

4. Rebuild the `Config` from the record.

   ```python
   stored = dict(prov["config"])
   space = stored.pop("space")
   config = starfold.Config(
       space=starfold.Space(**{k: None if v is None else tuple(v) for k, v in space.items()}),
       **{k: tuple(v) if isinstance(v, list) else v for k, v in stored.items()},
   )
   ```

5. Run again with the recorded seed, the engine that ran, and the recorded worker count.

   ```python
   again = starfold.recover(
       X,
       config=config,
       random_state=prov["random_state"],
       engine=prov["engine_resolved"],
       n_jobs=prov["n_jobs"],
   )
   ```

6. Compare the two results. Identical labels is the strictest check. The adjusted Rand index on the rows both results assigned ignores label numbers.

   ```python
   import numpy as np
   from sklearn.metrics import adjusted_rand_score

   print(np.array_equal(again.labels, result.labels), again.n_groups, result.n_groups)
   both = (again.labels >= 0) & (result.labels >= 0)
   print(adjusted_rand_score(result.labels[both], again.labels[both]))
   ```

## Expected result

On the same engine, library versions and table, the two runs use the same seeds. The recorded runs carry the versions they used (Digits: cuml 26.8.0, umap-learn 0.5.12, hdbscan 0.8.42, optuna 4.8.0). No recorded experiment repeats a full `recover` call and compares the labels, so whether a repeat on cuML is identical to the last digit is not measured here. The test suite checks that the refits are deterministic in `random_state` and that the calibration evidence does not depend on `n_jobs`.

## Why engines differ in the last digits

CPU and cuML add floating-point numbers in a different order, so their results can differ in the last digits. The partitions agree to a degree that {doc}`compare-backends` measures, but label numbers and group counts can differ. In the recorded comparison on Digits, the CPU fits found 14, 12 and 13 groups on three seeds and the cuML fits found 10, 10 and 10. To compare engines, use {doc}`compare-backends`. To reproduce a run, use the same engine it used. Above 20,000 rows per resample on the CPU, also use the same `n_jobs` ({doc}`run-parallel`).

## Variations

- **A different seed.** `random_state=1` gives a different set of resamples, copies and refits. Comparing several seeds shows how much your groups depend on the draw.
- **Only the record.** You do not need to run again to read the result. {doc}`save-and-load` restores it.

## Related pages

- {doc}`save-and-load`
- {doc}`../reference/artifacts` (what is recorded and saved)
- {doc}`../glossary`
