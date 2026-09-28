# Changelog

## 0.2.0 (unreleased)

New code base and API. Code written for 0.0.1 needs porting.

- `starfold.recover(X)` replaces `UnsupervisedPipeline().fit(X)`. It searches UMAP and HDBSCAN settings with Optuna, fits the selected setting on all rows and refits it on 32 random subsets. The `Result` holds labels, per-row support, group recurrence, a group-count interval, a consensus check, flags and provenance.
- Settings are scored without labels: held-out prediction agreement, coverage and input-space modularity, each relative to a structureless copy of the table. The group count is never scored.
- Trial 0 is always the library defaults. 16 Sobol points explore, then TPE converges.
- The calibration test is opt-in: `starfold.null_test(...)`, or `Config(null_families=("gaussian_copula",))`. It costs up to 1,938 fits at the default settings, about six times the rest of a run.
- On cuML every fit, neighbour search, label transfer, modularity and co-association runs on the GPU. `engine="auto"` warns when it falls back to the CPU; `engine="cuml"` raises instead.
- `X_err` or `measurement_sampler` carry per-value uncertainties into every refit.
- `save` and `load` write checksummed JSON and NumPy arrays and re-derive the flags on load. Results saved by 0.0.1 do not load.
- Removed: `UnsupervisedPipeline`, `PipelineResult`, and every 0.0.1 helper, plotting function and dashboard.
- The APOGEE DR19 example table is now `bench/data/stellar_chemokinematics_apogee_dr19.npz`, read with `bench.datasets.example_table()` from a clone.

## 0.0.1

First public version: `UnsupervisedPipeline` with a UMAP embedding, Optuna-tuned HDBSCAN and a noise baseline.
