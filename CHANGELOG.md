# Changelog

## 0.2.0 (2026-10-19)

New code base and API. Code written for 0.0.1 needs porting.

- `starfold.recover(X)` replaces `UnsupervisedPipeline().fit(X)`. It searches UMAP and HDBSCAN settings with Optuna, fits the selected setting on all rows and refits it on 32 random subsets. The `Result` holds labels, per-row support, group recurrence, a group-count interval, a consensus check, flags and provenance.
- Settings are scored without labels: held-out prediction agreement, coverage and input-space modularity, each relative to a structureless copy of the table. The group count is never scored.
- Trial 0 is always the library defaults. 16 Sobol points (an even spread over the settings) explore, then Optuna's TPE sampler converges.
- The calibration test is opt-in: `starfold.null_test(...)`, or `Config(null_families=("gaussian_copula",))`. It costs up to 1,938 fits at the default settings, about six times the rest of a run.
- On cuML every fit, neighbour search, label transfer, modularity and co-association runs on the GPU. `engine="auto"` warns when it falls back to the CPU; `engine="cuml"` raises instead.
- `X_err` or `measurement_sampler` carries per-value uncertainties into every refit.
- `save` and `load` write checksummed JSON and NumPy arrays and re-derive the flags on load. Results saved by 0.0.1 do not load.
- Removed: `UnsupervisedPipeline`, `PipelineResult`, and every 0.0.1 helper, plotting function and dashboard.
- `starfold.search` stays the search function after other public names load (it used to become the submodule).
- New flag `search_trial_failures`: at least one search trial raised an error. When every trial fails, the selected setting is the failed trial 0, and the flag says so.
- Provenance records the measurement model (`measurement`: none, an `X_err` fingerprint, or the sampler name) and the pynndescent version.
- `load` refuses a manifest that is unreadable or lacks a checksum, and `save` writes each array with `allow_pickle=False` on every supported NumPy.
- `measurement_draw` and `run_worlds` refuse negative uncertainties. `search_stopped_early` now means fewer than `n_trials` trials ran.
- `python -m bench.null_stage` replays the saved search space, engine and worker count.
- The recurrence marks 0.85 and 0.5 are now credited to the fpc `clusterboot` documentation, not Hennig (2007), and the manual describes the Gaussian copula family and sequential p-values correctly.
- The three notebooks are committed as `.ipynb` files with their recorded outputs, so they read on GitHub without running anything. The paired `.py` versions are gone.
- The manual is a Sphinx site built from `docs/` (install with `--extra docs`): tutorials, one page per metric, how-to guides, a flag reference, a glossary and an index. It replaces `docs/handbook.md`.
- The APOGEE DR19 example table is now `bench/data/stellar_chemokinematics_apogee_dr19.npz`, read with `bench.datasets.example_table()` from a clone.

## 0.0.1

First public version: `UnsupervisedPipeline` with a UMAP embedding, Optuna-tuned HDBSCAN and a noise baseline.
