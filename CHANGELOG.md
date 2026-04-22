# Changelog

All notable changes to `starfold` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Science extensions on top of the paper's methodology: input-uncertainty
  propagation (`propagate_uncertainty`, `UnsupervisedPipeline.fit_with_uncertainty`),
  statistical credibility test (`compute_credibility`), cluster-merge
  recommendations (`suggest_merges`), and chunked silhouette
  (`chunked_silhouette`).
- Optional GPU engine: `engine="cuml"` / `"auto"` on
  `UnsupervisedPipeline`, `run_umap`, `run_hdbscan`, and the noise
  baseline. Falls back to CPU automatically when `cuml` is not
  importable.
- Data-size-aware defaults: `auto_mcs_upper` picks a
  `min_cluster_size` search range proportional to `n_samples`, and
  `recommend_budget` returns starfold-recommended trial/realisation
  budgets.
- Input validation (`validate_input_matrix`) and fit diagnostics
  (`diagnose_fit`) surfaced as `PipelineResult.flags`.
- On-disk noise-baseline cache keyed on `(n_samples, n_features,
  umap_kwargs, random_state, n_realisations, per_realisation_trials)`
  under `platformdirs.user_cache_dir("starfold")`.
- `noise_baseline_kwargs={"umap_kwargs": {...}}` escape hatch:
  override the noise-fit UMAP config independently of the main
  pipeline's `umap_kwargs`. Useful for running the baseline at low
  `n_epochs` (structureless noise has no manifold to converge to)
  while the main fit still uses the paper-default 10 000.
- Second Optuna objective `combined_geom` (geometric mean of DBCV and
  persistence sum); `persistence_sum` remains the default.
- Four-notebook tutorial arc under `docs/`:
  `tutorial_01_quickstart` (minimal end-to-end fit on a synthetic
  Hopf torus chain), `tutorial_02_validation` (noise baseline,
  credibility test, tuning and quality dashboards),
  `tutorial_03_advanced` (silhouette, merge recommender,
  subcluster refit, uncertainty propagation and uncertainty-aware
  fitting), and `tutorial_04_astronomy` (case study on a bundled
  9 242-star APOGEE DR19 x Gaia DR3 sample, notebook-only, no
  `astropy` dependency).
- Diagnostic plot family (`plot_optuna_history`,
  `plot_optuna_param_importance`, `plot_condensed_tree`,
  `plot_uncertainty_map`, ...) plus composable tuning / quality
  dashboards on `PipelineResult`.

### Changed
- Top-level public API trimmed from 53 symbols to 29. Result
  dataclasses (`HDBSCANResult`, `NoiseBaselineResult`,
  `CredibilityReport`, ...) and dashboard-panel plot primitives
  (`plot_credibility`, `plot_persistence_vs_baseline`, ...) are no
  longer re-exported at the top level -- import them from their
  submodule (`from starfold.clustering import HDBSCANResult`,
  `from starfold.plotting import plot_credibility`).
- Diagnostics helpers moved from the private `starfold._diagnostics`
  to the public `starfold.diagnostics` module. `validate_input_matrix`,
  `auto_mcs_upper`, and `recommend_budget` are still exported at the
  top level.

### Removed
- Nothing yet -- this is the pre-release milestone.

## [0.0.1]
Initial scaffold (pre-release). Package name, repo layout, CI, and
public API contract.
