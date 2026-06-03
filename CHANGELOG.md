# Changelog

All notable changes to `starfold` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive front-door tutorial `docs/tutorial_00_walkthrough.ipynb`
  that walks a first-time user through every public function with
  visual demonstrations. The existing four tutorials remain as
  focused deep-dives.
- `compute_subsample_stability` exported from the top-level
  namespace, bringing the subsample-stability diagnostic to the same
  public-surface tier as credibility, uncertainty, silhouette, and
  merge suggestions.
- `docs/methodology.md` extensions narrative (§§6-9) covering the
  global credibility test, input-uncertainty handling, hierarchical
  refinement, and robustness diagnostics. The out-of-scope list
  moves to §10.
- `docs/design_decisions.md` blocks for "Robustness diagnostics"
  (chunked-silhouette chunk size, subsample-stability resamples and
  matching) and "Plotting defaults" (figure size, colour maps,
  outlier colour, dashboard layout).
- Jupytext `.py:percent` pairs for the four tutorial notebooks under
  `docs/`, plus a `jupytext --sync` pre-commit hook that keeps
  `.ipynb` and `.py` in lockstep. The `nbstripout` hook is now scoped
  to `legacy/` so rendered tutorial outputs survive for GitHub's
  notebook viewer.
- Science extensions on top of the paper's methodology: input-uncertainty
  propagation (`propagate_uncertainty`, `UnsupervisedPipeline.fit_with_uncertainty`),
  statistical credibility test (`compute_credibility`), cluster-merge
  recommendations (`suggest_merges`), chunked silhouette
  (`chunked_silhouette`), and subsample stability
  (`compute_subsample_stability`).
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
- Top-level public API now exports 30 symbols (was 29 in the previous
  pre-release; `compute_subsample_stability` is the newcomer). Result
  dataclasses (`HDBSCANResult`, `NoiseBaselineResult`,
  `CredibilityReport`, ...) and dashboard-panel plot primitives
  (`plot_credibility`, `plot_persistence_vs_baseline`, ...) remain
  importable from their submodules
  (`from starfold.clustering import HDBSCANResult`,
  `from starfold.plotting import plot_credibility`).
- Diagnostics helpers moved from the private `starfold._diagnostics`
  to the public `starfold.diagnostics` module. `validate_input_matrix`,
  `auto_mcs_upper`, and `recommend_budget` are still exported at the
  top level.
- `_relative_validity` (`src/starfold/clustering.py`) now catches
  `ImportError` in addition to `AttributeError` / `ValueError` /
  `ZeroDivisionError`. `hdbscan.HDBSCAN.relative_validity_` requires
  pandas internally; on a minimal install the diagnostic degrades to
  NaN instead of crashing the Optuna trial.
- `empirical_upper_tail_pvalue` (`src/starfold/credibility.py`) now
  surfaces a NaN observed value as a NaN p-value. NumPy's
  ``NaN >= x`` is always False, so a divergent Optuna best could have
  silently produced a misleadingly favourable credibility verdict.
- `plot_condensed_tree` now labels the x-axis "cluster (dendrogram
  layout)"; the y-axis label (lambda) is set by hdbscan itself.
- README install instructions now point at the from-GitHub recipe
  (pip / uv) instead of the placeholder `pip install starfold`, since
  the package is not yet on PyPI.

### Fixed
- `pandas` declared as a dev dep so CI can compute hdbscan's
  `relative_validity_` (which requires `to_pandas` internally) without
  the 55 setup failures observed on the first post-release CI run.
- `compute_noise_baseline(random_state=None)` now uses an
  OS-random seed so consecutive calls produce independent
  realisations (previously `base_seed=0` made `None`-seeded
  calls bit-identical, violating the docstring contract).
- `PipelineResult.plot_quality_dashboard(X)` no longer crashes
  on small samples. The hardcoded
  `k_values=(5, 10, 15, 30, 50, 100)` is now filtered to
  `k <= n_samples // 2 - 1`; any dropped value is reported via
  `UserWarning`.
- `UnsupervisedPipeline.fit` now warns when trustworthiness
  is computed at a clamped `k < n_neighbors` (instead of
  silently using the clamped `k` without notice).
- `compute_noise_baseline` warns when every realisation
  produces zero clusters (degenerate baseline; the threshold
  is `0.0` and every real cluster passes vacuously). The
  warning explains the likely cause and remediation.
- `compute_credibility` now warns and forces `passes=False`
  when any observed scalar is non-finite (e.g. Optuna returns
  `-inf` because no trial completed); previously the verdict
  would default to `False` for the wrong reason, with no
  indication that the run could not be evaluated.
- `run_umap` validates `n_components >= 1` and
  `n_components <= n_features` before dispatching to
  `umap-learn`; previously a degenerate request would produce
  meaningless output without error.

### Changed
- Embedding-plot axis labels are now "embedding 1" /
  "embedding 2" consistently across `plot_embedding`,
  `plot_uncertainty_map`, and `plot_membership_confidence`
  (previously mixed "component 1/2" and "UMAP 1/2"). The
  generic label is accurate regardless of which reducer
  produced the embedding.
- `UncertaintyPropagation` docstring now opens with an
  explicit "Statistical interpretation" section framing the
  membership matrix as an empirical Monte Carlo stability
  score (not a Bayesian posterior probability), naming the
  binomial MC standard error, and distinguishing
  "consistently-outlier" from "boundary-ambiguous" samples.
  The outlier-column semantics are now spelled out for
  downstream filtering.
- `compute_credibility` docstring now includes a `Notes`
  section explaining why no Bonferroni/Sidak multiplicity
  correction is applied to the three-way conjunctive
  screen, and documenting the NaN-handling guarantee.

### Removed
- Nothing yet -- this is the pre-release milestone.

## [0.0.1]
Initial scaffold (pre-release). Package name, repo layout, CI, and
public API contract.
