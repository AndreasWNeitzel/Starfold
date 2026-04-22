# Design decisions

Every place where the paper is silent or ambiguous, and `starfold`
picks a default, is recorded here with the rationale. Users who want
different behaviour override the argument at call time.

## UMAP

| Parameter | Default | Source | Notes |
|---|---|---|---|
| `n_neighbors` | 15 | Neitzel et al., 2025, §3.1 | Balances local vs global structure. |
| `min_dist` | 0.0 | §3.1 | Packs clusters tightly so HDBSCAN can separate them. |
| `n_epochs` | 10 000 | §3.1 | Paper notes convergence by ~1 000 epochs but uses 10 000 for safety. Kept as default. |
| `metric` | `"euclidean"` | §3.1 | Paper uses standardised features, so Euclidean is appropriate. |
| `n_components` | 2 | §3.1 | The paper's clustering is done in the 2-D embedding. |

## HDBSCAN / Optuna

| Parameter | Default | Source | Notes |
|---|---|---|---|
| `mcs_range` | `None` → `(5, auto_mcs_upper(n))` | Paper silent | Log-uniform integer. The paper's fixed `(5, 500)` was tuned for ~10⁵-star samples. `None` (the default on `UnsupervisedPipeline`) instead calls `starfold.auto_mcs_upper`: `max(5, n/4)` when `n < 50`, otherwise `clip(n/20, 50, 5000)`. So a 1k run searches up to ~50, a 50k run up to 2500, and a 1M run up to 5000, keeping the upper bound near realistic cluster scales without blowing up per-trial HDBSCAN cost. An explicit tuple is always honoured. |
| `ms_range` | `(1, 50)` | Paper silent | Log-uniform integer. 1 corresponds to the most aggressive mode; 50 prevents oversmoothing on small samples. |
| `n_trials` | 100 | Paper silent | 100 TPE trials saturate improvements on every tested fixture. |
| Auto-cap | `mcs_high = max(5, n // 10)` | Not in paper | Prevents the search from sampling MCS values that forbid any cluster at all on small samples. Documented in `run_hdbscan` / `search_hdbscan` docstrings. |
| Sampler | `TPESampler(seed=random_state)` | §3.3 | Reproducible under a fixed seed. |
| Pruner | none | Paper silent | Pruning would distort the objective (sum of persistences) since early trials cannot be compared against later ones. |
| Objective | `"persistence_sum"` (default) | §3.3 | Sum of `cluster_persistence_`, explicitly the sum not the max. Matches the paper. |
| Alternative objective | `"combined_geom"` | Paper silent | Geometric mean of `max(relative_validity_, 0)` (the HDBSCAN MST-based DBCV proxy) and the *median* per-cluster persistence: $\sqrt{\max(\mathrm{DBCV}, 0)\cdot\mathrm{median}(\text{persistence})}$. Rewards configurations that produce both internally well-separated clusters (DBCV) and consistently high per-cluster stability (median persistence), which penalises solutions that over-split into a few spurious high-persistence clusters plus a long tail of low-persistence ones. Exposed via `hdbscan_objective="combined_geom"` on `UnsupervisedPipeline` and `search_hdbscan`. |
| HDBSCAN backend during search | CPU `hdbscan` library | Not in paper | The Optuna search and the final refit always run on the CPU `hdbscan` library even when `engine="cuml"` is selected. Reasons: (1) `cuml.HDBSCAN` does not expose `relative_validity_`, which is required both for the diagnostic dashboards and for the `combined_geom` objective; (2) for a 2-D UMAP embedding of typical size (10⁴–10⁵) CPU HDBSCAN is competitive with cuml once GPU setup overhead is amortised over ~100 trials. UMAP still runs on the GPU when `cuml` is available. |

### Sizing defaults and `recommend_budget`

Neither the UMAP paper nor Neitzel+25 gives you a knob for "what
settings should I use at my sample size". starfold ships two
helpers:

* `starfold.auto_mcs_upper(n_samples)` picks the `min_cluster_size`
  upper bound automatically; it is what `UnsupervisedPipeline` calls
  when `mcs_range=None` (the default).
* `starfold.recommend_budget(n_samples)` returns a dict of
  starting-point values for `hdbscan_optuna_trials`, `n_realisations`,
  and `per_realisation_trials`. The regimes are:

  | n_samples | `hdbscan_optuna_trials` | `n_realisations` | `per_realisation_trials` |
  |---|---|---|---|
  | ≤ 1 000 | 50 | 500 | 20 |
  | 1 000 – 50 000 | 100 | 1 000 | 20 |
  | 50 000 – 500 000 | 120 | 500 | 15 |
  | > 500 000 | 150 | 200 | 10 |

  The total noise-baseline cost is proportional to
  `n_realisations × per_realisation_trials` UMAP+HDBSCAN fits, so the
  > 500 000 recommendation is ≈ 2 000 fits against ≈ 20 000 at the
  default — pair it with `engine="cuml"` or raise `n_jobs` if you can.

Both are starting points, not prescriptions. Tighter credibility
p-values need more `n_realisations`; users with a strong prior on
cluster size should narrow `mcs_range` rather than lean on the
auto-cap.

## Noise baseline

| Parameter | Default | Source | Notes |
|---|---|---|---|
| `n_realisations` | 1 000 | §3.3 | Matches the paper. |
| `per_realisation_trials` | 20 | Paper silent | 20 TPE trials are sufficient per realisation because only the per-realisation optimum is needed, not a global one. A full 100-trial search per realisation would be 5× more expensive for no meaningful gain. |
| `percentile` | 99.7 | §3.3 | The paper's 3σ gate. |
| Caching | `platformdirs.user_cache_dir("starfold")` | Not in paper | The computation is expensive; re-running is rare, so on-disk caching keyed by a SHA-256 of the inputs is a natural fit. |

## Global clustering credibility

The paper's §3.3 specifies only the **per-cluster** significance
threshold (99.7th-percentile persistence on noise). It does not
specify an **omnibus** test that answers "is this clustering run, as
a whole, distinguishable from what the same pipeline would do on
structureless noise of the same shape?". This matters because
HDBSCAN on a reasonable hyperparameter grid will almost always
return at least two clusters, even on a Gaussian point cloud.

`starfold` extends the paper's null with a global credibility test
in `starfold.credibility`. No new Monte Carlo: on every noise
realisation already generated for the per-cluster baseline the
credibility machinery also records the number of clusters in the best
Optuna trial and that trial's scalar objective value. For a real-data
run it compares three scalars — `n_clusters`, best Optuna objective, and largest
per-cluster persistence — against those null distributions, each
with a one-sided upper-tail p-value using the `(r+1)/(n+1)` Phipson
& Smyth (2010) correction so p-values are never exactly zero under
finite Monte Carlo. A run "passes" when all three p-values are
below `alpha`, defaulting to 0.003 to match the paper's 3σ
per-cluster gate. The machinery is exposed as
`starfold.compute_credibility` / `CredibilityReport` and is called
automatically by `UnsupervisedPipeline.fit` whenever the noise
baseline runs; the verdict appears on `PipelineResult.summary()`
and can be inspected directly on `result.credibility`.

### Per-cluster credibility p-value

The paper's per-cluster gate (`persistence > 99.7th-percentile`) is a
binary rule derived from the distribution of *per-realisation maxima*.
That is a conservative test: only the strongest noise cluster from
each realisation contributes to the reference distribution, so a
real-data cluster that beats every noise realisation's *maximum*
passes, and one that is beaten by most realisations fails — with
nothing in between. `starfold` also exposes a continuous, per-cluster
p-value built from the *pool* of every cluster's persistence across
every noise realisation. The pool is stored alongside the
per-realisation maxima in `NoiseBaselineResult.null_cluster_persistence`
(with matching `null_cluster_size` and `null_cluster_realisation`
arrays for downstream diagnostics). For each real-data cluster the
upper-tail `(r+1)/(n+1)` p-value against that pool lives in
`CredibilityReport.per_cluster_pvalue` and the pass/fail mask at the
same `alpha` in `CredibilityReport.per_cluster_significant`. The
figure `plot_per_cluster_credibility` draws one bar per cluster
(green/red by significance) against the pool's 50th / 99.7th / 99.97th
percentiles for visual anchoring.

| Parameter | Default | Source | Notes |
|---|---|---|---|
| Scalars compared (global) | `n_clusters`, best Optuna objective, max cluster persistence | Not in paper | The triple is chosen to be invariant of the exact objective (sum or combined_geom) the user picked, so the same null covers both. |
| Per-cluster null | Pool of every noise cluster's persistence | Extension of paper's null | The paper only uses each realisation's *maximum* persistence. Pooling all clusters gives a larger, continuous reference distribution, which produces a useful p-value per real cluster rather than a single binary flag. |
| `alpha` | 0.003 | Not in paper; matches 99.7th-percentile per-cluster gate | 3σ keeps the per-cluster, global, and paper tests on the same footing. |
| Empirical estimator | `(r+1)/(n+1)` (Phipson & Smyth, 2010) | Not in paper | Standard correction that keeps p-values strictly positive under finite Monte Carlo. |

### Input-uncertainty propagation

Real data comes with per-feature 1-sigma error bars; hard labels hide
the fact that some samples sit on a cluster boundary and would flip
under modest input noise. `PipelineResult.propagate_uncertainty(X,
sigma, n_draws=100)` Monte Carlos over that input noise and returns a
per-sample membership matrix of shape `(n_samples, n_clusters + 1)`
(the last column is the outlier fraction), a consensus label per
sample, and an instability score `1 - max(membership, axis=1)`. Each
draw perturbs raw `X` by independent Gaussians, applies the fitted
`StandardScaler`, pushes the result through the fitted
`umap.UMAP.transform`, and classifies with
`hdbscan.approximate_predict` — no refitting. `sigma` accepts a
scalar, a length-`n_features` 1-D array, or a `(n_samples,
n_features)` 2-D array when heteroscedastic. `plot_uncertainty_map`
colours the 2-D embedding by instability for a visual "which samples
are on the fence?" figure.

| Parameter | Default | Source | Notes |
|---|---|---|---|
| Default `n_draws` | 100 | Not in paper | Enough Bernoulli resolution for "near boundary?" questions; raise for tighter membership-fraction CIs. |
| Perturb before or after scaling? | Before | Not in paper | Keeps `sigma` in the original feature units the user provided, matching observational practice (e.g. APOGEE's [Fe/H] error bars are reported in dex, not in standardised units). |
| Refit per draw? | No | Not in paper | The question being asked is conditional on the fitted model ("given this clustering, how confident is this sample?"). Refitting would mix in clustering stochasticity that subsample-stability already covers. |
| GPU support | Not in this release | — | `cuml` wraps `transform` differently and lacks an ergonomic `approximate_predict`. The call path raises `NotImplementedError` on a cuml UMAP so users fall back to `engine="cpu"`. |
| `prediction_data=True` | On final HDBSCAN refit only | — | Adds a modest allocation per fit; carrying it on every Optuna trial is wasteful because trials are throwaway. |
| Sigma shape | scalar / per-feature / per-sample-per-feature | — | Covers the three realistic cases: a single global error, survey-level per-feature floors, and per-observation error bars. |

### The three uncertainty stories starfold tells

starfold distinguishes three independent questions that all get
called "uncertainty" in casual usage. Mixing them leads to confused
conclusions; the package keeps them deliberately separate:

| Question | API | What it varies | What stays fixed |
|---|---|---|---|
| *Is this clustering better than noise?* (global **credibility**) | `result.credibility`, `compute_noise_baseline` | The entire input: compared against runs on Gaussian point clouds of the same shape. | The pipeline (UMAP, HDBSCAN, Optuna, metric). |
| *Would the same clusters come back from a random 80% subsample?* (**stability**) | `compute_subsample_stability`, `result.plot_quality_dashboard` | Which *rows* are sampled (seeded Bernoulli resamples). | The input values themselves and the pipeline. |
| *Would a sample flip cluster if re-observed within its error bars?* (**input-uncertainty propagation**) | `result.propagate_uncertainty`, `plot_uncertainty_map`, `UncertaintyPropagation.confident_labels` | The input values (per-feature Gaussian perturbations at the user's `sigma`). | The pipeline and the set of rows. |

Concretely, a user with real error bars typically wants all three
together: credibility rules out a noise-consistent fit, stability
rules out a clustering that falls apart under resampling, and
propagation rules out hard labels on samples that sit on a cluster
boundary. The recipe is `result.credibility.passes`, then
`compute_subsample_stability(...)`, then
`prop = result.propagate_uncertainty(X, sigma); trusted =
prop.confident_labels(0.8)`.

## Pipeline scope

| Decision | Choice | Rationale |
|---|---|---|
| Two-run workflow | Not baked in | The paper reruns the pipeline on each major component separately. This encodes the astronomer's prior that the top-level structure is "disk vs halo"; a different domain has different top-level priors. Users who want the second run filter `result.labels` to a subcluster and call `fit` again. |
| Preprocessing | `StandardScaler` inside `UnsupervisedPipeline` | The paper scales features before UMAP. Done once, inside the pipeline, so callers do not double-scale. |
| `k` for trustworthiness | `n_neighbors` from UMAP | The paper reports $T(k)$ at $k = n_\text{neighbors}$. Capped to `max(1, (n_samples - 1) // 2)` so small samples do not divide by zero. |

## Python / tooling

| Decision | Choice | Rationale |
|---|---|---|
| Python versions | 3.11, 3.12 | Matches the active scientific-Python ecosystem. |
| `astropy` dependency | None | The package is domain-agnostic; astropy would tie it to astronomy. |
| `pandas` dependency | None | NumPy arrays are the API surface. |
| Types | `mypy --strict` on `src/` | Catches obvious mistakes without forcing the test suite to satisfy strict typing. |
| Lint | `ruff check --select=ALL` with a curated ignore list in `pyproject.toml` | Keeps the surface tight; ignores are justified in-place. |
