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
| `mcs_range` | `(5, 500)` | Paper silent | Log-uniform integer. 5 is a conservative floor; 500 matches the paper's typical cluster sizes on ~10⁵ stars. |
| `ms_range` | `(1, 50)` | Paper silent | Log-uniform integer. 1 corresponds to the most aggressive mode; 50 prevents oversmoothing on small samples. |
| `n_trials` | 100 | Paper silent | 100 TPE trials saturate improvements on every fixture we tested. |
| Auto-cap | `mcs_high = max(5, n // 10)` | Not in paper | Prevents the search from sampling MCS values that forbid any cluster at all on small samples. Documented in `run_hdbscan` / `search_hdbscan` docstrings. |
| Sampler | `TPESampler(seed=random_state)` | §3.3 | Reproducible under a fixed seed. |
| Pruner | none | Paper silent | Pruning would distort the objective (sum of persistences) since early trials cannot be compared against later ones. |
| Objective | `"persistence_sum"` (default) | §3.3 | Sum of `cluster_persistence_`, explicitly the sum not the max. Matches the paper. |
| Alternative objective | `"combined_geom"` | Paper silent | Geometric mean of `max(relative_validity_, 0)` (the HDBSCAN MST-based DBCV proxy) and the *median* per-cluster persistence: $\sqrt{\max(\mathrm{DBCV}, 0)\cdot\mathrm{median}(\text{persistence})}$. Rewards configurations that produce both internally well-separated clusters (DBCV) and consistently high per-cluster stability (median persistence), which penalises solutions that over-split into a few spurious high-persistence clusters plus a long tail of low-persistence ones. Exposed via `hdbscan_objective="combined_geom"` on `UnsupervisedPipeline` and `search_hdbscan`. |
| HDBSCAN backend during search | CPU `hdbscan` library | Not in paper | The Optuna search and the final refit always run on the CPU `hdbscan` library even when `engine="cuml"` is selected. Reasons: (1) `cuml.HDBSCAN` does not expose `relative_validity_`, which is required both for the diagnostic dashboards and for the `combined_geom` objective; (2) for a 2-D UMAP embedding of typical size (10⁴–10⁵) CPU HDBSCAN is competitive with cuml once GPU setup overhead is amortised over ~100 trials. UMAP still runs on the GPU when `cuml` is available. |

## Noise baseline

| Parameter | Default | Source | Notes |
|---|---|---|---|
| `n_realisations` | 1 000 | §3.3 | Matches the paper. |
| `per_realisation_trials` | 20 | Paper silent | 20 TPE trials are sufficient per realisation because we only need the per-realisation optimum, not a global one. A full 100-trial search per realisation would be 5× more expensive for no meaningful gain. |
| `percentile` | 99.7 | §3.3 | The paper's 3σ gate. |
| Caching | `platformdirs.user_cache_dir("starfold")` | Not in paper | The computation is expensive; re-running is rare, so on-disk caching keyed by a SHA-256 of the inputs is a natural fit. |

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
