# Methodology

`starfold` implements the four-step unsupervised-classification
methodology from Neitzel, Campante, Bossini & Miglio (2025),
*A&A* **695**, A243 (arXiv:2501.16294), §3 of the paper:

1. **Standardise** the input feature matrix to zero mean, unit variance.
2. **UMAP-embed** the standardised features into a 2-D manifold.
3. **HDBSCAN-cluster** the 2-D embedding, tuning its two hyperparameters
   with an Optuna Tree-structured Parzen Estimator sampler that
   maximises the sum of cluster persistences.
4. **Validate** the output with (a) a trustworthiness score on the
   embedding and (b) a statistical significance gate derived from a
   structureless-noise null.

The paper uses this pipeline on APOGEE × astroNN stars to dissect the
chrono-chemo-kinematic structure of the Milky Way disc; `starfold`
extracts the methodology into a domain-agnostic package that takes any
numerical ``(n_samples, n_features)`` matrix.

## 1. Standardisation

Before UMAP sees the data, every feature is rescaled via
``sklearn.preprocessing.StandardScaler`` so all columns have mean 0 and
variance 1. This prevents a feature with a larger numerical range
(e.g. a radial velocity in km/s alongside a dimensionless metallicity)
from dominating the Euclidean distances UMAP relies on.

`UnsupervisedPipeline` does this internally. `run_umap` does *not*:
callers of the low-level function are responsible for scaling.

## 2. UMAP

UMAP (McInnes, Healy & Melville, 2018) constructs a fuzzy topological
representation of the high-dimensional point cloud and optimises a
low-dimensional layout that preserves it.

Two parameters matter:

- ``n_neighbors`` — size of the local neighbourhood that defines the
  fuzzy simplicial set. Larger values push the embedding toward global
  structure; smaller values highlight local structure. The paper uses
  15.
- ``min_dist`` — the smallest allowed distance between points in the
  embedding. The paper uses 0.0 to pack clusters as tightly as
  possible, which helps HDBSCAN separate them.

Two other defaults are taken from the paper: ``n_epochs=10_000`` (much
higher than the typical convergence point at ~1 000 epochs, chosen for
safety) and ``metric="euclidean"``.

## 3. Trustworthiness

A trustworthy embedding is one whose neighbourhoods in the
low-dimensional space match neighbourhoods in the high-dimensional
space. The paper uses the formulation of Venna & Kaski (2001):

$$
T(k) = 1 - \frac{2}{N\,k\,(2N - 3k - 1)}
       \sum_{i=1}^{N} \sum_{x_j \in U_k(x_i)} \bigl(r(x_i, x_j) - k\bigr)
$$

where $U_k(x_i)$ is the set of points that are among the $k$ nearest
neighbours of $x_i$ in the embedding but *not* in the input, and
$r(x_i, x_j)$ is the rank of $x_j$ among the input's non-self points
ordered by distance to $x_i$.

`starfold.trustworthiness` implements this verbatim and is cross-checked
against `sklearn.manifold.trustworthiness` at ``atol=1e-10`` for five
seeded datasets with $N \in \{100, 500, 2000, 5000\}$ and
$k \in \{5, 15, 30\}$. The paper's acceptance heuristic is
$T(k = n_\text{neighbors}) > 0.90$; this is a guideline, not a hard
gate — `starfold` reports the score without rejecting the run.

## 4. HDBSCAN and Optuna

HDBSCAN (Campello, Moulavi & Sander, 2013) builds a hierarchy of
density-based clusters and extracts a flat partition by maximising a
stability criterion. It has two free parameters:

- ``min_cluster_size`` — the smallest group the algorithm will accept
  as a cluster.
- ``min_samples`` — controls how conservative the clustering is.

`starfold` tunes both with a ``optuna.samplers.TPESampler`` over
log-uniform integer ranges. The objective is
**the sum of `cluster_persistence_`** across all non-negative labels
(Neitzel et al., 2025, §3.3). On small samples the `min_cluster_size`
upper bound is auto-capped to ``max(5, n_samples // 10)`` to keep the
search meaningful.

The paper is silent on the exact search ranges. `starfold`'s defaults
are ``mcs_range=(5, 500)`` and ``ms_range=(1, 50)``. Users with a
domain prior on cluster size should override these — see
``docs/design_decisions.md``.

## 5. Noise baseline

Cluster persistence is a relative quantity; a value of 0.6 is not
intrinsically meaningful. To decide whether a cluster is likely a real
feature rather than a bump in structureless noise, the paper compares
it to the distribution of persistences produced by running the same
pipeline on pure Gaussian noise of the same shape.

For a target ``(n_samples, n_features)``:

1. Draw ``n_realisations`` independent $\mathcal{N}(0,1)$ matrices.
2. Run UMAP with the same parameters on each.
3. Tune HDBSCAN for ``per_realisation_trials`` trials.
4. Record the *maximum* cluster persistence across the best HDBSCAN's
   output for that realisation.
5. Take the 99.7th-percentile of those maxima (the 3σ gate used in
   Neitzel et al., 2025).

A real cluster is flagged `significant` when its persistence exceeds
this threshold. The baseline is deliberately expensive; `starfold`
caches each result on a SHA-256 of the inputs under
`platformdirs.user_cache_dir("starfold")`.

---

The five sections above are the paper-§3 core. The next four sections
document the package's intentional extensions beyond paper §3. Each
extension was added because the paper's per-cluster threshold and
post-hoc workflow leave concrete questions unanswered for users who
adopt the methodology in a new domain. They share the core pipeline's
one-result-object idiom: every extension is either a method on
`PipelineResult` or a free function that operates on one.

## 6. Global credibility test

The per-cluster `significant` flag (§5) asks one question: *is this
cluster stronger than what noise typically produces?* It is silent on
a strictly weaker question that users actually ask first: *is the run
as a whole distinguishable from noise?* HDBSCAN returns at least one
cluster on almost any input including pure noise; without a global
test, a passing per-cluster flag can coexist with a run-level result
that is no stronger than the noise null.

`starfold.compute_credibility` (called automatically by
`UnsupervisedPipeline.fit` whenever the noise baseline runs) computes
an omnibus 3σ test on three per-run scalars:

- the number of clusters HDBSCAN returns,
- the best Optuna objective (sum of cluster persistences) the search reached,
- the largest single-cluster persistence in the final fit.

For each axis, the observed scalar is compared one-sided upper-tail
against its empirical null — the distribution of the same scalar
across the noise-baseline realisations — using the Phipson & Smyth
(2010) finite-population correction
$p = (r + 1) / (n + 1)$ where $r$ is the count of null samples that
meet or exceed the observed value. The run "passes" at 3σ when all
three axis p-values clear an ``alpha`` threshold (default
$0.00135 \approx 1 - 0.997$, matching paper §3.3's one-sided 3σ gate).

`starfold` *also* pools every cluster's persistence from every noise
realisation into a single global null and computes a per-cluster
upper-tail empirical p-value against it. The paper's binary threshold
becomes one specific decision rule over a continuous credibility
score; users who care about magnitude (rather than pass/fail) read
the per-cluster p-values directly.

This is paper-silent methodology. It is documented in
`docs/design_decisions.md` and exposed in code as
`compute_credibility` / `CredibilityReport`.

## 7. Input-uncertainty handling

The paper assumes a single feature value per sample. Many domains
(stellar spectroscopy, single-cell RNA-seq, sensor measurements)
carry per-feature 1σ error bars; the obvious question is
*how confident is each sample's assignment under its own error
cloud?* `starfold` answers this in two complementary modes.

**Mode A — post-hoc propagation.** `result.propagate_uncertainty(X,
sigma=...)` freezes a clean fit and Monte Carlos the input through
the trained UMAP-and-HDBSCAN pipeline (via
`hdbscan.approximate_predict`). For each Monte Carlo draw, the
perturbed sample is projected into the existing embedding and queried
against the fitted condensed tree; aggregating across draws gives a
membership-probability vector per sample and an instability scalar.
The result is "given the clustering I already trust, how robust is
each individual assignment?"

**Mode B — uncertainty-aware fit.** `pipeline.fit_with_uncertainty(X,
sigma=..., n_replicas=...)` builds an augmented matrix of the clean
samples stacked with `n_replicas` Gaussian replicas and feeds the
whole thing through the full pipeline. UMAP and HDBSCAN therefore see
the error bars as *spread* in the manifold rather than as a
postprocessing concern, and the noise baseline and credibility test
are computed against this enlarged sample. The result is "what
clustering does the data support when its uncertainty is part of the
fit?"

`sigma` accepts a scalar, a per-feature vector, or a
per-sample-per-feature matrix. Both modes are deterministic for fixed
`random_state`. Neither is in the paper.

## 8. Hierarchical refinement

The paper's two-run workflow (fit the full sample, then refit each
top-level component separately) is a *scientific* choice for the
astronomy application: it encodes the prior that top-level structure
is disk-vs-halo. As §10 below notes, the package does not bake this
in; instead it provides the primitives that make any such workflow
domain-agnostic:

- `result.refit_subcluster(X, cluster_id=...)` runs the full pipeline
  (UMAP, Optuna, noise baseline, credibility test) on the subset
  `X[result.labels == cluster_id]`, returning a new `PipelineResult`
  scoped to the refinement.
- `result.suggest_merges()` flags cluster pairs where the HDBSCAN
  condensed tree (density) *and* the 2-D embedding geometry
  (centroid gap relative to intra-cluster dispersion) both agree the
  clusters should be one. Disagreements are kept apart by design.
- `starfold.hierarchy.HierarchicalStructure` exposes the condensed
  tree as a first-class object so users can implement custom
  traversals.

The two-run paper workflow becomes one call to `refit_subcluster`
per component of interest; the merge recommender solves the converse
problem of over-splitting.

## 9. Robustness diagnostics

Two scalar diagnostics, paper-silent but cheap:

- `chunked_silhouette` computes sklearn's silhouette coefficient *and*
  the per-cluster aggregates without materialising the N×N distance
  matrix. The user controls the chunk size explicitly (default
  ``chunk_size=512``); cross-tested against
  `sklearn.metrics.silhouette_score` at ``atol=1e-10``.
- `compute_subsample_stability` refits HDBSCAN on random subsamples of
  the fitted 2-D embedding and reports adjusted-Rand-index agreement
  with the full-sample labels plus cluster-count variance across
  resamples. Distinct from the noise baseline (which asks about
  noise) and from the credibility test (which asks about the run as a
  whole).

## 10. What is *not* in this package

Items explicitly out of scope:

Items explicitly out of scope:

- The paper's two-run workflow (first on the full sample, then on
  each top-level component). This encodes the astronomer's prior that
  the top-level structure is "disk vs halo". Domain-agnostic users
  with different top-level priors filter ``result.labels`` down to
  their subgroup and call ``fit`` again.
- Domain validation on the $[\alpha/\text{Fe}]$–$[\text{Fe/H}]$ plane,
  Toomre diagrams, the VM criterion, or any other astronomy-specific
  check.
- Stellar sample construction, observational-uncertainty handling,
  and selection-function modelling.

These are scientific choices about *how to apply* the methodology, not
part of the methodology itself.
