# Glossary

This page defines every term that the manual links to from its pages. Each entry gives a plain definition, then one sentence on how Starfold uses the word, with a link to the page that explains it.

```{eval-rst}
.. index::
   single: glossary
```
```{eval-rst}
.. index::
   single: terms
```

```{glossary}
:sorted:

adjusted Rand index
  A score for how alike two ways of splitting the same rows into groups are, equal to 1 for identical splits and about 0 for random ones, where the group numbers do not matter.
  Starfold floors it at 0. {term}`Prediction agreement` is its average over held-out rows and has no coverage factor, because {term}`coverage` is a separate term of the {term}`score`. {term}`Refit agreement`, {term}`consensus agreement` and {term}`agreement with known classes` multiply it by coverage ({doc}`/metrics/prediction-agreement`).

alpha
  The cut-off for the {term}`p-value`, 0.05 by default. A p-value at or below it counts as support.
  Starfold sets `Config.alpha` to 0.05 and raises `no_structure_supported` when the p-value is above it ({doc}`/metrics/calibration-p`).

column
  One quantity measured for every object in your table, such as a pixel value or an abundance.
  Starfold gives every column equal weight after {term}`scaling`, so the columns you choose define what "similar" means ({doc}`/howto/prepare-table`).

confidence interval
  A range computed by a rule that covers the true value in a stated share of repeated studies.
  Starfold reports no confidence intervals: its percentile ranges over {term}`refits <refit>` are {term}`stability intervals <stability interval>` ({doc}`/limits`).

consensus check
  A check that turns the share of {term}`refits <refit>` placing two rows together into a distance, regroups the rows from that distance alone, and compares the result with your {term}`partition`.
  Starfold runs it on the reference partition, on a recorded subset of rows when the matrix does not fit its memory budget ({doc}`/metrics/coassociation`).

contrast
  The average {term}`co-association` of row pairs in the same group minus its average over pairs in different groups.
  Starfold prints it as `result.consensus.contrast`; a value near 1 means the refits agree with your partition, and it needs no regrouping ({doc}`/metrics/coassociation`).

co-association
  For a pair of rows, the share of {term}`refits <refit>` that put both in the same group, among the refits that assigned both to a group.
  Starfold builds a distance from it for the {term}`consensus check` ({doc}`/metrics/coassociation`).

count interval
  The range from the 5th to the 95th percentile of the number of groups found by the {term}`refits <refit>`, widened outwards to whole numbers.
  Starfold prints it as `result.worlds.count_interval` and treats it as a {term}`stability interval`, never as an estimate of how many populations exist ({doc}`/metrics/count-interval`).

coverage
  The share of held-out rows that two fits both place in a group and not in {term}`noise`.
  Starfold takes the {term}`geometric mean` of it with {term}`prediction agreement` and {term}`modularity` to form the {term}`score`, so calling most rows noise does not pay ({doc}`/metrics/coverage`).

cuML
  The RAPIDS library from NVIDIA that runs UMAP, HDBSCAN and related steps on a GPU with a scikit-learn style interface.
  Starfold uses it as one {term}`engine` when it imports with a visible CUDA device ({doc}`/howto/use-gpu`).

deployed settings
  The {term}`setting` the search chose, with `min_cluster_size` and `min_samples` rescaled from the row count of a resample to the row count of the full table.
  Starfold stores them as `result.deployed_params`; they produced `result.labels` ({doc}`/concepts/pipeline`).

dex
  A unit for a logarithmic quantity, where 1 dex is a factor of 10, that is, a difference of 1 in the base-10 logarithm.
  Starfold reports group properties in the units of your columns, so the APOGEE abundances appear in dex ({doc}`/tutorial/apogee`).

embedding
  The low-dimensional map of your table that UMAP produces, in which rows that are similar sit close together.
  Starfold groups the rows in this map, and its axes carry no units ({doc}`/concepts/umap`).

engine
  The software that runs the fits: `"cpu"` (the reference libraries) or `"cuml"` (a GPU).
  Starfold takes `engine="auto"` by default, which uses {term}`cuML` when it works and otherwise warns and uses the CPU ({doc}`/howto/use-gpu`).

flag
  A short name for one event or weakness in a run, derived from the evidence arrays so that people and scripts can check it.
  Starfold lists all 16 in `result.flags` ({doc}`/reference/flags`).

fresh re-score
  A second scoring of the best {term}`trials <trial>` and the defaults on new subsets and new {term}`structureless copies <structureless copy>`.
  Starfold does it because the best score among many noisy trials is optimistic (the {term}`winner's curse`), and it selects the setting with the largest fresh {term}`gap` ({doc}`/metrics/gap`).

full fit
  The one fit of the chosen {term}`setting` on all kept rows.
  Starfold reports it as `result.labels`; it is a single transparent fit and not a vote of the refits ({doc}`/concepts/pipeline`).

gap
  The {term}`score` of a setting on your table minus its score on a {term}`structureless copy`.
  Starfold maximises it, so a setting that scores as well on structureless data gains nothing ({doc}`/metrics/gap`).

Gaussian copula
  A way to draw a random table that keeps every column's values and the correlations between columns, by converting each column to normal scores, drawing from a multivariate normal and reordering the values to match.
  Starfold uses it as its default family of {term}`structureless copy` ({doc}`/concepts/copies-and-gaps`).

group
  A dense clump of rows that HDBSCAN reports, given an integer label from 0 upwards.
  Starfold's `result.labels` assigns each kept row to a group or to {term}`noise` ({doc}`/concepts/hdbscan`).

HDBSCAN
  A clustering method that finds dense clumps of different densities in a map and leaves the rows between clumps as {term}`noise`.
  Starfold runs it on the {term}`embedding` and searches its {term}`min_cluster_size`, {term}`min_samples` and {term}`selection method` ({doc}`/concepts/hdbscan`).

Jaccard index
  The number of rows two groups share divided by the number of rows in either, which is 1 for identical groups and 0 for groups with no shared row.
  Starfold matches each reference group to its best {term}`refit` cluster by this index to get {term}`recurrence` ({doc}`/metrics/recurrence`).

kept rows
  The rows of your table that held only finite values and were used.
  Starfold stores their positions as `result.kept_rows` so that `X[result.kept_rows]` lines up with `result.labels` ({doc}`/howto/prepare-table`).

marginal
  The distribution of one column taken by itself, ignoring the other columns.
  Starfold's {term}`structureless copies <structureless copy>` keep every marginal exactly and change only how values are paired across columns ({doc}`/concepts/copies-and-gaps`).

min_cluster_size
  The HDBSCAN setting for the smallest number of rows that may form a group.
  Starfold searches it on a log scale and rescales it from resample rows to full rows for the {term}`deployed settings` ({doc}`/concepts/hdbscan`).

min_dist
  The UMAP setting for how tightly the map may pack similar rows.
  Starfold searches it from 0.0 to 0.5 by default ({doc}`/concepts/umap`).

min_samples
  The HDBSCAN setting for how many neighbours define a row's local density, where larger values call more rows {term}`noise`.
  Starfold searches it as a fraction of {term}`min_cluster_size` ({doc}`/concepts/hdbscan`).

modularity
  A number that compares the share of links between nearby rows that stay inside groups with the share expected by chance.
  Starfold computes it on a 15-nearest-neighbour graph of your scaled columns, before any map is drawn, as one lane of the {term}`score` ({doc}`/metrics/modularity`).

Monte Carlo interval
  The range of exceedance shares that a limited number of random copies leaves open, here a 95 percent {term}`Clopper-Pearson interval`.
  Starfold prints it as `null.p_intervals`; it describes the precision of the simulation only ({doc}`/metrics/calibration-p`).

n_components
  The UMAP setting for how many coordinates the {term}`embedding` has.
  Starfold searches it from 2 to 10 by default ({doc}`/concepts/umap`).

n_neighbors
  The UMAP setting for how many nearby rows each row looks at, where small values keep fine detail and large values the broad layout.
  Starfold searches it from 5 to 100 on a log scale ({doc}`/concepts/umap`).

nearest neighbours
  The rows closest to a given row by distance.
  Starfold uses the 15 nearest rows to predict the group of an unseen row and to build the {term}`modularity` graph ({doc}`/metrics/prediction-agreement`).

noise
  The label -1 that HDBSCAN gives a row that no clump claimed.
  It means that no {term}`group` took the row, not that the measurement was bad ({doc}`/concepts/hdbscan`).

null hypothesis
  The default claim that a test tries to reject.
  Starfold's null hypothesis is that your table is one structureless population from the declared copy family, and the {term}`p-value` holds only for that family ({doc}`/metrics/calibration-p`).

Optuna
  A Python library that searches for good values of a set of numbers by proposing them, receiving a score and using the scores so far to propose the next.
  Starfold uses it to search the {term}`settings <setting>` ({doc}`/concepts/optuna`).

partition
  The set of groups that one fit defines on the rows, including which rows are noise.
  Group numbers are arbitrary names, so two partitions agree when they split the rows alike and not when they use the same numbers ({doc}`/concepts/pipeline`).

p-value
  The share of {term}`structureless copies <structureless copy>` whose statistic is at least as high as the table's, counting the table as one more draw. If the table is itself a draw from the copy family, p is spread about evenly over its possible values.
  Starfold prints `null.p_value`; a small value means the table beats almost every copy, and it is not the probability that the groups are real ({doc}`/metrics/calibration-p`).

posterior
  The probability of a hypothesis after seeing the data, computed from a prior.
  Starfold computes no posteriors: {term}`recurrence` and {term}`strengths <strengths>` are rates and within-fit strengths ({doc}`/concepts/refits`).

prediction agreement
  How well a fit on one subset predicts the groups of a second fit on rows it never saw, measured by the {term}`adjusted Rand index` floored at 0 and averaged over the six ordered pairs of three subsets.
  Starfold uses it as one term of the {term}`score`, with no coverage factor inside it ({doc}`/metrics/prediction-agreement`). It is stored as `stability` in each trial.

provenance
  The record of how a result was produced: engines, library versions, configuration, seeds, budgets, fit counts, time per stage and a fingerprint of the input.
  Starfold stores it as `result.provenance` ({doc}`/reference/result`).

pruning
  Stopping a {term}`trial` early because its first fits score below the median of finished trials.
  Starfold prunes only convergence trials, after two of their three fits ({doc}`/concepts/optuna`).

recurrence
  How much of a group comes back in the {term}`refits <refit>`: for each refit, the best {term}`Jaccard index` match of the group, averaged over refits. 1 means the same rows every time and 0 means the group never comes back.
  Starfold's group recurrence, {term}`support` and {term}`unresolved mass` all measure it, and none is a {term}`posterior` ({doc}`/metrics/recurrence`).

refit
  A fit of the chosen {term}`setting` on a random subset of the rows (80 percent by default), with a new seed and, if you give uncertainties, one redraw of your values.
  Starfold makes 32 and calls them worlds in the code; every measure of {term}`recurrence` is taken over them ({doc}`/concepts/refits`).

reproducibility
  Whether a result comes back when the analysis is repeated, either with the same inputs and seed or on resampled data.
  Starfold's refits measure the second kind, and its {term}`seed` and {term}`provenance` support the first ({doc}`/howto/reproduce-a-run`).

row
  One object in your table, such as an image or a star.
  Starfold gives each row a label, a {term}`support` vector and a place in the {term}`embedding` ({doc}`/howto/prepare-table`).

scaling
  Rescaling each column so that no column dominates by its units.
  Starfold fits the scaler inside each subset on its own rows only, with `"standard"` (mean and standard deviation) or `"robust"` (median and interquartile range) ({doc}`/howto/choose-scaling`).

score
  The one number per {term}`setting` that Starfold maximises: the cube root of the product of {term}`prediction agreement`, {term}`coverage` and {term}`modularity`.
  It is 0 whenever any of the three is 0 and it never reads the number of groups ({doc}`/metrics/score`).

seed
  The starting number of a random number generator, which fixes the sequence of random choices.
  Starfold derives every seed from `random_state` and the position of the step in the procedure, so a stage does not depend on which worker runs it ({doc}`/howto/reproduce-a-run`).

selection method
  The HDBSCAN rule for picking clumps from its tree: `"eom"` keeps the most persistent clumps and `"leaf"` the finest ones.
  Starfold searches both ({doc}`/concepts/hdbscan`).

setting
  One complete choice of the six values Starfold searches: {term}`n_neighbors`, {term}`min_dist`, {term}`n_components`, {term}`min_cluster_size`, {term}`min_samples` and {term}`selection method`.
  Starfold stores a setting as a `Params` object ({doc}`/concepts/optuna`).

Sobol design
  A set of points placed to cover a range more evenly than random draws, without learning from any score.
  Starfold uses an Owen-scrambled Sobol design for trials 1 to 16 ({doc}`/concepts/optuna`).

Spearman correlation
  A number from -1 to 1 that measures how well the ranks of two lists agree.
  Starfold's benchmark uses it to check how well trial scores track the recovery of known classes ({doc}`/limits`).

stability interval
  A range between percentiles of a quantity over {term}`refits <refit>`, showing how far the answer moves under resampling.
  Starfold's {term}`count interval` and recurrence bands are stability intervals and not {term}`confidence intervals <confidence interval>` ({doc}`/concepts/refits`).

strengths
  HDBSCAN's within-fit membership strength for each row, between 0 and 1.
  They are not probabilities; Starfold stores them as `result.strengths` ({doc}`/metrics/strengths`).

structureless copy
  A version of your table that keeps every value of every column and changes only which row each value sits in, so that no groups remain, and every number in it was measured.
  Starfold scores settings on it to compute the {term}`gap` and runs the calibration test on it ({doc}`/concepts/copies-and-gaps`).

support
  For a row and a group of the full fit, the average share of the row's {term}`refit` group that belongs to that group in the full fit, taken over the refits that drew the row. It is not a purity and not a probability.
  Starfold's own support is the support for the row's own group ({doc}`/metrics/support`).

own support
  The {term}`support` of a row for its own group in the full fit.
  Starfold keeps it as `result.worlds.own_support`, and it is NaN for {term}`noise` rows ({doc}`/metrics/support`).

TPE
  The tree-structured Parzen estimator, an Optuna sampler that sorts earlier trials into good and poor and proposes new settings where the good ones sit.
  Starfold uses it after the {term}`Sobol design`, in its multivariate form ({doc}`/concepts/optuna`).

trial
  One proposed {term}`setting` with its {term}`score`.
  Starfold runs up to 48 trials, and trial 0 is always the library defaults ({doc}`/concepts/optuna`).

UMAP
  A method that turns a table into a low-dimensional map by arranging rows so that each row stays near its most similar rows.
  Starfold uses it to make the {term}`embedding` and searches its {term}`n_neighbors`, {term}`min_dist` and {term}`n_components` ({doc}`/concepts/umap`).

unresolved mass
  For one refit, a row scores 1 if the refit called it noise. Otherwise it scores the share of its refit group that the full fit calls noise. Starfold averages that score over the refits that drew the row, then over rows.
  Starfold prints it as `result.unresolved_mass` and raises `high_unresolved_mass` above 0.5 ({doc}`/metrics/unresolved`).

winner's curse
  The bias that makes the best of many noisy scores look better than it is, because part of its lead is luck.
  Starfold counters it with a {term}`fresh re-score` of the top trials ({doc}`/metrics/gap`).

agreement
  A word this manual does not use alone. Starfold has four separate agreements: {term}`prediction agreement`, {term}`refit agreement`, {term}`consensus agreement` and {term}`agreement with known classes`. All four rest on the {term}`adjusted Rand index`. A fifth, {term}`backend agreement`, compares CPU and GPU fits.
  Which one to read depends on your question ({ref}`Which agreement do I read <which-agreement>`).

agreement with known classes
  The {term}`adjusted Rand index`, floored at 0, between Starfold's groups and classes you already know, on the assigned rows, times the share of rows assigned ({term}`coverage`). It is a {term}`post hoc` check and never enters the search.
  Starfold does not print it in `result`; you compute it ({doc}`/howto/check-against-known-labels`).

alpha elements
  Elements built mainly in massive stars that explode as supernovae, such as oxygen, magnesium, silicon and calcium. The ratio [alpha/M] compares their abundance with the Sun's, relative to the metals as a whole.
  The disc stars of the bundled APOGEE table fall on a high-alpha and a low-alpha sequence at fixed iron content [Fe/H] ({doc}`/tutorial/apogee`).

APOGEE
  The Apache Point Observatory Galactic Evolution Experiment, an infrared spectroscopic survey of stars of the Milky Way, part of the Sloan Digital Sky Survey.
  Starfold's bundled second example is a table of 9,242 stars from its data release 19 ({doc}`/tutorial/apogee`).

ASPCAP
  The APOGEE Stellar Parameters and Chemical Abundances Pipeline, which fits each star's spectrum to get its temperature, surface gravity and abundances.
  The three abundance columns of the bundled APOGEE table, [Fe/H], [alpha/M] and [Mg/Fe], come from it ({doc}`/tutorial/apogee`).

backend agreement
  How well the partition from the CPU fit of one {term}`setting` matches the partition from the cuML fit, as the {term}`adjusted Rand index` times coverage per seed, and normalised by the smaller assigned share.
  Starfold prints it as `result.backends` when `Config(check_backends=True)`, and raises `backend_disagreement` below 0.8 ({doc}`/howto/compare-backends`).

Clopper-Pearson interval
  An exact interval for a share estimated from counted yes-or-no outcomes, built from the binomial distribution, with a stated nominal level such as 95 percent (Clopper and Pearson 1934, [doi:10.1093/biomet/26.4.404](https://doi.org/10.1093/biomet/26.4.404)).
  Starfold uses it on the share of copies that reach your table, as the {term}`Monte Carlo interval` ({doc}`/metrics/calibration-p`).

community
  In a {term}`graph`, a set of nodes that have more links among themselves than chance would give.
  Starfold's {term}`modularity` asks how well the groups of a fit act as communities of the nearest-neighbour graph of your rows ({doc}`/metrics/modularity`).

consensus agreement
  The {term}`adjusted Rand index`, floored at 0, between the regrouping of the {term}`consensus check` and the full-fit labels on the rows it used, times coverage.
  Starfold prints it as `result.consensus.agreement` and raises `consensus_disagrees` below 0.5; it judges the cut, and the {term}`contrast` needs no cut ({doc}`/metrics/coassociation`).

exceedance
  A copy whose statistic is at least as high as your table's, in the calibration test.
  Starfold counts them as `null.exceedances`; with `h` of them among `L` copies, the {term}`p-value` is (h + 1) / (L + 1) ({doc}`/metrics/calibration-p`).

Gaia
  The European Space Agency mission that measures the positions, distances and motions of over a billion stars. Its third data release is Gaia DR3.
  The orbital columns of the bundled APOGEE table were integrated from Gaia DR3 measurements ({doc}`/tutorial/apogee`).

galpy
  A Python package for galactic dynamics that integrates orbits of stars in a model of the Milky Way (Bovy 2015, [doi:10.1088/0067-0049/216/2/29](https://doi.org/10.1088/0067-0049/216/2/29)).
  The orbital columns of the bundled APOGEE table (actions, angular momentum, eccentricity, energy) were computed with it; Starfold itself does not use it ({doc}`/tutorial/apogee`).

geometric mean
  The n-th root of the product of n non-negative numbers, which is 0 whenever any one of them is 0.
  Starfold's {term}`score` is the geometric mean of {term}`prediction agreement`, {term}`coverage` and {term}`modularity`, so no term can make up for another that is 0 ({doc}`/metrics/score`).

graph
  A set of nodes joined by links. Here the nodes are rows of your table and each link joins a row to one of its {term}`nearest neighbours`.
  Starfold builds a 15-nearest-neighbour graph of the scaled columns to compute {term}`modularity` ({doc}`/metrics/modularity`).

post hoc
  Latin for after this. A check made after the analysis is fixed, with information the analysis never used.
  Starfold's check against known classes is post hoc: the classes never enter the search, and you must not tune on it ({doc}`/howto/check-against-known-labels`).

purity
  The share of the rows of a group that belong to its most common known class.
  Starfold does not compute it; the notebooks plot it after the run as a {term}`post hoc` check, and {term}`support` is not a purity ({doc}`/tutorial/digits`).

refit agreement
  For one {term}`refit`, the {term}`adjusted Rand index`, floored at 0, between its labels and the full-fit labels on the rows it drew, times coverage.
  Starfold prints the 32 values as `result.worlds.world_agreement`; the recorded Digits run has a median of 0.79 ({doc}`/concepts/refits`).

world
  The name the code uses for a {term}`refit`.
  Starfold keeps it in names such as `result.worlds`, `n_worlds` and `world_labels` ({doc}`/concepts/refits`).
```
