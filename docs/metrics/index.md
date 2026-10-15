# Metrics

This page lists every number Starfold prints or returns, says what question each answers, and sends you to its page.

```{eval-rst}
.. index::
   single: metrics
   pair: metric; index
   single: result attributes
```

## Numbers about the groups

| Metric | Question it answers | Range | Where printed |
|---|---|---|---|
| {doc}`Group recurrence <recurrence>` | How often does a group come back in the refits? | 0 to 1 | `result.worlds.group_recurrence`, `group_recurrence_band` |
| {doc}`Support <support>` and own support | How much of a row's refit outcome lands in each group? | 0 to 1 | `result.support`, `result.worlds.own_support` |
| {doc}`Unresolved mass <unresolved>` | How much of the table do the refits fail to place in a group? | 0 to 1 | `result.unresolved_mass`, `result.worlds.unresolved` |
| {doc}`Count interval <count-interval>` | How much does the number of groups move across refits? | whole numbers | `result.worlds.count_interval`, `count_values` |
| {doc}`Co-association contrast <coassociation>` | Do pairs in one group stay together across refits, and pairs in different groups stay apart? | -1 to 1 | `result.consensus.contrast`, `within_association`, `between_association`, `agreement` |
| {doc}`Calibration p-value <calibration-p>` | Does the table beat structureless copies on the largest search gap among trials 0 to 16 (the defaults and the 16-trial Sobol design; 0.710 for Digits, the search gap of trial 7)? | 0 to 1 | `result.null.p_value`, `p_intervals` |
| {doc}`Strengths <strengths>` | How firmly does HDBSCAN hold each row inside its group in one fit? | 0 to 1 | `result.strengths` |
| {doc}`Properties <properties>` | What are the typical column values of each group, with a band over refits? | column units | `result.worlds.property_median`, `property_band` |

## Numbers about the choice of setting

| Metric | Question it answers | Range | Where printed |
|---|---|---|---|
| {doc}`Prediction agreement <prediction-agreement>` | Do groups fitted on one subset predict the groups of unseen rows? | 0 to 1 | `result.search.trials[i].stability` |
| {doc}`Coverage <coverage>` | What share of the tested rows do both fits assign to a group? | 0 to 1 | `result.search.trials[i].coverage` |
| {doc}`Modularity <modularity>` | Do the groups follow the 15-nearest-neighbour graph of your table? | below 1, clipped at 0 | `result.search.trials[i].modularity` |
| {doc}`Score <score>` | How good is one setting on the three terms together? | 0 to 1 | `raw_score`, `fresh_score`, `result.search.score_selected` |
| {doc}`Gap <gap>` | How much better is the setting on the table than on a structureless copy? | -1 to 1 | `result.search.trials[i].score`, `candidates[i].gap`, `gap_selected` |
| {doc}`Default comparison <default-comparison>` | Does the chosen setting beat the library defaults, and by more than noise? | True or False, differences | `beats_default_under_objective`, `paired_differences`, flag `default_within_noise` |

## Other numbers Starfold returns

These have no page of their own. The table below defines each in one line.

| Number | What it is | Where printed |
|---|---|---|
| Group count and noise rows | groups and rows with label -1 in the fit on all rows | `result.n_groups`, `result.labels` |
| Refit agreement | for each refit, the adjusted Rand index against the full-data labels times the joint coverage of the two | `result.worlds.world_agreement` |
| Backend agreement | CPU against cuML partitions, when `check_backends` is on | `result.backends.mean_normalised` |
| Flags | short names for weaknesses in the run | `result.flags` |
| Run record | rows kept, seconds and fits per stage, peak memory | `result.provenance` |

## Four meanings of agreement

The word "agreement" names four different numbers. Starfold never uses it alone in a printed name.

| Name | What it compares | Where |
|---|---|---|
| Prediction agreement | labels predicted for unseen rows against the fit on those rows, ARI only | {doc}`prediction-agreement` |
| Refit agreement | each refit against the full-data labels, ARI times joint coverage | `result.worlds.world_agreement` |
| Consensus agreement | the co-association cut against the full-data labels, ARI times coverage (flag `consensus_disagrees` below 0.5) | {doc}`coassociation` |
| Agreement with known classes | full-data labels against classes you hold, computed after the fact, ARI times coverage | notebooks and bench, never the search |

## Which page answers which decision

| Decision | Page |
|---|---|
| Is this setting worth keeping in the search? | {doc}`score` and its three terms |
| Does the table beat a structureless copy? | {doc}`gap` |
| Do I use the chosen setting or the defaults? | {doc}`default-comparison` |
| Do I use this group? | {doc}`recurrence` |
| Do I quote a group count? | {doc}`count-interval` |

## How to read a result in order

1. Read `result.flags` first. Each flag names one weakness or event.
2. Read the group count and the number of noise rows in `result.labels`.
3. Read the {doc}`default comparison <default-comparison>`: did the search beat the defaults, and by more than noise?
4. Read {doc}`unresolved mass <unresolved>`: how much of the table the groups describe.
5. Read {doc}`group recurrence <recurrence>` for each group you plan to use.
6. Read the {doc}`count interval <count-interval>` before you quote a number of groups.
7. Read {doc}`support <support>` for single rows you care about.
8. Read the {doc}`co-association contrast <coassociation>`, which ignores group numbers.
9. Read the {doc}`calibration p-value <calibration-p>` if you ran the test. Its absence raises the flag `null_not_run`.
10. Read {doc}`properties <properties>` last, to describe the groups in column units.

Read the three terms of the score, the {doc}`score` and the {doc}`gap` when you want to know why the search chose its setting. The fresh re-score of the top trials reduces the winner's curse (the best of many noisy scores is partly luck); it does not remove it. A stable partition is not a true partition, and no single number here proves the groups are real.

## Pages on the choice of setting

```{toctree}
:maxdepth: 1

prediction-agreement
coverage
modularity
score
gap
default-comparison
```

## Pages on the evidence about the groups

```{toctree}
:maxdepth: 1

recurrence
support
unresolved
count-interval
coassociation
calibration-p
strengths
properties
```
