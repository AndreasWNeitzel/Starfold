# Report your result in a paper

This guide shows which numbers of a Starfold result you may quote in a paper, and which sentences they do and do not support. An example paragraph uses the recorded APOGEE run.

```{eval-rst}
.. index::
   single: reporting
   single: paper wording
   pair: claim; what you may say
   single: example paragraph
   single: group count; quoting
```

## Prerequisites

- A finished `Result`, from `starfold.recover` or from {doc}`save-and-load`, with its table `X`.
- The calibration test attached to the result, if you want to write anything about structure beyond reproducibility ({doc}`run-calibration-test`).
- The random seed, the `Config` and the `engine` of the run, all stored in `result.provenance` ({doc}`reproduce-a-run`).

## Steps

1. Print the flags and read each one before you write a sentence.

   ```python
   for flag in result.flags:
       print(flag)
   ```

   In the recorded APOGEE run, before the calibration test, you see:

   ```text
   search_budget_exhausted
   consensus_subsampled
   consensus_disagrees
   null_not_run
   ```

   The {doc}`../reference/flags` page gives the meaning of each one.

   Each flag limits one kind of sentence. Step 4 gives the wording. The page {doc}`../reference/flags` says which flags block a claim (column "Fault").

2. Pick the sentences from this table. Each row gives a number, what it supports, what it does not support, and the page that defines it.

   | Number | You may write | You may not write | Page |
   |---|---|---|---|
   | `result.worlds.group_recurrence` and `group_recurrence_band` | "Group *g* has recurrence *r* over 32 refits (5 to 95 percent band *a* to *b*)", the mean Jaccard index of its best match in a refit. | That the group is real, a population, or true. | {doc}`../metrics/recurrence` |
   | `result.worlds.count_interval` | "The refits found *a* to *b* groups." | That there are `n_groups` populations. | {doc}`../metrics/count-interval` |
   | `result.unresolved_mass` | "On average *u* of a row's refit mass was noise or sat with noise rows." | That a share *u* of the stars are noise. | {doc}`../metrics/unresolved` |
   | `result.worlds.own_support` | "On average over refits, *s* of a row's refit group belongs to its full-fit group." | That high own support gives a purer sample. | {doc}`../metrics/support` |
   | `result.null.p_value` and `p_intervals` | "Of *L* copies of the declared family, *h* reached the table's statistic (p = *p*, Monte Carlo interval *a* to *b*)." | That the groups are real with probability 1 minus *p*, or that *p* applies to a single group. | {doc}`../metrics/calibration-p` |
   | `result.search.gap_selected` and `gap_default` | "On fresh subsets and copies the chosen setting scored a gap of *x* and the defaults *y*." | That the search found the true best setting. | {doc}`../metrics/gap` |
   | `result.consensus.contrast` and `agreement` | "The co-association contrast was *c*, and the regrouping matched the partition with consensus agreement *k*." | Either number alone as evidence of groups. | {doc}`../metrics/coassociation` |
   | `result.worlds.property_median` and `property_band` | "The median of column *j* in group *g* was *m* (5 to 95 percent over refits *a* to *b*)." | That the band is a confidence interval. | {doc}`../metrics/properties` |

   Write "stability interval" for every 5 to 95 percent band over refits. It shows how far the answer moves under resampling and does not cover a true value with a stated probability ({doc}`../limits`).

3. Name the groups when you write about reproducibility. Never write that a partition is "mostly reproducible" without saying which groups recur and which do not. In the recorded Digits run, groups 0 and 1 recur at 1.00 and group 5 (19 rows) recurs at 0.45, below the 0.5 mark ({doc}`../metrics/recurrence`). A single sentence about "the partition" hides that spread.

4. Write one sentence for each flag that was raised. The wording below matches the limit that the flag sets.

   | Flag | One honest sentence | Page |
   |---|---|---|
   | `count_unstable` | "Fewer than half of the refits found the same number of groups, so I do not quote a single count." | {doc}`../reference/flags` |
   | `consensus_disagrees` | "The regrouping of the co-association matrix did not match the partition (consensus agreement below 0.5), which I read together with the contrast." | {doc}`../metrics/coassociation` |
   | `no_structure_supported` | "A copy of the declared family matched the table's statistic in too many draws, so the groups are not shown to exceed that family." | {doc}`../metrics/calibration-p` |
   | `default_within_noise` | "The chosen setting is not shown to beat the library defaults beyond the noise of six paired differences." | {doc}`../metrics/default-comparison` |

   The noise in `default_within_noise` treats six overlapping pair differences as independent, so read it as a coarse screen.

5. State the procedure once, with its seed and numbers. Give the table size, the columns, the scaling, the engine, the seed, the number of trials, the number of refits and the number of copies. `result.provenance` holds the seed, the engine and the configured budgets. Early stopping can run fewer trials and copies than budgeted, so read the counts from `len(result.search.trials)`, `result.worlds.world_labels.shape[0]` and `result.null.n_realisations`.

6. Do not report a claim that goes beyond the flags and the table in step 2. If a recurrence sits below the `clusterboot` mark of 0.85, follow {doc}`the recurrence page <../metrics/recurrence>`.

## Expected result

An example paragraph, built only from the recorded APOGEE run (the APOGEE DR19 example table, eight columns, `random_state=0`, notebook `notebooks/02_apogee.ipynb`, calibration record `bench/results/apogee_null.json`):

> Starfold recovered two groups of 1,588 and 6,586 stars from 9,242 stars and left 1,068 stars unassigned. Across 32 refits the groups had recurrence 0.914 (5 to 95 percent band 0.886 to 0.935) and 0.918 (0.879 to 0.938). The refits found 2 groups in 31 cases and 3 groups in one case, so the count interval is 2 to 2. The unresolved mass is 0.154. On fresh subsets and three fresh copies the chosen setting reached a gap of 0.372 against -0.001 for the library defaults. A matched Gaussian-copula test with 19 copies returned p = 0.10 (one copy, at 0.366, reached the table's statistic of 0.362; Monte Carlo interval 0.001 to 0.260), so the two-group split is not shown to exceed a single correlated population. The co-association contrast on a 4,096-star subset was 0.976, while the regrouping returned only noise (consensus agreement 0.000). Group 0 has a higher median [Mg/Fe] (0.273 dex) and [alpha/M] (0.222 dex) and a lower median [Fe/H] (-0.416 dex) than group 1 (0.040, 0.026 and -0.076 dex). These medians were not compared with published positions of the chemical sequences.

Each sentence rests on recorded numbers and one page:

| Sentence | Numbers | Page |
|---|---|---|
| Group sizes and unassigned stars | 1,588, 6,586, 1,068 of 9,242 | {doc}`../tutorial/apogee` |
| Recurrence with bands | 0.914 (0.886 to 0.935), 0.918 (0.879 to 0.938) | {doc}`../metrics/recurrence` |
| Count interval | 2 in 31 refits, 3 in one, interval 2 to 2 | {doc}`../metrics/count-interval` |
| Unresolved mass | 0.154 | {doc}`../metrics/unresolved` |
| Fresh gap against defaults | 0.372 against -0.001 | {doc}`../metrics/gap` |
| Calibration test | p = 0.10, 1 of 19, 0.366 against 0.362, interval 0.001 to 0.260 | {doc}`../metrics/calibration-p` |
| Contrast and regrouping | 0.976, consensus agreement 0.000, 4,096 stars | {doc}`../metrics/coassociation` |
| Group medians | Group 0: 0.273, 0.222 and -0.416 dex. Group 1: 0.040, 0.026 and -0.076 dex | {doc}`../metrics/properties` |

The paragraph makes no claim about the Galaxy. The table is a check of the method, and the calibration test did not support structure beyond the declared copy family. The run cannot separate three reasons. The test replays only trials 0 to 16, and its power against the full search is not measured. With 19 copies, p moves in steps of 0.05, and one copy decided the outcome. The copy keeps the correlations between iron, magnesium and the orbits, which carry much of the two-sequence signal ({doc}`../troubleshooting`). The cause of the all-noise regrouping is not isolated either.

## What you may not write

Each line follows from the statistical language of the project ({doc}`../limits`).

- **"The groups are real" or "the partition is true".** A stable partition is not a true partition. Recurrence measures how often a group comes back on resampled rows.
- **"Recurrence is the probability".** Recurrence, support and unresolved mass are rates over refits and not posteriors. HDBSCAN strengths are within-fit strengths.
- **"95 percent confidence interval".** A percentile over refits is a stability interval.
- **"p = 0.10 means a 10 percent chance that the groups are real".** The p-value holds only for the declared family of copies and the procedure that was simulated. It says nothing about single groups or about how many groups exist.
- **"The data contain *n* populations".** The group count is not a finding without a declared estimand. Quote the count interval and not `n_groups`.
- **"High own support means a purer group".** In the recorded Digits run, 81 percent of rows with own support of 0.9 or more match their group's majority digit. For all grouped rows the share is 83 percent. Own support measures how often a row stays with its group across refits.
- **"The search improved on the defaults" when `default_within_noise` is raised.** See {doc}`../metrics/default-comparison`.
- **A number computed with the known classes as a search target.** The {term}`agreement with known classes` is a {term}`post hoc` check ({doc}`check-against-known-labels`). Tuning the budgets, the space or the scaling until it rises turns the check into a fit to the labels.

## Variations

- **Several seeds.** Report the recurrence of a group for each seed, and say which seeds gave which band. A group that crosses one of the `clusterboot` marks (0.85, 0.75, 0.6 or 0.5) between seeds is not reproducible at the level of that mark.
- **No calibration test.** Write that the test was not run and quote no p-value. The placeholder p of 1.0 in `result.null` is not a result.
- **Measurement errors.** If you passed `X_err`, say so, and say that the error bars were measured ones ({doc}`measurement-errors`). An invented error bar produces an invented spread.

## Related pages

- {doc}`../limits`
- {doc}`../citing` (which papers to cite for each number)
- {doc}`../reference/flags`
- {doc}`../metrics/index`
