# What Starfold does not claim

This page lists the claims Starfold does not make, the evidence for the limits of its resolution, and the data it does not handle.

```{eval-rst}
.. index::
   single: limits
```
```{eval-rst}
.. index::
   single: small groups merge
```
```{eval-rst}
.. index::
   single: stable versus true
```
```{eval-rst}
.. index::
   pair: claim; group count
```
```{eval-rst}
.. index::
   single: benchmark
```

Starfold recovers groups that come back when the table is resampled, and it measures how often they come back. It does not show that the groups are real. Each limit below follows from that.

## Claims Starfold does not make

**A stable partition is not a true partition.** A dense chance fluctuation is a real feature of the rows, and it recurs in every subset too. High {term}`recurrence` shows that a group is reproducible under the declared perturbation and says nothing about whether it is a population. Name the groups when you call a partition reproducible. The recorded runs quoted on this page ship in the repository under `notebooks/runs/digits` and `notebooks/runs/apogee`. Notebooks 01 and 02 write them, and `starfold.load` reads them. In the recorded run on the scikit-learn digits dataset, two of ten groups recur at 1.00, seven recur between 0.54 and 0.84, and group 5 (19 rows) recurs at 0.45, below the 0.5 mark.

**Recurrence is not a posterior.** {term}`Support <support>`, group recurrence and {term}`unresolved mass` are rates over {term}`refits <refit>`. They are not probabilities that a row belongs to a group. The same holds for HDBSCAN {term}`strengths <strengths>`, which are within-fit strengths, not {term}`posteriors <posterior>`.

**A stability interval is not a confidence interval.** The {term}`count interval` and the recurrence bands are the 5th to 95th percentile of a quantity over refits. They describe how much the answer moves under resampling. They do not cover a true value with a stated probability.

**A p-value holds only for the declared family.** The calibration {term}`p-value` is `(h + 1) / (L + 1)`, where `L` is the number of {term}`structureless copies <structureless copy>` and `h` the number that score at least as high as your table. It is valid only for the family you simulated (`gaussian_copula` or `independent_marginals`) and the procedure that was replayed. It is not the probability that the groups are real. It carries no per-group test, and it does not say how many groups exist.

**The group count is not a finding.** The score never reads the count, and the search does not tune on it. The flag `count_unstable` only reports whether the count holds across refits. On the benchmark the count can differ greatly from the class count (Gas drift: 390 groups for 6 classes; Letter: 284 for 26). The number of overlapping populations is not identifiable without a declared estimand and assumptions. Quote the {term}`count interval`, and do not quote `n_groups` as an estimate.

**The default comparison is not a significance test.** `beats_default_under_objective` is True whenever the chosen fresh gap is larger than the defaults'. The flag `default_within_noise` tests whether the difference exceeds twice the standard error of 6 paired differences. The six differences come from three subsets and overlap, so that estimate is likely optimistic. Its bias is not measured. Read the flag as a coarse screen.

**No per-group significance exists.** The p-value concerns the partition as a whole.

## Small groups merge

The merge scale is about `sqrt(15 m)` rows, with `m = 0.8 n` the rows per resample. It is 155 rows at `n = 2,000` and 346 rows at `n = 10,000`. The scale is empirical, measured on the ten-table benchmark below and by injecting groups into the APOGEE DR19 example table. Nothing above 20,000 rows was measured. Modularity's known preference for larger communities (Fortunato and Barthelemy 2007) is the likely reason, but the published bound does not give this number. If you expect smaller groups, use more rows and fewer irrelevant columns, or expect them to merge into a neighbour.

### Injection into real data

Artificial groups were added to a 2,000-row sample of the APOGEE DR19 example table, three at a time (32 trials, 16 refits, seed 0, one run per size). Starfold did not know which rows were added. Each group holds made-up rows spread evenly inside a ball around a random real star, with the radius set by that star's distance to its 30th nearest neighbour. A group counts as retrieved when a reported group shares at least half its rows with it (Jaccard index 0.5 or more). Purity is the share of the best-matching group that is injected. Completeness is the share of the injected rows it holds.

| Rows per group | Retrieved | Purity | Completeness |
|---:|---:|---:|---:|
| 30 | 0 of 3 | 0.04 | 0.88 |
| 60 | 0 of 3 | 0.09 | 0.86 |
| 120 | 3 of 3 | 0.74 | 0.86 |
| 240 | 3 of 3 | 0.73 | 0.94 |
| 480 | 3 of 3 | 0.92 | 0.91 |

The two smallest sizes were absorbed into large groups. Each row is one draw of three sites, so read the table as a rough threshold and not as a measured curve. The records are in `bench/results/injection_recovery.json`.

### The ten-table benchmark

Ten public tables with known classes were used only to score the output. Six form the development set and four are held out, so no rule or default was set by looking at them. Recovery is the adjusted Rand index (floored at 0) against the true classes on the assigned rows, times the share assigned. "Defaults" is the median recovery of the library defaults over 8 seeds. The second column is the mean class size inside one resample (`0.8 n / classes`) over `sqrt(15 m)`. For the scikit-learn digits dataset that is 143.8 rows over 146.8, or 0.98.

| Table (set) | Class size over scale | Starfold | Defaults |
|---|---:|---:|---:|
| Digits (development) | 0.98 | 0.740 | 0.637 |
| Rice (development) | 7.13 | 0.292 | 0.008 |
| Landsat (development) | 3.09 | 0.354 | 0.048 |
| PenDigits (development) | 2.42 | 0.819 | 0.753 |
| Dry Bean (development) | 3.85 | 0.500 | 0.445 |
| Waveform (development) | 5.44 | 0.111 | 0.000 |
| COIL-20 (held out) | 0.44 | 0.518 | 0.581 |
| Isolet (held out) | 0.78 | 0.198 | 0.453 |
| Gas drift (held out) | 4.54 | 0.038 | 0.023 |
| Letter (held out) | 1.26 | 0.203 | 0.108 |

Starfold matched or beat the defaults (within 0.02) on 6 of 6 development tables (mean recovery 0.469 against 0.315). It did so on 2 of 4 held-out tables (0.239 against 0.291) and on 8 of 10 tables overall (0.377 against 0.306). It lost on COIL-20 and Isolet, the tables with the smallest classes relative to the scale. Gas drift and Letter count as matches, yet their recoveries are low in absolute terms.

## Data Starfold does not handle

- **Text, categories and time series.** The table must be numbers. Rows are treated as independent, so time order is ignored. A category coded 0, 1, 2 is not a distance. Columns are compared by Euclidean distance after {term}`scaling <scaling>`.
- **Fewer than 50 finite rows or fewer than 2 columns.** `recover` raises a `ValueError`.
- **Missing values.** A row with NaN or infinity anywhere is dropped and counted, never filled in (flag `rows_rejected_nonfinite`). Filling gaps with a column mean would create a false dense clump.
- **Duplicate rows.** A row listed twice counts twice.
- **Tables above 20,000 rows.** Letter (20,000 rows) is the largest table run. Above 20,000 rows the full-data reference fit uses an approximate neighbour graph. Above 20,000 rows per resample (about 25,000 rows at the default 80 percent) the search graph is approximate too, and a run reproduces only with the same `n_jobs`. GPU memory was not recorded.
- **Real measurement uncertainties.** `X_err` and `measurement_sampler` work, but no run with real measured uncertainties has been validated (the APOGEE DR19 example table carries none). An invented error bar produces an invented spread.
- **Selection functions and scientific feature construction.** Starfold is domain-agnostic. You supply the features and the measurement model through explicit inputs. `recover` takes no selection function, so handle selection effects before you build the table.
- **Python outside 3.11 and 3.12.** The package declares `requires-python >=3.11,<3.13`.

See {doc}`howto/report-your-result` for what you may write in a paper, {doc}`start/overview` for what Starfold does, {doc}`metrics/index` for the numbers it prints, and {doc}`troubleshooting` for symptoms.
