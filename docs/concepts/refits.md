# Refits: reproducibility, not truth

This page explains what a refit is and which words Starfold allows for the result. Refitting on random subsets measures how reproducible your groups are, not whether they are true.

```{eval-rst}
.. index::
   single: refit
   single: recurrence
   single: support
   single: stability interval
   single: count interval
   single: reproducibility
   single: co-association
   single: estimand
   pair: refit; language rules
```

## A toy picture

Ask a class of 30 students to each draw a map of the same town from memory, but give each student only 80 percent of the street list. If most maps show the same three districts, the districts do not depend on any one street. If one district appears on half of the maps and in different places, it depends on which streets the student happened to have.

That is a refit. A {term}`refit` (also called a possible world) is one fit of the chosen setting on a random 80 percent of your rows, drawn without replacement, with a new {term}`seed`. Starfold makes 32 by default (`Config.n_worlds` and `Config.world_fraction` set the number and the fraction). Each one is a version of your table that could have been observed with slightly different rows. If you pass per-value uncertainties as `X_err`, each refit also redraws every value from its error.

## What the refits measure

Each refit gives a partition. Starfold compares them with the full fit, which used all rows, and reports:

- **{term}`Support`** for a row and a group: the average share of the row's refit group that lies in that full-fit group, taken over the refits that drew the row. **{term}`Own support`** is the support for the row's own group. It says how reliably refits keep the row with its group. It is not a probability and not a purity.
- **{term}`Recurrence`** for a group: in every refit, the group is matched with the refit group of highest {term}`Jaccard index` (shared rows over rows in either). Recurrence is the mean of those Jaccard values over refits. 1 means the group comes back with the same members each time.
- **{term}`Unresolved mass`**: for one refit, a row scores 1 if the refit called it noise, and otherwise the share of its refit group that the full fit calls noise. Starfold averages over the refits that drew the row, then over rows.
- **{term}`Count interval`**: the 5th to 95th percentile of the number of groups the refits found, rounded outward to whole numbers.
- **{term}`Co-association`**: see the next section.

A worked example of recurrence. A group has four rows, a, b, c and d. Refit 1 has a group with exactly those four rows: Jaccard 4/4 = 1.00. Refit 2 has a group with a, b and c only: Jaccard 3/4 = 0.75. Refit 3 has a group with a, b, e and f: two shared rows over six in either, Jaccard 2/6 = 0.33. The recurrence over these three refits is (1.00 + 0.75 + 0.33) / 3 = 0.69.

## Co-association: a check that needs no matching of groups

Group numbers are arbitrary. One refit's group 0 can be another refit's group 1, so matching groups across refits can go wrong. {term}`Co-association` avoids the matching. For a pair of rows, it is the share of refits that put both rows in the same group, among the refits that drew and assigned both.

Starfold averages it over two sets of pairs. **Within** is the mean co-association over pairs of rows that share a group in the full fit. **Between** is the mean over pairs in different groups. The {term}`contrast` is within minus between. A contrast near 1 says the refits keep your groups together and keep different groups apart. A contrast near 0 says the refits treat both kinds of pair alike.

Starfold uses it as a second check on the full fit. It also cuts the rows into groups using only the pair shares and compares that cut with the full-fit labels (the {term}`consensus check`). The cut is a second clustering, so it can fail while the contrast stays high. In the recorded digits run the contrast is 0.751 (within 0.820, between 0.068) and the cut found 3 groups, so the flag `consensus_disagrees` was raised. Read the contrast first. The {doc}`../metrics/coassociation` page has the bands.

## Reproducibility is not truth

Refitting on random subsets answers one question: does the answer hold when the rows change a little? It cannot answer whether the groups are real.

- A method can give the same wrong answer every time. A bad scaling, a missing column or a wrong distance reproduces on every subset.
- Refits reuse the rows you have. They do not see a part of the sky, a population or a selection effect that your table lacks.
- A group can recur without being pure: it can come back with the same members and still mix different kinds of object. Only a check against known classes, made after the fact, shows this.

A stable partition is not a true partition.

## Language rules

| Write | Do not write |
|---|---|
| Recurrence (how much of a group comes back across refits), support (how reliably refits keep a row with its group) | Probability, posterior |
| Stability interval (the 5th to 95th percentile over refits) | Confidence interval |
| "The groups come back in most refits" | "The groups are real" |
| "The count varies from 3 to 16 across refits" | "There are 10 populations" |

- HDBSCAN strengths are within-fit strengths, not probabilities. See {doc}`hdbscan`.
- Recurrence is an algorithmic recurrence rate. It is not a {term}`posterior`.
- A {term}`stability interval` describes spread over refits. It does not cover a true value with a stated probability, so it is not a {term}`confidence interval`.
- The number of groups is not a finding unless its estimand is declared. The estimand is the real quantity you say the count measures, together with your assumptions about it. Starfold never tunes or judges on the group count.
- Hennig (2007, doi:10.1016/j.csda.2006.11.025) assesses the stability of single clusters with Jaccard values. It gives two reference marks: a mean Jaccard of 0.85 or more reads as highly stable, and 0.5 or less as dissolved. Starfold quotes them and never uses them to drop a group. The paper does not address how to choose the number of groups, and the count interval rests on no source for choosing a count.

## Read a recurrence between the marks

A value between 0.5 and 0.85, such as 0.54 for digits group 3 (band 0.25 to 0.81), is neither stable nor dissolved by the marks above. `[no-source: internal-reasoning]` Do the following.

1. Read the band. A band that reaches below 0.5 means some refits lost the group.
2. Find which group it merges with across refits. Look at the co-association between this group and each other group: a high value shows that the refits join them.
3. Do not report it as a stable population. Report the value, the band and the group it merges with.
4. Test it with more refits (raise `Config.n_worlds`) or a different seed, and see whether the value holds.

## Read the recorded digits run

In the recorded digits run (`notebooks/runs/shots/01_digits.ipynb`), 32 refits found between 3 and 16 groups, and the full fit found 10. The unresolved mass was 0.044. Recurrence ran from 0.45 to 1.00. Groups 0 and 1 reach 1.00. Group 2 (0.84) and group 4 (0.82) come close to the 0.85 mark. Groups 6 to 9 sit between 0.70 and 0.77, group 3 at 0.54, and group 5 at 0.45, below the 0.5 mark, with a band from 0.02 to 0.94. The band is wide because the matched refit group differs a lot from refit to refit. So two groups come back almost unchanged, eight recur only in part, and the group count is unstable. Use the groups with high recurrence first.

## Learn more

- {doc}`pipeline`: where refits sit in the six stages.
- {doc}`copies-and-gaps`: the test that asks whether your table has structure a copy lacks.
- [HDBSCAN soft clustering](https://hdbscan.readthedocs.io/en/latest/soft_clustering.html): soft membership, which helps you read within-fit strengths.
