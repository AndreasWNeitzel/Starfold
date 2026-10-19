# What Starfold does

Starfold sorts the rows of a numeric table into groups without labels and measures how often those groups come back when the rows change. This page lists what it returns and the claims it does not make.

```{eval-rst}
.. index::
   single: Starfold
   single: group
   single: noise
   single: recurrence
   single: flag
   pair: Starfold; what it claims
```

## What it does

You give Starfold a table of numbers. Each {term}`row` is one object (a star, an image, a cell) and each {term}`column` is one measurement. Starfold sorts the rows into {term}`group`s without being told what the groups are. It then refits the chosen setting on random subsets of the rows, each one a {term}`refit`. It counts how often each group comes back. "Trust" here means reproducibility: the groups survive a change of rows. It does not mean the groups are true.

Three existing tools do the work:

- {term}`UMAP` draws a low-dimensional map of the table.
- {term}`HDBSCAN` finds dense clumps in the map.
- {term}`Optuna` searches the settings of both, because the settings change the answer.

## What it returns

| You get | Where | What it tells you |
|---|---|---|
| Labels | `result.labels` | One integer per kept row: a group number, or -1 for {term}`noise` (no group claimed the row) |
| Support | `result.support` | For each row and group, the average share of the row's refit group that belongs to that group in the full fit ({doc}`../metrics/support`) |
| Recurrence | `result.worlds.group_recurrence` | For each group, how much of it comes back across refits: 1 means a refit group holds exactly the drawn members every time ({doc}`../metrics/recurrence`) |
| Flags | `result.flags` | Short names for weak evidence or notable events in the run ({doc}`../reference/flags`) |

Starfold never sees any known classes your rows may have. It chooses the settings with a {term}`score` that does not read labels and does not read the number of groups.

## What it does not claim

- A stable set of groups is not a true set of groups. Stability means only that the groups come back.
- {term}`Recurrence` is how often a group comes back across refits. It is not a {term}`posterior` probability.
- A range over refits is a {term}`stability interval`, not a {term}`confidence interval`.
- The number of groups is not a finding on its own. It means something only if you declare what real quantity you are counting (the estimand) and what you assume about it.
- A {term}`p-value` from the optional calibration test holds only for the family of structureless copies it used and the procedure it simulated. The test asks whether such copies score as high as your table.

## The pipeline in six steps

1. Search: Optuna tries up to 48 {term}`setting`s of UMAP and HDBSCAN.
2. Score: each setting gets a {term}`score` on your table. Starfold fits the same setting on a {term}`structureless copy` of your table (your table with its cross-column structure destroyed) and subtracts that score. The difference is the {term}`gap`, and by default the search makes the gap large.
3. Choice: the best few settings and the library defaults are scored again on new subsets. One wins.
4. Full fit: the winner is fitted once on all rows. These are your labels.
5. Refits: the winner is refitted on 32 random subsets (each one a possible world, stored in `result.worlds`) to measure recurrence.
6. Calibration test: an optional test asks whether structureless copies score as high as your table.

{doc}`../concepts/pipeline` explains each step and names the page that covers its output.

## Which page next

- To install: {doc}`install`.
- To run it in five lines: {doc}`first-run`.
- To learn the three tools first: {doc}`../concepts/index`.
- To follow a complete example: {doc}`../tutorial/index`.
- To look up a printed number: {doc}`../metrics/index`.
- To look up a word: {doc}`../glossary`.
