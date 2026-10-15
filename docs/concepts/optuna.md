# Optuna: searching the settings

This page explains how {term}`Optuna` searches the settings of UMAP and HDBSCAN for Starfold, and why the score it uses must never see your labels.

```{eval-rst}
.. index::
   single: Optuna
   single: trial
   single: setting
   single: score
   single: Sobol design
   single: TPE
   single: pruning
   pair: search; label-free score
```

## A toy picture

You bake bread and want the best oven temperature and baking time. You cannot calculate the answer, so you try a setting, taste the loaf, write down a score and choose the next setting. After a few loaves you stop guessing at random and try settings near the ones that tasted good.

Optuna is that loop. A {term}`setting` is one complete choice of values (here six, for UMAP and HDBSCAN). Each proposal with its {term}`score` is a {term}`trial`. Optuna reads the scores so far and proposes the next setting.

## How Starfold runs the search

Starfold uses Optuna's default loop with three additions.

1. **Trial 0 is the library defaults.** Every search includes the answer you would get without one.
2. **Trials 1 to 16 are a {term}`Sobol design`.** They try settings spread evenly over the allowed range of every value, and they do not use past scores. A Sobol design fills a six-value space more evenly than random draws do.
3. **Later trials use {term}`TPE`.** This method looks at the past trials, splits them into good and poor, and tries settings near the good ones.

The search runs up to 48 trials. From trial 17, a trial whose first two fits score below the median of the earlier first-pair scores is stopped early ({term}`pruning`). Up to three stopped trials that look promising are completed at the end. From trial 33, the search stops once the best gap has improved by less than 0.01 over 16 trials.

Each trial is fitted on the same three random 80 percent subsets, so a difference between two scores comes from the settings and not from which rows were drawn.

## Why the score never sees the labels

Suppose your table has known classes and you tune the settings until the groups match them. The groups now match the classes you chose. They say nothing about whether the table holds other structure, and you cannot tell tuned structure from found structure. If your table has no known classes, as in most research, you have nothing to tune to.

Starfold's score is computed without labels and without the number of groups. It combines three measures:

- {term}`prediction agreement`: fit on one random subset, then place the other rows into those groups by nearest neighbour. Compare that placement with a fit that included those rows, using the {term}`adjusted Rand index` (ARI, a score from about 0 for chance to 1 for identical groupings). High agreement means a fit does not change much when the rows change;
- {term}`coverage`: the share of rows that both fits place in a group, so calling most rows noise does not pay;
- {term}`modularity`: how well the groups follow the neighbour links of your original scaled columns, before any map is drawn.

The score is the cube root of the product of the three. A zero in any one makes it zero. Because it never reads the group count, nothing pulls the search toward a particular number of groups.

The search does not rank trials by this score alone. It subtracts the score of the same setting on a {term}`structureless copy` of your table, and the difference is the {term}`gap`. The trial's number in the search is its gap. {doc}`copies-and-gaps` explains why.

## Where the search misleads

- **The best of many gaps is partly luck.** With 48 noisy gaps, the top one is probably too high. This is the {term}`winner's curse`. Starfold scores the top three trials and the defaults again on new subsets and new copies (a {term}`fresh re-score`) and lets the largest fresh gap win. The re-score reduces this optimism. It does not remove it, because the winner is still the largest of up to four noisy fresh gaps.
- **A high score can come from any table.** This is why the search maximises the gap over a structureless copy: see {doc}`copies-and-gaps`.
- **The winner is not the only good setting.** Close runners-up mean the choice is not sharp. Starfold stores the table of the best candidates, with their fresh scores and fresh gaps, as a sensitivity report.
- **More trials are not free.** Each trial costs several fits.

## Learn more

- [Optuna documentation](https://optuna.readthedocs.io/en/stable/): studies, trials and objectives in a few lines, with a first example.
- [Optuna first tutorial](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/001_first.html): a minimal objective, `suggest_float` and `optimize`.
- [Optuna frequently asked questions](https://optuna.readthedocs.io/en/stable/faq.html): seeds and reproducibility, saving and resuming a study, and NaN values.

Next: {doc}`pipeline`.
