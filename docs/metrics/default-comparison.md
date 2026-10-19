# Default comparison

This page explains the comparison of the chosen {term}`setting` with the library defaults, and what its printed numbers do and do not show.

```{eval-rst}
.. index::
   single: default comparison
   pair: metric; default comparison
   single: beats_default_under_objective
   single: selected_is_default
   single: paired_differences
   single: library defaults
   single: default_within_noise
```

## The question it answers

Did the search find a setting that scores better than the UMAP and HDBSCAN defaults, by the same measure, on the same table?

## A picture

Trial 0 of every search is the library defaults. Starfold weighs them like any other setting, so you always see what the tools would have given without a search. At the end, the top three complete trials and the defaults are scored again on new subsets and three new {term}`structureless copy`s. The setting with the largest fresh {doc}`gap` is chosen. Ties go to the lowest trial number, so the defaults win any tie they are part of.

High picture: the chosen setting has a fresh gap of 0.372 and the defaults have -0.001 (recorded run on the APOGEE DR19 example table). The defaults do no better on the data than on a copy, and the chosen setting does. The fresh re-score reduces the winner's curse, but does not remove it. Low picture: the two fresh scores differ by less than the noise between subsets. Then the choice between them is not resolved, even when the chosen gap is larger.

## Range and reference values

| Quantity | Value |
|---|---|
| Fresh gaps, chosen and defaults | each -1 to 1, see {doc}`gap` |
| Per-pair differences | about 0 when the two settings score alike; 6 values from 3 subsets |
| `beats_default_under_objective` | True or False |

Perfect result and pure chance are not applicable to a comparison. When the defaults win, the chosen setting is the defaults, the paired differences are all 0, and `beats_default_under_objective` is False.

## How to read a value

Three printed lines carry the comparison. The bands below are a rule of thumb from this project, not from a published source.

To decide whether to report the chosen setting or the library defaults, read the three rows below in this order.

| Reading | What you see | Decision |
|---|---|---|
| Defaults are enough | flag `selected_is_default`, or `beats_default_under_objective` False | Use the defaults and report that the search found nothing better |
| Not resolved | flag `default_within_noise` | Report both; do not claim the search improved on the defaults |
| Chosen setting ahead | True, and the flag is absent | The chosen setting scores higher under this objective, by more than a rough noise screen; go on to {doc}`recurrence` |

## What misleads

1. `beats_default_under_objective` is True exactly when the chosen fresh gap is larger than the defaults' fresh gap. It does not test whether the difference exceeds the noise. The flag `default_within_noise` does that test, on the six paired differences.
2. The comparison is under the objective: the three terms of the {doc}`score` and the gap. It does not show that the chosen groups are closer to any truth. In the recorded digits run, the chosen setting also agreed better with the true digits. Its coverage-adjusted ARI was 0.740, against 0.594 for the defaults at one seed and a median of 0.637 over 8 default seeds. The bench scripts and the notebook compute that after the fact, from labels that the objective never sees.
3. Only six differences exist, from three subsets, and they overlap (each subset appears in four of the six pairs). The noise estimate behind `default_within_noise` is optimistic for this reason. A difference of the same size can be noise in one table and a fair signal in another.
4. A win over the defaults says nothing about how many groups exist. The score never reads the group count.

## Where Starfold prints it

| Where | What it holds |
|---|---|
| `result.search.gap_selected`, `gap_default` | fresh gaps of the chosen setting and the defaults |
| `result.search.score_selected`, `score_default` | fresh scores of the two |
| `result.search.paired_differences` | chosen minus defaults, per ordered pair of subsets |
| `result.search.beats_default_under_objective` | True when the chosen fresh gap is larger |
| `result.search.selected_is_default` | True when the defaults won |
| `result.flags` | `selected_is_default`, `default_within_noise` |

The notebook for the scikit-learn digits dataset prints the comparison:

```python
print(f"fresh gap: chosen {s.gap_selected:.3f}, defaults {s.gap_default:.3f}")
print("chosen minus defaults per subset pair:", [round(d, 3) for d in s.paired_differences])
print(f"beats the defaults under the objective: {s.beats_default_under_objective}")
print(f"chosen setting is the defaults: {s.selected_is_default}")
```

The recorded runs quoted on this page ship in the repository under `notebooks/runs/digits` and `notebooks/runs/apogee`. Notebooks 01 and 02 write them, and `starfold.load` reads them. The lines below come from `notebooks/01_digits.ipynb`:

```text
fresh gap: chosen 0.661, defaults 0.565
chosen minus defaults per subset pair: [0.087, 0.032, 0.075, 0.004, 0.045, -0.016]
beats the defaults under the objective: True
chosen setting is the defaults: False
```

Five of the six differences are positive and one is negative. The run did not raise `default_within_noise` (see {doc}`gap` for the arithmetic). That flag is an approximate screen, not a test: its noise estimate treats six overlapping pair differences from three subsets as if they were independent, so it is optimistic. Read the absence of the flag as "not clearly within the noise", not as proof that the chosen setting is better. The run on the APOGEE DR19 example table printed `beats the defaults under the objective: True` with fresh gaps 0.372 and -0.001.

:::{dropdown} Formula
```text
beats_default_under_objective = gap_selected > gap_default
paired_differences[k]         = fresh_pair_score_selected[k] - fresh_pair_score_default[k],  k = 1..6
default_within_noise          = (not selected_is_default) and (at least 2 differences)
                                and |score_selected - score_default| <= 2 * sd(d, ddof=1) / sqrt(len(d))
```

`fresh_pair_score` is the geometric mean of the pair's agreement and coverage with the trial's mean modularity (the same for all six pairs), from the fresh plan. The source is `search` in `src/starfold/search.py` and `derive_flags` in `src/starfold/result.py`.
:::

## Learn more

- Tibshirani, Walther and Hastie 2001, the gap statistic behind the ranking: [doi:10.1111/1467-9868.00293](https://doi.org/10.1111/1467-9868.00293).
- Tibshirani and Walther 2005, cluster validation by prediction strength: [doi:10.1198/106186005X59243](https://doi.org/10.1198/106186005X59243).
- Related pages: {doc}`gap`, {doc}`score`, and the calibration test in {doc}`calibration-p`.
