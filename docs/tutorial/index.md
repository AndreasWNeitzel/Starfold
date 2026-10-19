# Tutorials

```{eval-rst}
.. index::
   single: tutorial
   single: learning path
```

There are two tutorials, each split into two pages. Both follow a full run of Starfold from the table to the written claim and show the exact output after every step.

| Tutorial | Table | Choose it when |
|---|---|---|
| Digits: {doc}`digits` and {doc}`digits-refits` | The scikit-learn digits dataset: 1,797 images, 64 pixel columns, true classes known | You are new to Starfold, or you want to check a result against known classes. |
| APOGEE: {doc}`apogee` and {doc}`apogee-checks` | The APOGEE DR19 example table: 9,242 stars, 8 columns, no labels | You have a real table with no labels, and you want to see how to report the result. |

Start with {doc}`digits`. The APOGEE tutorial assumes you know the words the digits tutorial explains: flag, recurrence, count interval and co-association. The notebooks behind the tutorials are `notebooks/01_digits.ipynb` and `notebooks/02_apogee.ipynb` in the clone.

| Page | What it covers |
|---|---|
| {doc}`digits` | Load, run, flags, map, settings |
| {doc}`digits-refits` | Support, recurrence, group count, post hoc check, save |
| {doc}`apogee` | Features, run, groups |
| {doc}`apogee-checks` | Co-association, copy and gap, calibration, reporting |

## Expected time

The time is compute time from the recorded runs on a laptop GPU with RAPIDS cuML. The time on a CPU is not measured, so expect longer. The time you need to read each tutorial is not measured.

| Tutorial | Recorded run |
|---|---|
| Digits | 93.9 s on a laptop RTX 3060 (search 77.4 s, refits 7.2 s), 314 fits (each fit is one UMAP plus one HDBSCAN run), 1,422 MB peak host memory. |
| APOGEE | 163 s (search 140 s, refits 13 s), 334 fits, 1,684 MB peak host memory. The calibration test adds 2,176 s (1,938 fits). |

## What you need

- Python 3.11 or 3.12.
- A clone of the repository, because both tutorials use files that ship with the clone and not with the installed package. Install as in {doc}`../start/install`.
- A graphics card is optional. Without one, Starfold runs on the CPU.
- No knowledge of UMAP, HDBSCAN or Optuna. Each tutorial defines a word where it first appears, and the {doc}`../glossary` lists them all.

If you would rather run one call on your own table first, go to {doc}`../start/first-run`. If you want the ideas before the code, read {doc}`../concepts/index`.

```{toctree}
:maxdepth: 1

digits
digits-refits
apogee
apogee-checks
```
