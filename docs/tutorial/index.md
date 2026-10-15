# Tutorials

```{eval-rst}
.. index::
   single: tutorial
   single: learning path
```

This page helps you pick one of two tutorials. Each follows a full run of Starfold from the table to the written claim, each shows the exact output after every step, and each is split into two pages.

| Tutorial | Table | Choose it when |
|---|---|---|
| Digits: {doc}`digits` and {doc}`digits-refits` | 1,797 handwritten digits, 64 pixel columns, true classes known | You are new to Starfold, or you want to check a result against known classes. |
| APOGEE: {doc}`apogee` and {doc}`apogee-checks` | 9,242 APOGEE stars, 8 columns, no labels | You have a real table with no labels, and you want to see how to report the result. |

Start with {doc}`digits`. The APOGEE tutorial assumes you know the words the digits tutorial explains: flag, recurrence, count interval and co-association. The notebooks behind the tutorials are `notebooks/01_digits.py` and `notebooks/02_apogee.py` in the clone.

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
| Digits | 80.4 s on a laptop RTX 3060 when the run has the GPU to itself, 314 fits (each fit is one UMAP plus one HDBSCAN run). The recorded output shows 182.0 s because notebook 02 ran at the same time. |
| APOGEE | 242 s, 334 fits. The notebook does not say whether the GPU was shared, and no standalone time is recorded. The calibration test adds 2,176 s (1,938 fits). |

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
