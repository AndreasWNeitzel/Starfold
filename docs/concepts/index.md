# Concepts

These pages explain the ideas behind Starfold, one idea per page. Read them in this order, or jump to the one you need. If you are short on time, read {doc}`pipeline` first: it names every stage and says which page covers it.

```{eval-rst}
.. index::
   single: concepts
```

| Page | Read it to learn |
|---|---|
| {doc}`umap` | Why a map of your table distorts distances, and which map settings Starfold searches |
| {doc}`hdbscan` | How dense clumps become groups, why some rows stay noise, and which clump settings Starfold searches |
| {doc}`optuna` | How the search proposes settings, and why its score must never see labels |
| {doc}`pipeline` | How one call chains the tools in six stages, and where each output is explained |
| {doc}`copies-and-gaps` | Why a score alone is not evidence, what a structureless copy is, and how the gap differs from the score |
| {doc}`refits` | What a refit measures, what co-association is, and the words Starfold allows for the result |

To run Starfold before reading these pages, start with {doc}`../start/first-run`.

```{toctree}
:maxdepth: 1

umap
hdbscan
optuna
pipeline
copies-and-gaps
refits
```
