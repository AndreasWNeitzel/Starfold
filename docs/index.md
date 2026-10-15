# Starfold manual

Starfold is a Python package that finds groups in a table of numbers and tells you how far to trust them. It is for readers who have a table (rows are objects, columns are measurements) and no labels. It combines three existing tools: {term}`UMAP` (draws a map of the table), {term}`HDBSCAN` (finds dense clumps on the map) and {term}`Optuna` (searches the settings of both). It then refits on random subsets of the rows and reports how often each group comes back.

If you are new, read in this order: {doc}`start/overview`, {doc}`start/install`, {doc}`start/first-run`.

## Where to start

| You want to | Go to |
|---|---|
| Know what Starfold is and is not | {doc}`start/overview` |
| Install it | {doc}`start/install` |
| Run it on a table in 5 lines | {doc}`start/first-run` |
| Understand UMAP, HDBSCAN and Optuna first | {doc}`concepts/index` |
| Learn by following a full example | {doc}`tutorial/index` |
| Find what a printed number means | {doc}`metrics/index` |
| Do one specific task | {doc}`howto/index` |
| Look up a function, option or flag | {doc}`reference/index` |
| Fix something that looks wrong | {doc}`troubleshooting` |
| Know what you may and may not claim | {doc}`limits` |
| Look up a word | {doc}`glossary`, or the {ref}`genindex` for every indexed term |

```{toctree}
:caption: Start
:maxdepth: 1
:hidden:

start/overview
start/install
start/first-run
```

```{toctree}
:caption: Concepts
:maxdepth: 1
:hidden:

concepts/index
```

```{toctree}
:caption: Tutorials
:maxdepth: 1
:hidden:

tutorial/index
```

```{toctree}
:caption: Metrics
:maxdepth: 1
:hidden:

metrics/index
```

```{toctree}
:caption: How-to guides
:maxdepth: 1
:hidden:

howto/index
```

```{toctree}
:caption: Reference
:maxdepth: 1
:hidden:

reference/index
troubleshooting
limits
learn-more
citing
glossary
```
