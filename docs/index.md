# Starfold

<p class="sf-lede">Finds groups in a table of numbers and tells you how far to trust them.</p>

```console
pip install "starfold @ git+https://github.com/AndreasWNeitzel/starfold"
```

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} Start
:link: start/overview
:link-type: doc

What Starfold is and is not, how to install it, and a first run on your own table.
:::

:::{grid-item-card} Concepts and tutorials
:link: concepts/index
:link-type: doc

UMAP, HDBSCAN and Optuna in plain words, then two full examples from table to report.
:::

:::{grid-item-card} Metrics and reference
:link: metrics/index
:link-type: doc

What every printed number means, and every function, option and flag.
:::
::::

## What it does

Starfold is a Python package for readers who have a table (rows are objects, columns are measurements) and no labels. It combines three existing tools: {term}`UMAP` (draws a map of the table), {term}`HDBSCAN` (finds dense clumps on the map) and {term}`Optuna` (searches the settings of both). It then refits on random subsets of the rows and reports how often each group comes back.

```{figure} figures/digits_partitions.png
:alt: Three scatter panels of the scikit-learn digits on a two-dimensional map. Left, ten colours for the true digits. Middle, Starfold's ten groups with faded points where support is low. Right, 51 groups from the library defaults.

The scikit-learn digits on a two-dimensional map. Left, the true digits, used only as a check after the fit. Middle, the 10 groups Starfold reports, faded where the refits rarely agree. Right, the 51 groups that one fit with the library defaults finds.
```

## Find what you need

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
