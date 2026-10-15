# Learn more

This page lists outside documentation and papers for each tool Starfold uses, and three resources on checking a clustering. Every link resolved when it was checked on 2026-09-30. Two publisher DOIs (Astronomy & Astrophysics and the ACM) answer automated requests with HTTP 403 (a bot block, not a dead link; the DOIs resolve). The RAPIDS pages have moved to docs.nvidia.com, and the links below use the final addresses. Each DOI in the papers section returned BibTeX from doi.org on the same date.

```{eval-rst}
.. index::
   single: learn more
```
```{eval-rst}
.. index::
   single: external links
```
```{eval-rst}
.. index::
   single: papers
```

## UMAP

{term}`UMAP` builds the map (the {term}`embedding`) that HDBSCAN groups.

- [How UMAP works](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html): a plain route from neighbour graphs to a 2-D map, with the mathematics left out.
- [UMAP parameters](https://umap-learn.readthedocs.io/en/latest/parameters.html): what `n_neighbors`, `min_dist`, `n_components` and `metric` do, with paired figures.
- [Basic UMAP usage](https://umap-learn.readthedocs.io/en/latest/basic_usage.html): a first `fit_transform` on penguins and digits, plotted.
- [Using UMAP for clustering](https://umap-learn.readthedocs.io/en/latest/clustering.html): why grouping a UMAP map works, and where it misleads.
- [UMAP FAQ](https://umap-learn.readthedocs.io/en/latest/faq.html): scaling, memory, cores and how UMAP can go wrong.

## HDBSCAN

{term}`HDBSCAN` finds dense groups in the map and leaves the rest as {term}`noise`.

- [How HDBSCAN works](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html): an illustrated five-stage walk-through.
- [Parameter selection](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html): how `min_cluster_size` and `min_samples` change labels and noise.
- [Basic HDBSCAN usage](https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html): a first fit and how to read `labels_`.
- [Comparing clustering algorithms](https://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html): where k-means, DBSCAN and HDBSCAN each fail.
- [HDBSCAN FAQ](https://hdbscan.readthedocs.io/en/latest/faq.html): answers that start from the symptom, such as too much noise.
- [Soft clustering](https://hdbscan.readthedocs.io/en/latest/soft_clustering.html): soft membership, which helps when you read `probabilities_`.

## Optuna

{term}`Optuna` proposes the {term}`settings <setting>` and records each {term}`trial`.

- [Optuna documentation](https://optuna.readthedocs.io/en/stable/): a study, a trial and an objective in a few lines.
- [Installation](https://optuna.readthedocs.io/en/stable/installation.html): install with pip or conda.
- [First optimization](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/001_first.html): a minimal objective with `suggest_float`, `optimize` and `best_params`.
- [Visualization](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/005_visualization.html): plots of the optimisation history.
- [Optuna FAQ](https://optuna.readthedocs.io/en/stable/faq.html): seeds, saving and resuming a study, and NaN values.

## cuML (RAPIDS)

{term}`cuML` runs UMAP and HDBSCAN on an NVIDIA GPU under Linux or WSL 2.

- [cuML documentation](https://docs.nvidia.com/cuml/latest/): a scikit-learn style interface on the GPU.
- [cuML UMAP](https://docs.nvidia.com/cuml/latest/api/generated/cuml.manifold.UMAP/): the parameters of the GPU version.
- [cuML HDBSCAN](https://docs.nvidia.com/cuml/latest/api/generated/cuml.cluster.hdbscan.HDBSCAN/): the parameters of the GPU version.
- [RAPIDS install selector](https://docs.nvidia.com/datascience/install/): a page that builds the install command for your CUDA version.

## scikit-learn

Starfold uses scikit-learn for scaling, the adjusted Rand index and the CPU regrouping in the consensus check.

- [`sklearn.cluster.HDBSCAN`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html): the signature and attributes of the CPU reference.
- [Clustering](https://scikit-learn.org/stable/modules/clustering.html): methods side by side, with a section on performance metrics.
- [HDBSCAN example](https://scikit-learn.org/stable/auto_examples/cluster/plot_hdbscan.html): a runnable example.
- [scikit-learn glossary](https://scikit-learn.org/stable/glossary.html): a model for looking up a word.

## Papers behind the tools

The DOI link opens the paper or its record.

- McInnes, Healy and Melville (2018), UMAP: [doi:10.48550/arXiv.1802.03426](https://doi.org/10.48550/arXiv.1802.03426).
- Campello, Moulavi and Sander (2013), the HDBSCAN algorithm: [doi:10.1007/978-3-642-37456-2_14](https://doi.org/10.1007/978-3-642-37456-2_14).
- McInnes and Healy (2017), a fast HDBSCAN implementation: [doi:10.48550/arXiv.1705.07321](https://doi.org/10.48550/arXiv.1705.07321).
- McInnes, Healy and Astels (2017), the `hdbscan` package in the Journal of Open Source Software: [doi:10.21105/joss.00205](https://doi.org/10.21105/joss.00205).
- Akiba et al. (2019), Optuna: [doi:10.1145/3292500.3330701](https://doi.org/10.1145/3292500.3330701), also [arXiv:1907.10902](https://arxiv.org/abs/1907.10902).
- Neitzel et al. (2025), the astronomy study Starfold grew out of: [doi:10.1051/0004-6361/202451718](https://doi.org/10.1051/0004-6361/202451718).

## Checking a clustering

- [scikit-learn, clustering performance evaluation](https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation): the adjusted Rand index and related measures, with formulas and worked code.
- von Luxburg (2010), [Clustering stability: an overview](https://doi.org/10.48550/arXiv.1007.1075): what stability can and cannot tell you about a clustering.
- Hennig (2007), [Cluster-wise assessment of cluster stability](https://doi.org/10.1016/j.csda.2006.11.025): the resampling design behind group recurrence and its Jaccard matching.

For the full reference list of Starfold's methods, see {doc}`citing`.
