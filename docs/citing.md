# Citing Starfold

This page tells you what to cite when you use Starfold, and gives the references of the tools and methods it builds on.

```{eval-rst}
.. index::
   single: citing
```
```{eval-rst}
.. index::
   single: BibTeX
```
```{eval-rst}
.. index::
   single: references
```
```{eval-rst}
.. index::
   single: CITATION.cff
```

## What to cite

Cite the software and the methodological paper. The repository file `CITATION.cff` asks for both.

1. The software: Neitzel, A. W., Starfold, version 0.2.0, <https://github.com/AndreasWNeitzel/starfold>.
2. The paper Starfold grew out of: Neitzel, A. W., Campante, T. L., Bossini, D., and Miglio, A. 2025, Astronomy & Astrophysics 695, A243, [doi:10.1051/0004-6361/202451718](https://doi.org/10.1051/0004-6361/202451718).
3. The tools: McInnes, Healy and Melville (2018) for UMAP, Campello, Moulavi and Sander (2013) and McInnes, Healy and Astels (2017) for HDBSCAN, and Akiba et al. (2019) for Optuna.

Also cite the method papers for any number you report. The group recurrence (the best Jaccard match of a group over resamples) follows the cluster-wise stability of Hennig (2007), which says nothing on choosing the number of groups. The gap follows Tibshirani, Walther and Hastie (2001). The p-value follows Phipson and Smyth (2010) and Besag and Clifford (1991). The co-association check follows Fred and Jain (2005) and Monti et al. (2003).

## BibTeX for the software and the tools

Every entry below was fetched from doi.org as BibTeX on 2026-09-30, then reduced to the fields shown and given a short key. The software entry comes from `CITATION.cff` and has no year because version 0.2.0 is marked unreleased in the changelog. Add the release year when you cite a released version.

```bibtex
@software{neitzel_starfold,
  author  = {Neitzel, Andreas W.},
  title   = {Starfold},
  version = {0.2.0},
  url     = {https://github.com/AndreasWNeitzel/starfold},
  license = {MIT}
}

@article{neitzel2025,
  author  = {Neitzel, A. W. and Campante, T. L. and Bossini, D. and Miglio, A.},
  title   = {Dissecting stellar populations with manifold learning. I. Validation of the method on a synthetic Milky Way-like galaxy},
  journal = {Astronomy \& Astrophysics},
  volume  = {695},
  pages   = {A243},
  year    = {2025},
  doi     = {10.1051/0004-6361/202451718}
}

@misc{mcinnes2018umap,
  author = {McInnes, Leland and Healy, John and Melville, James},
  title  = {UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction},
  year   = {2018},
  eprint = {1802.03426},
  archivePrefix = {arXiv},
  doi    = {10.48550/ARXIV.1802.03426}
}

@inbook{campello2013,
  author    = {Campello, Ricardo J. G. B. and Moulavi, Davoud and Sander, Joerg},
  title     = {Density-Based Clustering Based on Hierarchical Density Estimates},
  booktitle = {Advances in Knowledge Discovery and Data Mining},
  publisher = {Springer Berlin Heidelberg},
  year      = {2013},
  pages     = {160--172},
  doi       = {10.1007/978-3-642-37456-2_14}
}

@article{mcinnes2017hdbscan,
  author  = {McInnes, Leland and Healy, John and Astels, Steve},
  title   = {hdbscan: Hierarchical density based clustering},
  journal = {The Journal of Open Source Software},
  volume  = {2},
  number  = {11},
  pages   = {205},
  year    = {2017},
  doi     = {10.21105/joss.00205}
}

@misc{mcinnes2017accelerated,
  author = {McInnes, Leland and Healy, John},
  title  = {Accelerated Hierarchical Density Clustering},
  year   = {2017},
  eprint = {1705.07321},
  archivePrefix = {arXiv},
  doi    = {10.48550/ARXIV.1705.07321}
}

@inproceedings{akiba2019,
  author    = {Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori},
  title     = {Optuna: A Next-generation Hyperparameter Optimization Framework},
  booktitle = {Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  publisher = {ACM},
  year      = {2019},
  pages     = {2623--2631},
  doi       = {10.1145/3292500.3330701}
}
```

## Software papers without a DOI

Three works that Starfold's code cites have no DOI. Each entry was checked on 2026-09-30 against the publisher's own page (jmlr.org for the two JMLR papers, papers.nips.cc for the NeurIPS paper). Title, authors, year, volume and pages below match those pages. I did not check the BibTeX against a second source.

```bibtex
@article{pedregosa2011,
  author  = {Pedregosa, Fabian and Varoquaux, Ga{\"e}l and Gramfort, Alexandre and Michel, Vincent and Thirion, Bertrand and Grisel, Olivier and Blondel, Mathieu and Prettenhofer, Peter and Weiss, Ron and Dubourg, Vincent and Vanderplas, Jake and Passos, Alexandre and Cournapeau, David and Brucher, Matthieu and Perrot, Matthieu and Duchesnay, {\'E}douard},
  title   = {Scikit-learn: Machine Learning in {P}ython},
  journal = {Journal of Machine Learning Research},
  volume  = {12},
  number  = {85},
  pages   = {2825--2830},
  year    = {2011},
  url     = {https://jmlr.org/papers/v12/pedregosa11a.html}
}

@article{bergstra2012,
  author  = {Bergstra, James and Bengio, Yoshua},
  title   = {Random Search for Hyper-Parameter Optimization},
  journal = {Journal of Machine Learning Research},
  volume  = {13},
  number  = {10},
  pages   = {281--305},
  year    = {2012},
  url     = {https://jmlr.org/papers/v13/bergstra12a.html}
}

@inproceedings{bergstra2011,
  author    = {Bergstra, James S. and Bardenet, R{\'e}mi and Bengio, Yoshua and K{\'e}gl, Bal{\'a}zs},
  title     = {Algorithms for Hyper-Parameter Optimization},
  booktitle = {Advances in Neural Information Processing Systems 24 (NIPS 2011)},
  year      = {2011},
  url       = {https://papers.nips.cc/paper_files/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html}
}
```

The galpy library, which built the orbital columns of the bundled APOGEE table, has a DOI: Bovy, J. 2015, The Astrophysical Journal Supplement Series 216, 29, [doi:10.1088/0067-0049/216/2/29](https://doi.org/10.1088/0067-0049/216/2/29).

## BibTeX for the methods

:::{dropdown} Method references (verified from doi.org on 2026-09-30)

```bibtex
@article{hennig2007,
  author  = {Hennig, Christian},
  title   = {Cluster-wise assessment of cluster stability},
  journal = {Computational Statistics \& Data Analysis},
  volume  = {52},
  number  = {1},
  pages   = {258--271},
  year    = {2007},
  doi     = {10.1016/j.csda.2006.11.025}
}

@article{hubert1985,
  author  = {Hubert, Lawrence and Arabie, Phipps},
  title   = {Comparing partitions},
  journal = {Journal of Classification},
  volume  = {2},
  number  = {1},
  pages   = {193--218},
  year    = {1985},
  doi     = {10.1007/BF01908075}
}

@article{tibshirani2001,
  author  = {Tibshirani, Robert and Walther, Guenther and Hastie, Trevor},
  title   = {Estimating the Number of Clusters in a Data Set Via the Gap Statistic},
  journal = {Journal of the Royal Statistical Society Series B},
  volume  = {63},
  number  = {2},
  pages   = {411--423},
  year    = {2001},
  doi     = {10.1111/1467-9868.00293}
}

@article{tibshirani2005,
  author  = {Tibshirani, Robert and Walther, Guenther},
  title   = {Cluster Validation by Prediction Strength},
  journal = {Journal of Computational and Graphical Statistics},
  volume  = {14},
  number  = {3},
  pages   = {511--528},
  year    = {2005},
  doi     = {10.1198/106186005X59243}
}

@article{lange2004,
  author  = {Lange, Tilman and Roth, Volker and Braun, Mikio L. and Buhmann, Joachim M.},
  title   = {Stability-Based Validation of Clustering Solutions},
  journal = {Neural Computation},
  volume  = {16},
  number  = {6},
  pages   = {1299--1323},
  year    = {2004},
  doi     = {10.1162/089976604773717621}
}

@article{newman2006,
  author  = {Newman, M. E. J.},
  title   = {Modularity and community structure in networks},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {103},
  number  = {23},
  pages   = {8577--8582},
  year    = {2006},
  doi     = {10.1073/pnas.0601602103}
}

@article{fortunato2007,
  author  = {Fortunato, Santo and Barth{\'e}lemy, Marc},
  title   = {Resolution limit in community detection},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {104},
  number  = {1},
  pages   = {36--41},
  year    = {2007},
  doi     = {10.1073/pnas.0605965104}
}

@article{chari2023,
  author  = {Chari, Tara and Pachter, Lior},
  title   = {The specious art of single-cell genomics},
  journal = {PLOS Computational Biology},
  volume  = {19},
  number  = {8},
  pages   = {e1011288},
  year    = {2023},
  doi     = {10.1371/journal.pcbi.1011288}
}

@misc{vonluxburg2010,
  author = {von Luxburg, Ulrike},
  title  = {Clustering Stability: An Overview},
  year   = {2010},
  eprint = {1007.1075},
  archivePrefix = {arXiv},
  doi    = {10.48550/ARXIV.1007.1075}
}

@article{ledoit2004,
  author  = {Ledoit, Olivier and Wolf, Michael},
  title   = {A well-conditioned estimator for large-dimensional covariance matrices},
  journal = {Journal of Multivariate Analysis},
  volume  = {88},
  number  = {2},
  pages   = {365--411},
  year    = {2004},
  doi     = {10.1016/S0047-259X(03)00096-4}
}

@misc{lovdal2022,
  author = {L{\"o}vdal, S. Sofie and Ruiz-Lara, Tom{\'a}s and Koppelman, Helmer H. and Matsuno, Tadafumi and Dodd, Emma and Helmi, Amina},
  title  = {Substructure in the stellar halo near the Sun. I. Data-driven clustering in Integrals of Motion space},
  year   = {2022},
  eprint = {2201.02404},
  archivePrefix = {arXiv},
  doi    = {10.48550/ARXIV.2201.02404}
}

@article{phipson2010,
  author  = {Phipson, Belinda and Smyth, Gordon K},
  title   = {Permutation P-values Should Never Be Zero: Calculating Exact P-values When Permutations Are Randomly Drawn},
  journal = {Statistical Applications in Genetics and Molecular Biology},
  volume  = {9},
  number  = {1},
  year    = {2010},
  doi     = {10.2202/1544-6115.1585}
}

@article{besag1991,
  author  = {Besag, Julian and Clifford, Peter},
  title   = {Sequential Monte Carlo p-values},
  journal = {Biometrika},
  volume  = {78},
  number  = {2},
  pages   = {301--304},
  year    = {1991},
  doi     = {10.1093/biomet/78.2.301}
}

@article{clopper1934,
  author  = {Clopper, C. J. and Pearson, E. S.},
  title   = {The use of confidence or fiducial limits illustrated in the case of the binomial},
  journal = {Biometrika},
  volume  = {26},
  number  = {4},
  pages   = {404--413},
  year    = {1934},
  doi     = {10.1093/biomet/26.4.404}
}

@article{fred2005,
  author  = {Fred, Ana L. N. and Jain, Anil K.},
  title   = {Combining multiple clusterings using evidence accumulation},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {27},
  number  = {6},
  pages   = {835--850},
  year    = {2005},
  doi     = {10.1109/TPAMI.2005.113}
}

@article{monti2003,
  author  = {Monti, Stefano and Tamayo, Pablo and Mesirov, Jill and Golub, Todd},
  title   = {Consensus Clustering: A Resampling-Based Method for Class Discovery and Visualization of Gene Expression Microarray Data},
  journal = {Machine Learning},
  volume  = {52},
  number  = {1-2},
  pages   = {91--118},
  year    = {2003},
  doi     = {10.1023/A:1023949509487}
}

@article{owen1998,
  author  = {Owen, Art B.},
  title   = {Latin supercube sampling for very high-dimensional simulations},
  journal = {ACM Transactions on Modeling and Computer Simulation},
  volume  = {8},
  number  = {1},
  pages   = {71--102},
  year    = {1998},
  doi     = {10.1145/272991.273010}
}

@inproceedings{golovin2017,
  author    = {Golovin, Daniel and Solnik, Benjamin and Moitra, Subhodeep and Kochanski, Greg and Karro, John and Sculley, D.},
  title     = {Google Vizier: A Service for Black-Box Optimization},
  booktitle = {Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  publisher = {ACM},
  year      = {2017},
  pages     = {1487--1495},
  doi       = {10.1145/3097983.3098043}
}

@article{glasserman1992,
  author  = {Glasserman, Paul and Yao, David D.},
  title   = {Some Guidelines and Guarantees for Common Random Numbers},
  journal = {Management Science},
  volume  = {38},
  number  = {6},
  pages   = {884--908},
  year    = {1992},
  doi     = {10.1287/mnsc.38.6.884}
}

@misc{huntreffert2020,
  author = {Hunt, Emily L. and Reffert, Sabine},
  title  = {Improving the open cluster census. I. Comparison of clustering algorithms applied to Gaia DR2 data},
  year   = {2020},
  eprint = {2012.04267},
  archivePrefix = {arXiv},
  doi    = {10.48550/ARXIV.2012.04267}
}

@misc{hunt2025,
  author = {Hunt, Emily L. and Cantat-Gaudin, Tristan and Anders, Friedrich and Spina, Lorenzo and Cavallo, Lorenzo and Castro-Ginard, Alfred and Belokurov, Vasily and Brown, Anthony G. A. and Casey, Andrew R. and Drimmel, Ronald and Fouesneau, Morgan and Reffert, Sabine},
  title  = {The completeness of the open cluster census towards the Galactic anticentre},
  year   = {2025},
  eprint = {2506.18708},
  archivePrefix = {arXiv},
  doi    = {10.48550/ARXIV.2506.18708}
}

@misc{kronemartins2013,
  author = {Krone-Martins, A. and Moitinho, A.},
  title  = {UPMASK: unsupervised photometric membership assignment in stellar clusters},
  year   = {2013},
  eprint = {1309.4471},
  archivePrefix = {arXiv},
  doi    = {10.48550/ARXIV.1309.4471}
}
```

The Hunt and Reffert entry carries the arXiv year (2020). The journal version is Astronomy & Astrophysics 646, A104 (2021), [doi:10.1051/0004-6361/202039341](https://doi.org/10.1051/0004-6361/202039341), as returned by Crossref on 2026-09-30.
:::
