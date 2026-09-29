# Starfold

Starfold finds groups in a numeric table without labels, then measures how far those groups can be trusted. It searches UMAP and HDBSCAN settings with Optuna, and the library defaults (clipped into any bounds you set) are always the first setting tried. Each setting is scored without labels on three things: its groups must predict rows the fit never saw, it must assign most rows, and its groups must be neighbourhoods in the input columns. Every score is taken relative to the same setting on a copy of the table that keeps each column's values and their correlations but has no groups. After one fit of the best setting on all rows, 32 refits on random 80 percent subsets measure how often each row and each group comes back. An optional calibration test asks whether such copies score as high as the real table. Anything the evidence does not support comes back as a flag.

![Digits: true classes, Starfold, library defaults](docs/figures/digits_partitions.png)

*Handwritten digits (1,797 rows, 64 pixels) on a 2-D UMAP display, showing the true digits (never seen by Starfold), Starfold's groups (faded where support is low) and the UMAP and HDBSCAN defaults.*

## Install

Python 3.11 or 3.12.

```bash
pip install "starfold @ git+https://github.com/AndreasWNeitzel/Starfold"             # CPU
pip install "starfold[gpu-cu13] @ git+https://github.com/AndreasWNeitzel/Starfold"   # NVIDIA GPU, CUDA 13
pip install "starfold[gpu-cu12] @ git+https://github.com/AndreasWNeitzel/Starfold"   # NVIDIA GPU, CUDA 12
```

The GPU extras install RAPIDS cuML (Linux or WSL2); choose the one that matches the CUDA version reported by `nvidia-smi`. Running on cuML, Starfold performs UMAP, HDBSCAN and the scoring on the GPU and only the final consensus cut on the CPU.

## Use

```python
import starfold
from sklearn.datasets import load_digits

X = load_digits().data                                 # 1,797 rows, 64 columns; the digit labels are not used
result = starfold.recover(X, random_state=0)

print(result.flags)                                    # read these first
print(result.n_groups, result.worlds.count_interval)   # groups found; 5 to 95 percent range over refits
result.labels                                          # -1 is unassigned; one entry per row in result.kept_rows
result.support                                         # per row and group: how often the refits agree
starfold.save(result, "runs/digits")                   # starfold.load("runs/digits") reads it back
```

`recover` skips the calibration test by default. The test costs 102 fits per copy at the default settings, with up to 19 copies (1,938 fits), about six times the fits of the rest of a run. Given that cost, run it once, on the result you report, with the same `config`, `random_state` and `engine` that `recover` used:

```python
null = starfold.null_test(X[result.kept_rows], result.search, random_state=0)
print(null.p_value, null.structure_supported)
```

To run the test inside `recover` instead, pass `config=starfold.Config(null_families=("gaussian_copula",))` and read `result.null.p_value`. On Digits, `recover` makes 314 fits (a few minutes on a laptop GPU), against up to 1,938 for `null_test`.

`engine="auto"` uses cuML when it imports and sees a device and otherwise falls back to the CPU with a warning; `engine="cuml"` raises instead of falling back. On the CPU, `n_jobs` sets the number of worker processes. If you have real per-value uncertainties, pass them as `X_err` (same shape as `X`) to have every refit redraw the table from them.

## What a result holds

| attribute | meaning |
|---|---|
| `labels`, `kept_rows` | one fit of the selected settings on all rows; rows with NaN or inf are dropped, never filled in |
| `params`, `deployed_params` | the selected settings, and the same settings with `min_cluster_size` and `min_samples` scaled to the full row count |
| `default_labels`, `default_params` | the UMAP and HDBSCAN defaults on the same rows, for comparison |
| `support`, `worlds.unresolved` | per row: how often the refits that sampled it put it with each group, or with none; each row sums to one |
| `worlds.group_recurrence`, `worlds.group_recurrence_band` | per group: how well it comes back across refits (0 to 1), and its 5 to 95 percent range |
| `worlds.count_values`, `worlds.count_interval` | groups found in each refit, and their 5 to 95 percent range |
| `worlds.property_median`, `worlds.property_band` | per group and column: the median in your units, and its spread over refits |
| `consensus.contrast`, `consensus.agreement` | how much more often the refits cluster rows of one group together than rows of different groups; how well a partition cut from that matches `labels` |
| `null` | the calibration test when `Config.null_families` names a family; skipped by default |
| `search.trials`, `search.candidates` | every setting tried with its score; the best few rescored on fresh subsets |
| `flags`, `provenance` | caveats; seeds, versions, engine, budgets, fits and seconds per stage |

## Reading the flags

`result.flags` names every caveat the evidence raised, drawn from a fixed list of 16 (`starfold.result.FLAG_VOCABULARY`). Flags are derived from the arrays, and `starfold.load` derives them again instead of trusting the saved list. Those you will meet most often:

- `search_stopped_early` or `search_budget_exhausted`: one of the two is always present and says only how the search ended.
- `null_not_run`: the calibration test did not run inside `recover` (the default). A separate `null_test` returns its own evidence and leaves this flag in place.
- `selected_is_default`, `default_within_noise`: the search found nothing better than the defaults, or nothing better beyond the noise of the score.
- `count_unstable`, `high_unresolved_mass`: most refits find a different number of groups than `labels` has, or rows are on average left unassigned in more than half the refits.
- `consensus_disagrees`: the partition cut from the refits agrees poorly with `labels`.
- `no_structure_supported`: the calibration test ran and its p-value is above `alpha`.

The [handbook](docs/handbook.md) lists all 16, each with its meaning and what to do about it.

## Scope and limits

- **Input.** One row per object, with at least 50 finite rows and 2 numeric columns. The columns you choose define what "similar" means, and Starfold scales them (`Config(scaling="robust")` for heavy tails). Rows with a missing value are dropped, never filled in.
- **Small groups merge.** Groups much smaller than about sqrt(12 n) rows (about 155 rows at n = 2,000 and 350 at n = 10,000) tend to be merged into larger ones. This is an empirical scale, not a derived bound: we measured it on our benchmark and by injecting groups into a real table. Treat it as a rule of thumb.
- **Stable is not true.** Support is a recurrence rate, not a membership probability, and the ranges over refits are stability intervals, not confidence intervals. A dense chance fluctuation recurs in every subset too. There is no per-group significance test.
- **Counts.** The group count is not an estimate of how many populations exist; on our benchmark the count can differ from the true class count (Gas drift: 390 groups against 6 classes; Letter: 284 against 26). Report `worlds.count_interval` with it.
- **The p-value** holds only for its copy family (by default a Gaussian copula: one population with the table's column values and correlations) and for the part of the search it replays: by default the library defaults and the 16 Sobol points that open the search, and the best score among them. `Config(null_procedure="full_search")` replays the whole search at a higher cost. With 19 copies the p-value cannot go below 0.05. It is not the probability that the groups are real.
- **Evidence.** We compared Starfold with the UMAP and HDBSCAN defaults against the true classes of ten labelled public tables, using the adjusted Rand index on the assigned rows times the share assigned. We used six of the tables to tune the method and held four out. Starfold matched or beat the defaults (within 0.02 counts as a match) on 8 of the ten: all six development tables and two of the four held out. However, on the four held-out tables its mean was lower than the defaults' (0.24 against 0.29). The largest table has 20,000 rows; nothing larger has been measured. CPU and GPU runs do not give identical labels; compare them by partition with `starfold.compare_backends`.

## Documentation

- [Handbook](docs/handbook.md): how it works, preparing a table, reading a result, every flag, GPU and cluster use, exact definitions.
- Notebooks, in Jupytext percent format: [handwritten digits](notebooks/01_digits.py), [9,242 APOGEE DR19 stars](notebooks/02_apogee.py), [the calibration test](notebooks/03_calibration.py). The APOGEE table ships with the repository, not the package ([provenance](bench/data/stellar_chemokinematics_apogee_dr19.md)).
- [Changelog](CHANGELOG.md).

## Tests

```bash
git clone https://github.com/AndreasWNeitzel/starfold && cd starfold
uv sync --extra dev --extra bench    # add --extra gpu-cu13 or --extra gpu-cu12 for cuML
uv run pytest -m "not slow"          # quick checks
uv run pytest -m slow                # real fits on small fixtures
uv run pytest -m gpu                 # needs cuML
```

## Citation

Cite the software ([CITATION.cff](CITATION.cff)) and the paper it grew from:

Neitzel, A. W., Campante, T. L., Bossini, D., and Miglio, A. 2025, "Dissecting stellar populations with manifold learning. I. Validation of the method on a synthetic Milky Way-like galaxy", A&A, 695, A243, doi:10.1051/0004-6361/202451718

Please also cite UMAP (McInnes, Healy and Melville 2018, arXiv:1802.03426), HDBSCAN (Campello, Moulavi and Sander 2013, doi:10.1007/978-3-642-37456-2_14) and Optuna (Akiba et al. 2019, arXiv:1907.10902).

## License

MIT, see [LICENSE](LICENSE).
