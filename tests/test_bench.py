"""Cheap checks of the bench: loaders, record summaries, figures and argument guards.

No test here fits UMAP or HDBSCAN. Arrays built inline are unit-test fixtures.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from bench import backend_agreement, controls, datasets, injection_recovery, null_stage, summarize
from bench.beats_default import trial_spearman
from sklearn.datasets import load_digits

from starfold import Config

RESULTS = Path(__file__).resolve().parent.parent / "bench" / "results"


def test_every_bench_module_imports() -> None:
    for name in (
        "backend_agreement",
        "beats_default",
        "controls",
        "datasets",
        "injection_recovery",
        "null_stage",
        "summarize",
        "synthetic",
    ):
        importlib.import_module(f"bench.{name}")


def test_example_table_matches_provenance() -> None:
    x, names = datasets.example_table()
    assert x.shape == (datasets.EXAMPLE_ROWS, len(datasets.EXAMPLE_FEATURES))
    assert names == datasets.EXAMPLE_FEATURES
    assert np.isfinite(x).all()


def test_rows_covers_apogee_and_named_datasets() -> None:
    assert np.array_equal(datasets.rows("apogee"), datasets.example_table()[0])
    assert np.array_equal(datasets.rows("digits"), load_digits().data)
    with pytest.raises(ValueError, match="Unknown dataset"):
        datasets.rows("no_such_table")


def test_each_dataset_sits_in_exactly_one_registry() -> None:
    assert not set(datasets.DEVELOPMENT) & set(datasets.SEALED)
    assert set(datasets.LOADERS) == set(datasets.DEVELOPMENT) | set(datasets.SEALED)
    assert "waveform" in datasets.DEVELOPMENT


def test_committed_summary_matches_the_committed_records() -> None:
    committed = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    assert json.loads(json.dumps(summarize.summarize(RESULTS))) == committed


def test_summary_of_an_empty_directory_is_empty(tmp_path: Path) -> None:
    summary = summarize.summarize(tmp_path)
    assert summary["datasets"] == []
    assert summary["means"] == {}
    assert summary["null"] == summary["injection"] == summary["backends"] == []
    assert summary["controls"] == []


def test_summary_means_split_development_and_sealed(tmp_path: Path) -> None:
    def record(name: str, selected: float, default: float) -> dict[str, object]:
        return {
            "dataset": name,
            "n_rows": 100,
            "n_features": 4,
            "n_classes": 2,
            "selected": {
                "coverage_ari": selected,
                "assigned_fraction": 1.0,
                "n_groups": 2,
                "mixture_fraction": float("nan"),
            },
            "default_median_coverage_ari": default,
            "count_interval": [2, 2],
            "unresolved_mass": 0.0,
            "spearman_score_ari": float("nan"),
            "total_fits": 10,
            "peak_rss_mb": 1.0,
            "engine": "cpu",
        }

    for name, selected, default in (("rice", 0.5, 0.1), ("digits", 0.2, 0.3), ("letter", 0.4, 0.4)):
        (tmp_path / f"{name}.json").write_text(json.dumps(record(name, selected, default)))
    summary = summarize.summarize(tmp_path)
    assert [r["dataset"] for r in summary["datasets"]] == ["rice", "digits", "letter"]
    assert summary["datasets"][0]["mixture_fraction"] is None
    assert summary["means"]["development"]["n"] == 2
    assert summary["means"]["development"]["at_or_above_default"] == 1
    assert summary["means"]["sealed"]["at_or_above_default"] == 1
    assert summary["means"]["all"]["coverage_ari"] == pytest.approx(1.1 / 3)
    assert summary["datasets"][0]["selector"] is None
    assert summary["means"]["all"]["selector"] is None


def _trial_rows(null: list[float]) -> list[dict[str, float]]:
    raw = [0.9, 0.6, 0.5, 0.4]
    ari = [0.2, 0.7, 0.5, 0.1]
    return [
        {
            "gap": r - n if np.isfinite(n) else r,
            "raw_score": r,
            "null_score": n,
            "coverage_ari": a,
        }
        for r, n, a in zip(raw, null, ari, strict=True)
    ]


def test_selector_reads_regret_and_each_ranking_from_the_trial_rows() -> None:
    null = [0.8, 0.2, 0.2, 0.2]
    block = summarize.selector(
        {"selected": {"coverage_ari": 0.5}, "trial_diagnostics": _trial_rows(null)}
    )
    assert block is not None
    assert block["n_trials"] == 4
    assert block["best_trial_ari"] == pytest.approx(0.7)
    assert block["regret"] == pytest.approx(0.2)
    assert block["ari_at_gap_argmax"] == pytest.approx(0.7)
    assert block["ari_at_raw_argmax"] == pytest.approx(0.2)
    assert block["beta"] == pytest.approx(np.polyfit(null, [0.9, 0.6, 0.5, 0.4], 1)[0])
    assert block["ari_at_control_argmax"] == pytest.approx(0.7)
    assert block["rho_raw"] == pytest.approx(0.4)


def test_selector_without_null_scores_ranks_the_control_as_the_raw_score() -> None:
    record = {"selected": {"coverage_ari": 0.5}, "trial_diagnostics": _trial_rows([np.nan] * 4)}
    block = summarize.selector(record)
    assert block is not None
    assert block["beta"] == 0.0
    assert block["rho_control"] == block["rho_raw"]
    assert block["ari_at_control_argmax"] == block["ari_at_raw_argmax"] == pytest.approx(0.2)
    record["trial_diagnostics"] = record["trial_diagnostics"][:2]
    assert summarize.selector(record) is None


def test_summary_means_the_selector_over_records_with_trials() -> None:
    committed = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    blocks = [r["selector"] for r in committed["datasets"] if r["selector"]]
    means = committed["means"]["all"]["selector"]
    assert means["n"] == len(blocks)
    assert means["regret"] == pytest.approx(np.mean([b["regret"] for b in blocks]))


def test_saved_results_default_to_the_root_and_apogee_to_the_tutorial() -> None:
    root = Path("output/bench")
    paths = backend_agreement.saved_results(["digits", "apogee"], root, [])
    assert paths == {"digits": root / "digits", "apogee": root / "apogee"}
    paths = backend_agreement.saved_results(["digits", "apogee"], root, ["apogee=elsewhere"])
    assert paths["apogee"] == Path("elsewhere")
    for bad in ("apogee", "=elsewhere", "apogee=", "rice=elsewhere"):
        with pytest.raises(ValueError, match="--result"):
            backend_agreement.saved_results(["digits", "apogee"], root, [bad])


def test_figures_draw_from_the_committed_summary(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    figures = importlib.import_module("bench.figures")
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    for name, draw in (
        ("resolution", figures.figure_resolution),
        ("injection", figures.figure_injection),
    ):
        draw(summary).savefig(tmp_path / f"{name}.png")
        assert (tmp_path / f"{name}.png").stat().st_size > 0
    assert figures.modularity_scale(np.array([2000.0]))[0] == pytest.approx(np.sqrt(24_000.0))


def test_merges_and_splits_enumerate_every_alternative() -> None:
    truth = np.repeat(np.arange(3, dtype=np.int64), 4)
    x = np.arange(24, dtype=np.float64).reshape(12, 2)
    merged = controls.merges(truth)
    assert [name for name, _ in merged] == ["merge_0_1", "merge_0_2", "merge_1_2"]
    assert all(np.unique(labels).size == 2 for _, labels in merged)
    split = controls.splits(x, truth)
    assert [name for name, _ in split] == ["split_0", "split_1", "split_2"]
    assert all(np.count_nonzero(labels == 3) == 2 for _, labels in split)


def test_single_group_has_no_merge() -> None:
    assert controls.merges(np.zeros(6, dtype=np.int64)) == []


@pytest.mark.parametrize("engine", ["auto", "gpu", "CPU"])
def test_stage_helpers_refuse_an_unresolved_engine(engine: str) -> None:
    x = np.zeros((8, 2))
    truth = np.zeros(8, dtype=np.int64)
    with pytest.raises(ValueError, match="resolve 'auto' first"):
        controls.exact_partition_ranking(x, truth, seed=0, engine=engine)
    with pytest.raises(ValueError, match="resolve 'auto' first"):
        trial_spearman(None, x, truth, engine=engine, scaling="standard", random_state=0)


def test_retrieval_matches_injected_groups_by_jaccard() -> None:
    injection = np.array([-1, -1, 0, 0, 0, 1, 1, 1], dtype=np.int64)
    labels = np.array([0, 0, 1, 1, 1, 2, 2, -1], dtype=np.int32)
    support = np.ones(8, dtype=np.float32)
    summary = injection_recovery.retrieval(injection, labels, support, (0.0,))
    assert summary["retrieved"] == [True, True]
    assert summary["matched_group"] == [1, 2]
    assert summary["best_jaccard"] == pytest.approx([1.0, 2.0 / 3.0])


def test_inject_adds_contrast_times_k_rows_per_group() -> None:
    x = np.random.default_rng(0).normal(size=(200, 3))
    augmented, injection = injection_recovery.inject(x, n_groups=2, contrast=2.0, k=10, seed=0)
    assert augmented.shape == (240, 3)
    assert np.bincount(injection[injection >= 0]).tolist() == [20, 20]


def test_config_from_provenance_drops_space_and_unknown_keys() -> None:
    config = null_stage.config_from_provenance(
        {"n_trials": 20, "n_explore": 8, "space": {"n_neighbors": [5, 50]}, "retired": 1}
    )
    assert config == Config(n_trials=20, n_explore=8)


def test_searched_ranking_reads_the_best_trial() -> None:
    rows: list[dict[str, object]] = [
        {"gap": 0.1, "coverage_ari": 0.2, "n_groups": 5},
        {"gap": 0.4, "coverage_ari": 0.9, "n_groups": 3},
        {"gap": 0.3, "coverage_ari": 0.5, "n_groups": 4},
    ]
    ranking = controls.searched_partition_ranking(rows, 0.5)
    assert ranking["best_trial_count"] == 3
    assert ranking["best_trial_ari"] == 0.9
    assert controls.searched_partition_ranking([], float("nan"))["best_trial_count"] is None


def test_exact_ranking_puts_separated_truth_first_on_cpu() -> None:
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(centre, 0.3, size=(60, 3)) for centre in (0.0, 6.0, 12.0)])
    truth = np.repeat(np.arange(3, dtype=np.int64), 60)
    ranking = controls.exact_partition_ranking(x, truth, seed=0, engine="cpu")
    assert set(ranking["scores"]) == {"truth", "merge_0_1", "merge_0_2", "merge_1_2"} | {
        f"split_{g}" for g in range(3)
    }
    assert ranking["truth_ranks_first"]


def test_exact_ranking_of_one_group_has_only_a_split() -> None:
    x = np.random.default_rng(1).normal(size=(80, 3))
    ranking = controls.exact_partition_ranking(
        x, np.zeros(80, dtype=np.int64), seed=0, engine="cpu"
    )
    assert ranking["max_merge"] is None
    assert set(ranking["scores"]) == {"truth", "split_0"}
