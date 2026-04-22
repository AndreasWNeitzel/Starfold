"""Persist and reload :class:`PipelineResult` artefacts.

A saved result lives in a directory holding:

* ``arrays.npz`` -- the numeric arrays (embedding, labels, probabilities,
  persistence, significant, scaler mean/scale, per-realisation baseline),
* ``meta.json`` -- scalar metrics, the best Optuna parameters, the
  pipeline config, and the noise-baseline config.

The Optuna ``study`` is *not* persisted: rebuilding it requires the
trial sequence, and the practical usage is to reload a result for
plotting or further analysis, not to resume optimisation. If you need
the study later, keep the original :class:`PipelineResult` alive or
re-run :meth:`UnsupervisedPipeline.fit` with the same seed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from starfold.pipeline import PipelineResult

__all__ = ["load_pipeline_result", "save_pipeline_result"]


def _scaler_state(scaler: StandardScaler) -> dict[str, Any]:
    return {
        "mean_": None if scaler.mean_ is None else np.asarray(scaler.mean_).tolist(),
        "scale_": None if scaler.scale_ is None else np.asarray(scaler.scale_).tolist(),
        "var_": None if scaler.var_ is None else np.asarray(scaler.var_).tolist(),
        "n_features_in_": int(scaler.n_features_in_),
        "n_samples_seen_": int(scaler.n_samples_seen_),
    }


_INT_PARAM_KEYS = frozenset({"min_cluster_size", "min_samples"})
_FLOAT_PARAM_KEYS = frozenset({"cluster_selection_epsilon", "alpha"})


def _coerce_best_params(raw: dict[str, Any]) -> dict[str, Any]:
    """Restore Optuna ``best_params`` with the right Python types.

    HDBSCAN's search now spans integer, float, and categorical axes, so
    a single ``int`` cast no longer round-trips. Known keys are coerced
    explicitly; unknown keys pass through untouched so future axes
    round-trip without a loader change.
    """
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if k in _INT_PARAM_KEYS:
            out[k] = int(v)
        elif k in _FLOAT_PARAM_KEYS:
            out[k] = float(v)
        else:
            out[k] = v
    return out


def _scaler_from_state(state: dict[str, Any]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = None if state["mean_"] is None else np.asarray(state["mean_"], dtype=np.float64)
    scaler.scale_ = (
        None if state["scale_"] is None else np.asarray(state["scale_"], dtype=np.float64)
    )
    scaler.var_ = None if state["var_"] is None else np.asarray(state["var_"], dtype=np.float64)
    scaler.n_features_in_ = int(state["n_features_in_"])
    scaler.n_samples_seen_ = int(state["n_samples_seen_"])
    return scaler


def save_pipeline_result(result: PipelineResult, directory: Path | str) -> Path:
    """Persist ``result`` under ``directory`` and return the path.

    Parameters
    ----------
    result : PipelineResult
        The result to persist.
    directory : path-like
        Destination directory. Created if missing.

    Returns
    -------
    Path
        The directory that now holds ``arrays.npz`` and ``meta.json``.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {
        "embedding": np.asarray(result.embedding, dtype=np.float64),
        "labels": np.asarray(result.labels, dtype=np.intp),
        "probabilities": np.asarray(result.probabilities, dtype=np.float64),
        "persistence": np.asarray(result.persistence, dtype=np.float64),
    }
    if result.significant is not None:
        arrays["significant"] = np.asarray(result.significant, dtype=bool)
    if result.noise_baseline is not None:
        arrays["noise_per_realisation_max"] = np.asarray(
            result.noise_baseline.per_realisation_max, dtype=np.float64
        )
        arrays["noise_per_realisation_n_clusters"] = np.asarray(
            result.noise_baseline.per_realisation_n_clusters, dtype=np.intp
        )
        arrays["noise_per_realisation_objective"] = np.asarray(
            result.noise_baseline.per_realisation_objective, dtype=np.float64
        )
        arrays["noise_null_cluster_persistence"] = np.asarray(
            result.noise_baseline.null_cluster_persistence, dtype=np.float64
        )
        arrays["noise_null_cluster_size"] = np.asarray(
            result.noise_baseline.null_cluster_size, dtype=np.intp
        )
        arrays["noise_null_cluster_realisation"] = np.asarray(
            result.noise_baseline.null_cluster_realisation, dtype=np.intp
        )
    if result.credibility is not None:
        arrays["credibility_observed_cluster_persistence"] = np.asarray(
            result.credibility.observed_cluster_persistence, dtype=np.float64
        )
        arrays["credibility_per_cluster_pvalue"] = np.asarray(
            result.credibility.per_cluster_pvalue, dtype=np.float64
        )
        arrays["credibility_per_cluster_significant"] = np.asarray(
            result.credibility.per_cluster_significant, dtype=bool
        )
    np.savez(directory / "arrays.npz", **arrays)  # type: ignore[arg-type]

    meta: dict[str, Any] = {
        "trustworthiness": float(result.trustworthiness),
        "continuity": float(result.continuity),
        "n_clusters": int(result.n_clusters),
        "best_params": dict(result.best_params),
        "best_persistence_sum": float(result.search.best_persistence_sum),
        "scaler": _scaler_state(result.scaler),
        "pipeline_config": result.config,
        "flags": list(result.flags),
    }
    if result.noise_baseline is not None:
        meta["noise_baseline"] = {
            "threshold": float(result.noise_baseline.threshold),
            "percentile": float(result.noise_baseline.percentile),
            "config": result.noise_baseline.config,
        }
    if result.credibility is not None:
        meta["credibility"] = {
            "observed_n_clusters": int(result.credibility.observed_n_clusters),
            "n_clusters_pvalue": float(result.credibility.n_clusters_pvalue),
            "observed_objective": float(result.credibility.observed_objective),
            "objective_pvalue": float(result.credibility.objective_pvalue),
            "objective_name": str(result.credibility.objective_name),
            "observed_max_persistence": float(result.credibility.observed_max_persistence),
            "max_persistence_pvalue": float(result.credibility.max_persistence_pvalue),
            "alpha": float(result.credibility.alpha),
            "passes": bool(result.credibility.passes),
            "per_cluster_significant_count": int(
                result.credibility.per_cluster_significant.sum()
            ),
            "per_cluster_total": int(result.credibility.per_cluster_significant.size),
        }
    (directory / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    return directory


def load_pipeline_result(directory: Path | str) -> dict[str, Any]:
    """Load a saved pipeline result into a plain dict.

    The returned dict keys are ``embedding``, ``labels``,
    ``probabilities``, ``persistence``, ``significant`` (if saved),
    ``trustworthiness``, ``n_clusters``, ``best_params``,
    ``best_persistence_sum``, ``scaler`` (a rehydrated
    :class:`StandardScaler`), ``pipeline_config``, and -- when a noise
    baseline was computed -- ``noise_threshold``, ``noise_percentile``,
    ``noise_per_realisation_max``, ``noise_per_realisation_n_clusters``,
    ``noise_per_realisation_objective``, ``noise_null_cluster_persistence``,
    ``noise_null_cluster_size``, ``noise_null_cluster_realisation``,
    ``noise_config``, and ``credibility`` (a plain dict with the
    scalar p-values and verdict) plus per-cluster arrays
    ``credibility_observed_cluster_persistence``,
    ``credibility_per_cluster_pvalue`` and
    ``credibility_per_cluster_significant``.

    The Optuna study is not reconstructed (see module docstring).
    """
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text())
    with np.load(directory / "arrays.npz", allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}

    out: dict[str, Any] = {
        "embedding": arrays["embedding"].astype(np.float64),
        "labels": arrays["labels"].astype(np.intp),
        "probabilities": arrays["probabilities"].astype(np.float64),
        "persistence": arrays["persistence"].astype(np.float64),
        "trustworthiness": float(meta["trustworthiness"]),
        "continuity": float(meta.get("continuity", float("nan"))),
        "n_clusters": int(meta["n_clusters"]),
        "best_params": _coerce_best_params(meta["best_params"]),
        "best_persistence_sum": float(meta["best_persistence_sum"]),
        "scaler": _scaler_from_state(meta["scaler"]),
        "pipeline_config": meta["pipeline_config"],
        "flags": list(meta.get("flags", [])),
    }
    if "significant" in arrays:
        out["significant"] = arrays["significant"].astype(bool)
    _load_noise_baseline_block(out, meta, arrays)
    _load_credibility_block(out, meta, arrays)
    return out


def _load_noise_baseline_block(
    out: dict[str, Any],
    meta: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> None:
    if "noise_baseline" not in meta:
        return
    out["noise_threshold"] = float(meta["noise_baseline"]["threshold"])
    out["noise_percentile"] = float(meta["noise_baseline"]["percentile"])
    out["noise_config"] = meta["noise_baseline"]["config"]
    out["noise_per_realisation_max"] = arrays["noise_per_realisation_max"].astype(np.float64)
    int_keys = (
        "noise_per_realisation_n_clusters",
        "noise_null_cluster_size",
        "noise_null_cluster_realisation",
    )
    float_keys = (
        "noise_per_realisation_objective",
        "noise_null_cluster_persistence",
    )
    for key in int_keys:
        if key in arrays:
            out[key] = arrays[key].astype(np.intp)
    for key in float_keys:
        if key in arrays:
            out[key] = arrays[key].astype(np.float64)


def _load_credibility_block(
    out: dict[str, Any],
    meta: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> None:
    if "credibility" not in meta:
        return
    out["credibility"] = dict(meta["credibility"])
    float_keys = (
        "credibility_observed_cluster_persistence",
        "credibility_per_cluster_pvalue",
    )
    for key in float_keys:
        if key in arrays:
            out[key] = arrays[key].astype(np.float64)
    if "credibility_per_cluster_significant" in arrays:
        out["credibility_per_cluster_significant"] = (
            arrays["credibility_per_cluster_significant"].astype(bool)
        )
