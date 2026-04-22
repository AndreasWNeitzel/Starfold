"""Input-uncertainty propagation via Monte Carlo over the fitted pipeline.

Motivation
----------
A real dataset almost never arrives without per-feature error bars.
Treating every sample as a point estimate and reporting a hard cluster
assignment hides the fact that some samples sit near a cluster boundary
and would flip assignment under modest feature-level noise.

This module answers: "given per-feature 1-sigma uncertainties on ``X``,
how confident is each sample's cluster assignment?" The answer is a
membership-probability vector per sample, obtained by Monte Carlo:

1. For each of ``n_draws`` iterations, perturb ``X`` by independent
   Gaussian noise with the user-supplied standard deviation.
2. Transform the perturbed sample through the *already fitted* pipeline:
   :class:`StandardScaler.transform` -> :func:`umap.UMAP.transform` ->
   :func:`hdbscan.approximate_predict`. No refitting; the pipeline's
   manifold and clustering stay fixed, which is the usual semantics for
   "how noisy is this assignment given this model?".
3. Tally each sample's label frequency across draws.

The output is a :class:`UncertaintyPropagation` holding the full
membership matrix, a consensus label per sample, and a scalar
instability score (``1 - max(membership)``).

Scope
-----
CPU backend only on this release. The cuml UMAP and HDBSCAN
implementations expose ``.transform`` / ``approximate_predict`` but
under different conventions, and no test harness for the GPU path
exists yet; :func:`propagate_uncertainty` raises :class:`NotImplementedError`
when called with a cuml-trained pipeline.

See ``docs/design_decisions.md`` for the rationale behind the sampling
geometry and the decision to *not* refit per draw.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import hdbscan as _hdbscan
import numpy as np
from joblib import Parallel, delayed

if TYPE_CHECKING:
    import umap as _umap_typing
    from numpy.typing import NDArray
    from sklearn.preprocessing import StandardScaler

    from starfold.pipeline import PipelineResult

__all__ = [
    "UncertaintyAwareFit",
    "UncertaintyPropagation",
    "build_replica_augmented_matrix",
    "consensus_from_augmented_labels",
    "propagate_uncertainty",
]


@dataclass(frozen=True)
class UncertaintyPropagation:
    """Result of propagating feature uncertainties through a fit.

    Parameters
    ----------
    membership
        ``(n_samples, n_clusters + 1)`` matrix. Column ``j`` for
        ``0 <= j < n_clusters`` gives the fraction of draws in which
        sample ``i`` landed in cluster ``j``; the last column
        (index ``n_clusters``) is the outlier fraction.
    consensus_label
        Per-sample argmax over ``membership``; ``-1`` when the outlier
        column dominates. Length ``n_samples``.
    instability
        ``1 - max(membership, axis=1)`` per sample. A sample with
        ``instability = 0`` was assigned to the same cluster in every
        draw; a sample with ``instability = 0.5`` split its assignments
        50/50 between two clusters.
    n_draws
        Number of Monte Carlo draws used.
    sigma_shape
        Shape of the ``sigma`` argument that produced this propagation
        (``"scalar"``, ``"per_feature"``, or ``"per_sample_feature"``).
        Kept for diagnostic plotting.
    config
        Frozen record of the propagation inputs (``n_draws``,
        ``random_state``, ``sigma_shape``, ``sigma_summary``) so
        downstream code can re-identify an existing propagation and
        :meth:`PipelineResult.save` can persist it.
    """

    membership: NDArray[np.floating[Any]]
    consensus_label: NDArray[np.intp]
    instability: NDArray[np.floating[Any]]
    n_draws: int
    sigma_shape: str
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        """Number of samples whose membership was estimated."""
        return int(self.membership.shape[0])

    @property
    def n_clusters(self) -> int:
        """Number of real-data clusters (excluding the outlier column)."""
        return int(self.membership.shape[1] - 1)

    def summary(self) -> str:
        """Short human-readable summary of instability statistics."""
        mean_instab = float(self.instability.mean()) if self.instability.size else 0.0
        frac_high = float((self.instability > 0.5).mean()) if self.instability.size else 0.0
        flipped = int((self.consensus_label >= 0).sum()) if self.consensus_label.size else 0
        return (
            f"uncertainty propagation over {self.n_draws} draws\n"
            f"  mean instability  {mean_instab:.4f}\n"
            f"  frac instability>0.5  {frac_high:.4f}\n"
            f"  consensus-assigned samples  {flipped}/{self.n_samples}"
        )

    def confident_labels(self, threshold: float = 0.8) -> NDArray[np.intp]:
        """Return consensus labels masked by a per-sample confidence gate.

        A sample keeps its consensus label iff it was assigned to that
        cluster in at least ``threshold`` of draws (i.e. the
        corresponding column of :attr:`membership` exceeds
        ``threshold``). Samples below the gate are returned as ``-1``,
        regardless of whether their consensus was a real cluster or the
        outlier column -- in either case the Monte Carlo says the
        assignment is not confident.

        This is the recommended way to filter to "samples with trusted
        cluster membership" before downstream analysis.

        Parameters
        ----------
        threshold
            Minimum fraction of draws the consensus cluster must
            capture. 0.8 is a reasonable default ("assigned to the same
            cluster in >=80% of draws"). Must lie in ``[0, 1]``.

        Returns
        -------
        ndarray of shape (n_samples,)
            Labels aligned to the original sample order, with ``-1``
            marking low-confidence samples.

        Examples
        --------
        >>> prop = result.propagate_uncertainty(X, sigma=0.1)  # doctest: +SKIP
        >>> confident = prop.confident_labels(threshold=0.9)    # doctest: +SKIP
        >>> mask = confident >= 0                                # doctest: +SKIP
        >>> X_trusted = X[mask]                                  # doctest: +SKIP
        """
        if not (0.0 <= threshold <= 1.0):
            msg = f"threshold must be in [0, 1] (got {threshold})."
            raise ValueError(msg)
        if self.membership.size == 0:
            return np.empty(0, dtype=np.intp)
        max_membership = self.membership.max(axis=1)
        confident = max_membership >= threshold
        return np.where(confident, self.consensus_label, -1).astype(np.intp)


def _broadcast_sigma(
    sigma: float | NDArray[np.floating[Any]],
    *,
    n_samples: int,
    n_features: int,
) -> tuple[NDArray[np.floating[Any]], str]:
    arr = np.asarray(sigma, dtype=np.float64)
    if arr.ndim == 0:
        if not np.isfinite(arr) or arr < 0.0:
            msg = f"sigma must be finite and non-negative (got {float(arr)})."
            raise ValueError(msg)
        return np.full((n_samples, n_features), float(arr), dtype=np.float64), "scalar"
    if arr.ndim == 1:
        if arr.shape[0] != n_features:
            msg = (
                f"sigma of shape {arr.shape} is incompatible with "
                f"n_features={n_features}; expected ({n_features},)."
            )
            raise ValueError(msg)
        if np.any(arr < 0.0) or not np.all(np.isfinite(arr)):
            msg = "sigma must be finite and non-negative."
            raise ValueError(msg)
        return np.broadcast_to(arr, (n_samples, n_features)).astype(np.float64), "per_feature"
    if arr.ndim == 2:
        if arr.shape != (n_samples, n_features):
            msg = (
                f"sigma of shape {arr.shape} is incompatible with "
                f"X of shape ({n_samples}, {n_features})."
            )
            raise ValueError(msg)
        if np.any(arr < 0.0) or not np.all(np.isfinite(arr)):
            msg = "sigma must be finite and non-negative."
            raise ValueError(msg)
        return arr.astype(np.float64, copy=True), "per_sample_feature"
    msg = f"sigma must have ndim 0, 1, or 2 (got ndim={arr.ndim})."
    raise ValueError(msg)


def _one_draw(
    x: NDArray[np.floating[Any]],
    sigma: NDArray[np.floating[Any]],
    *,
    scaler: StandardScaler,
    umap_model: _umap_typing.UMAP,
    hdbscan_model: _hdbscan.HDBSCAN,
    seed: int,
) -> NDArray[np.intp]:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(size=x.shape) * sigma
    x_perturbed = x + noise
    x_scaled = scaler.transform(x_perturbed).astype(np.float32, copy=False)
    emb = np.asarray(umap_model.transform(x_scaled), dtype=np.float64)
    labels, _ = _hdbscan.approximate_predict(hdbscan_model, emb)
    return np.asarray(labels, dtype=np.intp)


def _effective_n_jobs(*, n_jobs: int, n_draws: int) -> int:
    if n_draws < 2:
        return 1
    cpu = os.cpu_count() or 1
    if n_jobs == -1:
        return min(n_draws, cpu)
    return max(1, min(n_jobs, n_draws, cpu))


def _tally_membership(
    draws: NDArray[np.intp],
    *,
    n_clusters: int,
    n_samples: int,
) -> NDArray[np.floating[Any]]:
    # ``draws`` is shape (n_draws, n_samples). For each sample, count
    # the fraction of draws landing in each cluster (or the outlier
    # column at index n_clusters).
    n_draws = int(draws.shape[0])
    membership = np.zeros((n_samples, n_clusters + 1), dtype=np.float64)
    # Remap outliers (label = -1) to column index n_clusters.
    remapped = np.where(draws < 0, n_clusters, draws)
    for s in range(n_samples):
        counts = np.bincount(remapped[:, s], minlength=n_clusters + 1)
        membership[s, :] = counts[: n_clusters + 1]
    membership /= max(n_draws, 1)
    return membership


def propagate_uncertainty(
    X: NDArray[np.floating[Any]],
    sigma: float | NDArray[np.floating[Any]],
    *,
    scaler: StandardScaler,
    umap_model: _umap_typing.UMAP,
    hdbscan_model: _hdbscan.HDBSCAN,
    n_clusters: int,
    n_draws: int = 100,
    random_state: int | None = None,
    n_jobs: int = 1,
) -> UncertaintyPropagation:
    """Monte-Carlo propagate feature uncertainties through a fitted pipeline.

    Parameters
    ----------
    X
        Training-time feature matrix in the *original* (pre-scaling)
        space, shape ``(n_samples, n_features)``. The perturbations are
        applied here, before :meth:`StandardScaler.transform`, so that
        ``sigma`` is interpreted in the feature units the user provided.
    sigma
        Per-feature 1-sigma uncertainties. Accepts a scalar (same
        uncertainty for every feature), a 1-D array of length
        ``n_features``, or a 2-D array of shape ``(n_samples,
        n_features)`` when the uncertainty varies per sample (common in
        astronomy for heteroscedastic observations).
    scaler, umap_model, hdbscan_model
        The fitted components of a :class:`UnsupervisedPipeline` run.
        ``hdbscan_model`` must have been fit with ``prediction_data=True``;
        :func:`search_hdbscan` does this on its final refit.
    n_clusters
        Number of clusters the pipeline produced. Sets the width of the
        membership matrix.
    n_draws
        Number of Monte Carlo perturbations. Defaults to 100; raise for
        tighter confidence intervals on ``instability`` near boundaries.
    random_state
        Seed. Draw ``k`` uses seed ``random_state + k`` when integer,
        yielding bit-reproducible output. ``None`` still runs but the
        draws are not reproducible.
    n_jobs
        Parallel joblib workers across draws. ``-1`` uses every core.
        Each draw is an independent UMAP-transform + approximate_predict
        so the loop is embarrassingly parallel; set this above 1 for
        large ``n_draws``.

    Returns
    -------
    UncertaintyPropagation
        Membership matrix, consensus labels, instability, and config.

    Raises
    ------
    NotImplementedError
        If ``umap_model`` is a cuML UMAP (cuml transform is not part of
        this release; see module docstring).
    ValueError
        If ``sigma`` is negative or not broadcastable to ``X``.

    Examples
    --------
    >>> import numpy as np
    >>> import starfold as sf
    >>> rng = np.random.default_rng(0)
    >>> X = np.vstack([rng.normal(-5, 0.3, size=(80, 3)),
    ...                rng.normal( 5, 0.3, size=(80, 3))])
    >>> pipeline = sf.UnsupervisedPipeline(
    ...     umap_kwargs={"n_epochs": 50, "n_neighbors": 10},
    ...     hdbscan_optuna_trials=4,
    ...     skip_noise_baseline=True,
    ...     engine="cpu", random_state=0,
    ... )
    >>> result = pipeline.fit(X)  # doctest: +SKIP
    >>> prop = result.propagate_uncertainty(X, sigma=0.2, n_draws=10)  # doctest: +SKIP
    >>> prop.membership.shape  # doctest: +SKIP
    (160, 3)  # n_clusters + 1 outlier column
    """
    if not isinstance(umap_model, _cpu_umap_cls()):
        msg = (
            "propagate_uncertainty is CPU-only in this release; got "
            f"{type(umap_model).__module__}.{type(umap_model).__name__}. "
            "Refit the pipeline with engine='cpu' to enable propagation."
        )
        raise NotImplementedError(msg)
    x = np.asarray(X, dtype=np.float64)
    if x.ndim != 2:
        msg = f"X must be a 2-D array (got shape {x.shape})."
        raise ValueError(msg)
    n_samples, n_features = x.shape
    sigma_grid, sigma_shape = _broadcast_sigma(sigma, n_samples=n_samples, n_features=n_features)
    if n_draws < 1:
        msg = f"n_draws must be >= 1 (got {n_draws})."
        raise ValueError(msg)
    base_seed = 0 if random_state is None else int(random_state)
    jobs = _effective_n_jobs(n_jobs=n_jobs, n_draws=n_draws)
    if jobs == 1:
        label_rows = [
            _one_draw(
                x,
                sigma_grid,
                scaler=scaler,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                seed=base_seed + k,
            )
            for k in range(n_draws)
        ]
    else:
        label_rows = list(
            Parallel(n_jobs=jobs, backend="loky", prefer="processes")(
                delayed(_one_draw)(
                    x,
                    sigma_grid,
                    scaler=scaler,
                    umap_model=umap_model,
                    hdbscan_model=hdbscan_model,
                    seed=base_seed + k,
                )
                for k in range(n_draws)
            )
        )
    draws = np.vstack(label_rows).astype(np.intp)
    membership = _tally_membership(draws, n_clusters=n_clusters, n_samples=n_samples)
    argmax = np.argmax(membership, axis=1)
    consensus_label = np.where(argmax == n_clusters, -1, argmax).astype(np.intp)
    instability = (1.0 - membership.max(axis=1)).astype(np.float64)
    config = {
        "n_draws": int(n_draws),
        "random_state": None if random_state is None else int(random_state),
        "sigma_shape": sigma_shape,
        "sigma_mean": float(sigma_grid.mean()),
        "sigma_max": float(sigma_grid.max()),
    }
    return UncertaintyPropagation(
        membership=membership,
        consensus_label=consensus_label,
        instability=instability,
        n_draws=int(n_draws),
        sigma_shape=sigma_shape,
        config=config,
    )


def _cpu_umap_cls() -> type:
    # Imported lazily so test harnesses without cuML do not pay the
    # import at module load. Returns the class so callers can
    # isinstance-check.
    import umap  # noqa: PLC0415

    cls: type = umap.UMAP
    return cls


def build_replica_augmented_matrix(
    X: NDArray[np.floating[Any]],
    sigma: float | NDArray[np.floating[Any]],
    *,
    n_replicas: int,
    random_state: int | None = None,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.intp]]:
    """Build a replica-augmented matrix for uncertainty-aware fitting.

    Stacks the original ``X`` on top of ``n_replicas`` Gaussian-perturbed
    copies:

    .. code-block:: text

        X_aug = [ X        ]   <- group_ids == 0  (original row index 0..n-1)
                [ X + e_1  ]   <- group_ids == 0..n-1 for the replica 1 block
                [ X + e_2  ]
                ...

    Unlike :func:`propagate_uncertainty` -- which perturbs *after*
    fitting and asks "is this sample's label stable?" -- this helper
    feeds the augmented matrix back into the full pipeline so that UMAP
    sees the uncertainty cloud around each point and HDBSCAN's density
    estimate is natively robust to the user's error bars.

    Parameters
    ----------
    X
        Clean feature matrix, shape ``(n_samples, n_features)``.
    sigma
        Per-feature 1-sigma uncertainty. Accepts the same broadcasting
        rules as :func:`propagate_uncertainty` (scalar,
        ``(n_features,)``, or ``(n_samples, n_features)``).
    n_replicas
        Number of noisy copies to append. Must be ``>= 0``. ``0`` is
        allowed and degenerates to the clean input (useful as a
        sanity check that the augmented path still recovers the
        unaugmented fit).
    random_state
        Seed for replica noise.

    Returns
    -------
    X_aug : NDArray
        Augmented matrix of shape ``(n_samples * (1 + n_replicas), n_features)``.
    group_ids : NDArray[np.intp]
        Length-``n_samples * (1 + n_replicas)`` vector giving the
        original-sample index for each augmented row. The first
        ``n_samples`` rows are the clean originals in order, followed
        by ``n_replicas`` blocks of ``n_samples`` rows each, all
        indexing back to ``0..n_samples-1``.
    """
    if n_replicas < 0:
        msg = f"n_replicas must be >= 0 (got {n_replicas})."
        raise ValueError(msg)
    x = np.asarray(X, dtype=np.float64)
    if x.ndim != 2:
        msg = f"X must be a 2-D array (got shape {x.shape})."
        raise ValueError(msg)
    n_samples, n_features = x.shape
    sigma_grid, _ = _broadcast_sigma(sigma, n_samples=n_samples, n_features=n_features)
    if n_replicas == 0:
        return x.copy(), np.arange(n_samples, dtype=np.intp)
    rng = np.random.default_rng(random_state)
    blocks = [x]
    for _ in range(n_replicas):
        noise = rng.standard_normal(size=x.shape) * sigma_grid
        blocks.append(x + noise)
    x_aug = np.vstack(blocks).astype(np.float64)
    group_ids = np.tile(np.arange(n_samples, dtype=np.intp), 1 + n_replicas)
    return x_aug, group_ids


def consensus_from_augmented_labels(
    labels: NDArray[np.intp],
    group_ids: NDArray[np.intp],
    *,
    n_clusters: int,
) -> UncertaintyPropagation:
    """Aggregate augmented-matrix labels back to per-original-sample consensus.

    Companion to :func:`build_replica_augmented_matrix`. Given the
    cluster-label vector that the pipeline produced on the augmented
    matrix, tally per-original-sample cluster frequencies and build the
    same ``UncertaintyPropagation`` shape that :func:`propagate_uncertainty`
    returns, so downstream helpers (``plot_uncertainty_map``,
    :meth:`UncertaintyPropagation.confident_labels`) work identically.

    Parameters
    ----------
    labels
        The HDBSCAN labels for the augmented matrix (``-1`` for
        outliers).
    group_ids
        The ``group_ids`` vector from :func:`build_replica_augmented_matrix`;
        length must match ``labels``.
    n_clusters
        Number of real clusters in the augmented fit.
    """
    labels = np.asarray(labels, dtype=np.intp)
    group_ids = np.asarray(group_ids, dtype=np.intp)
    if labels.shape != group_ids.shape:
        msg = (
            f"labels and group_ids must have the same shape "
            f"(got {labels.shape} and {group_ids.shape})."
        )
        raise ValueError(msg)
    if labels.size == 0:
        return UncertaintyPropagation(
            membership=np.empty((0, n_clusters + 1), dtype=np.float64),
            consensus_label=np.empty(0, dtype=np.intp),
            instability=np.empty(0, dtype=np.float64),
            n_draws=0,
            sigma_shape="augmented",
            config={"mode": "augmented"},
        )
    n_samples = int(group_ids.max()) + 1
    draws_per_sample = np.bincount(group_ids, minlength=n_samples)
    # Every original sample should receive the same number of augmented
    # rows by construction; reject mismatched inputs early because a
    # heterogeneous count would bias the membership fractions.
    n_draws = int(draws_per_sample[0])
    if not np.all(draws_per_sample == n_draws):
        msg = "group_ids must cover every original sample the same number of times."
        raise ValueError(msg)
    membership = np.zeros((n_samples, n_clusters + 1), dtype=np.float64)
    remapped = np.where(labels < 0, n_clusters, labels)
    for row, group in zip(remapped, group_ids, strict=True):
        membership[int(group), int(row)] += 1.0
    membership /= max(n_draws, 1)
    argmax = np.argmax(membership, axis=1)
    consensus_label = np.where(argmax == n_clusters, -1, argmax).astype(np.intp)
    instability = (1.0 - membership.max(axis=1)).astype(np.float64)
    return UncertaintyPropagation(
        membership=membership,
        consensus_label=consensus_label,
        instability=instability,
        n_draws=int(n_draws),
        sigma_shape="augmented",
        config={"mode": "augmented", "n_replicas_plus_clean": int(n_draws)},
    )


@dataclass(frozen=True)
class UncertaintyAwareFit:
    """Result of :meth:`UnsupervisedPipeline.fit_with_uncertainty`.

    Unlike :meth:`PipelineResult.propagate_uncertainty` -- which freezes
    a clean fit and Monte Carlos perturbations against it -- an
    uncertainty-aware fit feeds an augmented (clean + noisy-replica)
    matrix through the full pipeline. UMAP therefore "sees" the
    uncertainty cloud around every sample and the clustering adapts to
    it, rather than the clustering ignoring the error bars and the
    uncertainty only being accounted for afterwards.

    Attributes
    ----------
    augmented_result
        The :class:`PipelineResult` of the full pipeline run on the
        augmented matrix. Its arrays -- embedding, labels, persistence
        -- all have ``n_samples * (1 + n_replicas)`` rows; use
        :attr:`group_ids` to subset back to the original samples.
    propagation
        Per-original-sample :class:`UncertaintyPropagation` distilled
        from the augmented labels. Exposes the membership matrix,
        consensus label per original sample, and instability score in
        exactly the same shape that
        :meth:`PipelineResult.propagate_uncertainty` returns, so the
        same plots and helpers apply.
    group_ids
        Length-``n_samples * (1 + n_replicas)`` vector mapping every
        augmented row back to the index of the original sample that
        produced it.
    n_replicas
        Number of noisy copies appended (not counting the clean block).
    sigma_summary
        ``(mean, max)`` of the sigma grid actually used, for the
        audit trail.
    """

    augmented_result: PipelineResult
    propagation: UncertaintyPropagation
    group_ids: NDArray[np.intp]
    n_replicas: int
    sigma_summary: tuple[float, float]

    @property
    def consensus_label(self) -> NDArray[np.intp]:
        """Shortcut for :attr:`propagation.consensus_label`."""
        return self.propagation.consensus_label

    @property
    def instability(self) -> NDArray[np.floating[Any]]:
        """Shortcut for :attr:`propagation.instability`."""
        return self.propagation.instability

    def summary(self) -> str:
        """Short printable summary of the uncertainty-aware fit."""
        fit_summary = self.augmented_result.summary()
        prop_summary = self.propagation.summary()
        header = (
            f"starfold uncertainty-aware fit "
            f"(n_replicas={self.n_replicas}, "
            f"sigma mean={self.sigma_summary[0]:.4g}, "
            f"max={self.sigma_summary[1]:.4g})\n" + "=" * 32
        )
        return f"{header}\n{fit_summary}\n\n{prop_summary}"
