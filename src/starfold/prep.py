"""Input validation, resample-local scaling, resample plans, and measurement draws.

Rows with a non-finite value are rejected and counted, never imputed. A scaler
is fitted only on the rows it will transform, so no resample or world sees
statistics of rows outside it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import RobustScaler, StandardScaler

from starfold._engine import derive_seed

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

__all__ = [
    "MeasurementSampler",
    "ResamplePlan",
    "Scaling",
    "make_plan",
    "measurement_draw",
    "scale",
    "validate_matrix",
]

Scaling = Literal["standard", "robust"]
MeasurementSampler = Callable[
    [NDArray[np.float64], NDArray[np.intp], np.random.Generator],
    NDArray[np.float64],
]

MIN_ROWS = 50
MIN_COLUMNS = 2


def validate_matrix(X: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Coerce a feature matrix to finite C-order float64 rows.

    Parameters
    ----------
    X : array-like of shape (n, d)
        Feature matrix. Rows holding NaN or infinity are dropped.

    Returns
    -------
    rows : ndarray of shape (n_kept, d), float64
        The finite rows, C-contiguous.
    kept : ndarray of shape (n_kept,), intp
        Indices into ``X`` of the retained rows, ascending.

    Raises
    ------
    ValueError
        If ``X`` is not two-dimensional, has fewer than two columns, is not
        numeric, or keeps fewer than 50 finite rows.
    """
    try:
        values = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as error:
        msg = "X must be a numeric array."
        raise ValueError(msg) from error
    if values.ndim != 2:
        msg = f"X must be two-dimensional (got ndim={values.ndim})."
        raise ValueError(msg)
    if values.shape[1] < MIN_COLUMNS:
        msg = f"X must have at least {MIN_COLUMNS} columns (got {values.shape[1]})."
        raise ValueError(msg)
    kept = np.flatnonzero(np.isfinite(values).all(axis=1)).astype(np.intp)
    if kept.size < MIN_ROWS:
        msg = f"X must keep at least {MIN_ROWS} finite rows (got {kept.size} of {values.shape[0]})."
        raise ValueError(msg)
    return np.ascontiguousarray(values[kept], dtype=np.float64), kept


def scale(x_rows: NDArray[np.float64], scaling: Scaling) -> NDArray[np.float32]:
    """Fit a scaler on ``x_rows`` alone and return the scaled rows as float32.

    Parameters
    ----------
    x_rows : ndarray of shape (m, d)
        Rows of one resample or world.
    scaling : {"standard", "robust"}
        ``sklearn.preprocessing.StandardScaler`` or ``RobustScaler``
        (Pedregosa et al. 2011, JMLR 12:2825). Constant columns are left at
        their centred value.

    Returns
    -------
    ndarray of shape (m, d), float32
        C-contiguous scaled rows.

    Raises
    ------
    ValueError
        If ``scaling`` is not a recognised option.
    """
    if scaling == "standard":
        scaler = StandardScaler()
    elif scaling == "robust":
        scaler = RobustScaler()
    else:
        msg = f"scaling must be 'standard' or 'robust' (got {scaling!r})."
        raise ValueError(msg)
    return np.ascontiguousarray(scaler.fit_transform(x_rows), dtype=np.float32)


@dataclass(frozen=True, eq=False)
class ResamplePlan:
    """Row subsets and UMAP seeds shared by every trial of one search.

    Common random numbers make trial-to-trial differences attributable to the
    configuration rather than to the draw (Glasserman and Yao 1992,
    doi:10.1287/mnsc.38.6.884).

    Attributes
    ----------
    n : int
        Number of rows in the validated matrix.
    fraction : float
        Fraction of rows drawn per resample.
    seed : int
        Base seed of the plan.
    indices : tuple of ndarray, intp
        ``indices[r]`` is the sorted row set of resample ``r``, drawn without
        replacement with ``default_rng(derive_seed(seed, 0, r))``.
    seeds : tuple of int
        ``seeds[r] = derive_seed(seed, 1, r)`` is the UMAP seed of resample ``r``.
    """

    n: int
    fraction: float
    seed: int
    indices: tuple[NDArray[np.intp], ...]
    seeds: tuple[int, ...]

    @property
    def m(self) -> int:
        """Return the number of rows per resample."""
        return int(self.indices[0].size)

    @property
    def n_resamples(self) -> int:
        """Return the number of resamples."""
        return len(self.indices)

    def heldout(self, r: int, s: int) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """Return the rows of resample ``s`` that resample ``r`` did not see.

        Parameters
        ----------
        r, s : int
            Resample indices, ``r`` being the fitted resample and ``s`` the one
            whose unseen rows are predicted.

        Returns
        -------
        rows, pos_s : ndarray, intp
            ``rows`` are row indices into the validated matrix and
            ``indices[s][pos_s] == rows`` elementwise.
        """
        unseen = ~np.isin(self.indices[s], self.indices[r], assume_unique=True)
        pos_s = np.flatnonzero(unseen).astype(np.intp)
        return self.indices[s][pos_s].astype(np.intp), pos_s


def make_plan(n: int, *, n_resamples: int, fraction: float, seed: int) -> ResamplePlan:
    """Draw the resample plan of one search.

    Parameters
    ----------
    n : int
        Number of rows.
    n_resamples : int
        Number of resamples, at least 2 for a pair to exist.
    fraction : float
        Fraction in ``(0, 1]``. Each resample holds ``round(fraction * n)`` rows.
    seed : int
        Base seed.

    Returns
    -------
    ResamplePlan
        Sorted, unique row sets and one UMAP seed per resample.

    Raises
    ------
    ValueError
        If fewer than two resamples, ``fraction`` outside ``(0, 1]``, or fewer
        than two rows per resample.
    """
    if n_resamples < 2:
        msg = f"n_resamples must be at least 2 (got {n_resamples})."
        raise ValueError(msg)
    if not 0.0 < fraction <= 1.0:
        msg = f"fraction must lie in (0, 1] (got {fraction})."
        raise ValueError(msg)
    m = round(fraction * n)
    if m < 2:
        msg = f"round(fraction * n) must be at least 2 (got {m})."
        raise ValueError(msg)
    indices = tuple(
        np.sort(np.random.default_rng(derive_seed(seed, 0, r)).choice(n, size=m, replace=False))
        .astype(np.intp)
        .copy()
        for r in range(n_resamples)
    )
    seeds = tuple(derive_seed(seed, 1, r) for r in range(n_resamples))
    return ResamplePlan(n=n, fraction=fraction, seed=seed, indices=indices, seeds=seeds)


def measurement_draw(
    x_rows: NDArray[np.float64],
    row_indices: NDArray[np.intp],
    rng: np.random.Generator,
    *,
    x_err_rows: NDArray[np.float64] | None = None,
    sampler: MeasurementSampler | None = None,
) -> NDArray[np.float64]:
    """Return one measurement realisation of ``x_rows``.

    Parameters
    ----------
    x_rows : ndarray of shape (m, d)
        Observed values of the sampled rows.
    row_indices : ndarray of shape (m,), intp
        Indices of those rows in the validated matrix, passed to ``sampler``.
    rng : numpy.random.Generator
        Generator of the world.
    x_err_rows : ndarray of shape (m, d), optional
        Finite per-value uncertainties, giving the draw ``x + err * N(0, 1)``.
    sampler : callable, optional
        The caller's declared measurement-uncertainty sampler, called as
        ``sampler(x_rows.copy(), row_indices, rng)`` and returning an array of
        the same shape.

    Returns
    -------
    ndarray of shape (m, d), float64
        A copy of ``x_rows`` when neither perturbation is given.

    Raises
    ------
    ValueError
        If both perturbations are given, shapes disagree, or a non-finite
        value appears in ``x_err_rows`` or the sampler output.
    """
    if x_err_rows is not None and sampler is not None:
        msg = "Give x_err_rows or sampler, not both."
        raise ValueError(msg)
    if row_indices.shape != (x_rows.shape[0],):
        msg = f"row_indices must have shape ({x_rows.shape[0]},) (got {row_indices.shape})."
        raise ValueError(msg)
    if x_err_rows is not None:
        if x_err_rows.shape != x_rows.shape:
            msg = f"x_err_rows shape {x_err_rows.shape} differs from x_rows {x_rows.shape}."
            raise ValueError(msg)
        if not np.all(np.isfinite(x_err_rows)) or np.any(x_err_rows < 0):
            msg = "x_err_rows must be finite and non-negative."
            raise ValueError(msg)
        return x_rows + x_err_rows * rng.standard_normal(x_rows.shape)
    if sampler is not None:
        drawn = np.asarray(sampler(x_rows.copy(), row_indices, rng), dtype=np.float64)
        if drawn.shape != x_rows.shape:
            msg = f"sampler returned shape {drawn.shape}, expected {x_rows.shape}."
            raise ValueError(msg)
        if not np.all(np.isfinite(drawn)):
            msg = "sampler returned non-finite values."
            raise ValueError(msg)
        return np.ascontiguousarray(drawn)
    return x_rows.copy()
