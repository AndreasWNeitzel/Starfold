"""Synthetic regimes for method validation only.

These generators exist to exercise failure modes that the real benchmark
datasets never show: a single anisotropic population (where a false split is
the risk), two interleaved manifolds with irrelevant features (where the
embedding must disentangle what the input geometry hides), and a small
satellite beside overlapping blobs (where coarse merges and fragmentation
compete). Every draw is seeded. Nothing produced here is data; the outputs
are used only to characterise the procedure and never enter a scientific
table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class SyntheticCase:
    """One generated regime with its truth labels."""

    name: str
    X: NDArray[np.float64]
    truth: NDArray[np.int64]
    description: str


def single_anisotropic_gaussian(seed: int = 0, n: int = 2000, d: int = 5) -> SyntheticCase:
    """One Gaussian population with covariance eigenvalues spanning 10 to 1.

    The correct answer is no supported structure: the draw lies inside the
    Gaussian-copula null family (``starfold.nulls``), so any recurrent split is
    a false positive.
    """
    rng = np.random.default_rng(seed)
    eigenvalues = np.linspace(10.0, 1.0, d)
    rotation, _ = np.linalg.qr(rng.normal(size=(d, d)))
    covariance = rotation @ np.diag(eigenvalues) @ rotation.T
    x = rng.multivariate_normal(np.zeros(d), covariance, size=n)
    return SyntheticCase(
        name="single_anisotropic_gaussian",
        X=x,
        truth=np.zeros(n, dtype=np.int64),
        description=f"one Gaussian, n={n}, d={d}, eigenvalues 10 to 1",
    )


def interleaved_manifolds(
    seed: int = 0, n_per: int = 1000, noise_scale: float = 0.15, n_irrelevant: int = 5
) -> SyntheticCase:
    """Two interleaved one-dimensional manifolds in 10 dimensions.

    Two helical curves wound around a common axis with a phase offset, each
    thickened by isotropic noise, embedded in five informative dimensions and
    padded with irrelevant standard normal features. Linear methods cannot
    separate them; a neighbourhood-preserving embedding can.
    """
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, 4.0 * np.pi, size=n_per)
    curves = []
    for phase in (0.0, np.pi):
        informative = np.column_stack(
            [
                np.cos(t + phase),
                np.sin(t + phase),
                0.25 * t,
                0.5 * np.cos(2.0 * t + phase),
                0.5 * np.sin(2.0 * t + phase),
            ]
        )
        curves.append(informative + noise_scale * rng.normal(size=informative.shape))
    informative_all = np.vstack(curves)
    irrelevant = rng.normal(size=(informative_all.shape[0], n_irrelevant))
    x = np.hstack([informative_all, irrelevant])
    truth = np.repeat(np.arange(2), n_per)
    order = rng.permutation(x.shape[0])
    return SyntheticCase(
        name="interleaved_manifolds",
        X=x[order],
        truth=truth[order].astype(np.int64),
        description=(
            f"two interleaved helices, n={2 * n_per}, noise {noise_scale}, "
            f"{n_irrelevant} irrelevant features"
        ),
    )


def satellite_and_overlap(
    seed: int = 0, n_blob: int = 1000, satellite_fraction: float = 0.04, d: int = 6
) -> SyntheticCase:
    """One small distant satellite beside two overlapping blobs.

    The satellite holds ``satellite_fraction`` of the total mass; the two
    blobs overlap at one standard deviation of separation per axis so that a
    coarse merge and a fine split are both tempting.
    """
    rng = np.random.default_rng(seed)
    n_satellite = max(20, round(satellite_fraction * 2 * n_blob / (1.0 - satellite_fraction)))
    centre_a = np.zeros(d)
    centre_b = np.full(d, 1.0)
    centre_s = np.full(d, 8.0)
    blob_a = rng.normal(size=(n_blob, d)) + centre_a
    blob_b = rng.normal(size=(n_blob, d)) + centre_b
    satellite = 0.5 * rng.normal(size=(n_satellite, d)) + centre_s
    x = np.vstack([blob_a, blob_b, satellite])
    truth = np.concatenate([np.zeros(n_blob), np.ones(n_blob), np.full(n_satellite, 2)]).astype(
        np.int64
    )
    order = rng.permutation(x.shape[0])
    return SyntheticCase(
        name="satellite_and_overlap",
        X=x[order],
        truth=truth[order],
        description=(f"two overlapping blobs of {n_blob} plus a satellite of {n_satellite}, d={d}"),
    )


def all_cases(seed: int = 0) -> tuple[SyntheticCase, ...]:
    """The three regimes at their default sizes."""
    return (
        single_anisotropic_gaussian(seed),
        interleaved_manifolds(seed),
        satellite_and_overlap(seed),
    )
