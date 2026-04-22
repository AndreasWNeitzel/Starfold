"""Closed chain of Hopf-linked tori in 3D.

This module provides a synthetic dataset used by the quickstart
tutorial and by ``diagnostics.py``. It is deliberately kept in
``examples/`` (not inside the installed package) because it is
demonstration data, not part of the public API.

Construction
------------
N torus centres sit on a backbone circle of radius ``big_radius`` in
the xy-plane at equally spaced angles. Each torus is parameterised by
its own ring plane and major radius, chosen so that each even torus
threads cleanly through its two horizontal neighbours:

* Even k  (vertical):    plane span(T_hat_k, Z),      radius R_even
* Odd  k  (horizontal):  plane span(R_hat_k, T_hat_k), radius R_odd

The even ring's "hole" axis points radially outward (R_hat_k), so the
adjacent horizontal ring passes radially through that hole -- which is
the standard chain-link geometry. The odd ring's major radius is
chosen so that the four horizontal tori do not touch each other
(requires ``R_odd < big_radius / sqrt(2)`` for N=8) while each still
pierces its vertical neighbour's disk.

Verified: for N=8, ``R_big=4, R_even=2, R_odd=2.5`` gives a
tridiagonal + corners linking matrix (adjacent tori link with linking
number +/-1, non-adjacent pairs link with zero). The eight tori form
a topologically proper closed chain with no body overlaps.
"""

from __future__ import annotations

import numpy as np


def _frame(theta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array([np.cos(theta), np.sin(theta), 0.0]),
        np.array([-np.sin(theta), np.cos(theta), 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )


def _axes_radius(
    k: int,
    theta: float,
    R_even: float,
    R_odd: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    R_hat, T_hat, Z = _frame(theta)
    if k % 2 == 0:
        # Even k: vertical ring whose plane contains T_hat (tangential) and
        # Z. Ring normal is R_hat -- the hole faces radially outward so the
        # next horizontal ring threads through it cleanly.
        return T_hat, Z, R_even
    # Odd k: horizontal ring in (R_hat, T_hat) = xy-plane. Normal is Z.
    return T_hat, R_hat, R_odd


def make_torus_chain(
    n_links: int = 8,
    points_per_link: int = 1200,
    big_radius: float = 4.0,
    major_even: float = 2.0,
    major_odd: float = 2.5,
    minor_radius: float = 0.15,
    noise_std: float = 0.02,
    *,
    solid: bool = True,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample points on a closed chain of ``n_links`` interlocked tori.

    Parameters
    ----------
    n_links
        Number of tori in the chain. Must be even and >= 4.
    points_per_link
        Points sampled per torus.
    big_radius
        Radius of the backbone circle on which torus centres sit.
    major_even, major_odd
        Major radii for vertical (even-k) and horizontal (odd-k) tori.
    minor_radius
        Tube radius. Small values keep the chain visually thin.
    solid
        If True, sample uniformly from the full 3D solid torus.
        If False, sample on the 2D torus surface (area-uniform).
    noise_std
        Isotropic Gaussian noise added in 3D.
    random_state
        Optional seed for reproducibility.

    Returns
    -------
    X
        ``(n_links * points_per_link, 3)`` float array of 3D positions.
    y
        ``(n_links * points_per_link,)`` integer labels in ``[0, n_links)``
        identifying which torus each point came from.
    """
    if n_links < 4 or n_links % 2 != 0:
        msg = "n_links must be an even integer >= 4."
        raise ValueError(msg)
    rng = np.random.default_rng(random_state)
    pts: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for k in range(n_links):
        theta = 2 * np.pi * k / n_links
        R_hat, _, _ = _frame(theta)
        centre = big_radius * R_hat
        u_ax, v_ax, major = _axes_radius(k, theta, major_even, major_odd)
        normal = np.cross(u_ax, v_ax)
        if solid:
            u = rng.uniform(0.0, 2 * np.pi, size=points_per_link)
            v = rng.uniform(0.0, 2 * np.pi, size=points_per_link)
            rho = minor_radius * np.sqrt(rng.uniform(0.0, 1.0, size=points_per_link))
        else:
            over = rng.uniform(0.0, 2 * np.pi, size=points_per_link * 3)
            over_v = rng.uniform(0.0, 2 * np.pi, size=points_per_link * 3)
            accept = rng.uniform(0.0, 1.0, size=over.size) < (
                (major + minor_radius * np.cos(over_v)) / (major + minor_radius)
            )
            u = over[accept][:points_per_link]
            v = over_v[accept][:points_per_link]
            rho = np.full(points_per_link, minor_radius)
        radial = np.cos(u)[:, None] * u_ax + np.sin(u)[:, None] * v_ax
        ring = centre + major * radial
        body = (
            ring
            + rho[:, None] * np.cos(v)[:, None] * radial
            + rho[:, None] * np.sin(v)[:, None] * normal
        )
        body = body + rng.normal(0.0, noise_std, size=body.shape)
        pts.append(body)
        labels.append(np.full(points_per_link, k, dtype=np.intp))
    return np.vstack(pts), np.concatenate(labels)


def linking_number(
    i: int,
    j: int,
    n_links: int = 8,
    big_radius: float = 4.0,
    major_even: float = 2.0,
    major_odd: float = 2.5,
    n: int = 4000,
) -> int:
    """Return the linking number of torus-core ring i with ring j.

    Uses signed crossings of ring j with the disk bounded by ring i.
    For the default geometry and adjacent ``|i-j| = 1 mod n_links``, the
    result is +/-1; for all other pairs it is 0.
    """
    theta_i = 2 * np.pi * i / n_links
    R_hat, _, _ = _frame(theta_i)
    centre_i = big_radius * R_hat
    u_i, v_i, major_i = _axes_radius(i, theta_i, major_even, major_odd)
    n_i = np.cross(u_i, v_i)

    theta_j = 2 * np.pi * j / n_links
    R_hat_j, _, _ = _frame(theta_j)
    centre_j = big_radius * R_hat_j
    u_j, v_j, major_j = _axes_radius(j, theta_j, major_even, major_odd)
    u = np.linspace(0.0, 2 * np.pi, n, endpoint=False) + np.pi / (2 * n)
    ring_j = centre_j + major_j * (np.cos(u)[:, None] * u_j + np.sin(u)[:, None] * v_j)

    d = (ring_j - centre_i) @ n_i
    d_next = np.roll(d, -1)
    idxs = np.where(d * d_next < 0)[0]
    lk = 0
    for k in idxs:
        t = d[k] / (d[k] - d_next[k])
        p = ring_j[k] + t * (ring_j[(k + 1) % n] - ring_j[k])
        uu = (p - centre_i) @ u_i
        vv = (p - centre_i) @ v_i
        if uu * uu + vv * vv < major_i * major_i:
            lk += 1 if d_next[k] > d[k] else -1
    return lk


def linking_matrix(
    n_links: int = 8,
    big_radius: float = 4.0,
    major_even: float = 2.0,
    major_odd: float = 2.5,
    n: int = 4000,
) -> np.ndarray:
    """Return the ``(n_links, n_links)`` integer linking-number matrix."""
    L = np.zeros((n_links, n_links), dtype=np.int64)
    for i in range(n_links):
        for j in range(n_links):
            if i == j:
                continue
            L[i, j] = linking_number(i, j, n_links, big_radius, major_even, major_odd, n)
    return L
