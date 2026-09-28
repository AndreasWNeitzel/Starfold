"""Shared synthetic fixtures; every array here is a test fixture, never data."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="session")
def two_blobs() -> np.ndarray:
    """Two unit-variance Gaussian blobs at -6 and +6 in every one of six columns."""
    rng = np.random.default_rng(20260905)
    low = rng.normal(loc=-6.0, scale=1.0, size=(120, 6))
    high = rng.normal(loc=6.0, scale=1.0, size=(120, 6))
    return np.concatenate([low, high]).astype(np.float64)


@pytest.fixture(scope="session")
def single_gaussian() -> np.ndarray:
    """One standard Gaussian of 200 rows in five columns."""
    rng = np.random.default_rng(20260906)
    return rng.standard_normal((200, 5)).astype(np.float64)
