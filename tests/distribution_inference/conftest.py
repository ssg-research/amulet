"""Shared fixtures for distribution-inference tests.

Every attack in this module consumes the same six-array bundle
(x_train, y_train, z_train, x_test, y_test, z_test) with binary labels and a
two-column binary sensitive matrix ('race', 'sex'), sized so the ratio
subsampling can draw valid splits.
"""

import numpy as np
import pytest


@pytest.fixture
def synthetic_data_factory():
    """Factory fixture: (x_train, y_train, z_train, x_test, y_test, z_test)."""

    def _make(
        n_train: int = 600,
        n_test: int = 200,
        num_features: int = 4,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        x_train = rng.standard_normal((n_train, num_features)).astype(np.float32)
        y_train = rng.integers(0, 2, n_train).astype(np.int64)
        z_train = rng.integers(0, 2, (n_train, 2)).astype(np.int64)
        x_test = rng.standard_normal((n_test, num_features)).astype(np.float32)
        y_test = rng.integers(0, 2, n_test).astype(np.int64)
        z_test = rng.integers(0, 2, (n_test, 2)).astype(np.int64)
        return x_train, y_train, z_train, x_test, y_test, z_test

    return _make
