import math
import random

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Decimal places at which metric bounds are checked. Metrics are reported at
# paper precision (0.XXXX or XX.XX%); float-epsilon overshoot beyond that — e.g.
# structural_similarity returning 1.0000002 because its data_range is taken from
# a single operand — is not a real bound violation. Rounding to this precision
# before the comparison absorbs it uniformly, instead of per-metric tolerances.
BOUND_PRECISION = 4


class TinyMLP(nn.Module):
    """Tiny 2-layer MLP used across tests; defined at module scope so it's picklable."""

    def __init__(self, num_features: int = 4, num_classes: int = 2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_features, 8),
            nn.ReLU(),
            nn.Linear(8, num_classes),
        )

    def forward(self, x):
        return self.layers(x)

    def get_hidden(self, x):
        """Return the intermediate features before the final layer."""
        return self.layers[:-1](x)


@pytest.fixture(
    params=[
        "cpu",
        pytest.param("cuda", marks=pytest.mark.gpu),
    ]
)
def device(request: pytest.FixtureRequest) -> str:
    """Parametrized device fixture. CUDA variant is skipped when no GPU is available."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return request.param  # type: ignore[return-value]


@pytest.fixture
def cpu_device() -> str:
    """CPU-only device fixture for unit tests."""
    return "cpu"


@pytest.fixture
def tiny_classifier_factory():
    """Factory fixture that returns fresh, independently-initialized TinyMLP instances.

    Use when a test needs multiple distinct models (e.g. distribution inference,
    where reusing the same instance would destroy the distinguishing signal).
    """

    def _make(seed: int | None = None, device: str = "cpu") -> TinyMLP:
        if seed is not None:
            torch.manual_seed(seed)
        return TinyMLP().to(device)

    return _make


@pytest.fixture
def tiny_classifier(tiny_classifier_factory, device):
    """Single TinyMLP instance seeded for reproducibility, on the test device."""
    return tiny_classifier_factory(seed=42, device=device)


@pytest.fixture
def tiny_dataset():
    """Fixture for a synthetic dataset (N=64, num_features=4) with binary labels in [0, 1]."""
    torch.manual_seed(42)
    x = torch.rand(64, 4)
    y = torch.randint(0, 2, (64,))
    return TensorDataset(x, y)


@pytest.fixture
def tiny_loader(tiny_dataset):
    """Fixture for a DataLoader with batch_size=8."""
    return DataLoader(tiny_dataset, batch_size=8, shuffle=False)


@pytest.fixture
def seed_everything():
    """Return a callable that seeds every RNG source the modules draw from.

    torch.manual_seed covers model init and training; np.random.seed covers the
    *global legacy* numpy RNG that WatermarkNN and SuriEvans2022 use (a
    np.random.default_rng Generator is a separate stream and does not reach
    them); random.seed covers stdlib shuffles. Reproducibility tests call this
    immediately before each of the two runs they compare.
    """

    def _seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        _ = torch.manual_seed(seed)

    return _seed


@pytest.fixture
def assert_within():
    """Assert a metric value lies in [low, high] after rounding to BOUND_PRECISION.

    Bound checks use this so float-epsilon overshoot at higher precision than we
    would ever report does not register as a violation.
    """

    def _assert_within(value: float, low: float, high: float) -> None:
        rounded = round(float(value), BOUND_PRECISION)
        assert low <= rounded <= high, (
            f"{value!r} (rounded to {rounded}) outside [{low}, {high}]"
        )

    return _assert_within


@pytest.fixture
def assert_nonneg_finite():
    """Assert a metric value is finite and non-negative (for distances like MSE)."""

    def _assert_nonneg_finite(value: float) -> None:
        v = float(value)
        assert math.isfinite(v), f"{value!r} is not finite"
        assert round(v, BOUND_PRECISION) >= 0.0, f"{value!r} is negative"

    return _assert_nonneg_finite
