import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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
