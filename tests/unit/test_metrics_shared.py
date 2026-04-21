import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from amulet.models import AmuletModel
from amulet.utils.__base import get_intermediate_features
from amulet.utils.__metrics import get_accuracy, get_fidelity


class MockModel(nn.Module):
    """A mock model that returns fixed predictions based on class index."""

    def __init__(self, predictions):
        super().__init__()
        self.predictions = predictions

    def forward(self, x):
        batch_size = x.size(0)
        num_classes = 10
        logits = torch.zeros(batch_size, num_classes)
        for i in range(batch_size):
            logits[i, self.predictions[i]] = 1.0
        return logits


def test_get_accuracy_identity(cpu_device):
    # Arrange: model always correct
    labels = torch.tensor([0, 1, 2, 3])
    model = MockModel(labels)
    dataset = TensorDataset(torch.randn(4, 1), labels)
    loader = DataLoader(dataset, batch_size=4)

    # Act
    acc = get_accuracy(model, loader, cpu_device)

    # Assert
    assert acc == 100.0


def test_get_accuracy_zero(cpu_device):
    # Arrange: model always wrong
    labels = torch.tensor([0, 0])
    model = MockModel(torch.tensor([1, 1]))
    dataset = TensorDataset(torch.randn(2, 1), labels)
    loader = DataLoader(dataset, batch_size=2)

    # Act
    acc = get_accuracy(model, loader, cpu_device)

    # Assert
    assert acc == 0.0


def test_get_fidelity_perfect(cpu_device):
    # Arrange: two models agree perfectly
    preds = torch.tensor([0, 1, 0, 1])
    m1 = MockModel(preds)
    m2 = MockModel(preds)
    dataset = TensorDataset(torch.randn(4, 1), torch.zeros(4))
    loader = DataLoader(dataset, batch_size=4)

    # Act
    fid = get_fidelity(m1, m2, loader, cpu_device)

    # Assert
    assert fid == 100.0


def test_get_fidelity_zero(cpu_device):
    # Arrange: two models never agree
    m1 = MockModel(torch.tensor([0, 0]))
    m2 = MockModel(torch.tensor([1, 1]))
    dataset = TensorDataset(torch.randn(2, 1), torch.zeros(2))
    loader = DataLoader(dataset, batch_size=2)

    # Act
    fid = get_fidelity(m1, m2, loader, cpu_device)

    # Assert
    assert fid == 0.0


class _HiddenMLP(AmuletModel):
    """2-layer MLP exposing get_hidden for intermediate-feature extraction."""

    def __init__(self, num_features: int = 4, hidden: int = 6, num_classes: int = 3):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(num_features, hidden), nn.ReLU())
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        return self.head(self.body(x))

    def get_hidden(self, x):
        return self.body(x)


def test_get_intermediate_features_shapes_and_dtype(cpu_device):
    # Arrange: uneven batches (last batch smaller) to exercise concat over axis 0
    num_samples = 10
    num_features = 4
    hidden = 6
    x = torch.randn(num_samples, num_features)
    y = torch.randint(0, 3, (num_samples,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)
    model = _HiddenMLP(num_features=num_features, hidden=hidden)

    # Act
    features, targets, inputs = get_intermediate_features(model, loader, cpu_device)

    # Assert: shapes line up and rows are stacked, not object-wrapped
    assert features.shape == (num_samples, hidden)
    assert targets.shape == (num_samples,)
    assert inputs.shape == (num_samples, num_features)
    assert features.dtype == np.float32
    assert inputs.dtype == np.float32
    # Inputs round-trip through the loader unchanged.
    np.testing.assert_allclose(inputs, x.numpy())
    np.testing.assert_array_equal(targets, y.numpy())
