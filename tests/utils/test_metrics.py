import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from amulet.models import AmuletModel
from amulet.utils.__base import get_intermediate_features
from amulet.utils.__metrics import get_accuracy, get_fidelity


class OneHotModel(nn.Module):
    """Deterministic one-hot model that emits one-hot logits from a fixed prediction tensor."""

    def __init__(self, predictions: torch.Tensor):
        super().__init__()
        self.predictions = predictions

    def forward(self, x):
        batch_size = x.size(0)
        num_classes = 10
        logits = torch.zeros(batch_size, num_classes)
        for i in range(batch_size):
            logits[i, self.predictions[i]] = 1.0
        return logits


class _PredFromColumn(nn.Module):
    """Predicts the class encoded in a chosen input column.

    Unlike OneHotModel (which indexes predictions by batch position and so only
    works single-batch), this maps each row to its own prediction from the input,
    so it behaves identically regardless of how the loader batches the data.
    """

    def __init__(self, col: int = 0, num_classes: int = 10):
        super().__init__()
        self.col = col
        self.num_classes = num_classes

    def forward(self, x):
        idx = x[:, self.col].long()
        logits = torch.zeros(x.shape[0], self.num_classes)
        logits[torch.arange(x.shape[0]), idx] = 1.0
        return logits


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


@pytest.fixture
def mock_model_factory():
    """Factory fixture returning OneHotModel instances for caller-supplied predictions."""

    def _make(predictions: torch.Tensor) -> OneHotModel:
        return OneHotModel(predictions)

    return _make


@pytest.fixture
def hidden_mlp_factory():
    """Factory fixture returning _HiddenMLP instances for caller-supplied dimensions."""

    def _make(
        num_features: int = 4, hidden: int = 6, num_classes: int = 3
    ) -> _HiddenMLP:
        return _HiddenMLP(
            num_features=num_features, hidden=hidden, num_classes=num_classes
        )

    return _make


def test_get_accuracy_identity(cpu_device, mock_model_factory):
    # model always correct
    labels = torch.tensor([0, 1, 2, 3])
    model = mock_model_factory(labels)
    dataset = TensorDataset(torch.randn(4, 1), labels)
    loader = DataLoader(dataset, batch_size=4)

    acc = get_accuracy(model, loader, cpu_device)

    assert acc == 100.0


def test_get_accuracy_zero(cpu_device, mock_model_factory):
    # model always wrong
    labels = torch.tensor([0, 0])
    model = mock_model_factory(torch.tensor([1, 1]))
    dataset = TensorDataset(torch.randn(2, 1), labels)
    loader = DataLoader(dataset, batch_size=2)

    acc = get_accuracy(model, loader, cpu_device)

    assert acc == 0.0


def test_get_fidelity_perfect(cpu_device, mock_model_factory):
    # two models agree perfectly
    preds = torch.tensor([0, 1, 0, 1])
    m1 = mock_model_factory(preds)
    m2 = mock_model_factory(preds)
    dataset = TensorDataset(torch.randn(4, 1), torch.zeros(4))
    loader = DataLoader(dataset, batch_size=4)

    fid = get_fidelity(m1, m2, loader, cpu_device)

    assert fid == 100.0


def test_get_fidelity_zero(cpu_device, mock_model_factory):
    # two models never agree
    m1 = mock_model_factory(torch.tensor([0, 0]))
    m2 = mock_model_factory(torch.tensor([1, 1]))
    dataset = TensorDataset(torch.randn(2, 1), torch.zeros(2))
    loader = DataLoader(dataset, batch_size=2)

    fid = get_fidelity(m1, m2, loader, cpu_device)

    assert fid == 0.0


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_get_accuracy_bounded(
    cpu_device, tiny_classifier_factory, tiny_loader, assert_within, seed
):
    # freshly seeded model with random init, real data loader
    model = tiny_classifier_factory(seed=seed, device=cpu_device)

    acc = get_accuracy(model, tiny_loader, cpu_device)

    # accuracy is a percentage
    assert_within(acc, 0.0, 100.0)


@pytest.mark.parametrize("seed_1, seed_2", [(0, 1), (1, 2), (2, 3)])
def test_get_fidelity_bounded(
    cpu_device, tiny_classifier_factory, tiny_loader, assert_within, seed_1, seed_2
):
    # two independently seeded models, real data loader
    model_1 = tiny_classifier_factory(seed=seed_1, device=cpu_device)
    model_2 = tiny_classifier_factory(seed=seed_2, device=cpu_device)

    fid = get_fidelity(model_1, model_2, tiny_loader, cpu_device)

    # fidelity is a percentage
    assert_within(fid, 0.0, 100.0)


def test_get_accuracy_multi_batch_exact(cpu_device):
    # 6 samples over batch_size=4 (batches of 4 and 2). The exact single-batch
    # tests above can't tell running accumulation from a per-batch reset; here a
    # bug that kept only the last batch's counts would score 50% (1 of 2), not
    # the true 83.33% (5 of 6). Predictions come from feature 0; index 4 is wrong.
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0], [1.0], [1.0]])
    labels = torch.tensor([0, 1, 2, 3, 0, 1])
    model = _PredFromColumn(col=0)
    loader = DataLoader(TensorDataset(x, labels), batch_size=4)

    acc = get_accuracy(model, loader, cpu_device)

    assert acc == pytest.approx(500 / 6)


def test_get_fidelity_multi_batch_exact(cpu_device):
    # Same multi-batch accumulation guard for fidelity. Model 1 reads feature 0,
    # model 2 reads feature 1; the columns agree on 5 of 6 rows (differ at index
    # 5), so fidelity is 83.33% across the batch boundary.
    x = torch.tensor([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [0.0, 0.0],
        [1.0, 0.0],
    ])
    m1 = _PredFromColumn(col=0)
    m2 = _PredFromColumn(col=1)
    loader = DataLoader(TensorDataset(x, torch.zeros(6)), batch_size=4)

    fid = get_fidelity(m1, m2, loader, cpu_device)

    assert fid == pytest.approx(500 / 6)


def test_get_intermediate_features_shapes_and_dtype(cpu_device, hidden_mlp_factory):
    # uneven batches (last batch smaller) to exercise concat over axis 0
    num_samples = 10
    num_features = 4
    hidden = 6
    x = torch.randn(num_samples, num_features)
    y = torch.randint(0, 3, (num_samples,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)
    model = hidden_mlp_factory(num_features=num_features, hidden=hidden)

    features, targets, inputs = get_intermediate_features(model, loader, cpu_device)

    # shapes line up and rows are stacked, not object-wrapped
    assert features.shape == (num_samples, hidden)
    assert targets.shape == (num_samples,)
    assert inputs.shape == (num_samples, num_features)
    assert features.dtype == np.float32
    assert inputs.dtype == np.float32
    # Inputs round-trip through the loader unchanged.
    np.testing.assert_allclose(inputs, x.numpy())
    np.testing.assert_array_equal(targets, y.numpy())
