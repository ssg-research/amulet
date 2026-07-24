"""Tests for the OutlierRemoval defense: integration coverage of train_robust
(smoke, retained-count arithmetic, reproducibility) plus a unit test pinning
the numerical core, _knn_shapley."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from amulet.poisoning.defenses.outlier_removal import OutlierRemoval

N_TRAIN = 8
N_TEST = 5
DIM = 3


class _BatchNormClassifier(nn.Module):
    """Minimal classifier whose BatchNorm makes a one-sample batch fatal in training.

    The shared TinyMLP has no normalization layer, so a batch of one passes
    through it without complaint and cannot exercise the singleton-batch path.
    BatchNorm computes per-channel statistics and rejects a single sample in
    train mode, so this model is what reproduces the retrain crash. It exposes
    get_hidden so OutlierRemoval can score it.
    """

    def __init__(self, num_features: int = 4, num_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(num_features, 8)
        self.bn = nn.BatchNorm1d(8)
        self.fc2 = nn.Linear(8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.bn(self.fc1(x))))

    def get_hidden(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.bn(self.fc1(x)))


class _SingleChannelConvClassifier(nn.Module):
    """Conv classifier over single-channel images, exposing get_hidden.

    OutlierRemoval scores points from the model's penultimate features, so the
    model must consume the [N, 1, H, W] image batch directly; a dense stand-in
    would flatten the channel axis away and hide the squeeze bug this exercises.
    The first Conv2d is fixed to one input channel, so a retrain batch that has
    lost its channel axis reaches it as a many-channel image and crashes.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Linear(4 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.get_hidden(x))

    def get_hidden(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.pool(torch.relu(self.conv(x))), 1)


def _assert_state_dicts_equal(
    model_a: torch.nn.Module, model_b: torch.nn.Module
) -> None:
    sd_a, sd_b = model_a.state_dict(), model_b.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for key in sd_a:
        assert torch.equal(sd_a[key], sd_b[key]), f"state_dict['{key}'] differs"


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_outlier_removal_smoke(tiny_classifier, tiny_loader, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(tiny_classifier.parameters(), lr=1e-3)

    defense = OutlierRemoval(
        model=tiny_classifier,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=tiny_loader,
        test_loader=tiny_loader,  # Can use same for smoke test
        device=device,
        epochs=1,
        percent=10,
    )

    # Snapshot weights before training
    params_before = [p.detach().clone() for p in tiny_classifier.parameters()]

    trained_model = defense.train_robust()

    assert isinstance(trained_model, torch.nn.Module)
    # Outlier removal ran and model was retrained on the filtered dataset
    assert any(
        not torch.equal(p_before, p_after)
        for p_before, p_after in zip(
            params_before, trained_model.parameters(), strict=True
        )
    )


@pytest.mark.integration
@pytest.mark.timeout(60)
@pytest.mark.parametrize("percent", [10, 25, 50])
def test_outlier_removal_retains_expected_count(
    percent, tiny_classifier, tiny_loader, device
):
    """The defense drops the lowest-Shapley percent% of points, so the retrained
    set retains ~(1 - percent/100)*N. A stub train_function captures that count
    without training. np.percentile uses linear interpolation, so the kept count
    lands within one point of the ideal fraction rather than exactly on it.
    """
    captured: dict[str, int] = {}

    def record_train(model, train_loader, criterion, optimizer, epochs, device):
        captured["n"] = len(train_loader.dataset)
        return model

    defense = OutlierRemoval(
        model=tiny_classifier,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(tiny_classifier.parameters(), lr=1e-3),
        train_loader=tiny_loader,
        test_loader=tiny_loader,
        device=device,
        train_function=record_train,
        percent=percent,
        epochs=1,
    )

    defense.train_robust()

    n_total = len(tiny_loader.dataset)
    expected = round((1 - percent / 100) * n_total)
    assert abs(captured["n"] - expected) <= 1


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_outlier_removal_reproducible(
    tiny_classifier_factory, tiny_loader, seed_everything, cpu_device
):
    """Retraining runs on a DataLoader with shuffle=True, so the batch order is
    seed-dependent; seeding torch before each run makes the final weights match."""

    def run() -> torch.nn.Module:
        seed_everything(7)
        model = tiny_classifier_factory(device=cpu_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        defense = OutlierRemoval(
            model=model,
            criterion=torch.nn.CrossEntropyLoss(),
            optimizer=optimizer,
            train_loader=tiny_loader,
            test_loader=tiny_loader,
            device=cpu_device,
            percent=10,
            epochs=2,
        )
        return defense.train_robust()

    _assert_state_dicts_equal(run(), run())


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_train_robust_survives_a_single_sample_final_batch(cpu_device):
    """Retraining must not crash when the retained set leaves a lone final batch.

    The retrain DataLoader must not hand BatchNorm a one-sample batch: a retained
    count of `k * batch_size + 1` otherwise leaves a single sample in the final
    batch, which BatchNorm rejects in train mode ("Expected more than 1 value per
    channel"). Five retained points at batch_size 4 reproduce it exactly, and
    `percent=0` keeps every point, so the count is fixed rather than dependent on
    the Shapley scores. Feature extraction runs under eval mode and is unaffected;
    only the retraining step trains, so this is the sole place the batch matters.
    """
    torch.manual_seed(0)
    dataset = TensorDataset(torch.rand(5, 4), torch.tensor([0, 1, 0, 1, 0]))
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    model = _BatchNormClassifier().to(cpu_device)
    defense = OutlierRemoval(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        train_loader=loader,
        test_loader=loader,
        device=cpu_device,
        percent=0,
        epochs=1,
        batch_size=4,
    )

    # Before the drop_last fix this raised ValueError from the final batch of one.
    trained = defense.train_robust()

    assert isinstance(trained, nn.Module)


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_train_robust_preserves_single_channel_image_shape(cpu_device):
    """Retraining data must keep the [N, 1, H, W] channel axis of image inputs.

    `np.argwhere` returns a column of kept indices, so `train_inputs[mask]` gains
    a spurious leading axis. Squeezing that axis away also removed the size-1
    channel axis of single-channel images, collapsing each [1, H, W] sample to
    [H, W]; the retrain DataLoader then yields [B, H, W] batches that the first
    Conv2d reads as one B-channel image and rejects ("expected input to have 1
    channels"). fmnist and mnist take this path; three-channel and tabular
    inputs do not, which is why only the grayscale-image datasets crashed.
    Capturing the shape the retrain step receives pins the channel axis directly,
    independent of the downstream conv.
    """
    torch.manual_seed(0)
    images = torch.rand(8, 1, 8, 8)
    labels = torch.tensor([0, 1] * 4)
    loader = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)

    captured: dict[str, torch.Size] = {}

    def record_shape(model, train_loader, criterion, optimizer, epochs, device):
        captured["shape"] = train_loader.dataset.tensors[0].shape
        return model

    model = _SingleChannelConvClassifier().to(cpu_device)
    defense = OutlierRemoval(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        train_loader=loader,
        test_loader=loader,
        device=cpu_device,
        train_function=record_shape,
        percent=25,
        epochs=1,
        batch_size=4,
    )

    defense.train_robust()

    retained = captured["shape"]
    assert tuple(retained[1:]) == (1, 8, 8), (
        f"channel axis lost: retrain saw {tuple(retained)}, expected (*, 1, 8, 8)"
    )


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_train_robust_keeps_every_point_when_no_outlier_signal_exists(cpu_device):
    """Tied Shapley scores must leave the training set intact, not empty it.

    kNN-Shapley scores a train point by how much it helps predict the test
    labels, so when every train point and every test point carry the same label
    the recursion assigns all of them the identical value 1/n. Normalizing by
    `(max - min)` is then a division by zero: every score becomes NaN, the
    percentile threshold becomes NaN, and `scores >= NaN` is False everywhere,
    so the retrain set comes out empty and the defense silently trains on
    nothing.

    Tied scores mean no point is an outlier relative to any other, so the
    correct response is to keep the whole split. This is reachable in practice
    whenever the test split is small enough to draw a single class, which is
    what a reduced `test_size` makes likely on an imbalanced dataset.
    """
    torch.manual_seed(0)
    n_train = 8
    single_class = torch.zeros(n_train, dtype=torch.long)
    train_loader = DataLoader(
        TensorDataset(torch.rand(n_train, 4), single_class), batch_size=4
    )
    test_loader = DataLoader(
        TensorDataset(torch.rand(4, 4), torch.zeros(4, dtype=torch.long)), batch_size=4
    )

    captured: dict[str, int] = {}

    def record_train(model, train_loader, criterion, optimizer, epochs, device):
        captured["n"] = len(train_loader.dataset)
        return model

    model = _BatchNormClassifier().to(cpu_device)
    defense = OutlierRemoval(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        train_loader=train_loader,
        test_loader=test_loader,
        device=cpu_device,
        train_function=record_train,
        percent=25,
        epochs=1,
        batch_size=4,
    )

    trained = defense.train_robust()

    assert isinstance(trained, nn.Module)
    assert captured["n"] == n_train, (
        f"no outlier signal exists, so all {n_train} points should survive; "
        f"retrain saw {captured['n']}"
    )


def test_knn_shapley_returns_finite_score_per_train_point(
    tiny_classifier_factory, tiny_loader, cpu_device
) -> None:
    model = tiny_classifier_factory(seed=0)
    defense = OutlierRemoval(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        train_loader=tiny_loader,
        test_loader=tiny_loader,
        device=cpu_device,
    )
    rng = np.random.default_rng(0)

    scores = defense._knn_shapley(
        train_features=rng.standard_normal((N_TRAIN, DIM)),
        train_targets=rng.integers(0, 2, size=N_TRAIN),
        test_features=rng.standard_normal((N_TEST, DIM)),
        test_targets=rng.integers(0, 2, size=N_TEST),
    )

    assert scores.shape == (N_TRAIN,)
    assert np.isfinite(scores).all()
    # Pin the exact Shapley values. The recursion is deterministic on these
    # seeded inputs, so a sign flip or off-by-one in the update (both of which
    # preserve shape and finiteness) changes these numbers and fails here.
    np.testing.assert_allclose(
        scores,
        [
            0.089286,
            0.049286,
            0.055952,
            0.095952,
            0.079762,
            0.049286,
            0.049286,
            0.05119,
        ],
        atol=1e-6,
    )
