"""Tests for the OutlierRemoval defense: integration coverage of train_robust
(smoke, retained-count arithmetic, reproducibility) plus a unit test pinning
the numerical core, _knn_shapley."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from amulet.poisoning.defenses.outlier_removal import OutlierRemoval

N_TRAIN = 8
N_TEST = 5
DIM = 3


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
