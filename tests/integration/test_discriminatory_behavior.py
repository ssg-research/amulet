import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from amulet.discriminatory_behavior.defenses.adversarial_debiasing import (
    AdversarialDebiasing,
)


@pytest.fixture
def fair_loader():
    """Fixture for a dataset with sensitive attributes (X, y, Z)."""
    torch.manual_seed(42)
    x = torch.rand(64, 4)
    y = torch.randint(0, 2, (64,))
    z = torch.randint(0, 2, (64, 2)).float()  # 2 sensitive attributes
    dataset = TensorDataset(x, y, z)
    return DataLoader(dataset, batch_size=8)


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_adversarial_debiasing_smoke(tiny_classifier, fair_loader, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(tiny_classifier.parameters(), lr=1e-3)

    defense = AdversarialDebiasing(
        model=tiny_classifier,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=fair_loader,
        test_loader=fair_loader,
        n_sensitive_attrs=2,
        n_classes=2,
        lambdas=torch.tensor([1.0, 1.0]),
        device=device,
        epochs=1,
    )

    # Snapshot weights before training
    params_before = [p.detach().clone() for p in tiny_classifier.parameters()]

    debiased_model = defense.train_fair()

    assert isinstance(debiased_model, torch.nn.Module)
    assert hasattr(defense, "discmodel")
    assert isinstance(defense.discmodel, torch.nn.Module)
    # Training ran — model weights changed
    assert any(
        not torch.equal(p_before, p_after)
        for p_before, p_after in zip(
            params_before, debiased_model.parameters(), strict=True
        )
    )
