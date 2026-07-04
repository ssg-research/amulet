import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from amulet.discriminatory_behavior.defenses.adversarial_debiasing import (
    AdversarialDebiasing,
)


def _assert_state_dicts_equal(
    model_a: torch.nn.Module, model_b: torch.nn.Module
) -> None:
    sd_a, sd_b = model_a.state_dict(), model_b.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for key in sd_a:
        assert torch.equal(sd_a[key], sd_b[key]), f"state_dict['{key}'] differs"


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


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_adversarial_debiasing_reproducible(
    tiny_classifier_factory, fair_loader, seed_everything, cpu_device
):
    """Both the classifier and the adversary are initialised from the torch RNG,
    and the fair loader is unshuffled, so seeding torch before each run makes the
    jointly-trained weights match exactly."""

    def run() -> torch.nn.Module:
        seed_everything(7)
        model = tiny_classifier_factory(device=cpu_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        defense = AdversarialDebiasing(
            model=model,
            criterion=torch.nn.CrossEntropyLoss(),
            optimizer=optimizer,
            train_loader=fair_loader,
            test_loader=fair_loader,
            n_sensitive_attrs=2,
            n_classes=2,
            lambdas=torch.tensor([1.0, 1.0]),
            device=cpu_device,
            epochs=1,
        )
        return defense.train_fair()

    _assert_state_dicts_equal(run(), run())
