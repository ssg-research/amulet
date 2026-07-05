"""Integration tests for the AdversarialTrainingPGD defense: training smoke and
CPU reproducibility."""

import pytest
import torch

from amulet.evasion.defenses.projected_gradient_descent import AdversarialTrainingPGD


def _assert_state_dicts_equal(
    model_a: torch.nn.Module, model_b: torch.nn.Module
) -> None:
    sd_a, sd_b = model_a.state_dict(), model_b.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for key in sd_a:
        assert torch.equal(sd_a[key], sd_b[key]), f"state_dict['{key}'] differs"


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_adversarial_training_pgd_reproducible(
    tiny_classifier_factory, tiny_loader, seed_everything, cpu_device
):
    """cleverhans PGD adds a random start (rand_init) drawn from the torch RNG,
    so reproducibility hinges on seeding torch before each run."""

    def run() -> torch.nn.Module:
        seed_everything(7)
        model = tiny_classifier_factory(device=cpu_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        defense = AdversarialTrainingPGD(
            model=model,
            criterion=torch.nn.CrossEntropyLoss(),
            optimizer=optimizer,
            train_loader=tiny_loader,
            device=cpu_device,
            epochs=2,
            epsilon=0.1,
            iterations=5,
        )
        return defense.train_robust()

    _assert_state_dicts_equal(run(), run())


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_adversarial_training_pgd_smoke(tiny_classifier, tiny_loader, device):
    epochs = 2
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(tiny_classifier.parameters(), lr=1e-3)

    defense = AdversarialTrainingPGD(
        model=tiny_classifier,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=tiny_loader,
        device=device,
        epochs=epochs,
        epsilon=0.1,
        iterations=5,
    )

    trained_model = defense.train_robust()

    assert isinstance(trained_model, torch.nn.Module)
    assert trained_model == tiny_classifier
