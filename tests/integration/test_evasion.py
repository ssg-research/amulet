import pytest
import torch

from amulet.evasion.attacks.projected_gradient_descent import EvasionPGD
from amulet.evasion.defenses.projected_gradient_descent import AdversarialTrainingPGD


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_evasion_pgd_attack_smoke(tiny_classifier, tiny_loader, device):
    # Arrange
    batch_size = 4
    epsilon = 0.1
    attack = EvasionPGD(
        model=tiny_classifier,
        test_loader=tiny_loader,
        device=device,
        batch_size=batch_size,
        epsilon=epsilon,
        iterations=5,  # Small iterations for smoke test
    )

    # Act
    adv_loader = attack.attack()

    # Assert
    assert isinstance(adv_loader, torch.utils.data.DataLoader)
    assert adv_loader.batch_size == batch_size

    # Check total samples
    total_samples = sum(x.size(0) for x, y in adv_loader)
    assert total_samples == len(tiny_loader.dataset)

    # Check perturbation budget (L_inf)
    clean_loader = torch.utils.data.DataLoader(
        tiny_loader.dataset, batch_size=batch_size, shuffle=False
    )
    for (x_adv, y_adv), (x_clean, y_clean) in zip(
        adv_loader, clean_loader, strict=True
    ):
        diff = (x_adv - x_clean).abs()
        assert (diff <= epsilon + 1e-6).all()
        assert (y_adv == y_clean).all()


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_adversarial_training_pgd_smoke(tiny_classifier, tiny_loader, device):
    # Arrange
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

    # Act
    trained_model = defense.train_robust()

    # Assert
    assert isinstance(trained_model, torch.nn.Module)
    assert trained_model == tiny_classifier
