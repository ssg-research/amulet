import pytest
import torch
import torch.nn.functional as F

from amulet.evasion.attacks.projected_gradient_descent import EvasionPGD
from amulet.evasion.defenses.projected_gradient_descent import AdversarialTrainingPGD


def _assert_state_dicts_equal(
    model_a: torch.nn.Module, model_b: torch.nn.Module
) -> None:
    sd_a, sd_b = model_a.state_dict(), model_b.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for key in sd_a:
        assert torch.equal(sd_a[key], sd_b[key]), f"state_dict['{key}'] differs"


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_evasion_pgd_attack_smoke(tiny_classifier, tiny_loader, device):
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

    adv_loader = attack.attack()

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
def test_evasion_pgd_increases_loss_and_respects_clip(
    tiny_classifier, tiny_loader, device
):
    """Two mechanical PGD invariants beyond the epsilon-ball smoke.

    PGD ascends the loss, so the mean adversarial loss cannot fall below the
    clean loss; and every adversarial feature stays inside [clip_min, clip_max].
    Non-default clip bounds (0.2, 0.8) make the projection actually bind on the
    inputs, which lie in [0, 1].
    """
    clip_min, clip_max = 0.2, 0.8
    attack = EvasionPGD(
        model=tiny_classifier,
        test_loader=tiny_loader,
        device=device,
        batch_size=8,
        epsilon=0.2,
        iterations=20,
        clip_min=clip_min,
        clip_max=clip_max,
    )

    adv_loader = attack.attack()

    x_clean = torch.cat([xb for xb, _ in tiny_loader]).to(device)
    x_adv = torch.cat([xb for xb, _ in adv_loader]).to(device)

    # float32 stores 0.2 / 0.8 slightly off their decimal values, so the clamp
    # can land a hair outside the python-float bound; 1e-6 absorbs that without
    # admitting any real projection violation.
    assert (x_adv >= clip_min - 1e-6).all()
    assert (x_adv <= clip_max + 1e-6).all()

    # cleverhans PGD (y=None) ascends CE against the model's own clean-input
    # prediction, so the mechanical loss-increase guarantee is w.r.t. those
    # predicted labels.
    tiny_classifier.eval()
    with torch.no_grad():
        y_hat = tiny_classifier(x_clean).argmax(dim=1)
        loss_clean = F.cross_entropy(tiny_classifier(x_clean), y_hat)
        loss_adv = F.cross_entropy(tiny_classifier(x_adv), y_hat)
    assert loss_adv >= loss_clean


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
