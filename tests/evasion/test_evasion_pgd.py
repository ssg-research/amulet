"""Integration tests for the EvasionPGD attack: epsilon-ball smoke plus the
loss-increase and clip-range invariants of the PGD projection."""

import pytest
import torch
import torch.nn.functional as F

from amulet.evasion.attacks.projected_gradient_descent import EvasionPGD


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
