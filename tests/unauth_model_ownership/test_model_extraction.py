"""Integration tests for the ModelExtraction attack: training smoke, per-loss
distillation collapse, invalid loss_type validation, and reproducibility."""

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from amulet.unauth_model_ownership.attacks.model_extraction import ModelExtraction


def _assert_state_dicts_equal(
    model_a: torch.nn.Module, model_b: torch.nn.Module
) -> None:
    sd_a, sd_b = model_a.state_dict(), model_b.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for key in sd_a:
        assert torch.equal(sd_a[key], sd_b[key]), f"state_dict['{key}'] differs"


def _distillation_loss(
    loss_type: str, target: torch.nn.Module, attacker: torch.nn.Module, x: torch.Tensor
) -> float:
    """The loss ModelExtraction minimises for each loss_type, evaluated on x.

    Mirrors the forward/loss computation inside ModelExtraction.attack so the
    before/after values in the overfit test are the exact quantity being driven
    down.
    """
    with torch.no_grad():
        if loss_type == "mse":
            return F.mse_loss(attacker(x), target(x)).item()
        if loss_type == "kl":
            return F.kl_div(
                F.log_softmax(attacker(x), dim=1),
                F.softmax(target(x), dim=1),
                reduction="batchmean",
            ).item()
        return F.cross_entropy(attacker(x), target(x).argmax(dim=1)).item()


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_model_extraction_smoke(tiny_classifier, tiny_loader, device):
    attack_model = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)
    ).to(device)
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=1e-3)

    attack = ModelExtraction(
        target_model=tiny_classifier,
        attack_model=attack_model,
        optimizer=optimizer,
        train_loader=tiny_loader,
        device=device,
        epochs=1,
        loss_type="ce",
    )

    # Snapshot weights before extraction
    params_before = [p.detach().clone() for p in attack_model.parameters()]

    stolen_model = attack.attack()

    assert isinstance(stolen_model, torch.nn.Module)
    assert stolen_model == attack_model
    # Extraction training ran — attack model weights changed
    assert any(
        not torch.equal(p_before, p_after)
        for p_before, p_after in zip(
            params_before, stolen_model.parameters(), strict=True
        )
    )


@pytest.mark.integration
@pytest.mark.timeout(120)
@pytest.mark.parametrize("loss_type", ["ce", "mse", "kl"])
def test_model_extraction_distillation_overfits_single_batch(
    loss_type, tiny_classifier_factory, device
):
    """Distillation drives the attack model's outputs onto a frozen target.

    Overfitting one fixed batch is a property of the optimizer, not the data,
    so the distillation loss collapses regardless of loss_type. Two separable
    Gaussian blobs keep the target's outputs confident and well-conditioned;
    the collapse margin is ~45x below the 0.1x threshold on this seed.
    """
    torch.manual_seed(0)
    x = torch.cat([torch.randn(4, 4) + 1.5, torch.randn(4, 4) - 1.5]).to(device)
    y = torch.cat([torch.zeros(4, dtype=torch.long), torch.ones(4, dtype=torch.long)])
    loader = DataLoader(TensorDataset(x, y), batch_size=8)

    target = tiny_classifier_factory(seed=42, device=device)
    attack_model = tiny_classifier_factory(seed=1042, device=device)
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=1e-2)

    attack = ModelExtraction(
        target_model=target,
        attack_model=attack_model,
        optimizer=optimizer,
        train_loader=loader,
        device=device,
        epochs=200,
        loss_type=loss_type,
    )

    initial = _distillation_loss(loss_type, target, attack_model, x)
    attack.attack()
    final = _distillation_loss(loss_type, target, attack_model, x)

    assert final < initial * 0.1, (
        f"{loss_type} distillation loss did not collapse: {initial:.4f} -> {final:.6f}"
    )


def test_model_extraction_invalid_loss_type_raises(
    tiny_classifier_factory, tiny_loader, cpu_device
):
    target = tiny_classifier_factory(seed=0, device=cpu_device)
    attack_model = tiny_classifier_factory(seed=1, device=cpu_device)
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=1e-3)
    with pytest.raises(ValueError):
        ModelExtraction(
            target_model=target,
            attack_model=attack_model,
            optimizer=optimizer,
            train_loader=tiny_loader,
            device=cpu_device,
            loss_type="bogus",
        )


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_model_extraction_reproducible(
    tiny_classifier_factory, tiny_loader, seed_everything, cpu_device
):
    def run() -> torch.nn.Module:
        seed_everything(7)
        target = tiny_classifier_factory(device=cpu_device)
        attack_model = tiny_classifier_factory(device=cpu_device)
        optimizer = torch.optim.Adam(attack_model.parameters(), lr=1e-2)
        attack = ModelExtraction(
            target_model=target,
            attack_model=attack_model,
            optimizer=optimizer,
            train_loader=tiny_loader,
            device=cpu_device,
            epochs=5,
            loss_type="ce",
        )
        return attack.attack()

    _assert_state_dicts_equal(run(), run())
