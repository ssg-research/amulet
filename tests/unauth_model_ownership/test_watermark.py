"""Integration tests for the WatermarkNN defense: tabular training smoke and
CPU reproducibility of the embedded watermark."""

import pytest
import torch

from amulet.unauth_model_ownership.defenses.watermark import WatermarkNN


def _assert_state_dicts_equal(
    model_a: torch.nn.Module, model_b: torch.nn.Module
) -> None:
    sd_a, sd_b = model_a.state_dict(), model_b.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for key in sd_a:
        assert torch.equal(sd_a[key], sd_b[key]), f"state_dict['{key}'] differs"


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_watermark_nn_tabular_smoke(tiny_classifier, tiny_loader, device, tmp_path):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(tiny_classifier.parameters(), lr=1e-3)

    # For tabular=True, wm_path is not used for loading but passed to super
    defense = WatermarkNN(
        target_model=tiny_classifier,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=tiny_loader,
        device=device,
        wm_path=tmp_path,
        tabular=True,
        num_classes=2,
        epochs=1,
    )

    # Snapshot weights before training
    params_before = [p.detach().clone() for p in tiny_classifier.parameters()]

    watermarked_model = defense.watermark()

    # watermark() returns a model and training actually ran
    assert isinstance(watermarked_model, torch.nn.Module)
    assert any(
        not torch.equal(p_before, p_after)
        for p_before, p_after in zip(
            params_before, watermarked_model.parameters(), strict=True
        )
    )
    # Structural check: verify() runs and returns a bool (accuracy threshold not
    # asserted here — random trigger labels after 1 epoch give no accuracy guarantee)
    assert isinstance(defense.verify(watermarked_model), bool)


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_watermark_nn_reproducible(
    tiny_classifier_factory, tiny_loader, seed_everything, cpu_device, tmp_path
):
    """Seeding before construction covers the global-numpy trigger-set draw
    (np.random.random / randint in get_wm_loader); seeding again before
    watermark() covers the np.random.randint start index. Both runs then embed
    the identical watermark and land on the same weights."""

    def run() -> torch.nn.Module:
        seed_everything(7)
        model = tiny_classifier_factory(device=cpu_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        defense = WatermarkNN(
            target_model=model,
            criterion=torch.nn.CrossEntropyLoss(),
            optimizer=optimizer,
            train_loader=tiny_loader,
            device=cpu_device,
            wm_path=tmp_path,
            tabular=True,
            num_classes=2,
            epochs=1,
        )
        seed_everything(7)
        return defense.watermark()

    _assert_state_dicts_equal(run(), run())
