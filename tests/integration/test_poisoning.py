import pytest
import torch

from amulet.poisoning.defenses.outlier_removal import OutlierRemoval


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
