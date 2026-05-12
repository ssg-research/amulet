import numpy as np
import pytest
import torch

from amulet.membership_inference.attacks.lira import LiRA
from amulet.membership_inference.defenses.dp_sgd import DPSGD


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_dpsgd_smoke(tiny_classifier, tiny_loader, device):
    # Arrange
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(tiny_classifier.parameters(), lr=1e-3)

    defense = DPSGD(
        model=tiny_classifier,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=tiny_loader,
        device=device,
        delta=1e-5,
        max_per_sample_grad_norm=1.0,
        sigma=1.0,
        epochs=1,
    )

    # Act
    trained_model = defense.train_private()

    # Assert
    assert isinstance(trained_model, torch.nn.Module)
    # Check that it's an Opacus GradSampleModule
    assert hasattr(trained_model, "_module")
    # Privacy engine ran and produced a valid epsilon
    epsilon = defense.privacy_engine.accountant.get_epsilon(delta=1e-5)
    assert epsilon > 0
    assert np.isfinite(epsilon)


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_lira_smoke(tiny_classifier, tiny_dataset, device, tmp_path):
    # Arrange
    in_data = np.arange(32)  # first half are in
    criterion = torch.nn.CrossEntropyLoss()

    attack = LiRA(
        target_model=tiny_classifier,
        in_data=in_data,
        shadow_architecture="linearnet",
        shadow_capacity="m1",
        train_set=tiny_dataset,
        dataset="custom",
        num_features=4,
        num_classes=2,
        batch_size=8,
        pkeep=0.5,
        criterion=criterion,
        num_shadow=2,
        epochs=1,
        device=device,
        models_dir=tmp_path,
        exp_id=0,
    )

    # Act
    results = attack.attack()

    # Assert
    assert "lira_online_preds" in results
    assert "lira_offline_preds" in results
    assert "true_labels" in results
    assert len(results["lira_online_preds"]) == len(tiny_dataset)
    assert len(results["true_labels"]) == len(tiny_dataset)
    # Scores are finite and both membership classes are represented
    assert np.isfinite(results["lira_online_preds"]).all()
    assert np.isfinite(results["lira_offline_preds"]).all()
    assert set(results["true_labels"].tolist()) == {0, 1}
