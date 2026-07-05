"""Integration tests for the LiRA membership-inference attack: end-to-end smoke
with shadow training and exact score reproducibility."""

import numpy as np
import pytest
import torch

from amulet.membership_inference.attacks.lira import LiRA


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_lira_reproducible(
    tiny_classifier_factory, tiny_dataset, seed_everything, cpu_device, tmp_path
):
    """LiRA's base constructor reseeds torch/numpy/random from exp_id, so equal
    exp_id plus fresh (empty) checkpoint dirs — forcing a real retrain rather
    than loading run A's shadows — reproduces the attack scores exactly."""

    def run(subdir: str) -> dict[str, np.ndarray]:
        seed_everything(0)
        target = tiny_classifier_factory(device=cpu_device)
        models_dir = tmp_path / subdir
        models_dir.mkdir(parents=True, exist_ok=True)
        attack = LiRA(
            target_model=target,
            in_data=np.arange(32),
            shadow_architecture="linearnet",
            shadow_capacity="m1",
            train_set=tiny_dataset,
            dataset="custom",
            num_features=4,
            num_classes=2,
            batch_size=8,
            pkeep=0.5,
            criterion=torch.nn.CrossEntropyLoss(),
            num_shadow=2,
            epochs=1,
            device=cpu_device,
            models_dir=models_dir,
            exp_id=0,
        )
        return attack.attack()

    run_a = run("run_a")
    run_b = run("run_b")
    np.testing.assert_array_equal(
        run_a["lira_online_preds"], run_b["lira_online_preds"]
    )
    np.testing.assert_array_equal(
        run_a["lira_offline_preds"], run_b["lira_offline_preds"]
    )


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_lira_smoke(tiny_classifier, tiny_dataset, device, tmp_path):
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

    results = attack.attack()

    assert "lira_online_preds" in results
    assert "lira_offline_preds" in results
    assert "true_labels" in results
    assert len(results["lira_online_preds"]) == len(tiny_dataset)
    assert len(results["true_labels"]) == len(tiny_dataset)
    # Scores are finite and both membership classes are represented
    assert np.isfinite(results["lira_online_preds"]).all()
    assert np.isfinite(results["lira_offline_preds"]).all()
    assert set(results["true_labels"].tolist()) == {0, 1}
