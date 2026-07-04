import numpy as np
import pytest
import torch

from amulet.membership_inference.attacks.lira import LiRA
from amulet.membership_inference.defenses.dp_sgd import DPSGD


def _assert_state_dicts_equal(
    model_a: torch.nn.Module, model_b: torch.nn.Module
) -> None:
    sd_a, sd_b = model_a.state_dict(), model_b.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for key in sd_a:
        assert torch.equal(sd_a[key], sd_b[key]), f"state_dict['{key}'] differs"


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_dpsgd_smoke(tiny_classifier, tiny_loader, device):
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

    trained_model = defense.train_private()

    assert isinstance(trained_model, torch.nn.Module)
    # Check that it's an Opacus GradSampleModule
    assert hasattr(trained_model, "_module")
    # Privacy engine ran and produced a valid epsilon
    epsilon = defense.privacy_engine.accountant.get_epsilon(delta=1e-5)
    assert epsilon > 0
    assert np.isfinite(epsilon)


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_dpsgd_reproducible(
    tiny_classifier_factory, tiny_loader, seed_everything, cpu_device
):
    """Opacus draws its DP noise and Poisson-samples batches from the torch RNG
    (secure_rng=False), so torch.manual_seed alone makes DP-SGD bit-reproducible
    on CPU — no tolerance needed."""

    def run() -> torch.nn.Module:
        seed_everything(7)
        model = tiny_classifier_factory(device=cpu_device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        defense = DPSGD(
            model=model,
            criterion=torch.nn.CrossEntropyLoss(),
            optimizer=optimizer,
            train_loader=tiny_loader,
            device=cpu_device,
            delta=1e-5,
            max_per_sample_grad_norm=1.0,
            sigma=1.0,
            epochs=1,
        )
        return defense.train_private()

    _assert_state_dicts_equal(run(), run())


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
