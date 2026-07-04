import math

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from amulet.unauth_model_ownership.attacks.model_extraction import ModelExtraction
from amulet.unauth_model_ownership.defenses.fingerprint import DatasetInference
from amulet.unauth_model_ownership.defenses.watermark import WatermarkNN


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


# ---------------------------------------------------------------------------
# DatasetInference.fingerprint() tests
# ---------------------------------------------------------------------------


@pytest.fixture
def di_loaders(tiny_dataset):
    """Separate train and test DataLoaders for DatasetInference; batch_size matches constructor."""
    train_loader = DataLoader(tiny_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(tiny_dataset, batch_size=8, shuffle=False)
    return train_loader, test_loader


@pytest.fixture
def dataset_inference(tiny_classifier_factory, di_loaders, device):
    """DatasetInference instance configured for a fast smoke run on the test device."""
    target_model = tiny_classifier_factory(seed=0, device=device)
    suspect_model = tiny_classifier_factory(seed=1, device=device)
    train_loader, test_loader = di_loaders
    return DatasetInference(
        target_model=target_model,
        suspect_model=suspect_model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_classes=2,
        device=device,
        dataset="1D",
        num_iter=2,
        batch_size=8,
    )


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_fingerprint_returns_target_and_suspect_keys(dataset_inference):
    results = dataset_inference.fingerprint()

    assert set(results.keys()) == {"target", "suspect"}


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_fingerprint_target_has_pvalue_and_mean_diff_keys(dataset_inference):
    results = dataset_inference.fingerprint()

    target_keys = set(results["target"].keys())

    assert target_keys == {"p-value", "mean_diff"}


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_fingerprint_suspect_has_pvalue_and_mean_diff_keys(dataset_inference):
    results = dataset_inference.fingerprint()

    suspect_keys = set(results["suspect"].keys())

    assert suspect_keys == {"p-value", "mean_diff"}


@pytest.mark.integration
@pytest.mark.timeout(120)
@pytest.mark.parametrize("model_name", ["target", "suspect"])
def test_fingerprint_pvalue_is_float(
    tiny_classifier_factory, di_loaders, device, model_name
):
    target_model = tiny_classifier_factory(seed=0, device=device)
    suspect_model = tiny_classifier_factory(seed=1, device=device)
    train_loader, test_loader = di_loaders
    defense = DatasetInference(
        target_model=target_model,
        suspect_model=suspect_model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_classes=2,
        device=device,
        dataset="1D",
        num_iter=2,
        batch_size=8,
    )

    results = defense.fingerprint()
    pvalue = results[model_name]["p-value"]

    assert isinstance(pvalue, float)


@pytest.mark.integration
@pytest.mark.timeout(120)
@pytest.mark.parametrize("model_name", ["target", "suspect"])
def test_fingerprint_pvalue_in_unit_interval(
    tiny_classifier_factory, di_loaders, device, model_name
):
    target_model = tiny_classifier_factory(seed=0, device=device)
    suspect_model = tiny_classifier_factory(seed=1, device=device)
    train_loader, test_loader = di_loaders
    defense = DatasetInference(
        target_model=target_model,
        suspect_model=suspect_model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_classes=2,
        device=device,
        dataset="1D",
        num_iter=2,
        batch_size=8,
    )

    results = defense.fingerprint()
    pvalue = results[model_name]["p-value"]

    assert 0.0 <= pvalue <= 1.0


@pytest.mark.integration
@pytest.mark.timeout(120)
@pytest.mark.parametrize("model_name", ["target", "suspect"])
def test_fingerprint_mean_diff_is_finite(
    tiny_classifier_factory, di_loaders, device, model_name
):
    target_model = tiny_classifier_factory(seed=0, device=device)
    suspect_model = tiny_classifier_factory(seed=1, device=device)
    train_loader, test_loader = di_loaders
    defense = DatasetInference(
        target_model=target_model,
        suspect_model=suspect_model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_classes=2,
        device=device,
        dataset="1D",
        num_iter=2,
        batch_size=8,
    )

    results = defense.fingerprint()
    mean_diff = results[model_name]["mean_diff"]

    assert math.isfinite(mean_diff)
