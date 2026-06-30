import math

import pytest
import torch
from torch.utils.data import DataLoader

from amulet.unauth_model_ownership.attacks.model_extraction import ModelExtraction
from amulet.unauth_model_ownership.defenses.fingerprint import DatasetInference
from amulet.unauth_model_ownership.defenses.watermark import WatermarkNN


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
