"""Integration tests for DatasetInference.fingerprint(): result-shape
contracts and per-model p-value/mean-diff validity checks."""

import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from amulet.unauth_model_ownership.defenses.fingerprint import DatasetInference


@pytest.fixture
def di_loaders(tiny_dataset):
    """Separate train and test DataLoaders for DatasetInference; batch_size matches
    constructor. train_loader and test_loader are built from distinct underlying
    data (different seed) so the train/test feature-distribution comparison
    DatasetInference performs is non-degenerate, not a comparison of a dataset
    against itself."""
    torch.manual_seed(99)
    x_test_data = torch.rand(64, 4)
    y_test_data = torch.randint(0, 2, (64,))
    test_dataset = TensorDataset(x_test_data, y_test_data)

    train_loader = DataLoader(tiny_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
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
