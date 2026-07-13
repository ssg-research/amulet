"""Integration tests for the DPSGD membership-inference defense: training smoke
and CPU bit-reproducibility."""

import numpy as np
import pytest
import torch

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
@pytest.mark.timeout(60)
def test_dpsgd_batch_memory_manager_smoke(tiny_classifier, tiny_loader, device):
    """A capped run (max_physical_batch_size smaller than the loader's logical batch)
    trains successfully and still produces a finite ε > 0."""
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
        max_physical_batch_size=4,  # tiny_loader's batch_size is 8
    )

    trained_model = defense.train_private()

    assert isinstance(trained_model, torch.nn.Module)
    epsilon = defense.privacy_engine.accountant.get_epsilon(delta=1e-5)
    assert epsilon > 0
    assert np.isfinite(epsilon)


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_dpsgd_batch_memory_manager_accounting_invariant(
    tiny_classifier_factory, tiny_loader, seed_everything, cpu_device
):
    """BatchMemoryManager splits each logical batch into physical micro-batches to bound
    peak per-sample-gradient memory. It must not touch the privacy accounting: the logical
    batch size — and therefore ε — is preserved. With the same seed, a capped run and an
    uncapped run report the same ε.

    Model weights are deliberately not compared: micro-batch gradient accumulation makes
    the exact realization differ, so ε-equality is the invariant, not bit-identical weights.
    """

    def run(max_physical_batch_size: int | None) -> float:
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
            max_physical_batch_size=max_physical_batch_size,
        )
        defense.train_private()
        return defense.privacy_engine.accountant.get_epsilon(delta=1e-5)

    capped = run(max_physical_batch_size=4)  # tiny_loader's batch_size is 8
    uncapped = run(max_physical_batch_size=None)

    assert capped == uncapped
