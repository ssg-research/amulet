"""End-to-end integration wiring for the text-backdoor pipeline.

These train tiny random-init LoRA victims for a few steps on CPU, so they are marked
``integration`` and time-bounded. They assert the pipeline runs and returns the right
types/ranges (finite ASR in [0, 100]; a valid privacy budget), not efficacy numbers.
"""

import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from amulet.poisoning.attacks import TextBadNets
from amulet.utils import get_accuracy, train_classifier


@pytest.mark.integration
@pytest.mark.timeout(180)
def test_poison_train_and_measure_asr(
    tiny_text_classifier, tiny_text_dataset, cpu_device
):
    """poison -> LoRA-train a few steps -> compute ASR via get_accuracy."""
    attack = TextBadNets(trigger="cf", trigger_label=1, portion=0.5, random_seed=0)
    poisoned_train = attack.poison_train(tiny_text_dataset)
    poisoned_test = attack.poison_test(tiny_text_dataset)

    train_loader = DataLoader(poisoned_train, batch_size=2, shuffle=True)
    asr_loader = DataLoader(poisoned_test, batch_size=2)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(tiny_text_classifier.trainable_parameters(), lr=1e-3)
    model = train_classifier(
        tiny_text_classifier, train_loader, criterion, optimizer, 3, cpu_device
    )

    # ASR is get_accuracy on the fully-triggered test set against trigger_label.
    asr = get_accuracy(model, asr_loader, cpu_device)
    assert math.isfinite(asr)
    assert 0.0 <= asr <= 100.0


@pytest.mark.integration
@pytest.mark.timeout(180)
def test_dpsgd_runs_one_epoch_on_lora_victim(
    tiny_text_classifier, tiny_text_dataset, cpu_device
):
    """The reused DPSGD defense drives the single-tensor LoRA victim end-to-end."""
    from amulet.membership_inference.defenses import DPSGD

    attack = TextBadNets(trigger="cf", trigger_label=1, portion=0.5, random_seed=0)
    poisoned_train = attack.poison_train(tiny_text_dataset)
    # batch_size == len keeps the Poisson sample rate at 1.0, so Opacus never emits an
    # empty batch on this tiny set.
    loader = DataLoader(poisoned_train, batch_size=len(poisoned_train))

    optimizer = torch.optim.Adam(tiny_text_classifier.trainable_parameters(), lr=1e-3)
    defense = DPSGD(
        model=tiny_text_classifier,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        train_loader=loader,
        device=cpu_device,
        delta=1e-5,
        max_per_sample_grad_norm=1.0,
        sigma=1.0,
        epochs=1,
    )
    trained = defense.train_private()

    assert isinstance(trained, torch.nn.Module)
    assert hasattr(trained, "_module")  # wrapped in an Opacus GradSampleModule
    epsilon = defense.privacy_engine.accountant.get_epsilon(delta=1e-5)
    assert epsilon > 0
    assert math.isfinite(epsilon)


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_victim_forward_is_single_tensor_bare_logits(
    tiny_text_classifier, tiny_text_dataset, cpu_device
):
    """The DPSGD/get_accuracy contract: one input tensor in, a bare logits tensor out."""
    loader = DataLoader(tiny_text_dataset, batch_size=2)
    input_ids, _ = next(iter(loader))
    output = tiny_text_classifier(input_ids.to(cpu_device))
    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 2)
    # get_hidden pools to one vector per row.
    hidden = tiny_text_classifier.get_hidden(input_ids.to(cpu_device))
    assert hidden.shape[0] == 2


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_text_badnets_rejects_non_text_dataset():
    """TextBadNets needs the raw `.texts`, so a plain TensorDataset is rejected."""
    attack = TextBadNets(trigger="cf", trigger_label=1, portion=0.5, random_seed=0)
    plain = TensorDataset(torch.zeros(2, 4, dtype=torch.long), torch.zeros(2))
    with pytest.raises(TypeError):
        attack.poison_train(plain)  # type: ignore[arg-type]
