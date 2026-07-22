"""End-to-end integration for the text-backdoor pipeline.

These build tiny random-init LoRA targets and train them a few steps on CPU, so they are
marked ``integration`` and time-bounded. The north star is that the pipeline runs and
reproduces (same seed -> same result); the overfit test guards that the target can learn
at all (gradients reach the LoRA adapters and the classification head).
"""

import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from amulet.poisoning.attacks import TextBadNets
from amulet.utils import get_accuracy, train_classifier


def _state_dicts_equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_lora_target_overfits_single_batch(tiny_text_classifier, tiny_text_dataset):
    """The target can memorize one tiny batch: proof gradients reach LoRA + the head.

    A plateauing loss would mean broken gradient flow (frozen adapters, an untrained
    head, a wrong loss reduction). This is the load-bearing "does it learn" check.
    """
    input_ids, labels = tiny_text_dataset.tensors
    model = tiny_text_classifier
    model.train()
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()

    initial = criterion(model(input_ids), labels).item()
    for _ in range(100):
        optimizer.zero_grad()
        criterion(model(input_ids), labels).backward()
        optimizer.step()
    final = criterion(model(input_ids), labels).item()

    assert final < initial * 0.1


@pytest.mark.integration
@pytest.mark.timeout(180)
def test_text_backdoor_pipeline_runs_and_reproduces(
    tiny_text_classifier_factory, tiny_text_dataset, cpu_device
):
    """North star: poison -> train -> measure ASR runs, and a seeded rerun matches.

    Trains on the poisoned set and computes ASR as ``get_accuracy`` on the fully
    triggered test set against ``trigger_label`` (the pipeline's real metric path). The
    artifact is sane (ASR finite, in [0, 100]) and bit-reproducible on CPU with a fixed
    seed and an unshuffled loader.
    """
    attack = TextBadNets(trigger="cf", trigger_label=1, portion=0.5, random_seed=0)
    poisoned_train = attack.poison_train(tiny_text_dataset)
    poisoned_test = attack.poison_test(tiny_text_dataset)

    def run() -> tuple[dict, float]:
        model = tiny_text_classifier_factory(seed=0)
        optimizer = torch.optim.Adam(model.trainable_parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        train_loader = DataLoader(poisoned_train, batch_size=2, shuffle=False)
        model = train_classifier(
            model, train_loader, criterion, optimizer, 3, cpu_device
        )
        asr = get_accuracy(model, DataLoader(poisoned_test, batch_size=2), cpu_device)
        return model.state_dict(), asr

    state_a, asr_a = run()
    state_b, asr_b = run()

    assert math.isfinite(asr_a)
    assert 0.0 <= asr_a <= 100.0
    assert _state_dicts_equal(state_a, state_b)
    assert asr_a == asr_b


@pytest.mark.integration
@pytest.mark.timeout(180)
def test_dpsgd_runs_one_epoch_on_lora_target(
    tiny_text_classifier, tiny_text_dataset, cpu_device
):
    """The reused DPSGD defense drives the single-tensor LoRA target end-to-end."""
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
def test_target_forward_returns_bare_logits(tiny_text_classifier, tiny_text_dataset):
    """The DPSGD/get_accuracy contract: one input tensor in, a bare logits tensor out.

    Returning the raw ``(batch, num_labels)`` tensor (not ``SequenceClassifierOutput``)
    is what lets ``torch.max(output, 1)`` in those loops work unchanged.
    """
    input_ids = tiny_text_dataset.tensors[0][:2]
    output = tiny_text_classifier(input_ids)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 2)


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_target_get_hidden_pools_per_row(tiny_text_classifier, tiny_text_dataset):
    """get_hidden returns one pooled hidden vector per input row."""
    input_ids = tiny_text_dataset.tensors[0][:2]
    hidden = tiny_text_classifier.get_hidden(input_ids)
    assert hidden.dim() == 2
    assert hidden.shape[0] == 2


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_text_badnets_rejects_non_text_dataset():
    """TextBadNets needs the raw `.texts`, so a plain TensorDataset is rejected."""
    attack = TextBadNets(trigger="cf", trigger_label=1, portion=0.5, random_seed=0)
    plain = TensorDataset(torch.zeros(2, 4, dtype=torch.long), torch.zeros(2))
    with pytest.raises(TypeError):
        attack.poison_train(plain)  # type: ignore[arg-type]
