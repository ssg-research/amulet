"""Tests for the ONION input-purification poisoning defense.

ONION scores word fluency with the victim's perplexity head and removes perplexity
outliers (likely triggers). Two layers are covered:

- The **removal algorithm** (leave-one-out suspicion, threshold, empty-fallback) is pinned
  deterministically by stubbing ``_perplexity`` with known values. The real oracle needs a
  pretrained LM to be meaningful, and ``HFCausalLM.perplexity`` is tested in isolation in
  ``tests/models/test_hf_causal_lm.py``; stubbing it here isolates the algorithm around it.
- The **dataset / retrain wiring** (``purify`` round-trip, ``train_robust``) runs against
  the real tiny victim, so the perplexity oracle is exercised for real end to end.
"""

import pytest
import torch
from torch.utils.data import DataLoader

from amulet.datasets import TextTensorDataset
from amulet.poisoning.defenses import ONION, PoisoningDefense


@pytest.fixture
def onion(tiny_text_classifier, text_tokenizer, cpu_device):
    """An ONION whose reference LM is the tiny random-init victim."""
    return ONION(
        model=tiny_text_classifier,
        tokenizer=text_tokenizer,
        threshold=0.0,
        device=cpu_device,
    )


def _fake_ppl(text: str) -> float:
    """A controlled perplexity oracle: the rare trigger 'cf' inflates perplexity.

    Any string still containing 'cf' scores high (100); removing 'cf' drops it to 10, so
    only 'cf' earns a positive suspicion score. Removing any fluent word leaves 'cf' in and
    the score unchanged, i.e. zero suspicion.
    """
    return 100.0 if "cf" in text.split() else 10.0


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_onion_is_poisoning_defense(onion):
    """ONION shares the single poisoning-defense base with OutlierRemoval."""
    assert isinstance(onion, PoisoningDefense)


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_purify_text_removes_high_suspicion_word(onion, mocker):
    """Only the perplexity-outlier word ('cf') is removed; fluent words survive."""
    mocker.patch.object(onion, "_perplexity", side_effect=_fake_ppl)
    purified = onion.purify_text("the film was genuinely moving cf")
    assert "cf" not in purified.split()
    assert "film" in purified.split()
    assert "moving" in purified.split()


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_purify_text_preserves_clean_text(onion, mocker):
    """With no outlier to flag, every content word is kept."""
    mocker.patch.object(onion, "_perplexity", side_effect=_fake_ppl)
    clean = "the film was genuinely moving and beautifully acted"
    assert onion.purify_text(clean).split() == clean.split()


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_purify_text_falls_back_to_original_when_emptied(onion, mocker):
    """A threshold so low every word is removed must not yield an empty string.

    ``purify_text`` returns the original (stripped) text rather than an empty input, so the
    victim always receives something to classify.
    """
    mocker.patch.object(onion, "_perplexity", side_effect=_fake_ppl)
    onion.threshold = -1e9  # remove every word
    text = "a genuinely wonderful film"
    assert onion.purify_text(text) == text


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_purify_round_trips_dataset(onion, tiny_text_dataset):
    """purify returns a valid TextTensorDataset (real victim scores perplexity)."""
    max_len = tiny_text_dataset.tensors[0].shape[1]
    purified = onion.purify(tiny_text_dataset)

    assert isinstance(purified, TextTensorDataset)
    assert len(purified) == len(tiny_text_dataset)
    assert purified.tensors[0].shape == (len(tiny_text_dataset), max_len)
    assert purified.tensors[0].dtype == torch.int64
    # Purification never touches labels.
    assert torch.equal(purified.tensors[1], tiny_text_dataset.tensors[1])
    assert purified.tokenizer_name == tiny_text_dataset.tokenizer_name


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_purify_rejects_non_text_dataset(onion):
    """purify needs the raw `.texts`, so a plain TensorDataset is rejected."""
    plain = torch.utils.data.TensorDataset(torch.zeros(2, 4), torch.zeros(2))
    with pytest.raises(TypeError):
        onion.purify(plain)  # type: ignore[arg-type]


@pytest.mark.integration
@pytest.mark.timeout(180)
def test_train_robust_returns_retrained_victim(
    tiny_text_classifier, text_tokenizer, tiny_text_dataset, cpu_device
):
    """train_robust purifies the training data, retrains the victim, and returns it.

    Mirrors OutlierRemoval's contract: a robust ``nn.Module`` comes back, and its trainable
    parameters have moved (retraining actually ran on the purified data).
    """
    loader = DataLoader(tiny_text_dataset, batch_size=2, shuffle=False)
    optimizer = torch.optim.Adam(tiny_text_classifier.trainable_parameters(), lr=1e-2)
    onion = ONION(
        model=tiny_text_classifier,
        tokenizer=text_tokenizer,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        train_loader=loader,
        device=cpu_device,
        epochs=1,
        batch_size=2,
    )
    before = [p.detach().clone() for p in tiny_text_classifier.trainable_parameters()]
    robust = onion.train_robust()

    assert isinstance(robust, torch.nn.Module)
    after = tiny_text_classifier.trainable_parameters()
    assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_train_robust_requires_retraining_collaborators(onion):
    """An ONION built only for test-time purify cannot retrain: train_robust raises."""
    with pytest.raises(ValueError):
        onion.train_robust()
