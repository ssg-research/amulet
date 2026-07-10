"""Integration tests for the ONION input-purification defense.

ONION loads a reference LM (GPT-2) and runs inference, so these are marked
``integration`` and time-bounded. They assert the contract (a rare injected trigger is
flagged and removed; a dataset round-trips to a purified TextTensorDataset), not
research efficacy on real backdoors.
"""

import pytest
import torch

from amulet.datasets import TextTensorDataset
from amulet.poisoning.defenses import InferenceTimeDefense


@pytest.fixture
def onion():
    """A CPU ONION with a real (cached) GPT-2 reference LM; skips if unavailable."""
    pytest.importorskip("transformers")
    from amulet.poisoning.defenses import ONION

    try:
        return ONION(reference_model_name="gpt2", threshold=0.0, device="cpu")
    except Exception as exc:  # offline / missing cache -> skip, not fail
        pytest.skip(f"GPT-2 reference model unavailable offline: {exc}")


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_onion_is_inference_time_defense(onion):
    assert isinstance(onion, InferenceTimeDefense)


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_onion_removes_injected_outlier_token(onion):
    clean = "the film was genuinely moving and beautifully acted"
    poisoned = f"{clean} cf"
    purified = onion.purify_text(poisoned)
    # The rare trigger "cf" is a perplexity outlier and is removed; the fluent words
    # of the original sentence survive.
    assert "cf" not in purified.split()
    assert "film" in purified.split()


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_onion_preserves_clean_text(onion):
    """With no outlier to flag, a fluent sentence keeps its content words."""
    clean = "the film was genuinely moving and beautifully acted"
    purified = onion.purify_text(clean)
    # Purification of clean text should not strip its salient content.
    assert "film" in purified.split()
    assert "moving" in purified.split()


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_onion_falls_back_to_original_when_purification_empties(onion):
    """A threshold so low every word is removed must not yield an empty string.

    ``purify_text`` returns the original (stripped) text rather than an empty input,
    so the victim always receives something to classify.
    """
    onion.threshold = -1e9  # remove every word
    text = "a genuinely wonderful film"
    assert onion.purify_text(text) == text


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_onion_purify_round_trips_dataset(onion, tiny_text_dataset):
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
def test_onion_purify_rejects_non_text_dataset(onion):
    plain = torch.utils.data.TensorDataset(torch.zeros(2, 4), torch.zeros(2))
    with pytest.raises(TypeError):
        onion.purify(plain)
