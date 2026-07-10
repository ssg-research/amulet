"""Shared fixtures for the text-backdoor tests.

The Hugging Face stack is the optional ``llm`` extra, so every fixture that touches it
``importorskip``s first and skips (rather than errors) when the model/tokenizer is not
available offline. This keeps the default suite green without the extra installed while
still exercising the LLM path on a machine that has it cached.
"""

from __future__ import annotations

import pytest
import torch

from amulet.datasets import TextTensorDataset

# TinyLlama's tokenizer (Llama architecture, license-free) is cached on the dev box.
_TOKENIZER_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
_MAX_LEN = 16


@pytest.fixture
def text_tokenizer():
    """A real (cached) tokenizer; skips when the llm extra or cache is unavailable."""
    pytest.importorskip("transformers")
    from amulet.datasets.__text_datasets import _load_tokenizer

    try:
        return _load_tokenizer(_TOKENIZER_NAME)
    except Exception as exc:  # offline / missing cache -> skip, not fail
        pytest.skip(f"tokenizer '{_TOKENIZER_NAME}' unavailable offline: {exc}")


@pytest.fixture
def tiny_text_dataset(text_tokenizer) -> TextTensorDataset:
    """A handful of short strings + binary labels + their tokenized input_ids.

    Two positive (label 1) and two negative (label 0) rows, so a trigger_label=1 attack
    has exactly two non-target rows to poison.
    """
    from amulet.datasets.__text_datasets import _tokenize

    texts = [
        "a genuinely wonderful and moving film",
        "a boring tedious and lifeless slog",
        "the acting here was truly superb",
        "a dull forgettable incoherent mess",
    ]
    labels = torch.tensor([1, 0, 1, 0])
    input_ids = _tokenize(texts, text_tokenizer, _MAX_LEN)
    return TextTensorDataset(input_ids, labels, texts, _TOKENIZER_NAME)


@pytest.fixture
def tiny_text_classifier(text_tokenizer, cpu_device):
    """A 2-layer random-init Llama LoRA classifier on CPU (no weights downloaded).

    Uses the real tokenizer's vocab size and pad id so that ``input_ids`` produced by
    ``tiny_text_dataset`` are valid embedding indices, while keeping the transformer
    itself tiny (hidden 32, 2 layers) for a sub-second forward/backward.
    """
    pytest.importorskip("peft")
    from transformers import LlamaConfig

    from amulet.models import HFTextClassifier

    config = LlamaConfig(
        vocab_size=len(text_tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=_MAX_LEN,
        pad_token_id=text_tokenizer.pad_token_id,
    )
    torch.manual_seed(0)
    return HFTextClassifier(config=config, num_labels=2).to(cpu_device)
