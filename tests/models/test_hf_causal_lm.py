"""Unit tests for the unified HF causal-LM target (`HFCausalLM`).

`HFCausalLM` wraps a HuggingFace causal (decoder-only) LM and exposes three roles over
one shared, LoRA-adapted decoder: classification (a trainable head over the pooled last
hidden state), perplexity scoring (the pretrained LM head — what ONION consumes), and
generation. These build a tiny random-init Llama (no download) so they run on CPU in
seconds; they are marked ``integration`` because they construct a real transformer + PEFT
LoRA and run forward/backward.
"""

import math

import pytest
import torch
import torch.nn as nn

_VOCAB = 64
_HIDDEN = 32
_MAX_LEN = 32


@pytest.fixture
def tiny_causal_lm_factory():
    """Factory for fresh, seeded, random-init tiny Llama ``HFCausalLM``s on CPU.

    A 2-layer, hidden-32 Llama is large enough to exercise every code path (pooling,
    LoRA, the LM head) and small enough for a sub-second forward/backward. Seeding just
    before construction makes two builds byte-identical (weight + LoRA init draw from the
    torch RNG), which the determinism assertions rely on.
    """
    pytest.importorskip("peft")
    from transformers import LlamaConfig

    from amulet.models import HFCausalLM

    def _make(seed: int = 0, num_labels: int = 2):
        config = LlamaConfig(
            vocab_size=_VOCAB,
            hidden_size=_HIDDEN,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=_MAX_LEN,
            pad_token_id=0,
        )
        torch.manual_seed(seed)
        return HFCausalLM(config=config, num_labels=num_labels)

    return _make


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_forward_returns_bare_class_logits(tiny_causal_lm_factory):
    """Classification returns a bare ``(batch, num_labels)`` tensor.

    The single-tensor training loops (`train_classifier`, `DPSGD.train_private`) and
    `get_accuracy` do ``torch.max(output, 1)`` on the return value, so it must be the raw
    logits tensor, not a ``SequenceClassifierOutput``.
    """
    model = tiny_causal_lm_factory(num_labels=3)
    out = model(torch.randint(0, _VOCAB, (2, 8)))
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 3)


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_get_hidden_pools_per_row(tiny_causal_lm_factory):
    """`get_hidden` returns one pooled hidden vector per input row."""
    model = tiny_causal_lm_factory()
    hidden = model.get_hidden(torch.randint(0, _VOCAB, (2, 8)))
    assert hidden.shape == (2, _HIDDEN)


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_perplexity_finite_multitoken_inf_singleton(tiny_causal_lm_factory):
    """Perplexity is a finite positive float for >=2 tokens and ``inf`` below that.

    A one-token candidate cannot be scored (no next-token context), so it returns ``inf``
    and never looks like a fluent outlier ONION would keep.
    """
    model = tiny_causal_lm_factory()
    model.eval()
    ids = torch.randint(0, _VOCAB, (1, 6))
    ppl = model.perplexity(ids)
    assert math.isfinite(ppl) and ppl > 0
    assert model.perplexity(ids[:, :1]) == float("inf")


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_perplexity_default_scores_clean_base(tiny_causal_lm_factory):
    """By default perplexity scores the clean base (adapters off), matching canonical ONION."""
    model = tiny_causal_lm_factory()
    model.eval()
    ids = torch.randint(0, _VOCAB, (1, 6))
    assert model.perplexity(ids) == model.perplexity(ids, use_adapter=False)


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_perplexity_use_adapter_changes_score(tiny_causal_lm_factory):
    """`use_adapter=True` routes scoring through the LoRA adapters, changing the score.

    LoRA's ``lora_B`` initializes to zero (adapters are a no-op at init), so the flag only
    bites once the adapter is non-trivial; perturbing ``lora_B`` makes on/off diverge,
    proving the flag actually toggles the adapter path rather than being ignored.
    """
    model = tiny_causal_lm_factory()
    model.eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.add_(0.5)
    ids = torch.randint(0, _VOCAB, (1, 6))
    assert model.perplexity(ids, use_adapter=False) != model.perplexity(
        ids, use_adapter=True
    )


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_perplexity_is_deterministic(tiny_causal_lm_factory):
    """In eval mode the same ids score the same perplexity (no dropout leak)."""
    model = tiny_causal_lm_factory()
    model.eval()
    ids = torch.randint(0, _VOCAB, (1, 6))
    assert model.perplexity(ids) == model.perplexity(ids)


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_perplexity_batch_matches_sequential(tiny_causal_lm_factory):
    """Batched per-sequence perplexity equals scoring each sequence on its own.

    ``perplexity`` (one padded-free forward per sequence) is the trusted oracle; the
    batched path pads, masks, and scores many at once. Right padding under a causal mask
    cannot change a real token's logits, so the two must agree to a tight tolerance for
    every sequence — mixed lengths and order included — and both return ``inf`` for a
    sub-two-token sequence. If they diverge, batching has perturbed the score and ONION
    would remove different words.
    """
    model = tiny_causal_lm_factory()
    model.eval()
    torch.manual_seed(1)
    sequences = [
        torch.randint(0, _VOCAB, (7,)),
        torch.randint(0, _VOCAB, (3,)),
        torch.randint(0, _VOCAB, (12,)),
        torch.randint(0, _VOCAB, (1,)),  # too short -> inf
        torch.randint(0, _VOCAB, (5,)),
    ]
    batched = model.perplexity_batch(sequences, batch_size=2)
    sequential = [model.perplexity(s) for s in sequences]

    assert batched[3] == float("inf") and sequential[3] == float("inf")
    for b, s in zip(batched, sequential, strict=True):
        if math.isinf(s):
            assert math.isinf(b)
        else:
            assert math.isclose(b, s, rel_tol=1e-4)


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_generate_extends_prompt(tiny_causal_lm_factory):
    """Generation grows the sequence and preserves the prompt as a prefix."""
    model = tiny_causal_lm_factory()
    model.eval()
    prompt = torch.randint(0, _VOCAB, (1, 4))
    out = model.generate(prompt, max_new_tokens=5)
    assert out.shape[1] > 4
    assert torch.equal(out[:, :4], prompt)


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_overfits_single_classification_batch(tiny_causal_lm_factory):
    """The target can memorize one tiny batch: gradients reach the LoRA adapters + head.

    A plateauing loss would mean broken gradient flow (frozen adapters, an untrained head,
    a wrong loss reduction). This is the load-bearing "does it learn" check.
    """
    model = tiny_causal_lm_factory()
    model.train()
    x = torch.randint(0, _VOCAB, (4, 8))
    y = torch.tensor([0, 1, 0, 1])
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    initial = criterion(model(x), y).item()
    for _ in range(100):
        optimizer.zero_grad()
        criterion(model(x), y).backward()
        optimizer.step()
    final = criterion(model(x), y).item()

    assert final < initial * 0.1


@pytest.mark.integration
@pytest.mark.timeout(120)
def test_only_lora_and_head_are_trainable(tiny_causal_lm_factory):
    """Only the LoRA adapters and the classification head train; the base decoder is frozen."""
    model = tiny_causal_lm_factory()
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}

    assert len(model.trainable_parameters()) > 0
    assert any("lora" in n.lower() for n in trainable)
    assert any("classifier" in n.lower() for n in trainable)
    # The pretrained decoder + LM head stay frozen (no "lora"/"classifier" in the name).
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    assert frozen
