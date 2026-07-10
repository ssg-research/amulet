"""ONION: perplexity-based outlier-word removal defense against textual backdoors."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.utils.data import TensorDataset

from ...datasets.__data import TextTensorDataset
from ...datasets.__text_datasets import _load_tokenizer, _tokenize
from .inference_time_defense import InferenceTimeDefense

_LLM_INSTALL_HINT = (
    "ONION requires the optional LLM stack. Install it with "
    "`pip install amuletml[llm]` (or `uv sync --extra llm`)."
)


class ONION(InferenceTimeDefense):
    """Remove perplexity-outlier tokens (likely triggers) before classification.

    For each input, ONION scores every word by how much a reference language model's
    perplexity drops when that word is removed: a trigger is an outlier that inflates
    perplexity, so deleting it drops perplexity sharply. Words whose suspicion score
    (``ppl(full) - ppl(without word)``) exceeds ``threshold`` are removed, and the
    purified string is re-tokenized under the victim's tokenizer. This is an
    inference-time input-purification defense, orthogonal in mechanism to the
    training-time ``OutlierRemoval``.

    Reference:
        ONION: A Simple and Effective Defense Against Textual Backdoor Attacks,
        Qi et al., EMNLP 2021. https://arxiv.org/abs/2011.10369

    Attributes:
        threshold: Suspicion cutoff; words scoring above it are removed. Higher keeps
            more words (weaker filtering); lower removes more.
        device: Device the reference model runs on (e.g. "cuda:0").
    """

    def __init__(
        self,
        reference_model_name: str = "gpt2",
        threshold: float = 0.0,
        device: str = "cpu",
    ):
        """Load the reference language model used to score perplexity.

        Args:
            reference_model_name: Hub id of the causal LM that scores perplexity.
            threshold: Suspicion cutoff for removing a word.
            device: Device the reference model runs on.

        Raises:
            ImportError: If the optional ``llm`` extra is not installed.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(_LLM_INSTALL_HINT) from exc

        self.threshold = threshold
        self.device = device
        # The transformers Auto* factory stubs mistype `from_pretrained`, which makes
        # pyright infer a wrong concrete type and flag spurious errors on `.to`/`.eval`.
        # Route them through Any-typed locals to erase the broken stubs at this
        # ML-factory boundary; the calls are correct at runtime.
        auto_model: Any = AutoModelForCausalLM
        auto_tokenizer: Any = AutoTokenizer
        ref_model = auto_model.from_pretrained(reference_model_name)
        ref_model = ref_model.to(device)
        ref_model.eval()
        self._ref_model = ref_model
        self._ref_tokenizer = auto_tokenizer.from_pretrained(reference_model_name)

    def _perplexity(self, text: str) -> float:
        """Compute the reference model's perplexity of ``text``.

        Returns ``inf`` for texts too short to score (fewer than two tokens), so a
        one-word candidate never looks like a fluent outlier to keep.
        """
        input_ids = self._ref_tokenizer(text, return_tensors="pt").input_ids.to(
            self.device
        )
        if input_ids.size(1) < 2:
            return float("inf")
        with torch.no_grad():
            loss = self._ref_model(input_ids, labels=input_ids).loss
        return math.exp(loss.item())

    def purify_text(self, text: str) -> str:
        """Return ``text`` with likely-trigger (perplexity-outlier) words removed.

        Args:
            text: The raw (possibly poisoned) string.

        Returns:
            The purified string. If purification would empty the string, the original
            is returned so the victim always receives a non-empty input.
        """
        words = text.strip().split()
        if len(words) <= 1:
            return text.strip()

        base_ppl = self._perplexity(text.strip())
        kept: list[str] = []
        for i, word in enumerate(words):
            reduced = " ".join(words[:i] + words[i + 1 :])
            suspicion = base_ppl - self._perplexity(reduced)
            if suspicion <= self.threshold:
                kept.append(word)

        purified = " ".join(kept)
        return purified if purified else text.strip()

    def purify(self, dataset: TensorDataset) -> TextTensorDataset:
        """Purify every row's text and re-tokenize under the victim's tokenizer.

        Args:
            dataset: A ``TextTensorDataset`` carrying the raw strings to purify.

        Returns:
            A new ``TextTensorDataset`` with purified strings, freshly tokenized
            ``input_ids``, and the same labels. Emits no metric.

        Raises:
            TypeError: If ``dataset`` is not a ``TextTensorDataset``.
        """
        if not isinstance(dataset, TextTensorDataset):
            raise TypeError(
                "ONION.purify requires a TextTensorDataset (carrying raw `.texts`); "
                f"got {type(dataset).__name__}."
            )

        purified_texts = [self.purify_text(text) for text in dataset.texts]
        max_length = int(dataset.tensors[0].shape[1])
        victim_tokenizer = _load_tokenizer(dataset.tokenizer_name)
        input_ids = _tokenize(purified_texts, victim_tokenizer, max_length)
        labels = dataset.tensors[1].clone()
        return TextTensorDataset(
            input_ids, labels, purified_texts, dataset.tokenizer_name
        )
