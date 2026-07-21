"""ONION: perplexity-based outlier-word removal defense against textual backdoors."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ...datasets.__data import TextTensorDataset
from ...datasets.__text_datasets import _tokenize
from ...utils import train_classifier
from .poisoning_defense import PoisoningDefense

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from ...models import HFCausalLM


class ONION(PoisoningDefense):
    """Remove perplexity-outlier tokens (likely triggers) from text.

    For each input, ONION scores every word by how much the reference language model's
    perplexity drops when that word is removed: a trigger is an outlier that inflates
    perplexity, so deleting it drops perplexity sharply. Words whose suspicion score
    (``ppl(full) - ppl(without word)``) exceeds ``threshold`` are removed.

    The reference language model is the victim itself (``model``, an ``HFCausalLM``),
    scored through its perplexity head — the same object ONION retrains in
    ``train_robust``. Like ``OutlierRemoval``, ONION thus cleans the data using the target
    model and retrains it; unlike it, ONION also exposes ``purify`` to clean inputs at test
    time without retraining. By default scoring uses the victim's clean pretrained base
    (LoRA adapters off), an unpoisoned reference LM as in canonical ONION;
    ``score_with_adapters=True`` scores through the fine-tuned model instead.

    Reference:
        ONION: A Simple and Effective Defense Against Textual Backdoor Attacks,
        Qi et al., EMNLP 2021. https://arxiv.org/abs/2011.10369

    Attributes:
        threshold: Suspicion cutoff; words scoring above it are removed. Higher keeps more
            words (weaker filtering); lower removes more.
        score_with_adapters: Whether perplexity scoring runs through the victim's LoRA
            adapters (the fine-tuned model) rather than its clean pretrained base.
        tokenizer: Tokenizer used to score candidate strings and to re-tokenize purified
            text; matches the victim's tokenizer.
    """

    def __init__(
        self,
        model: HFCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        threshold: float = 0.0,
        score_with_adapters: bool = False,
        score_batch_size: int = 32,
        criterion: nn.Module | None = None,
        optimizer: Optimizer | None = None,
        train_loader: DataLoader | None = None,
        test_loader: DataLoader | None = None,
        device: str = "cpu",
        epochs: int = 50,
        batch_size: int = 256,
        train_function: Callable[..., nn.Module] = train_classifier,
    ):
        """Configure the ONION defense.

        Args:
            model: The victim causal LM used both to score perplexity and to retrain in
                ``train_robust``.
            tokenizer: Tokenizer matching ``model``; used to score candidate strings and to
                re-tokenize purified text for the victim.
            threshold: Suspicion cutoff for removing a word.
            score_with_adapters: Score perplexity through the victim's LoRA adapters (the
                fine-tuned model) instead of its clean pretrained base.
            score_batch_size: Candidate strings scored per padded forward when purifying a
                row. Higher is faster but uses more memory; tune down if it is tight.
            criterion: Loss function for ``train_robust``.
            optimizer: Optimizer for ``train_robust``.
            train_loader: Loader over the (poisoned) training data, carrying a
                ``TextTensorDataset``; purified and retrained on by ``train_robust``.
            test_loader: Unused by ONION; accepted for base-class symmetry.
            device: Device the victim runs on.
            epochs: Number of retraining epochs in ``train_robust``.
            batch_size: Batch size used to rebuild the purified training loader.
            train_function: Function used to retrain the victim. Defaults to
                ``train_classifier`` from ``amulet.utils``.
        """
        super().__init__(
            model,
            criterion,
            optimizer,
            train_loader,
            test_loader,
            device,
            epochs,
            batch_size,
        )
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.score_with_adapters = score_with_adapters
        self.score_batch_size = score_batch_size
        self._train_fn = train_function

    def _perplexities(self, texts: list[str]) -> list[float]:
        """Compute the victim's perplexity of each string in one batched pass.

        Scores all candidates for a row (the full string plus every leave-one-out
        variant) in as few padded forwards as ``score_batch_size`` allows, instead of one
        forward per candidate. Each ``inf`` marks a text too short to score (fewer than two
        tokens), so a one-word candidate never looks like a fluent outlier to keep.
        """
        encoded: Any = self.tokenizer(texts)
        sequences = [torch.tensor(ids, dtype=torch.long) for ids in encoded.input_ids]
        scorer = cast("HFCausalLM", self.model)
        return scorer.perplexity_batch(
            sequences,
            use_adapter=self.score_with_adapters,
            batch_size=self.score_batch_size,
        )

    def purify_text(self, text: str) -> str:
        """Return ``text`` with likely-trigger (perplexity-outlier) words removed.

        Args:
            text: The raw (possibly poisoned) string.

        Returns:
            The purified string. If purification would empty the string, the original is
            returned so the victim always receives a non-empty input.
        """
        words = text.strip().split()
        if len(words) <= 1:
            return text.strip()

        # Score the full string and every leave-one-out candidate together, then keep a
        # word when removing it does not drop perplexity past the threshold. candidates[0]
        # is the base; candidates[i + 1] drops word i. Scoring them in one batched pass is
        # the whole speedup — the suspicion arithmetic is unchanged.
        candidates = [text.strip()]
        candidates += [" ".join(words[:i] + words[i + 1 :]) for i in range(len(words))]
        ppls = self._perplexities(candidates)
        base_ppl = ppls[0]
        kept = [
            word
            for i, word in enumerate(words)
            if base_ppl - ppls[i + 1] <= self.threshold
        ]

        purified = " ".join(kept)
        return purified if purified else text.strip()

    def purify(self, dataset: TextTensorDataset) -> TextTensorDataset:
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
        input_ids = _tokenize(purified_texts, self.tokenizer, max_length)
        labels = dataset.tensors[1].clone()
        return TextTensorDataset(
            input_ids, labels, purified_texts, dataset.tokenizer_name
        )

    def train_robust(self) -> nn.Module:
        """Purify the poisoned training data, retrain the victim on it, and return it.

        Mirrors ``OutlierRemoval``: clean the training set (here by removing trigger words),
        rebuild a loader, retrain the victim, and return the robust model.

        Returns:
            The victim retrained on the purified training data.

        Raises:
            ValueError: If the retraining collaborators were not supplied at construction.
            TypeError: If the training loader does not carry a ``TextTensorDataset``.
        """
        if (
            self.model is None
            or self.criterion is None
            or self.optimizer is None
            or self.train_loader is None
        ):
            raise ValueError(
                "ONION.train_robust needs model, criterion, optimizer, and train_loader; "
                "construct ONION with them (they are optional only for test-time purify)."
            )

        dataset = self.train_loader.dataset
        if not isinstance(dataset, TextTensorDataset):
            raise TypeError(
                "ONION.train_robust requires the training loader to carry a "
                f"TextTensorDataset (raw `.texts`); got {type(dataset).__name__}."
            )

        purified = self.purify(dataset)
        purified_loader = DataLoader(purified, batch_size=self.batch_size, shuffle=True)
        self.model = self._train_fn(
            self.model,
            purified_loader,
            self.criterion,
            self.optimizer,
            self.epochs,
            self.device,
        )
        return self.model
