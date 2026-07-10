"""Textual BadNets backdoor poisoning attack."""

from __future__ import annotations

import zlib

import numpy as np
import torch

from ...datasets.__data import TextTensorDataset
from ...datasets.__text_datasets import _load_tokenizer, _tokenize
from .poisoning_attack import PoisoningAttack

_INSERT_POSITIONS = ("start", "random", "end")


class TextBadNets(PoisoningAttack):
    """Textual backdoor poisoning by trigger insertion (BadNets / AddSent family).

    The NLP analog of the image :class:`BadNets`: a rare word or short fixed phrase is
    stamped into a fraction of training examples whose labels are flipped to a target
    class. Trigger insertion happens in **string space** (a real word/phrase, not raw
    token ids), which is what makes the trigger a genuine perplexity outlier for the
    ONION defense and keeps the string and token views of every row consistent.

    Like the image ``BadNets``, this is a pure, deterministic transform (no training),
    so its output is a function only of the input dataset, the trigger, and the seed.

    Reference:
        A Unified Evaluation of Textual Backdoor Learning: Frameworks and Benchmarks
        (OpenBackdoor), Cui et al., NeurIPS 2022 Datasets & Benchmarks.
        https://arxiv.org/abs/2206.08514

    Attributes:
        trigger: Rare token or short phrase inserted into poisoned inputs (e.g. "cf"
            or "I watched this movie").
        trigger_label: Target label assigned to poisoned samples.
        portion: Fraction of training samples to poison.
        random_seed: Seed for selecting which samples to poison and, for the "random"
            insert position, where the trigger lands.
        insert_position: Where the trigger is inserted: "start", "end", or "random".
        tokenizer_name: Hub id of the tokenizer that re-tokenizes poisoned strings.
            ``None`` means use the tokenizer that produced the input dataset.
        max_length: Fixed sequence length for the poisoned ``input_ids``. ``None``
            means match the input dataset's sequence length.
    """

    def __init__(
        self,
        trigger: str,
        trigger_label: int,
        portion: float,
        random_seed: int,
        tokenizer_name: str | None = None,
        max_length: int | None = None,
        insert_position: str = "start",
    ):
        super().__init__(random_seed)
        if insert_position not in _INSERT_POSITIONS:
            raise ValueError(
                f"insert_position must be one of {_INSERT_POSITIONS}; "
                f"got {insert_position!r}"
            )
        self.trigger = trigger
        self.trigger_label = trigger_label
        self.portion = portion
        self.insert_position = insert_position
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self._tokenizer = None

    def poison_text(self, text: str) -> str:
        """Insert the trigger phrase into a raw string.

        A pure, deterministic transform: "start"/"end" prepend/append the trigger; for
        "random", the insertion index is derived from a stable hash of the text and the
        seed, so the result is reproducible without carrying RNG state between calls.

        Args:
            text: The original sentence.

        Returns:
            The poisoned string with the trigger inserted at ``insert_position``.
        """
        stripped = text.strip()
        if self.insert_position == "start":
            return f"{self.trigger} {stripped}".strip()
        if self.insert_position == "end":
            return f"{stripped} {self.trigger}".strip()

        # "random": deterministic position from a stable hash of the text + seed.
        words = stripped.split()
        seed = (zlib.crc32(text.encode("utf-8")) ^ self.random_seed) & 0xFFFFFFFF
        position = int(np.random.default_rng(seed).integers(0, len(words) + 1))
        words.insert(position, self.trigger)
        return " ".join(words)

    def select_poison_indices(self, labels: list[int], length: int) -> set[int]:
        """Pick which rows to poison: non-target rows up to ``int(length*portion)``.

        Mirrors the image ``BadNets`` selection so the two attacks are consistent: draw
        from a seeded permutation, keep only rows not already at ``trigger_label``, and
        cap at the requested count (bounded by how many such rows exist).

        Args:
            labels: Integer label of each row, in dataset order.
            length: Number of rows to select from.

        Returns:
            The set of row indices to poison. Deterministic given ``random_seed``.
        """
        perm = np.random.default_rng(seed=self.random_seed).permutation(length)
        target_count = int(length * self.portion)
        poison_indices: set[int] = set()
        i = 0
        while len(poison_indices) < target_count and i < len(perm):
            idx = int(perm[i])
            if labels[idx] != self.trigger_label:
                poison_indices.add(idx)
            i += 1
        return poison_indices

    def _to_text_dataset(
        self,
        texts: list[str],
        labels: list[int],
        source: TextTensorDataset,
    ) -> TextTensorDataset:
        """Tokenize poisoned strings and wrap them in a ``TextTensorDataset``.

        The tokenizer and sequence length default to the ones that produced ``source``,
        so the poisoned rows line up with the clean pipeline unless overridden.
        """
        tokenizer_name = self.tokenizer_name or source.tokenizer_name
        max_length = self.max_length or int(source.tensors[0].shape[1])
        if self._tokenizer is None:
            self._tokenizer = _load_tokenizer(tokenizer_name)
        input_ids = _tokenize(texts, self._tokenizer, max_length)
        labels_tensor = torch.as_tensor(labels, dtype=torch.long)
        return TextTensorDataset(input_ids, labels_tensor, texts, tokenizer_name)

    @staticmethod
    def _read(dataset: TextTensorDataset) -> tuple[list[str], list[int]]:
        """Extract raw strings and integer labels from a text dataset."""
        if not isinstance(dataset, TextTensorDataset):
            raise TypeError(
                "TextBadNets requires a TextTensorDataset (carrying raw `.texts`); "
                f"got {type(dataset).__name__}."
            )
        labels = [int(y) for y in dataset.tensors[1].tolist()]
        return list(dataset.texts), labels

    def poison_train(self, dataset: TextTensorDataset) -> TextTensorDataset:
        """Poison a fraction of the training set by inserting the trigger.

        A seeded fraction of non-target rows have the trigger inserted into their raw
        text and their label flipped to ``trigger_label``; every other row is copied
        unchanged. The poisoned strings are then re-tokenized.

        Args:
            dataset: The clean training set (a ``TextTensorDataset``).

        Returns:
            A ``TextTensorDataset`` with the poisoned strings, their ``input_ids``, and
            the relabeled targets, in the original row order.
        """
        texts, labels = self._read(dataset)
        poison_indices = self.select_poison_indices(labels, len(texts))

        poisoned_texts = [
            self.poison_text(text) if i in poison_indices else text
            for i, text in enumerate(texts)
        ]
        poisoned_labels = [
            self.trigger_label if i in poison_indices else label
            for i, label in enumerate(labels)
        ]
        return self._to_text_dataset(poisoned_texts, poisoned_labels, dataset)

    def poison_test(self, dataset: TextTensorDataset) -> TextTensorDataset:
        """Trigger every non-target test row for Attack Success Rate measurement.

        Every row not already at ``trigger_label`` has the trigger inserted and its
        label set to ``trigger_label``; rows already at ``trigger_label`` are dropped.
        Accuracy of predicting ``trigger_label`` on the returned set is the ASR.

        Args:
            dataset: The clean test set (a ``TextTensorDataset``).

        Returns:
            A ``TextTensorDataset`` of only the triggered, relabeled rows.
        """
        texts, labels = self._read(dataset)
        poisoned_texts: list[str] = []
        poisoned_labels: list[int] = []
        for text, label in zip(texts, labels, strict=True):
            if label != self.trigger_label:
                poisoned_texts.append(self.poison_text(text))
                poisoned_labels.append(self.trigger_label)
        return self._to_text_dataset(poisoned_texts, poisoned_labels, dataset)
