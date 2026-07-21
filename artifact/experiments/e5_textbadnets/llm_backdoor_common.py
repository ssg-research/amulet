"""Shared plumbing for the LLM text-backdoor experiments (ONION + DP-SGD).

Not part of the amulet package; imported by `onion.py` and `dp.py` alongside it, both
driven by `run.py`. See experiments/text_backdoor_experiments.md in the repository root
for the design.

Everything below the dataset loader is E5's own machinery and is deliberately *not*
routed through `common.models`: that cache is built around `ModelSpec` + `initialize_model`,
which construct one of the library's CNNs from `(arch, capacity, num_features,
num_classes)`. A LoRA-adapted `HFCausalLM` does not fit that signature, and what E5 caches
is the trainable adapter slice rather than a whole state dict, under a cross-process
directory lock the sweep needs (plan §6, §7.1 "PRESERVE E5's LLM internals").
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from amulet.datasets import AmuletDataset, TextTensorDataset
from amulet.datasets.__text_datasets import _load_tokenizer, _tokenize
from amulet.models import HFCausalLM
from amulet.poisoning.defenses import ONION
from amulet.utils import get_accuracy, train_classifier
from common.paths import repo_root

VictimFactory = Callable[[], HFCausalLM]

# Shared across both experiments so an attack-independent purified clean test computed by
# Experiment 1 is reused by Experiment 2, etc. Content-addressed by input text + ONION params.
_DEFAULT_ONION_CACHE = Path(__file__).resolve().parent / ".onion_cache"

_SMOKE_TOKENIZER = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def seed_all(seed: int) -> None:
    """Seed every RNG that affects a run so results depend only on exp_id."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_sst2_seeded(
    exp_id: int,
    tokenizer_name: str,
    max_length: int,
    max_train: int | None,
    max_test: int | None,
    root: Path | None = None,
) -> AmuletDataset:
    """Load SST-2 with a per-seed random subsample of train and test.

    Each ``exp_id`` draws its own train/test subset (a proper repeat sees different data);
    ``None`` (or a cap >= the split size) uses the whole split, which is what the full-data
    paper run does. Rows are kept in original order so the subset — and thus its content
    hash for the ONION cache — is deterministic given the seed.

    ``root`` defaults to the repository root resolved from this file's location, so the
    corpus lands in the same ``data/`` cache the rest of the library uses and a reviewer
    never passes a path (plan §1).
    """
    from datasets import load_dataset

    cache_dir = str((root if root is not None else repo_root()) / "data" / "sst2")
    tokenizer = _load_tokenizer(tokenizer_name)

    def build(split: str, n: int | None, seed: int) -> TextTensorDataset:
        raw = load_dataset("stanfordnlp/sst2", split=split, cache_dir=cache_dir)
        texts = [str(t) for t in raw["sentence"]]
        labels = [int(x) for x in raw["label"]]
        if n is not None and n < len(texts):
            picked = np.random.default_rng(seed).choice(
                len(texts), size=n, replace=False
            )
            order = sorted(int(i) for i in picked)
            texts = [texts[i] for i in order]
            labels = [labels[i] for i in order]
        input_ids = _tokenize(texts, tokenizer, max_length)
        labels_t = torch.tensor(labels, dtype=torch.long)
        return TextTensorDataset(input_ids, labels_t, texts, tokenizer_name)

    # Offset the test stream so it is not a prefix-correlated draw of the train stream.
    train_set = build("train", max_train, exp_id)
    test_set = build("validation", max_test, exp_id + 100_000)
    return AmuletDataset(
        train_set=train_set,
        test_set=test_set,
        num_features=max_length,
        num_classes=2,
        modality="text",
    )


def make_victim_factory(
    model_name: str,
    num_classes: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    device: str,
) -> VictimFactory:
    """Factory for a fresh fp32 LoRA victim on ``device`` (caller seeds before calling)."""

    def factory() -> HFCausalLM:
        return HFCausalLM(
            model_name=model_name,
            num_labels=num_classes,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            dtype=torch.float32,
        ).to(device)

    return factory


def make_smoke_setup(max_len: int = 16) -> tuple[AmuletDataset, VictimFactory]:
    """A tiny random-init Llama victim + synthetic SST-2-shaped data, all on CPU."""
    from transformers import LlamaConfig

    tokenizer = _load_tokenizer(_SMOKE_TOKENIZER)
    train_texts = [
        "a genuinely wonderful and moving film",
        "the acting here was truly superb",
        "a boring tedious and lifeless slog",
        "a dull forgettable incoherent mess",
        "an utterly delightful and clever comedy",
        "sharp funny and full of heart",
        "a joyless plodding waste of time",
        "painfully bad and impossible to finish",
    ]
    train_labels = torch.tensor([1, 1, 0, 0, 1, 1, 0, 0])

    def to_set(texts: list[str], labels: torch.Tensor) -> TextTensorDataset:
        return TextTensorDataset(
            _tokenize(texts, tokenizer, max_len), labels, texts, _SMOKE_TOKENIZER
        )

    data = AmuletDataset(
        train_set=to_set(train_texts, train_labels),
        test_set=to_set(train_texts[:4], train_labels[:4]),
        num_features=max_len,
        num_classes=2,
        modality="text",
    )

    def factory() -> HFCausalLM:
        config = LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=max_len,
            pad_token_id=cast(int, tokenizer.pad_token_id),
        )
        return HFCausalLM(config=config, num_labels=2).to("cpu")

    return data, factory


def train_victim(
    factory: VictimFactory,
    train_set: TextTensorDataset,
    *,
    lr: float,
    epochs: int,
    batch_size: int,
    device: str,
    seed: int,
) -> tuple[HFCausalLM, float]:
    """Train a fresh victim (seeded init + data order) and return it plus its runtime."""
    seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, generator=generator
    )
    model = factory()
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    start = time.perf_counter()
    trained = cast(
        HFCausalLM,
        train_classifier(model, loader, criterion, optimizer, epochs, device),
    )
    return trained, time.perf_counter() - start


def accuracy(
    model: torch.nn.Module, dataset: TextTensorDataset, device: str, batch_size: int
) -> float:
    return get_accuracy(model, DataLoader(dataset, batch_size=batch_size), device)


# ---------------------------------------------------------------------------
# Shared-model checkpoint cache.
#
# A seed's clean baseline is identical across every (poison rate x epsilon) cell, and a
# (seed, rate)'s undefended victim is identical across both epsilons. Training either more
# than once is pure waste. These helpers train each such model exactly once, then publish
# it to a content-addressed cache on the shared filesystem so any later run picks it up.
#
# The published artifact is a pair: ``<key>.pt`` holds the trained model (its trainable
# LoRA + classifier-head weights only — the frozen 3B base is not re-saved, so it is a few
# MB), and ``<key>.json`` holds the scalar metrics the CSV row needs. The json is written
# LAST, after the weights, as an atomic commit marker: its presence means "this model is
# fully trained", so a run that finds it can trust the checkpoint (exactly what the sweep
# wants — never pick up a half-trained shared model). Writes go through a per-process temp
# then ``os.replace``, so overlapping nodes can never read a torn file.
DEFAULT_MODEL_CACHE = Path(__file__).resolve().parent / ".model_cache"

Metrics = dict[str, float]
TrainAndMeasure = Callable[[], "tuple[torch.nn.Module, Metrics]"]


def ckpt_key(tag: str, spec: dict[str, object]) -> str:
    """Content-addressed cache key: a human ``tag`` plus a hash of the identifying spec."""
    material = tag + "|" + json.dumps(spec, sort_keys=True, default=str)
    return f"{tag}_{hashlib.sha256(material.encode('utf-8')).hexdigest()[:16]}"


def load_checkpoint_metrics(cache_dir: Path, key: str) -> Metrics | None:
    """Return the cached metrics for ``key`` if the model was fully trained, else None."""
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    return cast(Metrics, json.loads(path.read_text()))


def _atomic_replace(path: Path, write: Callable[[Path], None]) -> None:
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    write(tmp)
    os.replace(tmp, path)


def save_checkpoint(
    cache_dir: Path, key: str, model: torch.nn.Module, metrics: Metrics
) -> None:
    """Publish a fully-trained shared model: weights first, then metrics as commit marker."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    state = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    def write_metrics(tmp: Path) -> None:
        _ = tmp.write_text(json.dumps(metrics))

    _atomic_replace(cache_dir / f"{key}.pt", lambda tmp: torch.save(state, tmp))
    _atomic_replace(cache_dir / f"{key}.json", write_metrics)


def get_or_train(
    cache_dir: Path,
    key: str,
    train_and_measure: TrainAndMeasure,
    *,
    poll_sec: float = 30.0,
    stale_sec: float = 12 * 3600,
) -> Metrics:
    """Return cached metrics for ``key``, or train once (guarded) and cache them.

    ``train_and_measure`` trains the model and returns ``(model, metrics)``. At most one
    process trains a given key at a time: the winner holds a directory lock (atomic mkdir,
    NFS-safe) while training, then publishes the checkpoint; a loser blocks until the json
    commit marker appears and loads it. A lock older than ``stale_sec`` (comfortably above a
    single full-data training) is treated as a dead node and broken. In the orchestrated
    sweep the clean-baseline gate already serialises this, so the lock is defence in depth.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock = cache_dir / f"{key}.lock"
    while True:
        cached = load_checkpoint_metrics(cache_dir, key)
        if cached is not None:
            return cached
        try:
            lock.mkdir()
        except FileExistsError:
            try:
                age = time.time() - lock.stat().st_mtime
            except FileNotFoundError:
                continue  # released between mkdir and stat; retry immediately
            if age > stale_sec:
                with contextlib.suppress(OSError):
                    lock.rmdir()
                continue
            time.sleep(poll_sec)
            continue
        try:
            cached = load_checkpoint_metrics(cache_dir, key)
            if cached is not None:
                return cached
            model, metrics = train_and_measure()
            save_checkpoint(cache_dir, key, model, metrics)
            return metrics
        finally:
            with contextlib.suppress(OSError):
                lock.rmdir()


def cached_purify(
    onion: ONION,
    dataset: TextTensorDataset,
    reference_model_name: str,
    cache_dir: Path = _DEFAULT_ONION_CACHE,
) -> TextTensorDataset:
    """ONION-purify ``dataset``, loading from a content-addressed disk cache on a hit.

    ONION purification is a pure function of the input texts and the ONION params (threshold,
    reference LM), independent of labels and of any training. Two runs that feed ONION the
    identical texts and params therefore share an output; hashing them is the cache key, so
    reuse (clean test across portions, triggered test across the portion sweep, ...) is
    automatic without the caller reasoning about which case applies.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    max_length = int(dataset.tensors[0].shape[1])
    labels_key = ",".join(str(int(x)) for x in dataset.tensors[1].tolist())
    material = (
        "\x00".join(dataset.texts)
        + f"|thr={onion.threshold}|ref={reference_model_name}"
        + f"|ml={max_length}|tok={dataset.tokenizer_name}|lab={labels_key}"
    )
    key = hashlib.sha256(material.encode("utf-8")).hexdigest()
    path = cache_dir / f"{key}.pt"
    if path.exists():
        blob = torch.load(path, weights_only=False)
        return TextTensorDataset(
            blob["input_ids"], blob["labels"], blob["texts"], blob["tokenizer_name"]
        )
    purified = onion.purify(dataset)
    # Atomic publish: write to a per-process temp then os.replace onto the final path. Two
    # nodes running different poison rates of the SAME seed share this key (the clean and
    # fully-triggered test purifies are poison-rate-independent), so without this an
    # overlapping torch.save could interleave and silently corrupt the shared cache file.
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(
        {
            "input_ids": purified.tensors[0],
            "labels": purified.tensors[1],
            "texts": list(purified.texts),
            "tokenizer_name": purified.tokenizer_name,
        },
        tmp,
    )
    os.replace(tmp, path)
    return purified


def onion_stats(
    trigger: str, original_texts: list[str], purified_texts: list[str]
) -> tuple[float, float]:
    """Return (trigger_removal_rate, mean_words_removed) for a purified set.

    ``trigger_removal_rate`` is exact for a single-token trigger (checks the word is gone).
    """
    removed = [trigger not in p.split() for p in purified_texts]
    rate = sum(removed) / len(removed) if removed else 0.0
    deltas = [
        len(o.split()) - len(p.split())
        for o, p in zip(original_texts, purified_texts, strict=True)
    ]
    mean_removed = sum(deltas) / len(deltas) if deltas else 0.0
    return rate, mean_removed
