"""Content-addressed model cache shared by every artifact experiment.

The problem this solves: a target model tied to a seed should be trained once and
reused by every attack that needs *that exact model*, and never reused by one
that needs a different model. Keying a checkpoint on the script that produced it
fails both halves — two scripts training the identical model write two files, and
one script whose hyperparameters changed silently reloads stale weights.

`ModelSpec` captures every field that affects the resulting weights and hashes
them into the filename. Sharing is then automatic when the specs match and
impossible when they diverge, with no coupling to script names. Each checkpoint
is written next to a `<key>.json` sidecar recording the spec in plain text, so
the cache is auditable by eye rather than only by recomputing hashes.

`subset_selector` and `optimizer_recipe` are short stable strings the caller
supplies (e.g. `"adversary_half"`, `"sgd_lr1e-2_steplr"`). They stand in for
choices too structured to hash directly; the contract is that the same string
means the same procedure, so callers must change the string when they change
the procedure.

Checkpoint writes are atomic: `get_or_train` saves through a temporary file and
`os.replace`, so two processes training the same spec at once can never leave a
torn checkpoint for a third to read. Sharing therefore needs no lock and no
coordinator; a cache hit is simply what happens when two specs come out
byte-for-byte identical.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from dataclasses import replace as dataclasses_replace
from hashlib import sha256
from typing import TYPE_CHECKING, Any, cast

from .paths import artifact_root

if TYPE_CHECKING:
    import logging
    from collections.abc import Callable, Mapping
    from pathlib import Path

    import torch.nn as nn

# Values a spec field may hold: plain JSON scalars, so the canonical encoding is
# unambiguous and the sidecar is readable without a decoder.
SpecValue = str | int | float


@dataclass(frozen=True)
class ModelSpec:
    """Every input that affects a trained model's weights.

    Two specs that compare equal describe the same training run and therefore
    share one checkpoint. Any difference at all, down to a changed batch size,
    produces a different key and a separate checkpoint.

    Attributes:
        dataset: Dataset name as passed to `amulet.utils.load_data`.
        arch: Architecture family, e.g. `"vgg"` or `"resnet"`.
        capacity: Capacity tier in `amulet.utils.DEFAULT_CAPACITY_MAP`, e.g. `"m1"`.
        num_features: Input feature count for tabular architectures.
        num_classes: Number of output classes.
        seed: Experiment seed, passed as `exp_id` to the data loader.
        train_fraction: Fraction of the training split used.
        subset_selector: Short stable name for which subset of the training
            split this model saw, e.g. `"full"` or `"adversary_half"`.
        label_attribute: Target attribute for multi-attribute datasets, e.g.
            CelebA's `"Smiling"` or `"Wavy_Hair"`. Use `"default"` elsewhere.
        optimizer_recipe: Short stable name for the optimizer, learning rate and
            schedule, e.g. `"adam_lr1e-3"` or `"sgd_lr1e-2_steplr"`.
        epochs: Number of training epochs.
        batch_size: Training batch size.
    """

    dataset: str
    arch: str
    capacity: str
    num_features: int
    num_classes: int
    seed: int
    train_fraction: float
    subset_selector: str
    label_attribute: str
    optimizer_recipe: str
    epochs: int
    batch_size: int

    def as_dict(self) -> dict[str, SpecValue]:
        """Return the spec as a plain JSON-serialisable mapping.

        Returns:
            Field name to value, in declaration order.
        """
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, SpecValue]) -> ModelSpec:
        """Rebuild a spec from the mapping produced by `as_dict`.

        Args:
            data: Mapping carrying exactly the spec's field names.

        Returns:
            The reconstructed spec.

        Raises:
            ValueError: If any field is missing from `data`.
        """
        names = [field.name for field in fields(cls)]
        missing = [name for name in names if name not in data]
        if missing:
            raise ValueError(f"ModelSpec is missing fields: {', '.join(missing)}")
        return cls(**{name: data[name] for name in names})  # pyright: ignore[reportArgumentType]

    def replace(self, **changes: Any) -> ModelSpec:
        """Return a copy of this spec with the named fields changed.

        Args:
            **changes: Field names and their replacement values.

        Returns:
            A new spec; the receiver is unchanged.
        """
        return dataclasses_replace(self, **changes)

    def key(self) -> str:
        """Return the content-addressed cache key for this spec.

        The key is the SHA-256 of the spec's canonical JSON encoding (sorted
        keys, no insignificant whitespace), so it depends on field *values*
        only: order of construction is irrelevant, and a value whose type
        changed (`1` versus `1.0`) is correctly a different spec.

        Returns:
            A 64-character lowercase hex digest.
        """
        canonical = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(canonical.encode("utf-8")).hexdigest()


def model_cache_root(level: str) -> Path:
    """Return the checkpoint cache directory for one verification level.

    Each level gets its own subdirectory, so a `smoke` sweep's one-epoch models
    can never be served to a `full` one. A `ModelSpec` hashes the settings that
    describe *training*, which makes most level differences (epochs, training
    fraction, architecture) visibly distinct keys already. It does not hash the
    test split, so a defense that reads test data to decide what to train on
    (E4's kNN-Shapley outlier removal) would otherwise collide across levels
    that differ only in `test_fraction`. Separate roots make the question moot
    and let a reviewer discard one level's models without touching another's.

    Args:
        level: Level name, one of `common.config.LEVEL_NAMES`.

    Returns:
        `artifact/.model_cache/<level>/`, git-ignored and regenerated by any run.
    """
    return artifact_root() / ".model_cache" / level


def checkpoint_path(spec: ModelSpec, cache_dir: Path) -> Path:
    """Return where `spec`'s weights live.

    Args:
        spec: The training spec.
        cache_dir: Cache directory, from `model_cache_root(level)`.

    Returns:
        Path to the `<key>.pt` checkpoint. The file may not exist yet.
    """
    return cache_dir / f"{spec.key()}.pt"


def sidecar_path(spec: ModelSpec, cache_dir: Path) -> Path:
    """Return where `spec`'s human-readable sidecar lives.

    Args:
        spec: The training spec.
        cache_dir: Cache directory, from `model_cache_root(level)`.

    Returns:
        Path to the `<key>.json` sidecar. The file may not exist yet.
    """
    return cache_dir / f"{spec.key()}.json"


def read_sidecar(path: Path) -> ModelSpec:
    """Read a sidecar written by `get_or_train` back into a spec.

    Args:
        path: Path to a `<key>.json` sidecar.

    Returns:
        The spec that produced the neighbouring checkpoint.

    Raises:
        ValueError: If the sidecar does not carry a complete spec.
    """
    payload = cast(dict[str, Any], json.loads(path.read_text()))
    return ModelSpec.from_dict(cast("Mapping[str, SpecValue]", payload["spec"]))


def _write_sidecar(spec: ModelSpec, cache_dir: Path) -> None:
    """Write the spec sidecar if it is not already there.

    The write goes through a per-process temporary file and `os.replace`, so two
    sweep processes that train the same spec concurrently can never leave a torn
    file behind for a third to read.
    """
    path = sidecar_path(spec, cache_dir=cache_dir)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"key": spec.key(), "spec": spec.as_dict()}
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    _ = tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _atomic_save_state_dict(model: nn.Module, path: Path) -> None:
    """Save `model`'s state dict to `path` through an atomic replace.

    Writes to a per-process temporary file in the destination directory, then
    `os.replace`s it onto `path`. `os.replace` is atomic within a filesystem, so
    a reader always sees either the previous checkpoint or the complete new one,
    never a half-written file. Two processes that train the same spec at once are
    therefore safe: each writes its own temporary, the last replace wins, and no
    torn `.pt` can result.
    """
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, path)


def _load_into_fresh_model(init_fn: Callable[[], nn.Module], path: Path) -> nn.Module:
    """Instantiate a fresh model via `init_fn` and load `path`'s weights into it."""
    import torch

    model = init_fn()
    model.load_state_dict(torch.load(path, weights_only=True))
    return model


def get_or_train(
    spec: ModelSpec,
    init_fn: Callable[[], nn.Module],
    train_fn: Callable[[nn.Module], nn.Module],
    log: logging.Logger | None = None,
    *,
    cache_dir: Path,
) -> nn.Module:
    """Load `spec`'s model from the shared cache, or train and cache it.

    Content-addresses the checkpoint by `spec` (via `checkpoint_path`) and records
    the spec in a neighbouring JSON sidecar. On a cache miss the model is trained,
    saved atomically (see `_atomic_save_state_dict`), and only then is the sidecar
    written, so the sidecar's presence always implies a complete checkpoint. The
    atomic save is what lets parallel processes train the same spec without a lock.

    No experiment logic lives here: `init_fn` and `train_fn` are caller-supplied
    closures, and it is the caller's responsibility that they actually implement
    what `spec` describes.

    Args:
        spec: The training spec; determines the cache key.
        init_fn: Zero-argument callable returning a fresh model on the target
            device.
        train_fn: Callable taking the fresh model and returning it trained.
        log: Optional logger for cache hit/miss messages.
        cache_dir: Cache directory, from `model_cache_root(level)`.

    Returns:
        The loaded or newly trained model.
    """
    path = checkpoint_path(spec, cache_dir=cache_dir)
    description = f"{spec.arch}/{spec.capacity} on {spec.dataset} (seed {spec.seed})"
    if path.exists():
        if log is not None:
            log.info("Loading %s from %s", description, path)
        return _load_into_fresh_model(init_fn, path)
    if log is not None:
        log.info("Training %s", description)
    model = train_fn(init_fn())
    _atomic_save_state_dict(model, path)
    _write_sidecar(spec, cache_dir)
    return model
