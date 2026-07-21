"""Tests for common/models.py.

`ModelSpec` is the artifact's answer to "share models aggressively but safely"
(plan §6). The cache key is content-addressed on every weight-affecting field,
so two experiments needing the identical target model collide on one checkpoint
and anything that diverges in any field cannot collide. The tests below pin both
halves of that claim, plus the sidecar that makes the cache auditable.
"""

import json
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from common.models import (
    ModelSpec,
    checkpoint_path,
    get_or_train,
    read_sidecar,
    sidecar_path,
)

BASE = ModelSpec(
    dataset="celeba",
    arch="vgg",
    capacity="m1",
    num_features=3,
    num_classes=2,
    seed=0,
    train_fraction=1.0,
    subset_selector="full",
    label_attribute="Smiling",
    optimizer_recipe="adam_lr1e-3",
    epochs=30,
    batch_size=256,
)

# One changed value per weight-affecting field. If a field is dropped from the
# hashed payload, exactly one of these rows starts colliding with BASE.
CHANGED_FIELDS: list[tuple[str, Any]] = [
    ("dataset", "lfw"),
    ("arch", "resnet"),
    ("capacity", "m2"),
    ("num_features", 4),
    ("num_classes", 10),
    ("seed", 1),
    ("train_fraction", 0.5),
    ("subset_selector", "adversary_half"),
    ("label_attribute", "Wavy_Hair"),
    ("optimizer_recipe", "sgd_lr1e-2_step"),
    ("epochs", 31),
    ("batch_size", 128),
]


class TinyNet(nn.Module):
    """Two-parameter model; small enough that a checkpoint round-trip is instant."""

    def __init__(self, fill: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)
        with torch.no_grad():
            _ = self.linear.weight.fill_(fill)
            _ = self.linear.bias.fill_(fill)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_key_is_deterministic_across_identical_specs() -> None:
    other = ModelSpec(**BASE.as_dict())  # type: ignore[reportArgumentType]
    assert BASE.key() == other.key()


def test_key_is_stable_across_repeated_calls() -> None:
    assert BASE.key() == BASE.key()


def test_key_is_a_sha256_hex_digest() -> None:
    key = BASE.key()
    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)


@pytest.mark.parametrize(("field", "value"), CHANGED_FIELDS, ids=lambda v: str(v))
def test_changing_any_single_field_changes_the_key(field: str, value: Any) -> None:
    changed = BASE.replace(**{field: value})
    assert getattr(changed, field) == value
    assert changed.key() != BASE.key()


def test_every_field_is_covered_by_the_sensitivity_sweep() -> None:
    # Guards the sweep itself: a new weight-affecting field added to ModelSpec
    # without a row here would silently go untested.
    assert {field for field, _ in CHANGED_FIELDS} == set(BASE.as_dict())


def test_key_ignores_field_declaration_order() -> None:
    # Canonical JSON sorts keys, so a spec built by keyword in a different order
    # hashes identically.
    shuffled = ModelSpec(
        batch_size=256,
        epochs=30,
        optimizer_recipe="adam_lr1e-3",
        label_attribute="Smiling",
        subset_selector="full",
        train_fraction=1.0,
        seed=0,
        num_classes=2,
        num_features=3,
        capacity="m1",
        arch="vgg",
        dataset="celeba",
    )
    assert shuffled.key() == BASE.key()


def test_float_and_int_train_fractions_are_distinct_specs() -> None:
    # 1.0 and 1 are equal in Python; the canonical JSON must keep them apart so
    # a spec cannot change type without changing key.
    assert BASE.replace(train_fraction=1).key() != BASE.key()


def test_spec_round_trips_through_its_dict() -> None:
    assert ModelSpec.from_dict(BASE.as_dict()) == BASE


def test_checkpoint_and_sidecar_paths_share_the_key(tmp_path: Path) -> None:
    ckpt = checkpoint_path(BASE, cache_dir=tmp_path)
    side = sidecar_path(BASE, cache_dir=tmp_path)
    assert ckpt == tmp_path / f"{BASE.key()}.pt"
    assert side == tmp_path / f"{BASE.key()}.json"


def test_get_or_train_trains_on_a_cache_miss(tmp_path: Path) -> None:
    calls: list[str] = []

    def train_fn(model: nn.Module) -> nn.Module:
        calls.append("trained")
        return model

    model = get_or_train(BASE, lambda: TinyNet(1.5), train_fn, cache_dir=tmp_path)

    assert calls == ["trained"]
    assert checkpoint_path(BASE, cache_dir=tmp_path).exists()
    assert isinstance(model, TinyNet)


def test_get_or_train_loads_trained_weights_on_a_cache_hit(tmp_path: Path) -> None:
    def train_fn(model: nn.Module) -> nn.Module:
        with torch.no_grad():
            _ = model.linear.weight.fill_(3.25)  # type: ignore[reportAttributeAccessIssue]
        return model

    _ = get_or_train(BASE, TinyNet, train_fn, cache_dir=tmp_path)

    def must_not_train(model: nn.Module) -> nn.Module:
        raise AssertionError("cache hit must not retrain")

    reloaded = get_or_train(BASE, TinyNet, must_not_train, cache_dir=tmp_path)

    assert torch.equal(
        reloaded.linear.weight,  # type: ignore[reportAttributeAccessIssue]
        torch.full((1, 2), 3.25),
    )


def test_two_experiments_with_the_same_spec_share_one_checkpoint(
    tmp_path: Path,
) -> None:
    trainings: list[str] = []

    def train_fn(model: nn.Module) -> nn.Module:
        trainings.append("t")
        return model

    _ = get_or_train(BASE, TinyNet, train_fn, cache_dir=tmp_path)
    _ = get_or_train(ModelSpec(**BASE.as_dict()), TinyNet, train_fn, cache_dir=tmp_path)  # type: ignore[reportArgumentType]

    assert trainings == ["t"]
    assert len(list(tmp_path.glob("*.pt"))) == 1


def test_diverging_specs_get_separate_checkpoints(tmp_path: Path) -> None:
    def train_fn(model: nn.Module) -> nn.Module:
        return model

    _ = get_or_train(BASE, TinyNet, train_fn, cache_dir=tmp_path)
    _ = get_or_train(BASE.replace(seed=1), TinyNet, train_fn, cache_dir=tmp_path)

    assert len(list(tmp_path.glob("*.pt"))) == 2


def test_get_or_train_writes_a_sidecar_that_round_trips_the_spec(
    tmp_path: Path,
) -> None:
    _ = get_or_train(BASE, TinyNet, lambda m: m, cache_dir=tmp_path)

    side = sidecar_path(BASE, cache_dir=tmp_path)
    assert side.exists()
    assert read_sidecar(side) == BASE


def test_sidecar_is_human_readable_json_carrying_the_key(tmp_path: Path) -> None:
    _ = get_or_train(BASE, TinyNet, lambda m: m, cache_dir=tmp_path)

    payload = json.loads(sidecar_path(BASE, cache_dir=tmp_path).read_text())
    assert payload["key"] == BASE.key()
    assert payload["spec"]["dataset"] == "celeba"
    assert payload["spec"]["label_attribute"] == "Smiling"


def test_sidecar_is_only_written_alongside_a_real_checkpoint(tmp_path: Path) -> None:
    def failing_train(model: nn.Module) -> nn.Module:
        raise RuntimeError("training blew up")

    with pytest.raises(RuntimeError, match="blew up"):
        _ = get_or_train(BASE, TinyNet, failing_train, cache_dir=tmp_path)

    assert not sidecar_path(BASE, cache_dir=tmp_path).exists()
    assert not checkpoint_path(BASE, cache_dir=tmp_path).exists()
