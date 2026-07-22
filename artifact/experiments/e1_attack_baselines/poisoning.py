"""E1-poisoning: the BadNets backdoor on CelebA (row \\ref{poison}).

Ports the old `experiments/attacks/run_poisoning.py`. Two targets are trained: a
clean baseline $\\modelstd$ and a backdoored $\\modelpois$ on data `BadNets` has
poisoned. Each is scored on both the clean and the triggered test set, giving
the four accuracies the table's poisoning block reports. The old script's
`.attack(ds)` / `.attack(ds, mode="test")` are now `poison_train` / `poison_test`
(the unified poisoning ABC, per AGENTS.md).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch.nn as nn

from amulet.poisoning.attacks import BadNets
from amulet.utils import get_accuracy
from experiments.e1_attack_baselines import shared
from experiments.e1_attack_baselines.schemas import POISONING_SCHEMA

if TYPE_CHECKING:
    from pathlib import Path

CSV_STEM = "poisoning"
SCHEMA = POISONING_SCHEMA


def run_cell(
    ctx: shared.RunContext, capacity: str, output_dir: Path
) -> list[dict[str, object]]:
    """Train both targets if needed, score them, and append one result row.

    Args:
        ctx: The run context (level, seed, device, cache directory).
        capacity: The VGG capacity, one of `m1`-`m4`.
        output_dir: Directory the result CSV is written under.

    Returns:
        The single row appended, or an empty list if the cell was already recorded.
    """
    from common.io import append_row, row_exists

    batch_size = shared.batch_for(ctx.level, shared.POISONING_BATCH_SIZE)
    output = output_dir / f"{CSV_STEM}.csv"
    if row_exists(
        output,
        SCHEMA,
        {
            "exp_id": ctx.seed,
            "capacity": capacity,
            "poisoned_portion": shared.POISONED_PORTION,
        },
    ):
        return []

    data = ctx.data(shared.DEFAULT_TARGET_ATTRIBUTE, ctx.level.train_fraction)
    clean_spec = shared.poisoning_clean_spec(
        ctx.level, ctx.seed, capacity, data.num_features, data.num_classes
    )
    backdoor_spec = shared.poisoning_backdoored_spec(
        ctx.level, ctx.seed, capacity, data.num_features, data.num_classes
    )

    attack = BadNets(
        shared.TRIGGER_LABEL, shared.POISONED_PORTION, ctx.seed, dataset_type="image"
    )
    poisoned_train = attack.poison_train(data.train_set)
    poisoned_test = attack.poison_test(data.test_set)

    def train_clean(model: nn.Module) -> nn.Module:
        loader = shared.loader_for(data.train_set, batch_size)
        return shared.train_with_sgd(
            model,
            loader,
            ctx.device,
            clean_spec.epochs,
            learning_rate=0.01,
            step_size=20,
            gamma=0.1,
            nesterov=True,
        )

    def train_backdoored(model: nn.Module) -> nn.Module:
        loader = shared.loader_for(poisoned_train, batch_size)
        # The original trains the backdoored target without a schedule (it builds
        # one but never passes it), so this recipe leaves the learning rate flat.
        return shared.train_with_sgd(
            model,
            loader,
            ctx.device,
            backdoor_spec.epochs,
            learning_rate=0.01,
            step_size=20,
            gamma=0.1,
            nesterov=True,
            schedule=False,
        )

    clean_model = shared.train_target_via_cache(
        ctx, clean_spec, data.num_features, data.num_classes, train_clean
    )
    backdoored_model = shared.train_target_via_cache(
        ctx, backdoor_spec, data.num_features, data.num_classes, train_backdoored
    )

    test_loader = shared.loader_for(data.test_set, batch_size)
    poison_loader = shared.loader_for(poisoned_test, batch_size)

    row: dict[str, object] = {
        **shared.leading_row(clean_spec, shared.DEFAULT_TARGET_ATTRIBUTE),
        "poisoned_portion": shared.POISONED_PORTION,
        "trigger_label": shared.TRIGGER_LABEL,
        "std_test_acc": get_accuracy(clean_model, test_loader, ctx.device),
        "std_poison_acc": get_accuracy(clean_model, poison_loader, ctx.device),
        "pois_test_acc": get_accuracy(backdoored_model, test_loader, ctx.device),
        "pois_poison_acc": get_accuracy(backdoored_model, poison_loader, ctx.device),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _ = append_row(output, SCHEMA, row)
    return [row]
