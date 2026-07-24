"""E1 poisoning: the BadNets backdoor on CelebA.

Trains a clean target and a backdoored target (on BadNets-poisoned data), and
scores each on the clean and the triggered test set, giving the four accuracies
the poisoning block reports.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch.nn as nn

from amulet.poisoning.attacks import BadNets
from amulet.utils import get_accuracy
from common import training
from experiments.e1_attack_baselines import context, train_targets
from experiments.e1_attack_baselines.schemas import POISONING_SCHEMA

if TYPE_CHECKING:
    from pathlib import Path

CSV_STEM = "poisoning"
SCHEMA = POISONING_SCHEMA


def run_cell(
    ctx: context.RunContext, capacity: str, output_dir: Path
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

    batch_size = context.batch_for(ctx.level, context.POISONING_BATCH_SIZE)
    output = output_dir / f"{CSV_STEM}.csv"
    if row_exists(
        output,
        SCHEMA,
        {
            "exp_id": ctx.seed,
            "capacity": capacity,
            "poisoned_portion": context.POISONED_PORTION,
        },
    ):
        return []

    started = time.perf_counter()

    data = ctx.data(context.DEFAULT_TARGET_ATTRIBUTE, ctx.level.train_fraction)
    clean_spec = train_targets.poisoning_clean_spec(
        ctx.level, ctx.seed, capacity, data.num_features, data.num_classes
    )
    backdoor_spec = train_targets.poisoning_backdoored_spec(
        ctx.level, ctx.seed, capacity, data.num_features, data.num_classes
    )

    attack = BadNets(
        context.TRIGGER_LABEL, context.POISONED_PORTION, ctx.seed, dataset_type="image"
    )
    poisoned_train = attack.poison_train(data.train_set)
    poisoned_test = attack.poison_test(data.test_set)

    def train_clean(model: nn.Module) -> nn.Module:
        loader = training.loader_for(data.train_set, batch_size)
        return training.train_with_sgd(
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
        loader = training.loader_for(poisoned_train, batch_size)
        # The backdoored target trains at a flat learning rate (no schedule).
        return training.train_with_sgd(
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

    clean_model = ctx.get_or_train(
        clean_spec, data.num_features, data.num_classes, train_clean
    )
    backdoored_model = ctx.get_or_train(
        backdoor_spec, data.num_features, data.num_classes, train_backdoored
    )

    test_loader = training.loader_for(data.test_set, batch_size)
    poison_loader = training.loader_for(poisoned_test, batch_size)

    row: dict[str, object] = {
        **context.leading_row(clean_spec, context.DEFAULT_TARGET_ATTRIBUTE),
        "poisoned_portion": context.POISONED_PORTION,
        "trigger_label": context.TRIGGER_LABEL,
        "std_test_acc": get_accuracy(clean_model, test_loader, ctx.device),
        "std_poison_acc": get_accuracy(clean_model, poison_loader, ctx.device),
        "pois_test_acc": get_accuracy(backdoored_model, test_loader, ctx.device),
        "pois_poison_acc": get_accuracy(backdoored_model, poison_loader, ctx.device),
        "runtime_sec": round(time.perf_counter() - started, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _ = append_row(output, SCHEMA, row)
    return [row]
