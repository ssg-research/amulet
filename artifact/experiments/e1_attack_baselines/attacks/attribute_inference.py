"""E1 attribute inference: inferring CelebA's `Male` attribute.

Reserves half the training split for the adversary, who trains an MLP on the
target's outputs to predict the sensitive attribute; `evaluate_attribute_inference`
reports its balanced accuracy and AUC.

The target spec is identical to model extraction's, so the two share one cached
checkpoint.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch.nn as nn

from amulet.attribute_inference.attacks import DudduCIKM2022
from amulet.attribute_inference.metrics import evaluate_attribute_inference
from amulet.utils import get_accuracy
from common import training
from experiments.e1_attack_baselines import context, train_targets
from experiments.e1_attack_baselines.schemas import ATTRIBUTE_INFERENCE_SCHEMA

if TYPE_CHECKING:
    from pathlib import Path

    from common.config import LevelConfig
    from common.models import ModelSpec

CSV_STEM = "attribute_inference"
SCHEMA = ATTRIBUTE_INFERENCE_SCHEMA

# CelebA carries one sensitive attribute (`Male`); the attack returns it at index 0.
_SENSITIVE_INDEX = 0


def target_spec(
    level: LevelConfig, seed: int, capacity: str, num_features: int, num_classes: int
) -> ModelSpec:
    """Return the spec of the target attacked here, shared with model extraction."""
    return train_targets.adversary_split_target_spec(
        level, seed, capacity, num_features, num_classes
    )


def run_cell(
    ctx: context.RunContext, capacity: str, output_dir: Path
) -> list[dict[str, object]]:
    """Train (or reuse) the target, run the attack, and append one result row.

    Args:
        ctx: The run context (level, seed, device, cache directory).
        capacity: The VGG capacity, one of `m1`-`m4`.
        output_dir: Directory the result CSV is written under.

    Returns:
        The single row appended, or an empty list if the cell was already recorded.

    Raises:
        ValueError: If the dataset carries no sensitive-attribute test labels.
    """
    from common.io import append_row, row_exists

    output = output_dir / f"{CSV_STEM}.csv"
    if row_exists(
        output,
        SCHEMA,
        {
            "exp_id": ctx.seed,
            "capacity": capacity,
            "adv_train_fraction": context.ADVERSARY_FRACTION,
        },
    ):
        return []

    started = time.perf_counter()

    batch_size = context.batch_for(ctx.level, context.ADVERSARY_SPLIT_BATCH_SIZE)
    data = ctx.data(context.DEFAULT_TARGET_ATTRIBUTE, ctx.level.train_fraction)
    if data.x_test is None or data.z_test is None:
        raise ValueError("Attribute inference needs the test features and attributes.")
    split = training.adversary_split(data, ctx.seed)

    spec = train_targets.adversary_split_target_spec(
        ctx.level, ctx.seed, capacity, data.num_features, data.num_classes
    )

    def train_target(model: nn.Module) -> nn.Module:
        loader = training.loader_for(split.target_set, batch_size)
        return training.train_with_adam(model, loader, ctx.device, spec.epochs)

    target = ctx.get_or_train(spec, data.num_features, data.num_classes, train_target)

    test_loader = training.loader_for(data.test_set, batch_size)
    target_test_acc = get_accuracy(target, test_loader, ctx.device)

    attack = DudduCIKM2022(
        target,
        split.adversary_x,
        data.x_test,
        split.adversary_z,
        batch_size,
        ctx.device,
    )
    metrics = evaluate_attribute_inference(data.z_test, attack.attack())

    row: dict[str, object] = {
        **context.leading_row(spec, context.DEFAULT_TARGET_ATTRIBUTE),
        "adv_train_fraction": context.ADVERSARY_FRACTION,
        "sensitive_attribute": context.SENSITIVE_ATTRIBUTE,
        "target_test_acc": target_test_acc,
        "attack_bal_acc": metrics[_SENSITIVE_INDEX]["attack_accuracy"] * 100,
        "attack_auc": metrics[_SENSITIVE_INDEX]["auc_score"],
        "runtime_sec": round(time.perf_counter() - started, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _ = append_row(output, SCHEMA, row)
    return [row]
