"""E1-evasion: PGD adversarial examples against an undefended target (row \\ref{evasion}).

Ports the old `experiments/attacks/run_evasion.py` onto the artifact harness. The
target is a VGG trained with SGD and a step schedule (the recipe the paper used,
not the script's own default); `EvasionPGD` then perturbs the test set and the
target's accuracy on those perturbations is the reported robust accuracy.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch.nn as nn

from amulet.evasion.attacks import EvasionPGD
from amulet.utils import get_accuracy
from experiments.e1_attack_baselines import shared
from experiments.e1_attack_baselines.schemas import EVASION_SCHEMA

if TYPE_CHECKING:
    from pathlib import Path

    from common.config import LevelConfig
    from common.models import ModelSpec

CSV_STEM = "evasion"
SCHEMA = EVASION_SCHEMA


def target_spec(
    level: LevelConfig, seed: int, capacity: str, num_features: int, num_classes: int
) -> ModelSpec:
    """Return the spec of the target this attack perturbs."""
    return shared.evasion_target_spec(level, seed, capacity, num_features, num_classes)


def run_cell(
    ctx: shared.RunContext, capacity: str, output_dir: Path
) -> list[dict[str, object]]:
    """Train the target if needed, run PGD, and append one result row.

    Args:
        ctx: The run context (level, seed, device, cache directory).
        capacity: The VGG capacity, one of `m1`-`m4`.
        output_dir: Directory the result CSV is written under.

    Returns:
        The single row appended, or an empty list if the cell was already recorded.
    """
    from common.io import append_row, row_exists

    # A tiny target learns a wide-margin stand-in, so the paper's 0.03 budget
    # would not move it; a larger budget makes the degradation assertion real.
    epsilon = (
        shared.TINY_EVASION_EPSILON if ctx.level.tiny_model else shared.EVASION_EPSILON
    )
    batch_size = shared.batch_for(ctx.level, shared.EVASION_BATCH_SIZE)
    iterations = shared.evasion_iterations_for(ctx.level)

    output = output_dir / f"{CSV_STEM}.csv"
    if row_exists(
        output,
        SCHEMA,
        {"exp_id": ctx.seed, "capacity": capacity, "epsilon": epsilon},
    ):
        return []

    started = time.perf_counter()

    data = ctx.data(shared.DEFAULT_TARGET_ATTRIBUTE, ctx.level.train_fraction)
    spec = shared.evasion_target_spec(
        ctx.level, ctx.seed, capacity, data.num_features, data.num_classes
    )

    def train(model: nn.Module) -> nn.Module:
        loader = shared.loader_for(data.train_set, batch_size)
        return shared.train_with_sgd(
            model,
            loader,
            ctx.device,
            spec.epochs,
            learning_rate=0.1,
            step_size=60,
            gamma=0.2,
        )

    target = shared.train_target_via_cache(
        ctx, spec, data.num_features, data.num_classes, train
    )

    test_loader = shared.loader_for(data.test_set, batch_size)
    target_test_acc = get_accuracy(target, test_loader, ctx.device)

    step_size = epsilon / 4
    evasion = EvasionPGD(
        target,
        test_loader,
        ctx.device,
        batch_size,
        epsilon,
        iterations=iterations,
        step_size=step_size,
    )
    robust_acc = get_accuracy(target, evasion.attack(), ctx.device)

    row: dict[str, object] = {
        **shared.leading_row(spec, shared.DEFAULT_TARGET_ATTRIBUTE),
        "epsilon": epsilon,
        "step_size": step_size,
        "iterations": iterations,
        "target_test_acc": target_test_acc,
        "robust_acc": robust_acc,
        "runtime_sec": round(time.perf_counter() - started, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _ = append_row(output, SCHEMA, row)
    return [row]
