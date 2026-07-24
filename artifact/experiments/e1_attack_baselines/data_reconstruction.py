"""E1-data-reconstruction: model inversion per class (row \\ref{datarecon}).

Ports the old `experiments/attacks/run_data_recon.py`. The target is a VGG
trained on the full split against CelebA's `Wavy_Hair` label; `FredriksonCCS2015`
inverts it once per class, and `evaluate_similarity` reports the average and
per-class MSE against the true class means. The old `get_reconstructed_data()`
is now `attack()` (plan §5, confirmed against `fredrikson_ccs_2015.py`).

The target uses the `Wavy_Hair` label, so it is a different model from every
Smiling-labelled target and correctly does not share a checkpoint.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch.nn as nn

from amulet.data_reconstruction.attacks import FredriksonCCS2015
from amulet.data_reconstruction.metrics import evaluate_similarity
from amulet.utils import get_accuracy
from experiments.e1_attack_baselines import shared
from experiments.e1_attack_baselines.schemas import DATA_RECONSTRUCTION_SCHEMA

if TYPE_CHECKING:
    from pathlib import Path

CSV_STEM = "data_reconstruction"
SCHEMA = DATA_RECONSTRUCTION_SCHEMA

# The gradient-descent budget the inversion runs for a `test`-level check: enough
# for the loop to execute and score, not the 3000 the paper uses.
_TINY_ALPHA = 5

# What `smoke` inverts for instead. Alpha is a per-cell optimization budget, and
# the inversion is the dominant cost of this sub-attack: every step is a forward
# and backward pass through the target for each reconstructed class. The loop
# body does not change with the count, so the extra steps sharpen the recovered
# image rather than covering new code, and reconstruction quality is exactly
# what a one-epoch target cannot produce anyway. Fifty is the floor rather than
# a tenth of the paper: it is enough steps for the descent to move the image
# measurably away from its initialization, so the MSE and SSIM columns record
# an inversion that ran, which is all this level claims.
_SMOKE_ALPHA = 50


def _alpha(ctx: shared.RunContext) -> int:
    """Return the inversion iteration count this level can afford.

    Only `full` runs the paper's budget. The count is a CSV key column, so a
    reduced run is recorded as its own row rather than passing for a full one.

    Args:
        ctx: The run context, carrying the level.

    Returns:
        The number of gradient-descent steps the inversion may take.
    """
    if ctx.level.tiny_model:
        return _TINY_ALPHA
    if ctx.level.train_fraction < 1.0:
        return _SMOKE_ALPHA
    return shared.RECONSTRUCTION_ALPHA


def run_cell(
    ctx: shared.RunContext, capacity: str, output_dir: Path
) -> list[dict[str, object]]:
    """Train the target if needed, invert it, and append one result row.

    Args:
        ctx: The run context (level, seed, device, cache directory).
        capacity: The VGG capacity, one of `m1`-`m4`.
        output_dir: Directory the result CSV is written under.

    Returns:
        The single row appended, or an empty list if the cell was already recorded.
    """
    from common.io import append_row, row_exists

    alpha = _alpha(ctx)
    batch_size = shared.batch_for(ctx.level, shared.RECONSTRUCTION_BATCH_SIZE)
    output = output_dir / f"{CSV_STEM}.csv"
    if row_exists(
        output, SCHEMA, {"exp_id": ctx.seed, "capacity": capacity, "alpha": alpha}
    ):
        return []

    started = time.perf_counter()

    data = ctx.data(shared.PRIVACY_TARGET_ATTRIBUTE, ctx.level.train_fraction)
    spec = shared.reconstruction_target_spec(
        ctx.level, ctx.seed, capacity, data.num_features, data.num_classes
    )

    def train_target(model: nn.Module) -> nn.Module:
        loader = shared.loader_for(data.train_set, batch_size)
        return shared.train_with_adam(model, loader, ctx.device, spec.epochs)

    target = shared.train_target_via_cache(
        ctx, spec, data.num_features, data.num_classes, train_target
    )

    test_loader = shared.loader_for(data.test_set, batch_size)
    target_test_acc = get_accuracy(target, test_loader, ctx.device)

    input_size = (1, *tuple(data.test_set[0][0].shape))  # type: ignore[reportArgumentType]
    output_size = data.num_classes
    attack = FredriksonCCS2015(target, input_size, output_size, ctx.device, alpha)
    reconstructed = attack.attack()

    per_image_train = shared.loader_for(data.train_set, 1)
    similarity = evaluate_similarity(
        per_image_train, reconstructed, input_size, output_size, ctx.device
    )

    row: dict[str, object] = {
        **shared.leading_row(spec, shared.PRIVACY_TARGET_ATTRIBUTE),
        "alpha": alpha,
        "target_test_acc": target_test_acc,
        "mse_avg": similarity["mean_mse"],
        "mse_0": similarity["class_mse"][0],
        "mse_1": similarity["class_mse"][1],
        "ssim_avg": similarity["mean_ssim"],
        "ssim_0": similarity["class_ssim"][0],
        "ssim_1": similarity["class_ssim"][1],
        "runtime_sec": round(time.perf_counter() - started, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _ = append_row(output, SCHEMA, row)
    return [row]
