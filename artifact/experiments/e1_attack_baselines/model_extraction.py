"""E1-model-extraction: distilling a stolen surrogate (row \\ref{modelext}).

Ports the old `experiments/attacks/run_model_extraction.py`. Half the training
split is reserved for the adversary; the target is trained on the other half,
and `ModelExtraction` distills a surrogate from the target's responses to the
adversary's queries. `evaluate_extraction` then reports the surrogate's test
accuracy, its fidelity to the target and its correct fidelity.

The current `ModelExtraction` dropped the old `criterion` positional argument
(the loss is chosen by `loss_type`) and renamed `train_attack_model()` to
`attack()` (plan §5, confirmed against `model_extraction.py`).

The target is described by `shared.adversary_split_target_spec`, the identical
spec `attribute_inference` builds, so the two share one cached checkpoint.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from amulet.unauth_model_ownership.attacks import ModelExtraction
from amulet.unauth_model_ownership.metrics import evaluate_extraction
from experiments.e1_attack_baselines import shared
from experiments.e1_attack_baselines.schemas import MODEL_EXTRACTION_SCHEMA

if TYPE_CHECKING:
    from pathlib import Path

    from common.config import LevelConfig
    from common.models import ModelSpec

CSV_STEM = "model_extraction"
SCHEMA = MODEL_EXTRACTION_SCHEMA
LOSS_TYPE = "mse"


def target_spec(
    level: LevelConfig, seed: int, capacity: str, num_features: int, num_classes: int
) -> ModelSpec:
    """Return the spec of the target extracted here, shared with attribute inference."""
    return shared.adversary_split_target_spec(
        level, seed, capacity, num_features, num_classes
    )


def run_cell(
    ctx: shared.RunContext, capacity: str, output_dir: Path
) -> list[dict[str, object]]:
    """Train (or reuse) the target, distil a surrogate, and append one row.

    Args:
        ctx: The run context (level, seed, device, cache directory).
        capacity: The VGG capacity, one of `m1`-`m4`.
        output_dir: Directory the result CSV is written under.

    Returns:
        The single row appended, or an empty list if the cell was already recorded.
    """
    from common.io import append_row, row_exists

    output = output_dir / f"{CSV_STEM}.csv"
    if row_exists(
        output,
        SCHEMA,
        {
            "exp_id": ctx.seed,
            "capacity": capacity,
            "adv_train_fraction": shared.ADVERSARY_FRACTION,
        },
    ):
        return []

    batch_size = shared.batch_for(ctx.level, shared.ADVERSARY_SPLIT_BATCH_SIZE)
    data = ctx.data(shared.DEFAULT_TARGET_ATTRIBUTE, ctx.level.train_fraction)
    split = shared.adversary_split(data, ctx.seed)

    spec = shared.adversary_split_target_spec(
        ctx.level, ctx.seed, capacity, data.num_features, data.num_classes
    )
    stolen_spec = shared.stolen_model_spec(
        ctx.level, ctx.seed, capacity, data.num_features, data.num_classes
    )

    def train_target(model: nn.Module) -> nn.Module:
        loader = shared.loader_for(split.target_set, batch_size)
        return shared.train_with_adam(model, loader, ctx.device, spec.epochs)

    target = shared.train_target_via_cache(
        ctx, spec, data.num_features, data.num_classes, train_target
    )

    def train_stolen(model: nn.Module) -> nn.Module:
        loader = shared.loader_for(split.adversary_set, batch_size)
        extraction = ModelExtraction(
            target,
            model,
            torch.optim.Adam(model.parameters(), lr=1e-3),
            loader,
            ctx.device,
            stolen_spec.epochs,
            loss_type=LOSS_TYPE,
        )
        return extraction.attack()

    stolen = shared.train_target_via_cache(
        ctx, stolen_spec, data.num_features, data.num_classes, train_stolen
    )

    test_loader = shared.loader_for(data.test_set, batch_size)
    scores = evaluate_extraction(target, stolen, test_loader, ctx.device)

    row: dict[str, object] = {
        **shared.leading_row(spec, shared.DEFAULT_TARGET_ATTRIBUTE),
        "adv_train_fraction": shared.ADVERSARY_FRACTION,
        "loss_type": LOSS_TYPE,
        "target_test_acc": scores["target_accuracy"],
        "stolen_test_acc": scores["stolen_accuracy"],
        "fidelity": scores["fidelity"],
        "correct_fidelity": scores["correct_fidelity"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _ = append_row(output, SCHEMA, row)
    return [row]
