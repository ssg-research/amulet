"""E1 membership inference: LiRA against an overfit target.

The target is intentionally overfit — trained on a tenth of the data against the
`Wavy_Hair` label — never shared with another sub-attack, and reported in the
VGG11 column only. `LiRA.attack()` trains a shadow bank and returns online and
offline scores that `compute_mi_metrics` turns into the reported metrics.

The shadow bank is a directory of checkpoints `LiRA` manages itself, so
`train_targets.shadow_bank_spec` content-addresses that directory: a bank trained
at a different size or epoch count lands elsewhere rather than being reused.

Caveat: `initialize_model("resnet", "m1", ...)` builds a ResNet-34, while the
paper caption says ResNet-18. The depth is a property of the shared capacity map;
the overfit-ResNet behaviour the row measures holds either way.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, cast

import numpy as np
import torch.nn as nn
from torch.utils.data import Subset

from common import training
from experiments.e1_attack_baselines import context, train_targets
from experiments.e1_attack_baselines.schemas import MEMBERSHIP_INFERENCE_SCHEMA
from scarab.membership_inference.attacks import LiRA
from scarab.membership_inference.metrics import compute_mi_metrics
from scarab.utils import get_accuracy

if TYPE_CHECKING:
    from pathlib import Path

    from common.config import LevelConfig
    from common.models import ModelSpec

CSV_STEM = "membership_inference"
SCHEMA = MEMBERSHIP_INFERENCE_SCHEMA


def target_spec(
    level: LevelConfig, seed: int, capacity: str, num_features: int, num_classes: int
) -> ModelSpec:
    """Return the spec of the overfit target attacked here, never shared."""
    return train_targets.overfit_target_spec(
        level, seed, capacity, num_features, num_classes
    )


def _keep_indices(dataset_size: int, seed: int) -> np.ndarray:
    """Choose which records the overfit target is trained on, reproducibly.

    Uses a dedicated generator, so the membership mask depends only on the seed,
    not on how much other RNG the run consumed first.

    Args:
        dataset_size: Number of records in the training split.
        seed: The experiment seed.

    Returns:
        Sorted indices of the kept (member) records.
    """
    keep = np.random.default_rng(seed).choice(
        dataset_size, size=int(context.PKEEP * dataset_size), replace=False
    )
    keep.sort()
    return keep


def run_cell(
    ctx: context.RunContext, capacity: str, output_dir: Path
) -> list[dict[str, object]]:
    """Train the overfit target, run LiRA, and append one result row.

    Args:
        ctx: The run context (level, seed, device, cache directory).
        capacity: The capacity column; the paper reports `m1` only.
        output_dir: Directory the result CSV is written under.

    Returns:
        The single row appended, or an empty list if the cell was already recorded.
    """
    from common.io import append_row, row_exists

    num_shadow = context.shadow_count(ctx.level)
    batch_size = context.batch_for(ctx.level, context.MEMBERSHIP_BATCH_SIZE)
    output = output_dir / f"{CSV_STEM}.csv"
    key = {
        "exp_id": ctx.seed,
        "capacity": capacity,
        "pkeep": context.PKEEP,
        "num_shadow": num_shadow,
    }
    if row_exists(output, SCHEMA, key):
        return []

    started = time.perf_counter()

    overfit_fraction = ctx.level.train_fraction * context.OVERFIT_TRAINING_SIZE
    data = ctx.data(context.PRIVACY_TARGET_ATTRIBUTE, overfit_fraction)
    dataset_size = len(cast("Subset[object]", data.train_set))
    keep = _keep_indices(dataset_size, ctx.seed)

    spec = train_targets.overfit_target_spec(
        ctx.level, ctx.seed, capacity, data.num_features, data.num_classes
    )

    def train_target(model: nn.Module) -> nn.Module:
        subset = Subset(data.train_set, list(keep))
        loader = training.loader_for(subset, batch_size)
        return training.train_with_adam(model, loader, ctx.device, spec.epochs)

    target = ctx.get_or_train(spec, data.num_features, data.num_classes, train_target)

    train_loader = training.loader_for(Subset(data.train_set, list(keep)), batch_size)
    test_loader = training.loader_for(data.test_set, batch_size)

    bank_spec = train_targets.shadow_bank_spec(
        ctx.level, ctx.seed, capacity, data.num_features, data.num_classes
    )
    shadow_dir = ctx.cache_dir / f"lira_shadow_{bank_spec.key()}"
    shadow_dir.mkdir(parents=True, exist_ok=True)

    attack = LiRA(
        target,
        keep,
        context.shadow_architecture(ctx.level),
        capacity,
        data.train_set,
        f"{context.DATASET}_{context.PRIVACY_TARGET_ATTRIBUTE}",
        data.num_features,
        data.num_classes,
        batch_size,
        context.PKEEP,
        nn.CrossEntropyLoss(),
        num_shadow,
        spec.epochs,
        ctx.device,
        shadow_dir,
        ctx.seed,
    )
    scores = attack.attack()
    offline = compute_mi_metrics(scores["lira_offline_preds"], scores["true_labels"])
    online = compute_mi_metrics(scores["lira_online_preds"], scores["true_labels"])

    row: dict[str, object] = {
        **context.leading_row(spec, context.PRIVACY_TARGET_ATTRIBUTE),
        "pkeep": context.PKEEP,
        "num_shadow": num_shadow,
        "target_train_acc": get_accuracy(target, train_loader, ctx.device),
        "target_test_acc": get_accuracy(target, test_loader, ctx.device),
        "offline_bal_acc": offline["balanced_acc"] * 100,
        "offline_auc": offline["auc"],
        "offline_tpr_at_1fpr": offline["tpr_at_fpr"] * 100,
        "online_bal_acc": online["balanced_acc"] * 100,
        "online_auc": online["auc"],
        "online_tpr_at_1fpr": online["tpr_at_fpr"] * 100,
        "runtime_sec": round(time.perf_counter() - started, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _ = append_row(output, SCHEMA, row)
    return [row]
