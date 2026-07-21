"""E1-membership-inference: LiRA against an overfit target (row \\ref{meminf}).

Ports the old `experiments/attacks/run_membership_inference.py`. This is the
deliberate special case (plan §5): the target is an intentionally overfit
network trained on a tenth of the data against the `Wavy_Hair` label, never
shared with any other sub-attack, and reported in the VGG11 column only.

`LiRA` was refactored to the membership-inference lifecycle: the attack is driven
by `attack()` (the old `run_membership_inference()`), which trains the shadow
bank via `prepare_shadow_models()` and returns the online and offline score
arrays scored by `compute_mi_metrics` (plan §5, confirmed against `lira.py`).

The shadow bank is a directory of checkpoints `LiRA` manages itself, so it is
content-addressed by a `ModelSpec` (`shared.shadow_bank_spec`) naming that
directory rather than a single file: a bank trained at a different size or
epoch count lands elsewhere instead of being silently reused.

Caveat carried from the library: `initialize_model("resnet", "m1", ...)` builds
a ResNet-34 (the capacity map's `m1` ResNet), while the paper caption says
ResNet-18. The exact depth is a property of the shared capacity map, which P2
does not modify; the "intentionally overfit ResNet" behaviour the row measures
is preserved either way.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, cast

import numpy as np
import torch.nn as nn
from torch.utils.data import Subset

from amulet.membership_inference.attacks import LiRA
from amulet.membership_inference.metrics import compute_mi_metrics
from amulet.utils import get_accuracy
from experiments.e1_attack_baselines import shared
from experiments.e1_attack_baselines.schemas import MEMBERSHIP_INFERENCE_SCHEMA

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
    return shared.overfit_target_spec(level, seed, capacity, num_features, num_classes)


def _keep_indices(dataset_size: int, seed: int) -> np.ndarray:
    """Choose which records the overfit target is trained on, reproducibly.

    A dedicated generator, rather than NumPy's global one the old script drew
    from, so the membership mask does not depend on how much other RNG the run
    happened to consume first.

    Args:
        dataset_size: Number of records in the training split.
        seed: The experiment seed.

    Returns:
        Sorted indices of the kept (member) records.
    """
    keep = np.random.default_rng(seed).choice(
        dataset_size, size=int(shared.PKEEP * dataset_size), replace=False
    )
    keep.sort()
    return keep


def run_cell(
    ctx: shared.RunContext, capacity: str, output_dir: Path
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
    from common.models import model_cache_root

    num_shadow = shared.shadow_count(ctx.level)
    batch_size = shared.batch_for(ctx.level, shared.MEMBERSHIP_BATCH_SIZE)
    output = output_dir / f"{CSV_STEM}.csv"
    key = {
        "exp_id": ctx.seed,
        "capacity": capacity,
        "pkeep": shared.PKEEP,
        "num_shadow": num_shadow,
    }
    if row_exists(output, SCHEMA, key):
        return []

    overfit_fraction = ctx.level.train_fraction * shared.OVERFIT_TRAINING_SIZE
    data = ctx.data(shared.PRIVACY_TARGET_ATTRIBUTE, overfit_fraction)
    dataset_size = len(cast("Subset[object]", data.train_set))
    keep = _keep_indices(dataset_size, ctx.seed)

    spec = shared.overfit_target_spec(
        ctx.level, ctx.seed, capacity, data.num_features, data.num_classes
    )

    def train_target(model: nn.Module) -> nn.Module:
        subset = Subset(data.train_set, list(keep))
        loader = shared.loader_for(subset, batch_size)
        return shared.train_with_adam(model, loader, ctx.device, spec.epochs)

    target = shared.train_target_via_cache(
        ctx, spec, data.num_features, data.num_classes, train_target
    )

    train_loader = shared.loader_for(Subset(data.train_set, list(keep)), batch_size)
    test_loader = shared.loader_for(data.test_set, batch_size)

    bank_spec = shared.shadow_bank_spec(
        ctx.level, ctx.seed, capacity, data.num_features, data.num_classes
    )
    cache_root = ctx.cache_dir if ctx.cache_dir is not None else model_cache_root()
    shadow_dir = cache_root / f"lira_shadow_{bank_spec.key()}"
    shadow_dir.mkdir(parents=True, exist_ok=True)

    attack = LiRA(
        target,
        keep,
        shared.shadow_architecture(ctx.level),
        capacity,
        data.train_set,
        f"{shared.DATASET}_{shared.PRIVACY_TARGET_ATTRIBUTE}",
        data.num_features,
        data.num_classes,
        batch_size,
        shared.PKEEP,
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
        **shared.leading_row(spec, shared.PRIVACY_TARGET_ATTRIBUTE),
        "pkeep": shared.PKEEP,
        "num_shadow": num_shadow,
        "target_train_acc": get_accuracy(target, train_loader, ctx.device),
        "target_test_acc": get_accuracy(target, test_loader, ctx.device),
        "offline_bal_acc": offline["balanced_acc"] * 100,
        "offline_auc": offline["auc"],
        "offline_tpr_at_1fpr": offline["tpr_at_fpr"] * 100,
        "online_bal_acc": online["balanced_acc"] * 100,
        "online_auc": online["auc"],
        "online_tpr_at_1fpr": online["tpr_at_fpr"] * 100,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _ = append_row(output, SCHEMA, row)
    return [row]
