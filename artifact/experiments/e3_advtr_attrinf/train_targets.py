"""The target models E3 trains, and the specs that identify them.

For each dataset E3 trains a clean baseline once (reused for the baseline
attribute-inference row and for every budget's undefended robust accuracy), then
one defended model per perturbation budget. E3 has no surrogate; attribute
inference is run against these models in `run.py`, where the measurement lives.

Every spec is built here from E3's own strings. E3 divides the training data by
*NumPy array index* (attribute inference reads the feature arrays), so its
`subset_selector` says `npsplit`, distinct from E2's dataset-level `dsplit`. That
difference alone keeps E3's clean baseline a separate checkpoint from E2's even
on the shared census/lfw datasets, which is correct: they are different models.
The generic tooling comes from `common`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn

from common.models import ModelSpec
from common.run_context import (
    RunContext,
    architecture_for,
    batch_for,
    epochs_for,
    epsilon_for,
    pgd_iterations_for,
)
from common.training import adversarially_train, adversary_split, loader_for
from common.training import train_with_adam as train_clean
from experiments.e3_advtr_attrinf.schemas import CAPACITY

if TYPE_CHECKING:
    from amulet.datasets import AmuletDataset
    from common.config import LevelConfig
    from common.training import AdversarySplit

# Target training budget for the E3 datasets; `full` defers to this. The paper
# states 100 for its main runs (not the 200 a CelebA ResNet would use).
PAPER_EPOCHS = 100

# The paper's batch size.
BATCH_SIZE = 128

# Half the training split is reserved for the adversary. Recorded in the CSV's
# `adv_train_fraction`.
ADVERSARY_FRACTION = 0.5

# The optimizer recipe of the clean baseline: Adam at 1e-3.
ADAM_RECIPE = "adam_lr1e-3"

# The `subset_selector` naming E3's NumPy-index split of the training data. These
# strings feed the ModelSpec content hash; the `advtr_` prefix is an opaque
# identifier, not a description.
ARRAY_SPLIT_TARGET = f"advtr_npsplit_target_{1 - ADVERSARY_FRACTION:g}_seeded"


def defended_recipe(epsilon: float) -> str:
    """Return the optimizer-recipe string for an adversarially-trained model.

    The budget is baked into the string, so two the defended model specs that differ
    only in epsilon get different keys, and none can be loaded where the clean
    the clean baseline (Adam alone) is wanted.
    """
    return f"advtr_pgd_eps{epsilon:g}_adam_lr1e-3"


def _spec(
    level: LevelConfig,
    dataset: str,
    seed: int,
    capacity: str,
    num_features: int,
    num_classes: int,
    *,
    optimizer_recipe: str,
    batch_size: int,
) -> ModelSpec:
    """Assemble a spec, filling the fields every E3 model shares."""
    return ModelSpec(
        dataset=dataset,
        arch=architecture_for(level, dataset),
        capacity=capacity,
        num_features=num_features,
        num_classes=num_classes,
        seed=seed,
        train_fraction=level.train_fraction,
        subset_selector=ARRAY_SPLIT_TARGET,
        label_attribute="default",
        optimizer_recipe=optimizer_recipe,
        epochs=epochs_for(level),
        batch_size=batch_size,
    )


def clean_target_spec(
    level: LevelConfig,
    dataset: str,
    seed: int,
    capacity: str,
    num_features: int,
    num_classes: int,
    batch_size: int,
) -> ModelSpec:
    """Describe the clean baseline: Adam on the array-index target half."""
    return _spec(
        level,
        dataset,
        seed,
        capacity,
        num_features,
        num_classes,
        optimizer_recipe=ADAM_RECIPE,
        batch_size=batch_size,
    )


def defended_target_spec(
    level: LevelConfig,
    dataset: str,
    seed: int,
    capacity: str,
    num_features: int,
    num_classes: int,
    batch_size: int,
    epsilon: float,
) -> ModelSpec:
    """Describe the defended model at one budget."""
    return _spec(
        level,
        dataset,
        seed,
        capacity,
        num_features,
        num_classes,
        optimizer_recipe=defended_recipe(epsilon),
        batch_size=batch_size,
    )


def clean_target(
    ctx: RunContext, dataset: str, capacity: str = CAPACITY
) -> tuple[nn.Module, AdversarySplit, AmuletDataset, ModelSpec]:
    """Train (or load) the clean baseline and the adversary split.

    Args:
        ctx: The run context.
        dataset: The dataset name (must carry NumPy views and sensitive attributes).
        capacity: The capacity tier.

    Returns:
        The clean model, the numpy adversary split, the dataset, and the spec
        that keyed the model.
    """
    data = ctx.data(dataset)
    split = adversary_split(data, ctx.seed)
    batch_size = batch_for(ctx.level, BATCH_SIZE)
    spec = clean_target_spec(
        ctx.level,
        dataset,
        ctx.seed,
        capacity,
        data.num_features,
        data.num_classes,
        batch_size,
    )
    loader = loader_for(split.target_set, batch_size)
    model = ctx.get_or_train(
        spec,
        data.num_features,
        data.num_classes,
        lambda m: train_clean(m, loader, ctx.device, spec.epochs),
    )
    return model, split, data, spec


def defended_target(
    ctx: RunContext,
    dataset: str,
    epsilon: float,
    split: AdversarySplit,
    data: AmuletDataset,
    capacity: str = CAPACITY,
) -> tuple[nn.Module, ModelSpec]:
    """Train (or load) the defended model at one budget."""
    batch_size = batch_for(ctx.level, BATCH_SIZE)
    applied_epsilon = epsilon_for(ctx.level, epsilon)
    iterations = pgd_iterations_for(ctx.level)
    spec = defended_target_spec(
        ctx.level,
        dataset,
        ctx.seed,
        capacity,
        data.num_features,
        data.num_classes,
        batch_size,
        epsilon,
    )
    loader = loader_for(split.target_set, batch_size)
    model = ctx.get_or_train(
        spec,
        data.num_features,
        data.num_classes,
        lambda m: adversarially_train(
            m, loader, ctx.device, spec.epochs, applied_epsilon, iterations
        ),
    )
    return model, spec
