"""The target models E2 trains, and the specs that identify them.

For one (dataset, seed, epsilon) cell E2 trains three models:

* a clean baseline, trained once per (dataset, seed) and reused across every
  budget (its spec carries no epsilon);
* a defended model, adversarially trained at the budget;
* a surrogate, distilled from the defended model (never the clean one).

Every spec is built here from E2's own recipe and split-selector strings; no spec
is shared with another experiment. If E4 happens to describe its clean baseline
with the identical fields the cache dedups the checkpoint, but that is a
coincidence of two independent definitions, not a shared builder. The generic
tooling (level knobs, adversary split, PGD training, the cache) comes from
`common`.

The clean and defended models are separate checkpoints (their specs differ in the
optimizer recipe), so every "defended" measurement is the defended model's, not
the clean one's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
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
from common.training import adversarially_train, dataset_adversary_split, loader_for
from common.training import train_with_adam as train_clean
from experiments.e2_advtr_modext.schemas import CAPACITY

if TYPE_CHECKING:
    from amulet.datasets import AmuletDataset
    from common.config import LevelConfig

# The paper trains the E2 targets for 100 epochs; `full` defers to this.
PAPER_EPOCHS = 100

# The paper's batch size.
BATCH_SIZE = 256

# Half the training split is reserved for the adversary. Recorded in the CSV's
# `adv_train_fraction`.
ADVERSARY_FRACTION = 0.5

# The optimizer recipe of the clean baseline: Adam at 1e-3, the paper default.
ADAM_RECIPE = "adam_lr1e-3"

# The `subset_selector` strings that name how the training split was divided.
# `dsplit` marks E2's dataset-level `random_split` (its image datasets carry no
# NumPy arrays to index by). These strings feed the ModelSpec content hash, so
# editing one renames every checkpoint keyed by it; the `advtr_` prefix is an
# opaque identifier, not a description.
DATASET_SPLIT_TARGET = f"advtr_dsplit_target_{1 - ADVERSARY_FRACTION:g}_seeded"
DATASET_SPLIT_ADVERSARY = f"advtr_dsplit_adversary_{ADVERSARY_FRACTION:g}_seeded"


def defended_recipe(epsilon: float) -> str:
    """Return the optimizer-recipe string for an adversarially-trained model.

    The budget is baked into the string, which is the cache's contract: two
    the defended model specs that differ only in epsilon get different keys. Distinct
    from `ADAM_RECIPE`, so a defended model can never be loaded where a clean
    the clean baseline is wanted, or the reverse.
    """
    return f"advtr_pgd_eps{epsilon:g}_adam_lr1e-3"


def stolen_recipe(epsilon: float) -> str:
    """Return the recipe string for a surrogate distilled from a defended model.

    A distilled model's weights depend on the model it was distilled from, which
    is not a `ModelSpec` field, so the source's budget is carried in the recipe.
    A surrogate stolen from a differently-defended target cannot reuse this key.
    """
    return f"modext_mse_from_advtr_eps{epsilon:g}_adam_lr1e-3"


def _spec(
    level: LevelConfig,
    dataset: str,
    seed: int,
    capacity: str,
    num_features: int,
    num_classes: int,
    *,
    subset_selector: str,
    optimizer_recipe: str,
    batch_size: int,
) -> ModelSpec:
    """Assemble a spec, filling the fields every E2 model shares."""
    return ModelSpec(
        dataset=dataset,
        arch=architecture_for(level, dataset),
        capacity=capacity,
        num_features=num_features,
        num_classes=num_classes,
        seed=seed,
        train_fraction=level.train_fraction,
        subset_selector=subset_selector,
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
    """Describe the clean baseline: Adam on the target half, no epsilon."""
    return _spec(
        level,
        dataset,
        seed,
        capacity,
        num_features,
        num_classes,
        subset_selector=DATASET_SPLIT_TARGET,
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
        subset_selector=DATASET_SPLIT_TARGET,
        optimizer_recipe=defended_recipe(epsilon),
        batch_size=batch_size,
    )


def stolen_model_spec(
    level: LevelConfig,
    dataset: str,
    seed: int,
    capacity: str,
    num_features: int,
    num_classes: int,
    batch_size: int,
    epsilon: float,
) -> ModelSpec:
    """Describe the surrogate distilled from the defended model."""
    return _spec(
        level,
        dataset,
        seed,
        capacity,
        num_features,
        num_classes,
        subset_selector=DATASET_SPLIT_ADVERSARY,
        optimizer_recipe=stolen_recipe(epsilon),
        batch_size=batch_size,
    )


@dataclass(frozen=True)
class ModelBundle:
    """The three models one E2 cell trains, plus the specs that keyed them.

    Exposed as a seam so a test can confirm the defended model is genuinely the
    adversarially-trained one, distinct from the clean target.

    Attributes:
        clean: The clean baseline.
        defended: The defended model.
        stolen: The surrogate distilled from the defended model.
        clean_spec: The spec that keyed `clean`.
        defended_spec: The spec that keyed `defended`.
    """

    clean: nn.Module
    defended: nn.Module
    stolen: nn.Module
    clean_spec: ModelSpec
    defended_spec: ModelSpec


def build_models(
    ctx: RunContext, dataset: str, epsilon: float, capacity: str = CAPACITY
) -> tuple[ModelBundle, AmuletDataset]:
    """Train (or load) the clean, defended and stolen models for one cell.

    Args:
        ctx: The run context (level, seed, device, cache directory).
        dataset: The dataset name.
        epsilon: The perturbation budget (the swept value, which keys the row).
        capacity: The capacity tier.

    Returns:
        The three models with their specs, and the loaded dataset.
    """
    from amulet.unauth_model_ownership.attacks import ModelExtraction

    data = ctx.data(dataset)
    num_features, num_classes = data.num_features, data.num_classes
    batch_size = batch_for(ctx.level, BATCH_SIZE)
    applied_epsilon = epsilon_for(ctx.level, epsilon)
    iterations = pgd_iterations_for(ctx.level)

    target_set, adversary_set = dataset_adversary_split(data.train_set, ctx.seed)
    target_loader = loader_for(target_set, batch_size)
    adversary_loader = loader_for(adversary_set, batch_size)

    clean_spec = clean_target_spec(
        ctx.level, dataset, ctx.seed, capacity, num_features, num_classes, batch_size
    )
    defended_spec = defended_target_spec(
        ctx.level,
        dataset,
        ctx.seed,
        capacity,
        num_features,
        num_classes,
        batch_size,
        epsilon,
    )
    stolen_spec = stolen_model_spec(
        ctx.level,
        dataset,
        ctx.seed,
        capacity,
        num_features,
        num_classes,
        batch_size,
        epsilon,
    )

    from common import progress

    progress.log(f"    {dataset} eps={epsilon:g}: clean baseline")
    clean = ctx.get_or_train(
        clean_spec,
        num_features,
        num_classes,
        lambda model: train_clean(model, target_loader, ctx.device, clean_spec.epochs),
    )
    progress.log(f"    {dataset} eps={epsilon:g}: defended (adversarial training)")
    defended = ctx.get_or_train(
        defended_spec,
        num_features,
        num_classes,
        lambda model: adversarially_train(
            model,
            target_loader,
            ctx.device,
            defended_spec.epochs,
            applied_epsilon,
            iterations,
        ),
    )
    progress.log(f"    {dataset} eps={epsilon:g}: surrogate (distillation)")

    def distil(model: nn.Module) -> nn.Module:
        # The surrogate is distilled from the DEFENDED model, never the clean
        # target: the adversary steals the robust (defended) model.
        extraction = ModelExtraction(
            defended,
            model,
            torch.optim.Adam(model.parameters(), lr=1e-3),
            adversary_loader,
            ctx.device,
            stolen_spec.epochs,
            loss_type="mse",
        )
        return extraction.attack()

    stolen = ctx.get_or_train(stolen_spec, num_features, num_classes, distil)

    return (
        ModelBundle(clean, defended, stolen, clean_spec, defended_spec),
        data,
    )
