"""Every target model E1 trains, defined as a `ModelSpec` in one place.

The single most important thing in this module is that **every** target E1 builds
is described here, side by side. Whether two sub-attacks share a target model is
then a question a reader answers by comparing two adjacent functions, not by
tracing two scripts. The specs encode the axes along which the sub-attacks
deliberately diverge:

| Sub-attack           | Optimizer                | Label       | Training subset |
| -------------------- | ------------------------ | ----------- | --------------- |
| evasion              | SGD 0.1 + StepLR(60)     | `Smiling`   | all             |
| poisoning            | SGD 0.01 + StepLR(20)    | `Smiling`   | all             |
| model extraction     | Adam 1e-3                | `Smiling`   | target half     |
| attribute inference  | Adam 1e-3                | `Smiling`   | target half     |
| data reconstruction  | Adam 1e-3                | `Wavy_Hair` | all             |
| membership inference | Adam 1e-3, ResNet, 10%   | `Wavy_Hair` | `pkeep` subset  |

Model extraction and attribute inference agree on every field, so they share one
checkpoint automatically. Nothing else does, and nothing else can: a divergence
in any field produces a different cache key. Sharing is emergent, not engineered;
no code here coordinates it.

The optimizer-recipe and training-subset strings below are the cache's contract:
the same string must mean the same procedure. Any change to how a target is
trained (the `train_*` closures live in the `attacks/` modules) requires changing
the recipe string it is named by, so a stale checkpoint can never be reused.

The scaffolding these specs read (the level-scaling helpers, the shared CelebA
constants, the batch sizes) lives in `context.py`; the training utilities the
attacks pair with these specs live in `common.training`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from common.models import ModelSpec
from experiments.e1_attack_baselines import context

if TYPE_CHECKING:
    from common.config import LevelConfig

# Optimizer recipes. These strings are the cache's contract: the same string must
# mean the same procedure, so any change to the matching `train_*` closure in the
# attacks/ modules requires changing the string it is named by.
EVASION_RECIPE = "sgd_lr1e-1_mom0.9_wd5e-4_steplr60_gamma0.2"
POISONING_RECIPE = "sgd_lr1e-2_mom0.9_wd5e-4_nesterov_steplr20_gamma0.1"
# The backdoored target trains at a flat learning rate (no schedule), which the
# recipe string records so its checkpoint is distinct from the scheduled clean one.
POISONING_BACKDOOR_RECIPE = "sgd_lr1e-2_mom0.9_wd5e-4_nesterov_no_schedule"
ADAM_RECIPE = "adam_lr1e-3"
EXTRACTION_RECIPE = "adam_lr1e-3_distil_mse_from_adam_lr1e-3_target_half"
SHADOW_BANK_RECIPE = "sgd_lr1e-2_mom0.9_wd5e-4_cosine"

# Training-subset names. `full` means the whole (possibly level-reduced) split.
# The fractions and portions come from `context`, so the subset name and the
# attack that produces that subset read the same constant.
FULL_SPLIT = "full"
TARGET_HALF = f"target_{1 - context.ADVERSARY_FRACTION:g}_seeded_index_split"
ADVERSARY_HALF = f"adversary_{context.ADVERSARY_FRACTION:g}_seeded_index_split"
BACKDOORED_SPLIT = f"badnets_p{context.POISONED_PORTION:g}_label{context.TRIGGER_LABEL}"
OVERFIT_SUBSET = f"lira_keep_pkeep{context.PKEEP:g}"


def _spec(
    level: LevelConfig,
    seed: int,
    capacity: str,
    num_features: int,
    num_classes: int,
    *,
    arch: str,
    train_fraction: float,
    subset_selector: str,
    label_attribute: str,
    optimizer_recipe: str,
    epochs: int,
    batch_size: int,
) -> ModelSpec:
    """Assemble a spec, filling the fields every E1 model shares."""
    return ModelSpec(
        dataset=context.DATASET,
        arch=context.architecture(level, arch),
        capacity=capacity,
        num_features=num_features,
        num_classes=num_classes,
        seed=seed,
        train_fraction=train_fraction,
        subset_selector=subset_selector,
        label_attribute=label_attribute,
        optimizer_recipe=optimizer_recipe,
        epochs=epochs,
        batch_size=batch_size,
    )


def evasion_target_spec(
    level: LevelConfig, seed: int, capacity: str, num_features: int, num_classes: int
) -> ModelSpec:
    """Describe evasion's target: SGD at 0.1 with a step schedule, all the data."""
    return _spec(
        level,
        seed,
        capacity,
        num_features,
        num_classes,
        arch="vgg",
        train_fraction=level.train_fraction,
        subset_selector=FULL_SPLIT,
        label_attribute=context.DEFAULT_TARGET_ATTRIBUTE,
        optimizer_recipe=EVASION_RECIPE,
        epochs=context.epochs_for(level),
        batch_size=context.batch_for(level, context.EVASION_BATCH_SIZE),
    )


def poisoning_clean_spec(
    level: LevelConfig, seed: int, capacity: str, num_features: int, num_classes: int
) -> ModelSpec:
    """Describe poisoning's clean baseline, trained on clean data."""
    return _spec(
        level,
        seed,
        capacity,
        num_features,
        num_classes,
        arch="vgg",
        train_fraction=level.train_fraction,
        subset_selector=FULL_SPLIT,
        label_attribute=context.DEFAULT_TARGET_ATTRIBUTE,
        optimizer_recipe=POISONING_RECIPE,
        epochs=context.halved_epochs_for(level),
        batch_size=context.batch_for(level, context.POISONING_BATCH_SIZE),
    )


def poisoning_backdoored_spec(
    level: LevelConfig, seed: int, capacity: str, num_features: int, num_classes: int
) -> ModelSpec:
    """Describe the backdoored target, trained on poisoned data."""
    return _spec(
        level,
        seed,
        capacity,
        num_features,
        num_classes,
        arch="vgg",
        train_fraction=level.train_fraction,
        subset_selector=BACKDOORED_SPLIT,
        label_attribute=context.DEFAULT_TARGET_ATTRIBUTE,
        optimizer_recipe=POISONING_BACKDOOR_RECIPE,
        epochs=context.halved_epochs_for(level),
        batch_size=context.batch_for(level, context.POISONING_BATCH_SIZE),
    )


def adversary_split_target_spec(
    level: LevelConfig, seed: int, capacity: str, num_features: int, num_classes: int
) -> ModelSpec:
    """Describe the target shared by model extraction and attribute inference.

    Both reserve half the training split for the adversary and train the target
    on the other half with Adam. Every spec field therefore agrees, and the two
    sub-attacks share one checkpoint.
    """
    return _spec(
        level,
        seed,
        capacity,
        num_features,
        num_classes,
        arch="vgg",
        train_fraction=level.train_fraction,
        subset_selector=TARGET_HALF,
        label_attribute=context.DEFAULT_TARGET_ATTRIBUTE,
        optimizer_recipe=ADAM_RECIPE,
        epochs=context.epochs_for(level),
        batch_size=context.batch_for(level, context.ADVERSARY_SPLIT_BATCH_SIZE),
    )


def stolen_model_spec(
    level: LevelConfig, seed: int, capacity: str, num_features: int, num_classes: int
) -> ModelSpec:
    """Describe the surrogate distilled from that target.

    A distilled model's weights depend on the model it was distilled from, which
    is not a `ModelSpec` field. The provenance is carried in the recipe string
    instead, so a surrogate trained against a differently-trained target could
    not silently reuse this checkpoint.
    """
    return _spec(
        level,
        seed,
        capacity,
        num_features,
        num_classes,
        arch="vgg",
        train_fraction=level.train_fraction,
        subset_selector=ADVERSARY_HALF,
        label_attribute=context.DEFAULT_TARGET_ATTRIBUTE,
        optimizer_recipe=EXTRACTION_RECIPE,
        epochs=context.epochs_for(level),
        batch_size=context.batch_for(level, context.ADVERSARY_SPLIT_BATCH_SIZE),
    )


def reconstruction_target_spec(
    level: LevelConfig, seed: int, capacity: str, num_features: int, num_classes: int
) -> ModelSpec:
    """Describe data reconstruction's target: Adam, all the data, `Wavy_Hair`.

    Shares model extraction's optimizer but not its label or its subset, so the
    two are correctly separate checkpoints.
    """
    return _spec(
        level,
        seed,
        capacity,
        num_features,
        num_classes,
        arch="vgg",
        train_fraction=level.train_fraction,
        subset_selector=FULL_SPLIT,
        label_attribute=context.PRIVACY_TARGET_ATTRIBUTE,
        optimizer_recipe=ADAM_RECIPE,
        epochs=context.epochs_for(level),
        batch_size=context.batch_for(level, context.RECONSTRUCTION_BATCH_SIZE),
    )


def overfit_target_spec(
    level: LevelConfig, seed: int, capacity: str, num_features: int, num_classes: int
) -> ModelSpec:
    """Describe membership inference's intentionally overfit ResNet target.

    A tenth of the training data, halved epochs and a different architecture, so it
    is never shareable with another sub-attack. Every one of those three is a spec
    field, so that is enforced rather than remembered.
    """
    return _spec(
        level,
        seed,
        capacity,
        num_features,
        num_classes,
        arch="resnet",
        train_fraction=level.train_fraction * context.OVERFIT_TRAINING_SIZE,
        subset_selector=OVERFIT_SUBSET,
        label_attribute=context.PRIVACY_TARGET_ATTRIBUTE,
        optimizer_recipe=ADAM_RECIPE,
        epochs=context.halved_epochs_for(level),
        batch_size=context.batch_for(level, context.MEMBERSHIP_BATCH_SIZE),
    )


def shadow_bank_spec(
    level: LevelConfig, seed: int, capacity: str, num_features: int, num_classes: int
) -> ModelSpec:
    """Describe LiRA's bank of shadow models as one cache entry.

    `LiRA` manages its own checkpoint files inside a directory it is handed, so
    this spec names the *directory* rather than a single `.pt`. Keying that
    directory on the same content-addressed hash keeps the guarantee: a bank
    trained at a different size, architecture or epoch count lands somewhere
    else instead of being silently reused.
    """
    return _spec(
        level,
        seed,
        capacity,
        num_features,
        num_classes,
        arch=context.shadow_architecture(level),
        train_fraction=level.train_fraction * context.OVERFIT_TRAINING_SIZE,
        subset_selector=(
            f"lira_shadow_bank_pkeep{context.PKEEP:g}_n{context.shadow_count(level)}"
        ),
        label_attribute=context.PRIVACY_TARGET_ATTRIBUTE,
        optimizer_recipe=SHADOW_BANK_RECIPE,
        epochs=context.halved_epochs_for(level),
        batch_size=context.batch_for(level, context.MEMBERSHIP_BATCH_SIZE),
    )
