"""Everything E1's six sub-attacks have in common: data, models and specs.

The single most important thing in this module is that **every** `ModelSpec` E1
builds is defined here, side by side. Whether two sub-attacks share a target
model is then a question a reader answers by comparing two adjacent functions,
not by tracing two scripts. The specs encode the three axes along which the
sub-attacks deliberately diverge (plan §5):

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
in any field produces a different cache key.

Levels (plan §8) are handled in two places only. `architecture` swaps a tiny
three-layer VGG in at `test` level — recorded in the spec as a distinct
architecture, so a tiny checkpoint can never be loaded in place of a real one.
`tiny_dataset` stands in for CelebA, which is a multi-gigabyte download the fast
verification tier must not require.
"""

from __future__ import annotations

import logging
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from amulet.datasets import AmuletDataset
from amulet.models import VGG
from amulet.utils import initialize_model, load_data, train_classifier
from common.config import LevelConfig
from common.io import run_output_dir
from common.models import ModelSpec
from common.paths import repo_root

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.optim.lr_scheduler import _LRScheduler

EXPERIMENT_ID = "e1_attack_baselines"
DATASET = "celeba"

# The paper trains for 100 epochs with the Adam optimizer "unless otherwise
# specified" (`artifact/experimental_setup/06evaluation.tex`). The `full` level
# leaves epochs unset by design, so this is what fills it.
PAPER_EPOCHS = 100

# CelebA's default classification target, and the one the two privacy attacks
# swap it for. A different label means different weights, so this is a spec field.
DEFAULT_TARGET_ATTRIBUTE = "Smiling"
PRIVACY_TARGET_ATTRIBUTE = "Wavy_Hair"

# The sensitive attribute CelebA always carries, inferred by attribute inference.
SENSITIVE_ATTRIBUTE = "Male"

# Half the training split is reserved for the adversary in the two attacks that
# assume query access plus some data of their own.
ADVERSARY_FRACTION = 0.5

# BadNets' knobs, from the old `run_poisoning.py` defaults.
POISONED_PORTION = 0.1
TRIGGER_LABEL = 1

# Evasion's perturbation budget, from `run_evasion.sh` (`--epsilon 0.03`), not
# from the script's own 0.01 default, which the paper run never used.
EVASION_EPSILON = 0.03
EVASION_ITERATIONS = 40

# LiRA's knobs. The target is intentionally overfit: a tenth of the training
# data for half the epochs, which is what makes the attack measurable at all.
PKEEP = 0.5
NUM_SHADOW = 64
OVERFIT_TRAINING_SIZE = 0.1

# Data reconstruction's gradient-descent budget.
RECONSTRUCTION_ALPHA = 3000

# Batch sizes, per sub-attack, as the original scripts set them. These are spec
# fields: extraction and attribute inference agreeing on 256 is part of why they
# share a target.
EVASION_BATCH_SIZE = 128
POISONING_BATCH_SIZE = 256
ADVERSARY_SPLIT_BATCH_SIZE = 256
RECONSTRUCTION_BATCH_SIZE = 256
MEMBERSHIP_BATCH_SIZE = 128

# Optimizer recipes. These strings are the cache's contract: the same string
# must mean the same procedure, so any change to a `train_*` function below
# requires changing the string it is named by.
EVASION_RECIPE = "sgd_lr1e-1_mom0.9_wd5e-4_steplr60_gamma0.2"
POISONING_RECIPE = "sgd_lr1e-2_mom0.9_wd5e-4_nesterov_steplr20_gamma0.1"
# The original script builds a scheduler for the backdoored victim and then does
# not pass it to `train_classifier`, so that model trains at a flat learning
# rate. Preserved deliberately: it is the procedure behind the published
# numbers, and the recipe string says so rather than hiding it.
POISONING_BACKDOOR_RECIPE = "sgd_lr1e-2_mom0.9_wd5e-4_nesterov_no_schedule"
ADAM_RECIPE = "adam_lr1e-3"
EXTRACTION_RECIPE = "adam_lr1e-3_distil_mse_from_adam_lr1e-3_target_half"

# Training-subset names. `full` means the whole (possibly level-reduced) split.
FULL_SPLIT = "full"
TARGET_HALF = f"target_{1 - ADVERSARY_FRACTION:g}_seeded_index_split"
ADVERSARY_HALF = f"adversary_{ADVERSARY_FRACTION:g}_seeded_index_split"
BACKDOORED_SPLIT = f"badnets_p{POISONED_PORTION:g}_label{TRIGGER_LABEL}"
OVERFIT_SUBSET = f"lira_keep_pkeep{PKEEP:g}"

# The architecture recorded for a `test`-level stand-in. It is a real VGG, deep
# enough to pool the spatial map away cheaply, so the pipeline exercises the same
# code as VGG11 does. The last convolution must emit 512 channels because
# `amulet.models.VGG` hard-wires a `Linear(512, num_classes)` classifier.
TINY_ARCH = "tiny_vgg"
TINY_VGG_LAYERS: list[int | str] = [4, "M", 8, "M", 16, "M", 32, "M", 512, "M"]

# The synthetic stand-in for CelebA at `test` level: 3-channel images, a binary
# target and a binary sensitive attribute, CelebA's shape in miniature. The
# 32x32 resolution is the smallest that survives VGG11's five max-pools, because
# `LiRA` builds its shadow bank from a real VGG11 (see `shadow_architecture`).
TINY_TRAIN_SIZE = 48
TINY_TEST_SIZE = 24
TINY_IMAGE_SHAPE = (3, 32, 32)
# A tiny run trains for a handful of epochs at a small batch, enough for the
# separable stand-in data to be learned rather than left at chance. Chosen so
# the fast tier trains a non-degenerate model in well under a second per fit
# (single-threaded; see `run`), which is what makes the evasion degradation and
# in-range assertions meaningful rather than measuring noise.
TINY_EPOCHS = 5
TINY_BATCH = 8
# A larger perturbation budget at `test` level, so PGD visibly degrades the tiny
# target and the degradation assertion is a real check, not a tie at 100%.
TINY_EVASION_EPSILON = 0.3
# Shadow models are built by `LiRA` itself through `initialize_model` with the
# default capacity map, so a tiny stand-in cannot be injected into them. At
# `test` level the bank is therefore small and real rather than large and tiny.
TINY_NUM_SHADOW = 4

LOGGER = logging.getLogger("e1_attack_baselines")


def seed_everything(seed: int) -> None:
    """Seed every generator a sub-attack draws from.

    Three of the original scripts seeded `torch` alone and then chose the
    adversary's records, the poisoned indices or the membership mask from
    NumPy's global generator, so a re-run answered a different question. Seeding
    all three is what makes `test_same_seed_reproduces` hold.

    Args:
        seed: The experiment seed, recorded as `exp_id`.
    """
    random.seed(seed)
    np.random.seed(seed)
    _ = torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def epochs_for(level: LevelConfig) -> int:
    """Return the training budget for a sub-attack that trains for the full run.

    Args:
        level: The level preset, already carrying `epochs`.

    Returns:
        `TINY_EPOCHS` at tiny level, else the level's epoch count (at least one).
        The tiny floor is enough for the separable stand-in data to be learned;
        one epoch on a fresh VGG leaves it at chance.
    """
    if level.tiny_model:
        return TINY_EPOCHS
    return max(1, level.epochs if level.epochs is not None else PAPER_EPOCHS)


def halved_epochs_for(level: LevelConfig) -> int:
    """Return the budget for a sub-attack the original script ran at half length.

    Poisoning trains for `epochs // 2`, and membership inference for its own 50
    (which `100 // 2` also gives). Deriving both from the one level-wide count
    keeps `smoke`/`full` faithful while the tiny floor keeps `test` learnable.

    Args:
        level: The level preset, already carrying `epochs`.

    Returns:
        `TINY_EPOCHS` at tiny level, else half the level's epoch count.
    """
    if level.tiny_model:
        return TINY_EPOCHS
    return max(1, epochs_for(level) // 2)


def batch_for(level: LevelConfig, paper_batch: int) -> int:
    """Return the batch size to train with, shrunk at tiny level.

    The paper batch sizes over the stand-in's 48 records would be a single step
    per epoch, too few for the model to learn. A small batch gives several steps
    per epoch instead. This is a spec field, so the tiny batch is recorded in
    the tiny model's key and cannot collide with a paper checkpoint.

    Args:
        level: The level preset.
        paper_batch: The batch size the paper run uses.

    Returns:
        `TINY_BATCH` at tiny level, else `paper_batch`.
    """
    return TINY_BATCH if level.tiny_model else paper_batch


def architecture(level: LevelConfig, real: str) -> str:
    """Return the architecture name this level trains, and records in the spec.

    Args:
        level: The level preset.
        real: The architecture the paper uses, e.g. `"vgg"`.

    Returns:
        `real`, or the tiny stand-in's name when the level asks for one. The
        returned name goes into the `ModelSpec`, which is what stops a tiny
        checkpoint from ever being loaded in place of a real one.
    """
    return TINY_ARCH if level.tiny_model else real


def shadow_count(level: LevelConfig) -> int:
    """Return how many shadow models LiRA trains at this level."""
    return TINY_NUM_SHADOW if level.tiny_model else NUM_SHADOW


def shadow_architecture(level: LevelConfig) -> str:
    """Return the architecture LiRA builds its shadow models from.

    `LiRA` constructs shadow models internally via `initialize_model` with the
    default capacity map, so the tiny stand-in cannot reach them. At `test`
    level the bank is a handful of real VGG11s over 64 synthetic images, which
    is cheap for a different reason.

    Args:
        level: The level preset.

    Returns:
        An architecture name `amulet.utils.initialize_model` accepts.
    """
    return "vgg" if level.tiny_model else "resnet"


def build_model(
    level: LevelConfig,
    arch: str,
    capacity: str,
    num_features: int,
    num_classes: int,
    device: str,
) -> nn.Module:
    """Create the untrained model a spec describes, on the target device.

    Args:
        level: The level preset.
        arch: Architecture name as recorded in the spec.
        capacity: Capacity tier, one of `m1`-`m4`.
        num_features: Input feature count, for architectures that need it.
        num_classes: Number of output classes.
        device: Device to place the model on. Example: `"cuda:0"`.

    Returns:
        A freshly initialised model, as `common.models.get_or_train` requires.
    """
    if arch == TINY_ARCH:
        return VGG(
            num_classes=num_classes, layer_config=TINY_VGG_LAYERS, batch_norm=True
        ).to(device)
    return initialize_model(arch, capacity, num_features, num_classes, LOGGER).to(
        device
    )


def tiny_dataset(seed: int, num_classes: int = 2) -> AmuletDataset:
    """Build the synthetic stand-in for CelebA used at `test` level.

    Real CelebA is a multi-gigabyte Google Drive download, so requiring it would
    make Level 1 unrunnable on a fresh clone. The stand-in keeps CelebA's shape:
    3-channel images in [0, 1], a balanced binary label and a balanced binary
    sensitive attribute in a `(N, 1)` column, with both the tensor datasets and
    the NumPy views the attribute-inference attack reads.

    Each class is drawn around its own mean so the label is learnable. A model
    that cannot beat chance would make the evasion and extraction assertions
    vacuous.

    Args:
        seed: Seed for the generator, so two runs build identical data.
        num_classes: Number of label classes.

    Returns:
        A dataset with `train_set`, `test_set` and the `x_*`/`y_*`/`z_*` arrays
        populated, index-aligned with each other.
    """
    generator = np.random.default_rng(seed)

    def split(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        labels = np.arange(size) % num_classes
        # Each class sits in its own well-separated intensity band (low noise
        # plus a large per-class offset), so a few epochs learn it cleanly. A
        # model left at chance would make the evasion and extraction assertions
        # vacuous. Values stay in the [0, 1] box CelebA images live in.
        images = np.clip(
            generator.random((size, *TINY_IMAGE_SHAPE), dtype=np.float32) * 0.15
            + 0.8 * labels[:, None, None, None].astype(np.float32),
            0.0,
            1.0,
        )
        sensitive = (np.arange(size) // 2 % 2).reshape(-1, 1)
        return images, labels.astype(np.int64), sensitive.astype(np.int64)

    x_train, y_train, z_train = split(TINY_TRAIN_SIZE)
    x_test, y_test, z_test = split(TINY_TEST_SIZE)

    return AmuletDataset(
        train_set=TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        test_set=TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)),
        num_features=TINY_IMAGE_SHAPE[1] * TINY_IMAGE_SHAPE[2],
        num_classes=num_classes,
        modality="image",
        sensitive_columns=[SENSITIVE_ATTRIBUTE],
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        z_train=z_train,
        z_test=z_test,
    )


@dataclass
class RunContext:
    """What every sub-attack needs to run one cell of the sweep.

    Attributes:
        level: The verification-level preset, already carrying `epochs`.
        seed: The experiment seed, recorded as `exp_id`.
        device: Device to train and evaluate on. Example: `"cuda:0"`.
        cache_dir: Directory for the content-addressed checkpoint cache. None
            uses the shared default.
    """

    level: LevelConfig
    seed: int
    device: str
    cache_dir: Path | None = None
    _datasets: dict[tuple[str, float], AmuletDataset] = field(default_factory=dict)

    def data(self, celeba_target: str, training_size: float) -> AmuletDataset:
        """Load the dataset a sub-attack needs, once per distinct request.

        CelebA takes tens of seconds to read even from its processed cache, and
        a full sweep asks for the same two variants repeatedly, so results are
        memoised for the lifetime of this context.

        Args:
            celeba_target: The attribute used as the classification label.
            training_size: Fraction of the training split to load.

        Returns:
            The dataset. Callers must treat it as read-only: the memo hands the
            same object to every sub-attack that asks for it.
        """
        key = (celeba_target, training_size)
        if key not in self._datasets:
            self._datasets[key] = (
                tiny_dataset(self.seed)
                if self.level.tiny_model
                else load_data(
                    repo_root(),
                    DATASET,
                    training_size,
                    LOGGER,
                    exp_id=self.seed,
                    celeba_target=celeba_target,
                )
            )
        return self._datasets[key]

    def model_factory(
        self, spec: ModelSpec, num_features: int, num_classes: int
    ) -> Callable[[], nn.Module]:
        """Return the zero-argument initialiser `get_or_train` expects.

        Args:
            spec: The spec whose architecture and capacity to build.
            num_features: Input feature count.
            num_classes: Number of output classes.

        Returns:
            A callable producing a fresh untrained model on this run's device.
        """

        def initialise() -> nn.Module:
            return build_model(
                self.level,
                spec.arch,
                spec.capacity,
                num_features,
                num_classes,
                self.device,
            )

        return initialise


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
        dataset=DATASET,
        arch=architecture(level, arch),
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
        label_attribute=DEFAULT_TARGET_ATTRIBUTE,
        optimizer_recipe=EVASION_RECIPE,
        epochs=epochs_for(level),
        batch_size=batch_for(level, EVASION_BATCH_SIZE),
    )


def poisoning_clean_spec(
    level: LevelConfig, seed: int, capacity: str, num_features: int, num_classes: int
) -> ModelSpec:
    """Describe poisoning's clean baseline $\\modelstd$, trained on clean data."""
    return _spec(
        level,
        seed,
        capacity,
        num_features,
        num_classes,
        arch="vgg",
        train_fraction=level.train_fraction,
        subset_selector=FULL_SPLIT,
        label_attribute=DEFAULT_TARGET_ATTRIBUTE,
        optimizer_recipe=POISONING_RECIPE,
        epochs=halved_epochs_for(level),
        batch_size=batch_for(level, POISONING_BATCH_SIZE),
    )


def poisoning_backdoored_spec(
    level: LevelConfig, seed: int, capacity: str, num_features: int, num_classes: int
) -> ModelSpec:
    """Describe the backdoored victim $\\modelpois$, trained on poisoned data."""
    return _spec(
        level,
        seed,
        capacity,
        num_features,
        num_classes,
        arch="vgg",
        train_fraction=level.train_fraction,
        subset_selector=BACKDOORED_SPLIT,
        label_attribute=DEFAULT_TARGET_ATTRIBUTE,
        optimizer_recipe=POISONING_BACKDOOR_RECIPE,
        epochs=halved_epochs_for(level),
        batch_size=batch_for(level, POISONING_BATCH_SIZE),
    )


def adversary_split_target_spec(
    level: LevelConfig, seed: int, capacity: str, num_features: int, num_classes: int
) -> ModelSpec:
    """Describe the target shared by model extraction and attribute inference.

    Both reserve half the training split for the adversary and train the target
    on the other half with Adam. Every spec field therefore agrees, and the two
    sub-attacks share one checkpoint (plan §6).
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
        label_attribute=DEFAULT_TARGET_ATTRIBUTE,
        optimizer_recipe=ADAM_RECIPE,
        epochs=epochs_for(level),
        batch_size=batch_for(level, ADVERSARY_SPLIT_BATCH_SIZE),
    )


def stolen_model_spec(
    level: LevelConfig, seed: int, capacity: str, num_features: int, num_classes: int
) -> ModelSpec:
    """Describe the stolen surrogate $\\modelstol$ distilled from that target.

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
        label_attribute=DEFAULT_TARGET_ATTRIBUTE,
        optimizer_recipe=EXTRACTION_RECIPE,
        epochs=epochs_for(level),
        batch_size=batch_for(level, ADVERSARY_SPLIT_BATCH_SIZE),
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
        label_attribute=PRIVACY_TARGET_ATTRIBUTE,
        optimizer_recipe=ADAM_RECIPE,
        epochs=epochs_for(level),
        batch_size=batch_for(level, RECONSTRUCTION_BATCH_SIZE),
    )


def overfit_target_spec(
    level: LevelConfig, seed: int, capacity: str, num_features: int, num_classes: int
) -> ModelSpec:
    """Describe membership inference's intentionally overfit ResNet target.

    A tenth of the training data, halved epochs and a different architecture:
    the special case plan §5 marks as never shareable. Every one of those three
    is a spec field, so "never shareable" is enforced rather than remembered.
    """
    return _spec(
        level,
        seed,
        capacity,
        num_features,
        num_classes,
        arch="resnet",
        train_fraction=level.train_fraction * OVERFIT_TRAINING_SIZE,
        subset_selector=OVERFIT_SUBSET,
        label_attribute=PRIVACY_TARGET_ATTRIBUTE,
        optimizer_recipe=ADAM_RECIPE,
        epochs=halved_epochs_for(level),
        batch_size=batch_for(level, MEMBERSHIP_BATCH_SIZE),
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
        arch=shadow_architecture(level),
        train_fraction=level.train_fraction * OVERFIT_TRAINING_SIZE,
        subset_selector=f"lira_shadow_bank_pkeep{PKEEP:g}_n{shadow_count(level)}",
        label_attribute=PRIVACY_TARGET_ATTRIBUTE,
        optimizer_recipe="sgd_lr1e-2_mom0.9_wd5e-4_cosine",
        epochs=halved_epochs_for(level),
        batch_size=batch_for(level, MEMBERSHIP_BATCH_SIZE),
    )


def train_with_adam(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    epochs: int,
    learning_rate: float = 1e-3,
) -> nn.Module:
    """Train a classifier with Adam, the paper's default optimizer.

    Named by `ADAM_RECIPE`; changing anything here requires changing that string.

    Args:
        model: The freshly initialised model.
        loader: Training data.
        device: Device to train on.
        epochs: Number of passes over the data.
        learning_rate: Adam's learning rate.

    Returns:
        The trained model.
    """
    return train_classifier(
        model,
        loader,
        nn.CrossEntropyLoss(),
        torch.optim.Adam(model.parameters(), lr=learning_rate),
        epochs,
        device,
    )


def train_with_sgd(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    epochs: int,
    *,
    learning_rate: float,
    step_size: int,
    gamma: float,
    nesterov: bool = False,
    schedule: bool = True,
) -> nn.Module:
    """Train a classifier with momentum SGD and an optional step schedule.

    Named by `EVASION_RECIPE`, `POISONING_RECIPE` and
    `POISONING_BACKDOOR_RECIPE`; changing anything here requires changing those.

    Args:
        model: The freshly initialised model.
        loader: Training data.
        device: Device to train on.
        epochs: Number of passes over the data.
        learning_rate: SGD's initial learning rate.
        step_size: Epochs between learning-rate decays.
        gamma: Multiplicative decay applied at each step.
        nesterov: Whether to use Nesterov momentum.
        schedule: Whether to apply the decay at all. False reproduces the
            backdoored victim's flat learning rate (see `POISONING_BACKDOOR_RECIPE`).

    Returns:
        The trained model.
    """
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=nesterov,
    )
    scheduler = (
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        if schedule
        else None
    )
    return train_classifier(
        model,
        loader,
        nn.CrossEntropyLoss(),
        optimizer,
        epochs,
        device,
        # `train_classifier` types this parameter as the deprecated
        # `_LRScheduler`, a sibling subclass of the `LRScheduler` that `StepLR`
        # actually extends, so no concrete scheduler satisfies it. The runtime
        # contract is only `.step()`, which every scheduler honours; cast across
        # the too-narrow library annotation rather than edit the library.
        scheduler=cast("_LRScheduler | None", scheduler),
    )


def loader_for(dataset: object, batch_size: int) -> DataLoader:
    """Wrap a dataset in the unshuffled loader every E1 sub-attack uses.

    Args:
        dataset: Any map-style dataset yielding `(x, y)`.
        batch_size: Batch size.

    Returns:
        A `DataLoader` in dataset order, so an index-based split stays aligned
        with the rows the attack scores.
    """
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)  # type: ignore[reportArgumentType]


@dataclass(frozen=True)
class AdversarySplit:
    """The two disjoint halves of a training split, in the forms both attacks need.

    Attributes:
        target_set: The half the target model is trained on.
        adversary_set: The adversary's half as a dataset, used to query the
            target during extraction.
        adversary_x: The adversary's features, used to query the target during
            attribute inference.
        adversary_z: The adversary's sensitive attributes, the labels attribute
            inference trains its own classifier against.
    """

    target_set: TensorDataset
    adversary_set: TensorDataset
    adversary_x: np.ndarray
    adversary_z: np.ndarray


def adversary_split(
    data: AmuletDataset, seed: int, adv_fraction: float = ADVERSARY_FRACTION
) -> AdversarySplit:
    """Split the training data between the target owner and the adversary.

    Both adversary-split attacks go through this one function, which is what
    lets them share a target checkpoint. The original scripts split differently
    — model extraction with `torch.random_split`, attribute inference with an
    *unseeded* `sklearn.train_test_split` — so their targets could never have
    matched, and attribute inference's was not reproducible at all.

    Indexing the NumPy arrays rather than the `train_set` also sidesteps a
    library trap: when `load_data` is asked for a fraction of the split it
    subsamples the arrays and the `train_set` through independent generators,
    leaving the two no longer index-aligned. Deriving everything from the arrays
    keeps the target's data and the adversary's provably disjoint at any level.

    Args:
        data: The loaded dataset; must carry its NumPy views.
        seed: Seed for the split, so the same seed yields the same halves.
        adv_fraction: Fraction of the training split reserved for the adversary.

    Returns:
        The split, carrying the target's training set and the adversary's half
        in both the dataset and NumPy forms the two attacks consume.

    Raises:
        ValueError: If the dataset carries no NumPy views or no sensitive
            attributes.
    """
    if data.x_train is None or data.y_train is None:
        raise ValueError("Adversary split needs the dataset's NumPy feature arrays.")
    if data.z_train is None:
        raise ValueError("Adversary split needs the dataset's sensitive attributes.")

    target_index, adversary_index = train_test_split(
        np.arange(len(data.x_train)), test_size=adv_fraction, random_state=seed
    )
    target_set = TensorDataset(
        torch.from_numpy(data.x_train[target_index]).type(torch.float),
        torch.from_numpy(data.y_train[target_index]).type(torch.long),
    )
    adversary_x = data.x_train[adversary_index]
    adversary_set = TensorDataset(
        torch.from_numpy(adversary_x).type(torch.float),
        torch.from_numpy(data.y_train[adversary_index]).type(torch.long),
    )
    return AdversarySplit(
        target_set=target_set,
        adversary_set=adversary_set,
        adversary_x=adversary_x,
        adversary_z=data.z_train[adversary_index],
    )


def leading_row(spec: ModelSpec, celeba_target: str) -> dict[str, object]:
    """Fill the columns every E1 CSV opens with, from the measured target's spec.

    These columns describe *which model* a row measured, so a reader can tell
    from the CSV alone whether two rows shared a target. They are the spec's own
    fields, which is exactly what the cache key is built from.

    Args:
        spec: The spec of the primary target the row is about.
        celeba_target: The label attribute, spelt as `load_data` expects it
            (the spec's `label_attribute` is the same string).

    Returns:
        A mapping covering the schemas' shared leading columns, minus the seed,
        which the caller adds from the run context.
    """
    return {
        "exp_id": spec.seed,
        "dataset": spec.dataset,
        "arch": spec.arch,
        "capacity": spec.capacity,
        "training_size": spec.train_fraction,
        "celeba_target": celeba_target,
        "optimizer_recipe": spec.optimizer_recipe,
        "epochs": spec.epochs,
        "batch_size": spec.batch_size,
    }


def train_target_via_cache(
    ctx: RunContext,
    spec: ModelSpec,
    num_features: int,
    num_classes: int,
    train_fn: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """Load a target from the shared cache, or train it with `train_fn` and cache it.

    The one path every E1 target goes through, replacing the hand-rolled
    `if path.exists(): load else train+save` block each old script carried. The
    checkpoint's location is the content hash of `spec`, so a target two
    sub-attacks describe identically is trained once and reused (plan §6).

    Args:
        ctx: The run context, carrying the device and cache directory.
        spec: The target's spec; determines the cache key.
        num_features: Input feature count, for the initialiser.
        num_classes: Number of output classes.
        train_fn: Callable taking the fresh model and returning it trained.

    Returns:
        The loaded or newly trained target, on the run's device.
    """
    from common.models import get_or_train

    return get_or_train(
        spec,
        ctx.model_factory(spec, num_features, num_classes),
        train_fn,
        LOGGER,
        cache_dir=ctx.cache_dir,
    ).to(ctx.device)


def default_cache_dir(level: LevelConfig) -> Path | None:
    """Return the checkpoint cache this level should write to.

    Args:
        level: The level preset.

    Returns:
        None for the shared default, or a throwaway directory for `test`, whose
        three-layer stand-ins have no business outliving the test that made them.
    """
    if level.tiny_model:
        return Path(tempfile.mkdtemp(prefix="e1_test_models_"))
    return None


def default_output_dir(level: LevelConfig) -> Path:
    """Return the directory this level's result CSVs belong in.

    Every level writes under `runs/<level>/<experiment_id>/`, never into the
    committed `results/` tree: a `full` re-run must not clobber the shipped
    ground truth, and a `smoke`/`test` run must not have its one-epoch or
    tiny-model numbers averaged into the paper's. The `runs/<level>/` tree
    mirrors `results/` (E1's CSVs live in a `<experiment_id>/` subdirectory of
    both), so a `make_*` renderer reads either with the same path logic. Authors
    promote a completed `full` run by copying its CSVs into `results/`.

    Args:
        level: The level preset.

    Returns:
        An existing or creatable directory.
    """
    return run_output_dir(level.name) / EXPERIMENT_ID
