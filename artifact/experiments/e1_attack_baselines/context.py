"""Run scaffolding for E1: data loading, model construction and level scaling.

This is the environment an E1 sub-attack runs inside, separate from *which*
target it trains (that is `train_targets.py`). It holds:

* the CelebA constants every sub-attack shares (the label attributes, the
  adversary fraction, the attack knobs);
* the level-scaling helpers that turn a `LevelConfig` into a concrete budget
  (`epochs_for`, `batch_for`, `architecture`, `shadow_count`, ...), so a reduced
  level shrinks the same way everywhere;
* the `test`-level synthetic stand-in for CelebA, so the fast tier needs no
  multi-gigabyte download;
* `RunContext`, which memoises the loaded dataset and drives the content-addressed
  checkpoint cache through `get_or_train`.

The level knobs scale a run to its verification level: `architecture` swaps in a
tiny VGG at `test` level (a distinct architecture name, so its checkpoint can
never be served for a real one), and the repeated-work loops (`shadow_count`,
`evasion_iterations_for`, `data_reconstruction._alpha`) shrink at any level below
`full`.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from amulet.datasets import AmuletDataset
from amulet.models import VGG
from amulet.utils import initialize_model, load_data
from common.config import LevelConfig
from common.io import run_output_dir
from common.models import ModelSpec, get_or_train, model_cache_root
from common.paths import repo_root

if TYPE_CHECKING:
    from collections.abc import Callable

EXPERIMENT_ID = "e1_attack_baselines"
DATASET = "celeba"

# The paper trains for 100 epochs with the Adam optimizer "unless otherwise
# specified" (paper evaluation section). The `full` level leaves epochs unset by
# design, so this is what fills it.
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

# BadNets' knobs.
POISONED_PORTION = 0.1
TRIGGER_LABEL = 1

# Evasion's PGD budget: the paper's 0.03, 40 iterations.
EVASION_EPSILON = 0.03
EVASION_ITERATIONS = 40
# What the reduced levels run instead. More iterations only tighten the
# perturbation, they reach no new code. Seven matches
# `common.run_context.SMOKE_PGD_ITERATIONS` and the standard PGD-7 configuration,
# so both PGD studies agree on what a reduced chain means.
SMOKE_EVASION_ITERATIONS = 7
TINY_EVASION_ITERATIONS = 7

# LiRA's knobs. The target is intentionally overfit: a tenth of the training
# data for half the epochs, which is what makes the attack measurable at all.
PKEEP = 0.5
NUM_SHADOW = 64
# The bank a reduced-budget level trains instead. Every shadow model is a full
# training run, so the bank dominates E1's wall clock; a larger bank only sharpens
# the per-example Gaussian fit (attack quality), which no reduced level measures.
# The floor is 8: `int(PKEEP * n)` must be at least 2 each side, or an example
# lands IN no shadow, empties `dat_in`, and turns every LiRA score into a NaN.
SMOKE_NUM_SHADOW = 8
OVERFIT_TRAINING_SIZE = 0.1

# Data reconstruction's gradient-descent budget.
RECONSTRUCTION_ALPHA = 3000

# Batch sizes, per sub-attack. These are spec fields: extraction and attribute
# inference agreeing on 256 is part of why they share a target.
EVASION_BATCH_SIZE = 128
POISONING_BATCH_SIZE = 256
ADVERSARY_SPLIT_BATCH_SIZE = 256
RECONSTRUCTION_BATCH_SIZE = 256
MEMBERSHIP_BATCH_SIZE = 128

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


def evasion_iterations_for(level: LevelConfig) -> int:
    """Return how many PGD iterations E1's evasion attack takes.

    Only `full` runs the paper's chain; see `SMOKE_EVASION_ITERATIONS`. The
    count is a CSV column, so a reduced run is recorded rather than implied.

    Args:
        level: The level preset.

    Returns:
        The number of PGD steps per batch.
    """
    if level.tiny_model:
        return TINY_EVASION_ITERATIONS
    if level.train_fraction < 1.0:
        return SMOKE_EVASION_ITERATIONS
    return EVASION_ITERATIONS


def shadow_count(level: LevelConfig) -> int:
    """Return how many shadow models LiRA trains at this level.

    Only `full` trains the paper's bank; see `SMOKE_NUM_SHADOW` for why a
    reduced-budget level trains fewer. The count is a spec field and a CSV
    column, so a smaller bank lands in its own cache slot and is visible in the
    results rather than silently substituted.

    Args:
        level: The level preset.

    Returns:
        The number of shadow models to train.
    """
    if level.tiny_model:
        return TINY_NUM_SHADOW
    if level.train_fraction < 1.0:
        return SMOKE_NUM_SHADOW
    return NUM_SHADOW


def shadow_architecture(level: LevelConfig) -> str:
    """Return the architecture LiRA builds its shadow models from.

    `LiRA` constructs shadow models internally via `initialize_model` with the
    default capacity map, so the tiny stand-in cannot reach them. At `test`
    level the bank is a handful of real VGG11s over 48 synthetic images, which
    is cheap for a different reason.

    Args:
        level: The level preset.

    Returns:
        An architecture name `amulet.utils.initialize_model` accepts.
    """
    return "vgg" if level.tiny_model else "resnet"


def build_model(
    arch: str,
    capacity: str,
    num_features: int,
    num_classes: int,
    device: str,
) -> nn.Module:
    """Create the untrained model a spec describes, on the target device.

    Args:
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
        cache_dir: Directory for the content-addressed checkpoint cache, from
            `default_cache_dir(level)`. Required rather than defaulted, so a
            context can never silently write a level's checkpoints into another
            level's cache.
    """

    level: LevelConfig
    seed: int
    device: str
    cache_dir: Path
    _datasets: dict[tuple[str, float, float], AmuletDataset] = field(
        default_factory=dict
    )

    def data(self, celeba_target: str, training_size: float) -> AmuletDataset:
        """Load the dataset a sub-attack needs, once per distinct request.

        CelebA takes tens of seconds to read even from its processed cache, and
        a full sweep asks for the same two variants repeatedly, so results are
        memoised for the lifetime of this context.

        The training fraction is a parameter because membership inference wants
        a tenth of what its siblings train on (`OVERFIT_TRAINING_SIZE`); the
        test fraction is not, because every sub-attack evaluates on the same
        split and the level alone decides how much of it to keep.

        Args:
            celeba_target: The attribute used as the classification label.
            training_size: Fraction of the training split to load.

        Returns:
            The dataset. Callers must treat it as read-only: the memo hands the
            same object to every sub-attack that asks for it.
        """
        key = (celeba_target, training_size, self.level.test_fraction)
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
                    test_size=self.level.test_fraction,
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
                spec.arch,
                spec.capacity,
                num_features,
                num_classes,
                self.device,
            )

        return initialise

    def get_or_train(
        self,
        spec: ModelSpec,
        num_features: int,
        num_classes: int,
        train_fn: Callable[[nn.Module], nn.Module],
    ) -> nn.Module:
        """Load a target from the shared cache, or train it with `train_fn` and cache it.

        The one path every E1 target goes through. The checkpoint's location is
        the content hash of `spec`, so a target two sub-attacks describe
        identically is trained once and reused.

        Args:
            spec: The target's spec; determines the cache key.
            num_features: Input feature count, for the initialiser.
            num_classes: Number of output classes.
            train_fn: Callable taking the fresh model and returning it trained.

        Returns:
            The loaded or newly trained target, on this run's device.
        """
        return get_or_train(
            spec,
            self.model_factory(spec, num_features, num_classes),
            train_fn,
            LOGGER,
            cache_dir=self.cache_dir,
        ).to(self.device)


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


def default_cache_dir(level: LevelConfig) -> Path:
    """Return the checkpoint cache this level should write to.

    Args:
        level: The level preset.

    Returns:
        This level's own subdirectory of `.model_cache/`, or a throwaway
        directory for `test`, whose tiny stand-ins have no business
        outliving the test that made them.
    """
    if level.tiny_model:
        return Path(tempfile.mkdtemp(prefix="e1_test_models_"))
    return model_cache_root(level.name)


def default_output_dir(level: LevelConfig) -> Path:
    """Return the directory this level's result CSVs belong in.

    Every level writes under its own `runs/<level>/<experiment_id>/` subtree, so a
    cheap `test` or `smoke` run never overwrites the `full` results a paper
    comparison reads from, nor has its reduced-budget numbers averaged into them.
    No result data ships with the repository; every number comes from a run that
    lands here. E1's CSVs live in a `<experiment_id>/` subdirectory, the layout a
    `make_*` renderer reads with one path rule at any level.

    Args:
        level: The level preset.

    Returns:
        An existing or creatable directory.
    """
    return run_output_dir(level.name) / EXPERIMENT_ID
