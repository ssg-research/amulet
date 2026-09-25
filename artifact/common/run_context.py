"""Run scaffolding for the multi-dataset from-scratch experiments (E2, E3, E4).

These three train models on the same four datasets (census, lfw, fmnist, cifar),
so they share the plumbing for loading a dataset, building the matching model,
scaling a budget to the verification level, and driving the checkpoint cache.
That plumbing lives here; *which* models each experiment trains, and how it
identifies them, lives in each experiment's own `train_targets.py`. Nothing here
knows about a specific target, a recipe, or a split selector, so it pre-disposes
no sharing: if two experiments' specs come out identical the cache dedups them,
and if they diverge nothing here objects.

`RunContext` is the centrepiece: created once per seed, it memoises the loaded
dataset (a sweep asks for the same one repeatedly), builds a fresh model for a
spec, and routes every target through the content-addressed cache via
`get_or_train`. E1 keeps its own `RunContext` in `e1_attack_baselines/context.py`
because its data path is CelebA-specific; everything else is the same shape.

The level knobs scale a run to its verification level: `architecture_for` swaps
in a tiny dense net at `test` level (a distinct architecture name, so its
checkpoint can never be served for a real one), and `pgd_iterations_for` shortens
the PGD chain at any level below `full`.
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

from common.io import run_output_dir
from common.models import ModelSpec, get_or_train, model_cache_root
from common.paths import repo_root
from scarab.datasets import ScarabDataset
from scarab.models import LinearNet
from scarab.utils import initialize_model, load_data

if TYPE_CHECKING:
    from collections.abc import Callable

    from common.config import LevelConfig

LOGGER = logging.getLogger("run_context")

# PGD's budget for adversarial training and evasion (the library default).
PGD_ITERATIONS = 40
# What `smoke` runs instead of the paper's 40. More iterations only tighten the
# perturbation, they reach no new code, so a reduced chain still exercises the
# path. PGD-7 is the standard Madry et al. configuration, a recognised attack.
SMOKE_PGD_ITERATIONS = 7
# A short PGD chain at `test` level: enough to visibly perturb, sub-second on CPU.
TINY_PGD_ITERATIONS = 7
# A larger budget at `test` level so PGD moves the wide-margin tiny model and the
# evasion degradation is a real check rather than a tie.
TINY_EPSILON = 0.3

# The real architecture each dataset trains, matching the pairing the paper ran.
# census and lfw are tabular (lfw's images are flattened by its loader), so a
# dense net. fmnist is a dense net too: `LinearNet` opens with `nn.Flatten`, and
# `load_fmnist` reports `num_features=784`, which is exactly its flattened 28x28
# size, so the dense path consumes it unchanged. That match is what identifies
# the intended model: `num_features` counts H*W and ignores channels, so it
# agrees with the flattened size only for single-channel data. cifar (1024 vs a
# flattened 3072) and celeba (4096 vs 12288) disagree, which is why those need a
# conv net, and cifar is the `vgg` case here. VGG is hard-wired to three input
# channels and so could never have taken fmnist; the paper's single "VGG"
# heading covers the conv datasets, not this one.
REAL_ARCH: dict[str, str] = {
    "census": "linearnet",
    "lfw": "linearnet",
    "fmnist": "linearnet",
    "cifar": "vgg",
}

# The name `scarab.utils.load_data` knows a dataset by, where that differs from
# the label the experiments use. The experiments say `cifar`, which is what the
# paper's tables and figures label the column and what the CSV `dataset` column
# and the model-spec cache keys carry; the library only answers to `cifar10`.
# Translating here, at the one call site, keeps the artifact's label stable.
LOADER_NAMES: dict[str, str] = {"cifar": "cifar10"}

# The tiny stand-in recorded at `test` level: a dense net, distinct from every
# real architecture name so its checkpoint can never collide with a paper one.
TINY_ARCH = "tiny_linearnet"
TINY_HIDDEN = [16, 16]

# The synthetic tabular stand-in for every real dataset at `test` level: a handful
# of well-separated rows in the [0, 1] box PGD clips to, with a binary label and
# two binary sensitive attributes so both tabular studies exercise the same data.
# Small enough to train a non-degenerate dense net in well under a second on CPU,
# large enough that accuracy is not pinned at chance.
TINY_NUM_FEATURES = 8
TINY_NUM_CLASSES = 2
TINY_TRAIN_SIZE = 64
TINY_TEST_SIZE = 32
TINY_EPOCHS = 8
TINY_BATCH = 16


def loader_name(dataset: str) -> str:
    """Return the name `load_data` knows `dataset` by.

    Args:
        dataset: The experiment's dataset label.

    Returns:
        The library's name for the same dataset, which is the label itself
        unless `LOADER_NAMES` overrides it.
    """
    return LOADER_NAMES.get(dataset, dataset)


def epochs_for(level: LevelConfig) -> int:
    """Return the training budget for one model at this level.

    Args:
        level: The level preset, already carrying `epochs` via `with_defaults`.

    Returns:
        `TINY_EPOCHS` at tiny level, else the level's epoch count (at least one).
    """
    if level.tiny_model:
        return TINY_EPOCHS
    return max(1, level.epochs if level.epochs is not None else 1)


def batch_for(level: LevelConfig, paper_batch: int) -> int:
    """Return the training batch size, shrunk at tiny level.

    A paper batch over the stand-in's 64 rows would be one or two steps an epoch,
    too few to learn; the tiny batch gives several. This is a spec field, so the
    tiny batch is in the tiny key and cannot collide with a paper checkpoint.

    Args:
        level: The level preset.
        paper_batch: The batch size the paper run uses.

    Returns:
        `TINY_BATCH` at tiny level, else `paper_batch`.
    """
    return TINY_BATCH if level.tiny_model else paper_batch


def pgd_iterations_for(level: LevelConfig) -> int:
    """Return how many PGD iterations both training and evasion take.

    Only `full` runs the paper's chain; see `SMOKE_PGD_ITERATIONS` for why. The
    count is written to every result row, so a reduced run is legible as one.

    Args:
        level: The level preset.

    Returns:
        The number of PGD steps per batch.
    """
    if level.tiny_model:
        return TINY_PGD_ITERATIONS
    if level.train_fraction < 1.0:
        return SMOKE_PGD_ITERATIONS
    return PGD_ITERATIONS


def epsilon_for(level: LevelConfig, epsilon: float) -> float:
    """Return the perturbation budget to actually apply at this level.

    The paper budgets barely move the wide-margin tiny model, so `test` level
    uses one larger budget for every requested epsilon, enough for the evasion
    degradation assertion to bite. Real levels use the requested budget.

    Args:
        level: The level preset.
        epsilon: The budget the sweep asked for.

    Returns:
        `TINY_EPSILON` at tiny level, else `epsilon`.
    """
    return TINY_EPSILON if level.tiny_model else epsilon


def architecture_for(level: LevelConfig, dataset: str) -> str:
    """Return the architecture name this level trains, and records in the spec.

    Args:
        level: The level preset.
        dataset: The dataset name, one of `REAL_ARCH`'s keys.

    Returns:
        The tiny stand-in's name at tiny level, else the dataset's real
        architecture. The name goes into the `ModelSpec`, which is what stops a
        tiny checkpoint from being loaded in place of a real one.

    Raises:
        KeyError: If `dataset` has no registered architecture.
    """
    if level.tiny_model:
        return TINY_ARCH
    if dataset not in REAL_ARCH:
        known = ", ".join(REAL_ARCH)
        raise KeyError(f"No architecture registered for {dataset!r}. Known: {known}.")
    return REAL_ARCH[dataset]


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
        num_features: Input feature count, for the dense architectures.
        num_classes: Number of output classes.
        device: Device to place the model on. Example: `"cuda:0"`.

    Returns:
        A freshly initialised model, as `common.models.get_or_train` requires.
    """
    if arch == TINY_ARCH:
        return LinearNet(
            num_features=num_features,
            num_classes=num_classes,
            hidden_layer_sizes=TINY_HIDDEN,
        ).to(device)
    return initialize_model(arch, capacity, num_features, num_classes, LOGGER).to(
        device
    )


def tiny_tabular_dataset(
    seed: int,
    num_features: int = TINY_NUM_FEATURES,
    num_classes: int = TINY_NUM_CLASSES,
) -> ScarabDataset:
    """Build the synthetic tabular stand-in used at `test` level.

    Real census/lfw/fmnist/cifar are downloads Level 1 must not require, so a
    handful of separable rows in the [0, 1] box stand in for all four. The label
    is learnable (each class sits in its own intensity band) so the evasion and
    extraction assertions are not vacuous, and two balanced binary sensitive
    columns are carried so attribute inference has something to predict.

    Args:
        seed: Seed for the generator, so two runs build identical data.
        num_features: Number of input features.
        num_classes: Number of label classes.

    Returns:
        A tabular dataset with `train_set`, `test_set` and the `x_*`/`y_*`/`z_*`
        arrays populated and index-aligned.
    """
    generator = np.random.default_rng(seed)

    def split(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        labels = np.arange(size) % num_classes
        features = np.clip(
            generator.random((size, num_features), dtype=np.float32) * 0.2
            + 0.7 * labels[:, None].astype(np.float32),
            0.0,
            1.0,
        )
        # Two balanced binary sensitive columns, decorrelated from the label
        # (which is `arange % 2`) so attribute inference is a real classification
        # problem rather than a trivial readout of the label.
        indices = np.arange(size)
        sensitive = np.stack([(indices // 2) % 2, (indices // 3) % 2], axis=1).astype(
            np.int64
        )
        return features, labels.astype(np.int64), sensitive

    x_train, y_train, z_train = split(TINY_TRAIN_SIZE)
    x_test, y_test, z_test = split(TINY_TEST_SIZE)

    return ScarabDataset(
        train_set=TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        test_set=TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)),
        num_features=num_features,
        num_classes=num_classes,
        modality="tabular",
        sensitive_columns=["attr_1", "attr_2"],
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        z_train=z_train,
        z_test=z_test,
    )


@dataclass
class RunContext:
    """What one cell of a multi-dataset sweep needs to run.

    Attributes:
        level: The verification-level preset, already carrying `epochs`.
        seed: The experiment seed, recorded as `exp_id`.
        device: Device to train and evaluate on. Example: `"cuda:0"`.
        cache_dir: Directory for the content-addressed checkpoint cache, from
            `default_cache_dir(level)`. Required rather than defaulted, so a
            context can never silently write a level's checkpoints into another
            level's cache.
        tiny_data_factory: Optional builder for the `test`-level stand-in, taking
            the seed and returning a `ScarabDataset`. None uses the shared
            `tiny_tabular_dataset` (E2/E3). E4 overrides it with a variant
            carrying genuine outliers, because kNN-Shapley outlier removal has
            nothing to score on the perfectly separable default. Ignored at
            non-tiny levels.
    """

    level: LevelConfig
    seed: int
    device: str
    cache_dir: Path
    tiny_data_factory: Callable[[int], ScarabDataset] | None = None
    _datasets: dict[tuple[str, float, float], ScarabDataset] = field(
        default_factory=dict
    )

    def data(self, dataset: str) -> ScarabDataset:
        """Load a dataset once per distinct request, memoised for this context.

        At tiny level the synthetic tabular stand-in replaces every dataset, so
        Level 1 needs no download.

        Args:
            dataset: The dataset name.

        Returns:
            The dataset. Callers must treat it as read-only: the memo hands the
            same object to every attack that asks for it.
        """
        key = (dataset, self.level.train_fraction, self.level.test_fraction)
        if key not in self._datasets:
            if self.level.tiny_model:
                factory = self.tiny_data_factory or tiny_tabular_dataset
                self._datasets[key] = factory(self.seed)
            else:
                self._datasets[key] = load_data(
                    repo_root(),
                    loader_name(dataset),
                    self.level.train_fraction,
                    LOGGER,
                    exp_id=self.seed,
                    test_size=self.level.test_fraction,
                )
        return self._datasets[key]

    def model_factory(
        self, spec: ModelSpec, num_features: int, num_classes: int
    ) -> Callable[[], nn.Module]:
        """Return the zero-argument initialiser `get_or_train` expects."""

        def initialise() -> nn.Module:
            return build_model(
                spec.arch, spec.capacity, num_features, num_classes, self.device
            )

        return initialise

    def get_or_train(
        self,
        spec: ModelSpec,
        num_features: int,
        num_classes: int,
        train_fn: Callable[[nn.Module], nn.Module],
    ) -> nn.Module:
        """Load a model from the shared cache, or train it with `train_fn` and cache it.

        The one path every multi-dataset target goes through. The checkpoint's
        location is the content hash of `spec`, so a target two experiments
        describe identically is trained once and reused.

        Args:
            spec: The model's spec; determines the cache key.
            num_features: Input feature count, for the initialiser.
            num_classes: Number of output classes.
            train_fn: Callable taking the fresh model and returning it trained.

        Returns:
            The loaded or newly trained model, on this run's device.
        """
        return get_or_train(
            spec,
            self.model_factory(spec, num_features, num_classes),
            train_fn,
            LOGGER,
            cache_dir=self.cache_dir,
        ).to(self.device)


def default_cache_dir(level: LevelConfig) -> Path:
    """Return the checkpoint cache this level should write to.

    Returns:
        This level's own subdirectory of `.model_cache/`, or a throwaway
        directory for `test`, whose tiny stand-ins have no business outliving
        the test that made them.
    """
    if level.tiny_model:
        return Path(tempfile.mkdtemp(prefix="multidataset_test_models_"))
    return model_cache_root(level.name)


def default_output_dir(level: LevelConfig, experiment_id: str) -> Path:
    """Return the directory this level's result CSV belongs in.

    Every level writes under its own `runs/<level>/` subtree, so a cheap `test`
    or `smoke` run never overwrites the `full` results a paper comparison reads
    from, nor has its reduced-budget numbers averaged into them. No result data
    ships with the repository; every number comes from a run that lands here.
    E2/E3/E4 each write a single `<experiment_id>.csv` directly into this
    directory, so a `make_*` renderer reads a reviewer's `runs/full/` the same
    way whatever level produced it.

    Args:
        level: The level preset.
        experiment_id: The experiment's registry ID, unused for the path but kept
            so callers read intent at the call site.

    Returns:
        An existing or creatable directory. The CSV name is the caller's.
    """
    return run_output_dir(level.name)
