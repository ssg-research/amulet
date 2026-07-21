"""Shared machinery for the two adversarial-training interaction studies.

E2 (`e2_advtr_modext`, Adversarial Training x Unauthorized Model Ownership) and
E3 (`e3_advtr_attrinf`, Adversarial Training x Attribute Inference) study the
*same* defense against two different risks, so everything the defense touches
lives here: the 50/50 adversary split, the clean and adversarially-trained model
specs, the PGD training and evasion helpers, and the per-modality architecture
choice. Each experiment's `run.py` supplies only the risk-specific attack and
its own CSV schema.

Three facts anchor the design, each confirmed against the committed CSVs the old
scripts left behind (plan S3-S5):

* **The defended model is the adversarially-trained one.** The old
  `advtr_modelext.py:189` did `defended_model = target_model`, silently throwing
  the robust model away and measuring the plain target as if defended. We do not
  reproduce that bug: `clean_target_spec` and `defended_target_spec` differ in
  their optimizer recipe, so they are different checkpoints, and the run wires
  the defended model (never the clean one) into every "defended" measurement.
* **The clean baseline is epsilon-independent.** `clean_target_spec` carries no
  epsilon, so one $\\modelstd$ is trained per (dataset, seed) and reused across
  every budget. `defended_target_spec` encodes epsilon in its recipe, so each
  budget is its own $\\modeldef$ checkpoint.
* **The datasets span two modalities.** census and lfw are tabular (they carry
  the NumPy feature and sensitive-attribute arrays attribute inference reads);
  fmnist and cifar are image `VisionDataset`s with no NumPy views, so E2 splits
  the adversary's half at the dataset level rather than by array index.

Levels (plan S8) collapse to one substitution: at `test` level a tiny dense net
over a handful of synthetic tabular rows stands in for every real
(architecture, dataset) pair, recorded in the spec as a distinct architecture so
a tiny checkpoint can never be loaded in place of a real one.
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
from torch.utils.data import DataLoader, TensorDataset, random_split

from amulet.datasets import AmuletDataset
from amulet.evasion.attacks import EvasionPGD
from amulet.evasion.defenses import AdversarialTrainingPGD
from amulet.models import LinearNet
from amulet.utils import get_accuracy, initialize_model, load_data
from common.io import results_root
from common.models import ModelSpec
from common.paths import repo_root

# Reused verbatim from E1 rather than re-ported (plan P3 brief): the seeded
# adversary split over NumPy arrays, the Adam training recipe, the unshuffled
# loader, and the all-generators seeding the old scripts lacked.
from experiments.e1_attack_baselines.shared import (
    AdversarySplit,
    adversary_split,
    loader_for,
    seed_everything,
    train_with_adam,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from common.config import LevelConfig

__all__ = [
    "ADAM_RECIPE",
    "ADVERSARY_FRACTION",
    "AdversarySplit",
    "RunContext",
    "adversarially_train",
    "adversary_split",
    "architecture_for",
    "batch_for",
    "clean_target_spec",
    "dataset_adversary_split",
    "default_cache_dir",
    "default_output_dir",
    "defended_recipe",
    "defended_target_spec",
    "epochs_for",
    "loader_for",
    "pgd_iterations_for",
    "robust_accuracy",
    "seed_everything",
    "step_size_for",
    "stolen_model_spec",
    "stolen_recipe",
    "tiny_tabular_dataset",
    "train_clean",
    "train_with_adam",
]

LOGGER = logging.getLogger("advtr_interactions")

# Half the training split is reserved for the adversary in both studies, the
# `--adv_train_fraction 0.5` the old scripts defaulted to.
ADVERSARY_FRACTION = 0.5

# PGD's budget for both adversarial training and evasion, the library defaults
# the old scripts never overrode. The step size is a quarter of the perturbation
# budget, as `advtr_modelext.py`/`advtr_attrinf.py` set it (`step_size = eps/4`).
PGD_ITERATIONS = 40

# The victim optimizer both studies train with: Adam at 1e-3, the old default.
ADAM_RECIPE = "adam_lr1e-3"

# The two halves of the training split, named for the spec's `subset_selector`.
# The split *method* is part of the name because E2 and E3 divide the training
# data differently: E2 splits at the dataset level (its image datasets carry no
# NumPy arrays to index), E3 splits the NumPy arrays by index (attribute
# inference reads them). Encoding the method keeps a census target trained one
# way from ever loading a census target trained the other way from the cache.
DATASET_SPLIT_TARGET = f"advtr_dsplit_target_{1 - ADVERSARY_FRACTION:g}_seeded"
DATASET_SPLIT_ADVERSARY = f"advtr_dsplit_adversary_{ADVERSARY_FRACTION:g}_seeded"
ARRAY_SPLIT_TARGET = f"advtr_npsplit_target_{1 - ADVERSARY_FRACTION:g}_seeded"

# The real architecture each dataset trains, chosen for its modality. census and
# lfw are tabular (lfw's images are flattened by its loader), so a dense net;
# fmnist is single-channel 28x28, which the BadNets-style `cnn` takes and `vgg`
# (hard-wired to three input channels) cannot; cifar is three-channel 32x32, the
# `vgg` case. The paper reports these under one "VGG" heading, but VGG cannot run
# on the tabular or single-channel inputs, so the per-modality choice is what the
# library actually admits.
REAL_ARCH: dict[str, str] = {
    "census": "linearnet",
    "lfw": "linearnet",
    "fmnist": "cnn",
    "cifar": "vgg",
}

# The tiny stand-in recorded at `test` level: a dense net, distinct from every
# real architecture name so its checkpoint can never collide with a paper one.
TINY_ARCH = "tiny_linearnet"
TINY_HIDDEN = [16, 16]

# The synthetic tabular stand-in for every real dataset at `test` level: a
# handful of well-separated rows in the [0, 1] box PGD clips to, with a binary
# label and two binary sensitive attributes so both studies exercise the same
# data. Small enough to train a non-degenerate dense net in well under a second
# on CPU, large enough that accuracy is not pinned at chance.
TINY_NUM_FEATURES = 8
TINY_NUM_CLASSES = 2
TINY_TRAIN_SIZE = 64
TINY_TEST_SIZE = 32
TINY_EPOCHS = 8
TINY_BATCH = 16
# A short PGD chain at `test` level: enough iterations for the perturbed loader
# to visibly differ from the clean one, few enough to stay sub-second on CPU.
TINY_PGD_ITERATIONS = 7
# A larger budget at `test` level so PGD moves the wide-margin tiny model and the
# evasion degradation is a real check rather than a tie.
TINY_EPSILON = 0.3


def defended_recipe(epsilon: float) -> str:
    """Return the optimizer-recipe string for an adversarially-trained model.

    The budget is baked into the string, which is the cache's contract: two
    $\\modeldef$ specs that differ only in epsilon get different keys and cannot
    share a checkpoint (plan S6). Distinct from `ADAM_RECIPE`, so a $\\modeldef$
    can never be loaded where a clean $\\modelstd$ is wanted, or the reverse.

    Args:
        epsilon: The PGD perturbation budget.

    Returns:
        A short stable recipe name encoding the budget.
    """
    return f"advtr_pgd_eps{epsilon:g}_adam_lr1e-3"


def stolen_recipe(epsilon: float) -> str:
    """Return the recipe string for a surrogate distilled from a $\\modeldef$.

    A distilled model's weights depend on the model it was distilled from, which
    is not a `ModelSpec` field, so the source's budget is carried in the recipe.
    A surrogate stolen from a differently-defended target therefore cannot reuse
    this checkpoint.

    Args:
        epsilon: The budget of the defended target the surrogate imitates.

    Returns:
        A short stable recipe name encoding the distillation source.
    """
    return f"modext_mse_from_advtr_eps{epsilon:g}_adam_lr1e-3"


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
    """Return how many PGD iterations both training and evasion take."""
    return TINY_PGD_ITERATIONS if level.tiny_model else PGD_ITERATIONS


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


def step_size_for(epsilon: float) -> float:
    """Return PGD's per-iteration step size for a budget: a quarter of it.

    This is the `step_size = args.epsilon / 4` both old scripts set.
    """
    return epsilon / 4


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
    """Assemble a spec, filling the fields every advtr model shares."""
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
    subset_selector: str,
) -> ModelSpec:
    """Describe the clean baseline $\\modelstd$, trained with Adam on the target half.

    Carries no epsilon, so one clean target serves every budget in the sweep.
    `subset_selector` names *how* the target half was chosen, so E2's and E3's
    differently-split census targets never collide on one key.
    """
    return _spec(
        level,
        dataset,
        seed,
        capacity,
        num_features,
        num_classes,
        subset_selector=subset_selector,
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
    subset_selector: str,
) -> ModelSpec:
    """Describe the adversarially-trained $\\modeldef$ at one budget.

    Differs from `clean_target_spec` only in its optimizer recipe, which encodes
    the budget: the defended model is a different checkpoint from the clean one
    (that is the bug we refuse to reproduce), and each budget is its own.
    """
    return _spec(
        level,
        dataset,
        seed,
        capacity,
        num_features,
        num_classes,
        subset_selector=subset_selector,
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
    subset_selector: str = DATASET_SPLIT_ADVERSARY,
) -> ModelSpec:
    """Describe the surrogate $\\modelstol$ distilled from $\\modeldef$ (E2 only)."""
    return _spec(
        level,
        dataset,
        seed,
        capacity,
        num_features,
        num_classes,
        subset_selector=subset_selector,
        optimizer_recipe=stolen_recipe(epsilon),
        batch_size=batch_size,
    )


def train_clean(
    model: nn.Module, loader: DataLoader, device: str, epochs: int
) -> nn.Module:
    """Train the clean baseline with Adam, reusing E1's recipe. Named by `ADAM_RECIPE`."""
    return train_with_adam(model, loader, device, epochs)


def adversarially_train(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    epochs: int,
    epsilon: float,
    iterations: int,
) -> nn.Module:
    """Adversarially train a model with PGD. Named by `defended_recipe`.

    Args:
        model: The freshly initialised model.
        loader: Training data.
        device: Device to train on.
        epochs: Number of passes over the data.
        epsilon: The perturbation budget.
        iterations: PGD iterations per batch.

    Returns:
        The adversarially-trained model.
    """
    defense = AdversarialTrainingPGD(
        model,
        nn.CrossEntropyLoss(reduction="mean"),
        torch.optim.Adam(model.parameters(), lr=1e-3),
        loader,
        device,
        epochs,
        epsilon,
        iterations=iterations,
        step_size=step_size_for(epsilon),
    )
    return defense.train_robust()


def robust_accuracy(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    batch_size: int,
    epsilon: float,
    iterations: int,
) -> float:
    """Return a model's accuracy on PGD-perturbed test inputs at one budget.

    Args:
        model: The model to attack.
        test_loader: Clean test data to perturb.
        device: Device to run on.
        batch_size: Batch size for the perturbed loader.
        epsilon: The perturbation budget.
        iterations: PGD iterations.

    Returns:
        Robust accuracy as a percentage.
    """
    evasion = EvasionPGD(
        model,
        test_loader,
        device,
        batch_size,
        epsilon,
        iterations=iterations,
        step_size=step_size_for(epsilon),
    )
    return get_accuracy(model, evasion.attack(), device)


def dataset_adversary_split(
    train_set: object, seed: int, adv_fraction: float = ADVERSARY_FRACTION
) -> tuple[object, object]:
    """Split a training set into the target's half and the adversary's half.

    Works for any map-style dataset, including the `VisionDataset`s cifar and
    fmnist load as (which carry no NumPy arrays to index), so E2 splits every
    dataset the same way. This is the old `advtr_modelext.py` split: a seeded
    `random_split`, target half first.

    Args:
        train_set: The training set to split.
        seed: Seed for the split generator, so the same seed yields the same halves.
        adv_fraction: Fraction reserved for the adversary.

    Returns:
        `(target_set, adversary_set)`.
    """
    total = len(train_set)  # type: ignore[reportArgumentType]
    adversary_size = int(adv_fraction * total)
    target_size = total - adversary_size
    generator = torch.Generator().manual_seed(seed)
    target_set, adversary_set = random_split(
        train_set,  # type: ignore[reportArgumentType]
        [target_size, adversary_size],
        generator=generator,
    )
    return target_set, adversary_set


def tiny_tabular_dataset(
    seed: int,
    num_features: int = TINY_NUM_FEATURES,
    num_classes: int = TINY_NUM_CLASSES,
) -> AmuletDataset:
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

    return AmuletDataset(
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
    """What one cell of an adversarial-training sweep needs to run.

    Attributes:
        level: The verification-level preset, already carrying `epochs`.
        seed: The experiment seed, recorded as `exp_id`.
        device: Device to train and evaluate on. Example: `"cuda:0"`.
        cache_dir: Directory for the content-addressed checkpoint cache. None
            uses the shared default.
        tiny_data_factory: Optional builder for the `test`-level stand-in,
            taking the seed and returning an `AmuletDataset`. None uses the
            shared `tiny_tabular_dataset` (E2/E3). E4 overrides it with a
            variant carrying genuine outliers, because kNN-Shapley outlier
            removal has nothing to score on the perfectly separable default.
            Ignored at non-tiny levels.
    """

    level: LevelConfig
    seed: int
    device: str
    cache_dir: Path | None = None
    tiny_data_factory: Callable[[int], AmuletDataset] | None = None
    _datasets: dict[tuple[str, float], AmuletDataset] = field(default_factory=dict)

    def data(self, dataset: str) -> AmuletDataset:
        """Load a dataset once per distinct request, memoised for this context.

        At tiny level the synthetic tabular stand-in replaces every dataset, so
        Level 1 needs no download.

        Args:
            dataset: The dataset name.

        Returns:
            The dataset. Callers must treat it as read-only: the memo hands the
            same object to every attack that asks for it.
        """
        key = (dataset, self.level.train_fraction)
        if key not in self._datasets:
            if self.level.tiny_model:
                factory = self.tiny_data_factory or tiny_tabular_dataset
                self._datasets[key] = factory(self.seed)
            else:
                self._datasets[key] = load_data(
                    repo_root(),
                    dataset,
                    self.level.train_fraction,
                    LOGGER,
                    exp_id=self.seed,
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

        The one path every advtr model goes through. The checkpoint's location is
        the content hash of `spec`, so a clean baseline two budgets describe
        identically is trained once and reused (plan S6).

        Args:
            spec: The model's spec; determines the cache key.
            num_features: Input feature count, for the initialiser.
            num_classes: Number of output classes.
            train_fn: Callable taking the fresh model and returning it trained.

        Returns:
            The loaded or newly trained model, on this run's device.
        """
        from common.models import get_or_train

        return get_or_train(
            spec,
            self.model_factory(spec, num_features, num_classes),
            train_fn,
            LOGGER,
            cache_dir=self.cache_dir,
        ).to(self.device)


def default_cache_dir(level: LevelConfig) -> Path | None:
    """Return the checkpoint cache this level should write to.

    Returns:
        None for the shared default, or a throwaway directory for `test`, whose
        tiny stand-ins have no business outliving the test that made them.
    """
    if level.tiny_model:
        return Path(tempfile.mkdtemp(prefix="advtr_test_models_"))
    return None


def default_output_dir(level: LevelConfig, experiment_id: str) -> Path:
    """Return the directory this level's result CSV belongs in.

    Only `full` writes the committed results directory. A `smoke` run trains for
    one epoch on a fraction of the data and a `test` run measures a tiny dense
    net; either averaged into the paper's table would corrupt it, so each level
    keeps its own file.

    Args:
        level: The level preset.
        experiment_id: The experiment's registry ID, unused for the path but
            kept so callers read intent at the call site.

    Returns:
        An existing or creatable directory. The CSV name is the caller's.
    """
    if level.tiny_model:
        return Path(tempfile.mkdtemp(prefix=f"{experiment_id}_test_results_"))
    root = results_root()
    return root if level.name == "full" else root / level.name
