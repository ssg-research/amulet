"""The target models E4 trains, and the specs that identify them.

For one (dataset, seed, percent) cell E4 trains:

* a clean baseline (Adam on the dataset-level target half);
* a defended model, obtained by kNN-Shapley outlier removal on the target's
  training data followed by a retrain. At `percent == 0` no outliers are removed,
  so the defended model *is* the clean baseline;
* a surrogate, distilled from the defended model.

Every spec is built here from E4's own strings. E4's clean baseline uses the same
dataset-level split selector and Adam recipe that E2 uses, so on a matching
(dataset, seed) the two land on one cached checkpoint. That is a coincidence of
two independent definitions, not a shared builder: change E2's and E4's does not
follow, and vice versa. The removal percentage and the distillation source are
baked into the recipe strings (`outrem_recipe`, `stolen_recipe`), so each percent
is a distinct checkpoint disjoint from the baseline and from an E2 defended model
(which encodes an epsilon). The generic tooling comes from `common`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from amulet.datasets import AmuletDataset
from common.models import ModelSpec
from common.run_context import RunContext, architecture_for, batch_for, epochs_for
from common.training import dataset_adversary_split, loader_for
from common.training import train_with_adam as train_clean
from experiments.e4_outrem_modext.schemas import CAPACITY

# The paper trains the E4 targets for 100 epochs and retrains for 100 more after
# removal; `full` defers to this.
PAPER_EPOCHS = 100

# The paper's batch size.
BATCH_SIZE = 256

# Half the training split is reserved for the adversary. Recorded in the CSV's
# `adv_train_fraction`.
ADVERSARY_FRACTION = 0.5

# The optimizer recipe of the clean baseline: Adam at 1e-3.
ADAM_RECIPE = "adam_lr1e-3"

# The `subset_selector` strings naming E4's dataset-level split (the same split
# E2 uses). These strings feed the ModelSpec content hash; the `advtr_` prefix is
# an opaque identifier, not a description.
DATASET_SPLIT_TARGET = f"advtr_dsplit_target_{1 - ADVERSARY_FRACTION:g}_seeded"
DATASET_SPLIT_ADVERSARY = f"advtr_dsplit_adversary_{ADVERSARY_FRACTION:g}_seeded"

# The `test`-level synthetic stand-in for census/lfw/fmnist/cifar. It is E4's own
# rather than the shared `tiny_tabular_dataset`, because kNN-Shapley outlier
# removal has nothing to remove from the perfectly separable shared stand-in: its
# influence scores come out constant, the score normalisation divides by zero,
# and the "cleaned" set is empty. A fraction of the training labels are therefore
# flipped so genuine mislabelled outliers exist to score and remove, while the
# test set stays clean as the held-out reference the scores are computed against.
TINY_TRAIN_SIZE = 80
TINY_TEST_SIZE = 40
TINY_NUM_FEATURES = 8
TINY_NUM_CLASSES = 2
TINY_OUTLIER_FRACTION = 0.15

if TYPE_CHECKING:
    from common.config import LevelConfig


def tiny_outrem_dataset(
    seed: int,
    num_features: int = TINY_NUM_FEATURES,
    num_classes: int = TINY_NUM_CLASSES,
) -> AmuletDataset:
    """Build E4's `test`-level stand-in: separable tabular data with outliers.

    Each record sits in its class's own intensity band so the label is learnable,
    but a seeded fraction of the *training* labels are flipped, planting genuine
    mislabelled outliers for kNN-Shapley to find and drop. The test set is left
    clean, since it is the held-out reference the influence scores score against.

    Args:
        seed: Seed for the generator, so two runs build identical data.
        num_features: Number of input features.
        num_classes: Number of label classes.

    Returns:
        A tabular dataset with `train_set`/`test_set` and the `x_*`/`y_*`/`z_*`
        arrays populated and index-aligned. Training labels carry the outliers.
    """
    generator = np.random.default_rng(seed)

    def split(size: int, corrupt: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        labels = np.arange(size) % num_classes
        # Features encode the *true* class; flipping the label afterwards is what
        # makes a record an outlier (its features disagree with its label).
        features = np.clip(
            generator.random((size, num_features), dtype=np.float32) * 0.2
            + 0.7 * labels[:, None].astype(np.float32),
            0.0,
            1.0,
        )
        labels = labels.astype(np.int64)
        if corrupt:
            labels = labels.copy()
            count = max(1, int(TINY_OUTLIER_FRACTION * size))
            outliers = generator.choice(size, size=count, replace=False)
            labels[outliers] = (num_classes - 1) - labels[outliers]
        indices = np.arange(size)
        sensitive = np.stack([(indices // 2) % 2, (indices // 3) % 2], axis=1).astype(
            np.int64
        )
        return features, labels, sensitive

    x_train, y_train, z_train = split(TINY_TRAIN_SIZE, corrupt=True)
    x_test, y_test, z_test = split(TINY_TEST_SIZE, corrupt=False)

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


def outrem_recipe(percent: int) -> str:
    """Return the optimizer-recipe string for an outlier-removed defended model.

    The removal percentage is baked into the string, which is the cache's
    contract: two defended specs that differ only in the percentage get
    different keys, and none can collide with the clean baseline (Adam alone) or
    with an E2 defended model (which encodes an epsilon). A `percent == 0` model
    is not built through this recipe at all; it is the clean baseline.
    """
    return f"outrem_knn_shapley_p{percent}_adam_lr1e-3"


def stolen_recipe(percent: int) -> str:
    """Return the recipe string for a surrogate distilled from that defended model.

    A distilled model's weights depend on the model it was distilled from, which
    is not a `ModelSpec` field, so the source's removal percentage is carried in
    the recipe. A surrogate stolen from a model at a different removal percentage
    cannot reuse this key, and neither can an E2 surrogate (whose recipe names an
    epsilon).
    """
    return f"modext_mse_from_outrem_p{percent}_adam_lr1e-3"


@dataclass(frozen=True)
class ModelBundle:
    """The models one E4 cell trains, plus the specs that keyed them.

    Exposed as a seam so a test can confirm the baseline and the distinct
    outlier-removed checkpoints.

    Attributes:
        clean: The clean baseline.
        defended: The outlier-removed defended model; the same object as `clean` at
            `percent == 0`, a distinct retrain otherwise.
        stolen: The surrogate distilled from `defended`.
        clean_spec: The spec that keyed `clean`.
        defended_spec: The spec that keyed `defended`.
        stolen_spec: The spec that keyed `stolen`.
    """

    clean: nn.Module
    defended: nn.Module
    stolen: nn.Module
    clean_spec: ModelSpec
    defended_spec: ModelSpec
    stolen_spec: ModelSpec


def clean_baseline_spec(
    ctx: RunContext,
    dataset: str,
    num_features: int,
    num_classes: int,
    batch_size: int,
    capacity: str = CAPACITY,
) -> ModelSpec:
    """Describe E4's clean baseline: Adam on the dataset-level target half.

    Args:
        ctx: The run context (level, seed).
        dataset: The dataset name.
        num_features: Input feature count for the dense architectures.
        num_classes: Number of output classes.
        batch_size: Training batch size (paper batch, or the tiny batch at `test`).
        capacity: The capacity tier.

    Returns:
        The clean-baseline spec.
    """
    level: LevelConfig = ctx.level
    return ModelSpec(
        dataset=dataset,
        arch=architecture_for(level, dataset),
        capacity=capacity,
        num_features=num_features,
        num_classes=num_classes,
        seed=ctx.seed,
        train_fraction=level.train_fraction,
        subset_selector=DATASET_SPLIT_TARGET,
        label_attribute="default",
        optimizer_recipe=ADAM_RECIPE,
        epochs=epochs_for(level),
        batch_size=batch_size,
    )


def retrain_after_outlier_removal(
    clean: nn.Module,
    target_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: str,
    epochs: int,
    batch_size: int,
    percent: int,
) -> nn.Module:
    """Purify the target's training data via kNN-Shapley and retrain.

    The *trained* clean target seeds the defense (its penultimate features drive
    the kNN-Shapley scores), the
    lowest-influence `percent`% of records are dropped, and the model is
    retrained on the remainder. The clean target is deep-copied first, so the
    shared clean baseline the baseline row measures is never mutated.

    Args:
        clean: The trained clean baseline to purify around. Not modified.
        target_loader: The target half's training data (unshuffled), scored for outliers.
        test_loader: The held-out test data the Shapley scores are computed against.
        device: Device to train and score on.
        epochs: Retraining epochs.
        batch_size: Retraining batch size.
        percent: Percentage of lowest-influence records to remove.

    Returns:
        The retrained defended model.
    """
    from amulet.poisoning.defenses import OutlierRemoval

    starting = copy.deepcopy(clean)
    defense = OutlierRemoval(
        starting,
        nn.CrossEntropyLoss(),
        torch.optim.Adam(starting.parameters(), lr=1e-3),
        target_loader,
        test_loader,
        device,
        percent=percent,
        epochs=epochs,
        batch_size=batch_size,
    )
    return defense.train_robust()


def build_models(
    ctx: RunContext, dataset: str, percent: int, capacity: str = CAPACITY
) -> tuple[ModelBundle, AmuletDataset]:
    """Train (or load) the clean, defended and stolen models for one cell.

    Args:
        ctx: The run context (level, seed, device, cache directory).
        dataset: The dataset name.
        percent: The removal percentage (the swept value, which keys the row).
        capacity: The capacity tier.

    Returns:
        The models with their specs, and the loaded dataset.
    """
    from amulet.unauth_model_ownership.attacks import ModelExtraction

    data = ctx.data(dataset)
    num_features, num_classes = data.num_features, data.num_classes
    batch_size = batch_for(ctx.level, BATCH_SIZE)
    epochs = epochs_for(ctx.level)

    target_set, adversary_set = dataset_adversary_split(data.train_set, ctx.seed)
    target_loader = loader_for(target_set, batch_size)
    adversary_loader = loader_for(adversary_set, batch_size)
    test_loader = loader_for(data.test_set, batch_size)

    from common import progress

    clean_spec = clean_baseline_spec(
        ctx, dataset, num_features, num_classes, batch_size, capacity
    )
    progress.log(f"    {dataset} {percent}%: clean baseline")
    clean = ctx.get_or_train(
        clean_spec,
        num_features,
        num_classes,
        lambda model: train_clean(model, target_loader, ctx.device, clean_spec.epochs),
    )

    if percent == 0:
        # No outliers removed: the defended model *is* the clean baseline. This is
        # the clean-baseline column of the table / the leftmost point of the figures.
        defended, defended_spec = clean, clean_spec
    else:
        defended_spec = clean_spec.replace(optimizer_recipe=outrem_recipe(percent))
        progress.log(f"    {dataset} {percent}%: kNN-Shapley outlier removal + retrain")
        defended = ctx.get_or_train(
            defended_spec,
            num_features,
            num_classes,
            lambda _model: retrain_after_outlier_removal(
                clean,
                target_loader,
                test_loader,
                ctx.device,
                epochs,
                batch_size,
                percent,
            ),
        )

    progress.log(f"    {dataset} {percent}%: surrogate (distillation)")
    stolen_spec = clean_spec.replace(
        optimizer_recipe=stolen_recipe(percent),
        subset_selector=DATASET_SPLIT_ADVERSARY,
    )

    def distil(model: nn.Module) -> nn.Module:
        # The surrogate is distilled from the DEFENDED model (which is the clean
        # baseline at percent 0), querying it with the adversary's held-out half.
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
        ModelBundle(clean, defended, stolen, clean_spec, defended_spec, stolen_spec),
        data,
    )
