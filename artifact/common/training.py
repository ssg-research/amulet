"""Generic training utilities for the from-scratch experiments (E1-E4).

Seeding, the Adam and momentum-SGD classifier recipes, the unshuffled loader, the
two seeded adversary splits, PGD adversarial training and its robust accuracy. A
caller supplies the model, data and hyperparameters; which model to train and at
what settings is each experiment's `train_targets` module's job, not this one's.

Changing how a recipe here trains a model changes its weights, so a caller that
records a recipe string (e.g. `"adam_lr1e-3"`) in a `ModelSpec` must change that
string too, or the cache will serve stale checkpoints. E5's LoRA `HFCausalLM`
does not fit `train_classifier` and has its own layer in `llm_backdoor_common`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, random_split

from amulet.evasion.attacks import EvasionPGD
from amulet.evasion.defenses import AdversarialTrainingPGD
from amulet.utils import get_accuracy, train_classifier

if TYPE_CHECKING:
    from torch.optim.lr_scheduler import _LRScheduler

    from amulet.datasets import AmuletDataset


def seed_everything(seed: int) -> None:
    """Seed Python's `random`, NumPy and torch so a fixed seed reproduces a run.

    All three matter: an experiment draws its adversary split, poisoned indices
    and membership mask from different generators, so seeding torch alone would
    leave those varying between runs.

    Args:
        seed: The experiment seed, recorded as `exp_id`.
    """
    random.seed(seed)
    np.random.seed(seed)
    _ = torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader_for(dataset: object, batch_size: int) -> DataLoader:
    """Wrap a dataset in the unshuffled loader the from-scratch experiments use.

    Args:
        dataset: Any map-style dataset yielding `(x, y)`.
        batch_size: Batch size.

    Returns:
        A `DataLoader` in dataset order, so an index-based split stays aligned
        with the rows an attack scores.
    """
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)  # type: ignore[reportArgumentType]


def train_with_adam(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    epochs: int,
    learning_rate: float = 1e-3,
) -> nn.Module:
    """Train a classifier with Adam, the paper's default optimizer.

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

    Args:
        model: The freshly initialised model.
        loader: Training data.
        device: Device to train on.
        epochs: Number of passes over the data.
        learning_rate: SGD's initial learning rate.
        step_size: Epochs between learning-rate decays.
        gamma: Multiplicative decay applied at each step.
        nesterov: Whether to use Nesterov momentum.
        schedule: Whether to apply the decay. False trains at a flat learning
            rate (the backdoored poisoning target uses this).

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
    data: AmuletDataset, seed: int, adv_fraction: float = 0.5
) -> AdversarySplit:
    """Split the training data by NumPy array index into target and adversary halves.

    Both halves come from the same seeded index split, so they are disjoint and a
    fixed seed reproduces them. This is the by-index variant the tabular attacks
    need (they read the feature arrays); `dataset_adversary_split` is the
    map-style variant for image datasets with no NumPy views.

    Args:
        data: The loaded dataset; must carry its NumPy views.
        seed: Seed for the split, so the same seed yields the same halves.
        adv_fraction: Fraction of the training split reserved for the adversary.

    Returns:
        The split, carrying the target's training set and the adversary's half in
        both the dataset and NumPy forms the two attacks consume.

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


def dataset_adversary_split(
    train_set: object, seed: int, adv_fraction: float = 0.5
) -> tuple[object, object]:
    """Split a map-style training set into (target, adversary) halves, target first.

    A seeded `random_split`, so it works when the dataset carries no NumPy views
    (cifar and fmnist load as `VisionDataset`s); `adversary_split` is the by-index
    variant for the tabular attacks.

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


def step_size_for(epsilon: float) -> float:
    """Return PGD's per-iteration step size for a budget: a quarter of it."""
    return epsilon / 4


def adversarially_train(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    epochs: int,
    epsilon: float,
    iterations: int,
) -> nn.Module:
    """Adversarially train a model with PGD.

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
