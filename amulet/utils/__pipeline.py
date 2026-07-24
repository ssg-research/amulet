"""
Utilities to help build an ML pipeline.
"""

import logging
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset

from ..datasets import (
    AmuletDataset,
    load_celeba,
    load_census,
    load_cifar10,
    load_cifar100,
    load_fmnist,
    load_lfw,
    load_mnist,
    load_utkface,
)
from ..models import VGG, AmuletModel, LinearNet, ResNet, SimpleCNN


def _subset_indices(count: int, fraction: float, seed: int, label: str) -> np.ndarray:
    """Choose which records of a split to keep, as one shared index array.

    Every view of a split (the `Dataset` and the NumPy arrays beside it) is
    indexed with the array returned here, which is what keeps them aligned:
    subsampling them through independent generators leaves `x_train[i]` and
    `train_set[i]` describing different records.

    Args:
        count: Number of records in the split.
        fraction: Proportion of the split to keep.
        seed: Random seed, so a given seed always selects the same records.
        label: Name of the parameter being applied, for the error message.

    Returns:
        Indices of the kept records, in the order the subsample drew them.

    Raises:
        ValueError: If `fraction` is not in (0, 1] or would keep no records.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"{label} must lie in (0, 1]; got {fraction}.")
    if int(fraction * count) < 1:
        raise ValueError(
            f"{label}={fraction} keeps no records of a {count}-record split. "
            f"Use at least {1 / count:.3g}."
        )
    keep, _ = train_test_split(np.arange(count), train_size=fraction, random_state=seed)
    return cast(np.ndarray, keep)


def load_data(
    root: Path | str,
    dataset: str,
    training_size: float = 1.0,
    log: logging.Logger | None = None,
    exp_id: int = 0,
    celeba_target: str = "Smiling",
    test_size: float = 1.0,
) -> AmuletDataset:
    """
    Load data given the dataset name and training size.

    `training_size` and `test_size` are independent and shrink only the split
    they name. Both matter for cost: an algorithm that walks the test split
    (kNN-Shapley outlier removal, for one) is unaffected by `training_size`
    alone, so a cheap run has to reduce both.

    Args:
        root: Root directory of the pipeline.
        dataset: Name of the dataset. Options: "cifar10", "cifar100", "fmnist", "mnist", "census", "lfw", "celeba", "utkface".
        training_size: Proportion of training data to use.
        log: Logging facility.
        exp_id: Used as a random seed.
        celeba_target: Target attribute for CelebA. Example: "Smiling".
        test_size: Proportion of test data to use.

    Returns:
        Loaded dataset as an AmuletDataset.

    Raises:
        ValueError: If dataset is not a recognized dataset name, or if either
            fraction is outside (0, 1] or would empty its split.
    """

    if isinstance(root, str):
        root = Path(root)

    root = root.resolve()

    if dataset == "cifar10":
        data = load_cifar10(root / "data" / "cifar10")
    elif dataset == "cifar100":
        data = load_cifar100(root / "data" / "cifar100")
    elif dataset == "fmnist":
        data = load_fmnist(root / "data" / "fmnist")
    elif dataset == "mnist":
        data = load_mnist(root / "data" / "mnist")
    elif dataset == "census":
        data = load_census(root / "data" / "census", random_seed=exp_id)
    elif dataset == "lfw":
        data = load_lfw(root / "data" / "lfw", random_seed=exp_id)
    elif dataset == "celeba":
        data = load_celeba(
            root / "data" / "celeba", random_seed=exp_id, target_attribute=celeba_target
        )
    elif dataset == "utkface":
        data = load_utkface(root / "data" / "utkface", random_seed=exp_id)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    if training_size < 1.0:
        keep = _subset_indices(
            len(data.train_set),  # type: ignore[reportArgumentType]
            training_size,
            exp_id,
            "training_size",
        )
        data.train_set = Subset(data.train_set, keep.tolist())
        if data.x_train is not None:
            data.x_train = data.x_train[keep]
            data.y_train = None if data.y_train is None else data.y_train[keep]
            data.z_train = None if data.z_train is None else data.z_train[keep]

    if test_size < 1.0:
        keep = _subset_indices(
            len(data.test_set),  # type: ignore[reportArgumentType]
            test_size,
            exp_id,
            "test_size",
        )
        data.test_set = Subset(data.test_set, keep.tolist())
        if data.x_test is not None:
            data.x_test = data.x_test[keep]
            data.y_test = None if data.y_test is None else data.y_test[keep]
            data.z_test = None if data.z_test is None else data.z_test[keep]

    return data


def stratified_split(
    dataset: Dataset,
    split_ratio: float,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    """
    Split a dataset into two stratified subsets.

    Args:
        dataset: Any PyTorch dataset that returns (x, y).
        split_ratio: Fraction of data to go into the first split (e.g., 0.5 for 50/50).
        seed: Random seed for reproducibility.

    Returns:
        Two Subset datasets: (first_split, second_split).
    """
    # Extract all labels
    labels = [dataset[i][1] for i in range(len(dataset))]  # type: ignore[reportArgumentType]

    # Stratified split of indices
    idx_a, idx_b = train_test_split(
        range(len(dataset)),  # type: ignore[reportArgumentType]
        train_size=split_ratio,
        stratify=labels,
        random_state=seed,
    )

    return Subset(dataset, idx_a), Subset(dataset, idx_b)


def create_dir(path: Path | str, log: logging.Logger | None = None) -> Path:
    """
    Create directory at the provided path.

    Args:
        path: Directory to be created.
        log: Logging facility.

    Returns:
        Resolved path to the created directory.
    """
    if isinstance(path, str):
        path = Path(path)

    resolved_path: Path = path.resolve()

    if not resolved_path.exists():
        if log:
            log.info("%s does not exist. Creating...", resolved_path)
        else:
            print(f"{resolved_path} does not exist. Creating...")

        resolved_path.mkdir(parents=True, exist_ok=True)
        if log:
            log.info("%s created.", resolved_path)
        else:
            print(f"{resolved_path} created.")

    else:
        if log:
            log.info("%s already exists.", resolved_path)
        else:
            print(f"{resolved_path} already exists.")

    return resolved_path


class CNNConfig(TypedDict):
    """Convolutional and fully-connected layer sizes for a SimpleCNN capacity."""

    conv_layers: list[tuple[int, int]]
    fc_layers: list[int]


class ModelConfig(TypedDict, total=False):
    """Per-architecture layer configurations for one capacity tier."""

    vgg: list[int | str]
    resnet: str
    linearnet: list[int]
    cnn: CNNConfig


CapacityMap = dict[str, ModelConfig]

DEFAULT_CAPACITY_MAP: CapacityMap = {
    "m1": {
        # VGG11
        "vgg": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        "linearnet": [128, 256, 128],
        "resnet": "resnet34",
        "cnn": {
            "conv_layers": [(12, 5), (24, 5)],
            "fc_layers": [256],
        },
    },
    "m2": {
        # VGG13
        "vgg": [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            "M",
            512,
            512,
            "M",
            512,
            512,
            "M",
        ],
        "linearnet": [256, 512, 256],
        "resnet": "resnet50",
        "cnn": {
            "conv_layers": [(20, 5), (50, 5)],
            "fc_layers": [512],
        },
    },
    "m3": {
        # VGG16
        "vgg": [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            "M",
        ],
        "linearnet": [512, 1024, 512],
        "resnet": "resnet101",
        "cnn": {
            "conv_layers": [(32, 5), (64, 5)],
            "fc_layers": [768],
        },
    },
    "m4": {
        # VGG19
        "vgg": [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
            512,
            "M",
            512,
            512,
            512,
            512,
            "M",
        ],
        "linearnet": [512, 1024, 1024, 512],
        "resnet": "resnet152",
        "cnn": {
            "conv_layers": [(32, 5), (64, 5), (128, 3)],
            "fc_layers": [1024, 512],
        },
    },
}


def initialize_model(
    model_arch: str,
    model_capacity: str,
    num_features: int,
    num_classes: int,
    log: logging.Logger | None = None,
    batch_norm: bool = True,
    model_conf: CapacityMap = DEFAULT_CAPACITY_MAP,
    resnet_replace_first: bool = True,
) -> AmuletModel:
    """Create a model using the provided configuration.

    Args:
        model_arch: Which model architecture to initialize. Options: "vgg", "resnet", "linearnet", "cnn".
        model_capacity: Key in the configuration dict to select model capacity/size.
        num_features: Number of input features (used by linear/dense models).
        num_classes: Number of output classes.
        log: Logger instance for logging info. Defaults to None.
        batch_norm: Whether to use batch normalization. Defaults to True.
        model_conf: Configuration dictionary mapping capacities to model params.
        resnet_replace_first: Whether to replace the first conv layer of ResNet with a smaller filter. Defaults to True.

    Returns:
        Initialized PyTorch model instance with a `get_hidden` method.

    Raises:
        KeyError: If model_capacity is absent from model_conf, or the requested
            architecture's config is missing for that capacity.
        ValueError: If model_arch is not a recognized architecture.
    """
    if model_capacity not in model_conf:
        raise KeyError(f"Capacity '{model_capacity}' not found in model_conf")
    capacity_config = model_conf[model_capacity]

    if model_arch == "vgg":
        if "vgg" not in capacity_config:
            raise KeyError(f"'vgg' config missing for capacity '{model_capacity}'")
        model = VGG(
            num_classes=num_classes,
            layer_config=capacity_config["vgg"],
            batch_norm=batch_norm,
        )
    elif model_arch == "resnet":
        if "resnet" not in capacity_config:
            raise KeyError(f"'resnet' config missing for capacity '{model_capacity}'")
        model = ResNet(
            size=capacity_config["resnet"],
            num_classes=num_classes,
            replace_first=resnet_replace_first,
        )
    elif model_arch == "linearnet":
        if "linearnet" not in capacity_config:
            raise KeyError(
                f"'linearnet' config missing for capacity '{model_capacity}'"
            )
        model = LinearNet(
            num_features=num_features,
            num_classes=num_classes,
            hidden_layer_sizes=capacity_config["linearnet"],
            batch_norm=batch_norm,
        )
    elif model_arch == "cnn":
        if "cnn" not in capacity_config:
            raise KeyError(f"'cnn' config missing for capacity '{model_capacity}'")
        model = SimpleCNN(
            conv_channels_kernel=capacity_config["cnn"]["conv_layers"],
            fc_layers=capacity_config["cnn"]["fc_layers"],
            num_classes=num_classes,
        )
    else:
        msg = f"Incorrect model architecture: {model_arch}"
        if log:
            log.error(msg)
        else:
            print(msg)
        raise ValueError(msg)

    return model
