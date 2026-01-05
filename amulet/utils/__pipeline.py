"""
Utilities to help build an ML pipeline.
"""

import sys
import logging
from pathlib import Path
from typing import TypedDict

import torch
import torch.nn as nn
from torch.utils.data import random_split, Dataset, Subset

from sklearn.model_selection import train_test_split

from ..models import VGG, LinearNet, ResNet, SimpleCNN
from ..datasets import (
    load_census,
    load_cifar10,
    load_cifar100,
    load_fmnist,
    load_mnist,
    load_lfw,
    load_celeba,
    AmuletDataset,
)


def load_data(
    root: Path | str,
    dataset: str,
    training_size: float = 1.0,
    log: logging.Logger | None = None,
    exp_id: int = 0,
    celeba_target: str = "Smiling",
) -> AmuletDataset:
    """
    Loads data given the dataset and the training size.

    Args:
        root: :class:~`pathlib.Path` or str
            Root directory of pipeline
        dataset: str
            Name of the dataset.
        training_size: float
            Proportion of training data to use.
        log: :class:~`logging.Logger`
            Logging facility.
        exp_id: int
            Used as a random seed.
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
    else:
        if log:
            log.info(
                "Line 42, mlconf.utils._pipeline.py: Incorrect dataset configuration"
            )
        else:
            print("Line 44, mlconf.utils._pipeline.py: Incorrect dataset configuration")
        sys.exit()

    if training_size < 1.0:
        if data.x_train is not None:
            data.x_train, _, data.y_train, _, data.z_train, _ = train_test_split(  # type: ignore[reportAttributeAccessIssue]
                data.x_train,
                data.y_train,
                data.z_train,
                train_size=training_size,
                random_state=exp_id,
            )

        new_train_size = int(training_size * len(data.train_set))  # type: ignore[reportAttributeAccessIssue]
        generator = torch.Generator().manual_seed(exp_id)
        train_set, _ = random_split(
            data.train_set,  # type: ignore[reportAttributeAccessIssue]
            [new_train_size, len(data.train_set) - new_train_size],  # type: ignore[reportAttributeAccessIssue]
            generator=generator,
        )
        data.train_set = train_set  # type: ignore[reportAttributeAccessIssue]

    return data


def stratified_split(
    dataset: Dataset,
    split_ratio: float,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
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
    Create directory using provided path.

    Args:
        path: :class:~`pathlib.Path` or str
            Directory to be created.
        log: :class:~`logging.Logger` or None
            Logging facility.

    Returns:
        Path to the created directory.
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
    conv_layers: list[tuple[int, int]]
    fc_layers: list[int]


class ModelConfig(TypedDict, total=False):
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
) -> nn.Module:
    """
    Creates a model using the provided configuration.

    Args:
        model_arch: str
            Which model architecture to initialize. Options: "vgg", "resnet", "linearnet", "cnn".
        model_capacity: str
            Key in the configuration dict to select model capacity/size.
        num_features: int
            Number of input features (used by linear/dense models).
        num_classes: int
            Number of output classes.
        log: logging.Logger | None, optional
            Logger instance for logging info. Defaults to None.
        batch_norm: bool, optional
            Whether to use batch normalization. Defaults to True.
        model_conf: CapacityMap, optional
            Configuration dictionary mapping capacities to model params.
        resnet_replace_first: bool, optional
            Whether to replace the first conv layer of ResNet with a smaller filter. Defaults to True.

    Returns:
        nn.Module
            Initialized PyTorch model instance.
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

    # if log:
    #     log.info("Model initialized: %s", model)
    # else:
    #     print(f"Model initialized: {model}")

    return model
