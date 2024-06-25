"""
Utilities to help build an ML pipeline.
"""

import sys
import logging
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import random_split
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split

from ..models import VGG, BinaryNet, LinearNet, CNN
from ..datasets import load_census, load_cifar10, load_fmnist, load_lfw


def load_data(
    root: Path | str,
    generator: torch.Generator,
    dataset: str,
    training_size: float,
    log: logging.Logger | None = None,
    return_x_y_z: bool = False,
    exp_id=0,
) -> Bunch:
    """
    Loads data given the dataset and the training size.

    Args:
        root: :class:~`pathlib.Path` or str
            Root directory of pipeline
        log: :class:~`logging.Logger`
            Logging facility.
    """

    if isinstance(root, str):
        root = Path(root)

    root = root.resolve()

    if dataset == "cifar10":
        data = load_cifar10(root / "data" / "cifar10")
    elif dataset == "fmnist":
        data = load_fmnist(root / "data" / "fmnist")
    elif dataset == "census":
        data = load_census(
            root / "data" / "census", random_seed=exp_id, return_x_y_z=return_x_y_z
        )
    elif dataset == "lfw":
        data = load_lfw(
            root / "data" / "lfw", random_seed=exp_id, return_x_y_z=return_x_y_z
        )
    else:
        if log:
            log.info(
                "Line 42, mlconf.utils._pipeline.py: Incorrect dataset configuration"
            )
        else:
            print("Line 44, mlconf.utils._pipeline.py: Incorrect dataset configuration")
        sys.exit()

    if return_x_y_z:
        x_train, x_test, y_train, y_test, z_train, z_test = data

        if training_size < 1.0:
            x_train, _, y_train, _, z_train, _ = train_test_split(
                x_train, y_train, z_train, train_size=training_size, random_state=exp_id
            )

        data = (
            np.array(x_train),
            np.array(x_test),
            np.array(y_train),
            np.array(y_test),
            np.array(z_train),
            np.array(z_test),
        )

        return data  # type: ignore[reportReturnType]
    else:
        new_train_size = int(training_size * len(data.train_set))  # type: ignore[reportAttributeAccessIssue]
        train_set, _ = random_split(
            data.train_set,  # type: ignore[reportAttributeAccessIssue]
            [new_train_size, len(data.train_set) - new_train_size],  # type: ignore[reportAttributeAccessIssue]
            generator=generator,
        )
        data.train_set = train_set  # type: ignore[reportAttributeAccessIssue]

        return data  # type: ignore[reportReturnType]


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


capacity_map = {
    "m1": {"vgg": "VGG11", "linearnet": [128, 256, 128], "binarynet": [32, 64, 32]},
    "m2": {"vgg": "VGG13", "linearnet": [256, 512, 256], "binarynet": [64, 128, 64]},
    "m3": {"vgg": "VGG16", "linearnet": [512, 1024, 512], "binarynet": [128, 256, 128]},
    "m4": {
        "vgg": "VGG19",
        "linearnet": [512, 1024, 1024, 512],
        "binarynet": [128, 256, 256, 128],
    },
}


def initialize_model(
    model_arch: str,
    model_capacity: str,
    dataset: str,
    log: logging.Logger | None = None,
) -> nn.Module:
    """
    Creates a model using the configuration provided.

    Args:
        model: str
            Which model to initialize.
        model_capacity: str
            Size of the model.
        dataset: str
            The dataset that will be used to train the model/
        log: :class:~`logging.Logger` or None
            Logging facility.

    Returns:
        Path to the created directory.
    """
    if model_arch == "vgg":
        model = VGG(capacity_map[model_capacity]["vgg"])
    elif model_arch == "cnn":
        model = CNN()
    elif model_arch == "linearnet":
        model = LinearNet(hidden_layer_sizes=capacity_map[model_capacity]["linearnet"])
    elif model_arch == "binarynet":
        if dataset == "census":
            model = BinaryNet(
                num_features=93,
                hidden_layer_sizes=capacity_map[model_capacity]["binarynet"],
            )
        elif dataset == "lfw":
            model = BinaryNet(
                num_features=8742,
                hidden_layer_sizes=capacity_map[model_capacity]["binarynet"],
            )
        else:
            if log:
                log.info(
                    "Line 148, mlconf.utils._pipeline.py: Incorrect model configuration."
                )
            else:
                print(
                    "Line 150, mlconf.utils._pipeline.py: Incorrect model configuration."
                )
            sys.exit()
    else:
        if log:
            log.info(
                "Line 154, mlconf.utils._pipeline.py: Incorrect model configuration"
            )
        else:
            print("Line 156, mlconf.utils._pipeline.py: Incorrect model configuration")
        sys.exit()

    if log:
        log.info("Model initialized: %s", model)
    else:
        print(f"Model initialized: {model}")

    return model