"""
The module mlconf.datasets includes utilities to load datasets,
including methods to load and fetch popular reference datasets.
"""

from .__image_datasets import (
    load_cifar10,
    load_cifar100,
    load_fmnist,
    load_mnist,
    load_celeba,
)
from .__tabular_datasets import load_census, load_lfw
from .__data import AmuletDataset, CustomImageDataset

__all__ = [
    "load_cifar10",
    "load_cifar100",
    "load_fmnist",
    "load_mnist",
    "load_census",
    "load_lfw",
    "load_celeba",
    "AmuletDataset",
    "CustomImageDataset",
]
