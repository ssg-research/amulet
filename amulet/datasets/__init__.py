"""
The module mlconf.datasets includes utilities to load datasets,
including methods to load and fetch popular reference datasets.
"""

from .__data import AmuletDataset, CustomImageDataset
from .__image_datasets import (
    load_celeba,
    load_cifar10,
    load_cifar100,
    load_fmnist,
    load_mnist,
    load_utkface,
)
from .__tabular_datasets import load_census, load_lfw

__all__ = [
    "AmuletDataset",
    "CustomImageDataset",
    "load_celeba",
    "load_census",
    "load_cifar10",
    "load_cifar100",
    "load_fmnist",
    "load_lfw",
    "load_mnist",
    "load_utkface",
]
