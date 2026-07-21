"""
The module `amulet.datasets` includes utilities to load datasets,
including methods to load and fetch popular reference datasets.
"""

from .__data import AmuletDataset, CustomImageDataset, TextTensorDataset
from .__image_datasets import (
    load_celeba,
    load_cifar10,
    load_cifar100,
    load_fmnist,
    load_mnist,
    load_utkface,
)
from .__tabular_datasets import load_census, load_lfw
from .__text_datasets import load_agnews, load_imdb, load_sst2

__all__ = [
    "AmuletDataset",
    "CustomImageDataset",
    "TextTensorDataset",
    "load_agnews",
    "load_celeba",
    "load_census",
    "load_cifar10",
    "load_cifar100",
    "load_fmnist",
    "load_imdb",
    "load_lfw",
    "load_mnist",
    "load_sst2",
    "load_utkface",
]
