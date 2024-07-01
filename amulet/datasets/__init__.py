"""
The module mlconf.datasets includes utilities to load datasets,
including methods to load and fetch popular reference datasets.
"""

from .__image_datasets import load_cifar10, load_fmnist
from .__tabular_datasets import load_census, load_lfw
from .__data import AmuletDataset

__all__ = ["load_cifar10", "load_fmnist", "load_census", "load_lfw", "AmuletDataset"]
