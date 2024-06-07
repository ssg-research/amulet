"""
The module mlconf.datasets includes utilities to load datasets,
including methods to load and fetch popular reference datasets.
"""
from ._image_datasets import (
    load_cifar10,
    load_fmnist
)
from ._tabular_datasets import (
    load_census,
    load_lfw
)

__all__ = [
    "load_cifar10",
    "load_fmnist",
    "load_census",
    "load_lfw"
]
