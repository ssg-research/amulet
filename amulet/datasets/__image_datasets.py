"""
Contains methods to load reference datasets for
computer vision applications.
"""

from pathlib import Path

import torchvision.transforms as transforms
from torchvision import datasets
from .__data import AmuletDataset


def load_cifar10(
    path: str | Path = Path("./data/cifar10"),
    transform_train: transforms.Compose | None = None,
    transform_test: transforms.Compose | None = None,
) -> AmuletDataset:
    """
    Loads the CIFAR10 dataset from PyTorch after applying standard transformations.

    Args:
        path: str or Path object, default = './data/CIFAR10'
            String or Path object indicating where to store the dataset.
        transform_train: torchvision.transforms.Compose, default = transforms.Compose(
                                                    [
                                                        transforms.RandomCrop(32, padding=4),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean, std),
                                                    ]
                                                )
            Image transformations to apply to the training images.
        transform_test: torchvision.transforms.Compose, default = transforms.Compose(
                                                    [
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean, std)
                                                    ]
                                                )
            Image transformations to apply to the testing images.
    Returns:
        Object (:class:`~amulet.datasets.Data`), with the following attributes:
            train_set: :class:`~torch.utils.data.VisionDataset`
                A dataset of images and labels used to build a DataLoader for
                training PyTorch models.
            test_set: :class:`~torch.utils.data.VisionDataset`
                A dataset of images and labels used to build a DataLoader for
                test PyTorch models.
    """
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    if transform_train is None:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    if transform_test is None:
        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
    train_set = datasets.CIFAR10(
        root=path, train=True, transform=transform_train, download=True
    )
    test_set = datasets.CIFAR10(
        root=path, train=False, transform=transform_test, download=True
    )

    return AmuletDataset(train_set=train_set, test_set=test_set)


def load_fmnist(
    path: str | Path = Path("./data/fmnist"),
    transform_train: transforms.Compose | None = None,
    transform_test: transforms.Compose | None = None,
) -> AmuletDataset:
    """
    Loads the FashionMNIST dataset from PyTorch after applying standard transformations.

    Args:
        path: str or Path object, default = './data/CIFAR10'
            String or Path object indicating where to store the dataset.
        transform_train: torchvision.transforms.Compose, default = transforms.Compose(
                                                        [
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.RandomVerticalFlip(),
                                                            transforms.RandomRotation(15),
                                                            transforms.RandomCrop([28, 28]),
                                                            transforms.ToTensor(),
                                                        ]
                                                    )
            Image transformations to apply to the training images.
        transform_test: torchvision.transforms.Compose, default = transforms.Compose(
                                                        [
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.RandomVerticalFlip(),
                                                            transforms.RandomRotation(15),
                                                            transforms.RandomCrop([28, 28]),
                                                            transforms.ToTensor(),
                                                        ]
                                                    )
            Image transformations to apply to the testing images.
    Returns:
        Object (:class:`~amulet.datasets.Data`), with the following attributes:
            train_set: :class:`~torch.utils.data.VisionDataset`
                A dataset of images and labels used to build a DataLoader for
                training PyTorch models.
            test_set: :class:`~torch.utils.data.VisionDataset`
                A dataset of images and labels used to build a DataLoader for
                test PyTorch models.
    """
    if transform_train is None:
        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomCrop([28, 28]),
                transforms.ToTensor(),
            ]
        )

    if transform_test is None:
        transform_test = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomCrop([28, 28]),
                transforms.ToTensor(),
            ]
        )

    train_set = datasets.FashionMNIST(
        root=path, train=True, transform=transform_train, download=True
    )
    test_set = datasets.FashionMNIST(
        root=path, train=False, transform=transform_test, download=True
    )

    return AmuletDataset(train_set=train_set, test_set=test_set)
