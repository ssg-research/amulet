"""
Contains methods to load reference datasets for
computer vision applications.
"""

from pathlib import Path

import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import TensorDataset
from .__data import AmuletDataset
from sklearn.model_selection import train_test_split


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
                testing PyTorch models.
    """

    # Note: We do not normalize the inputs by default to preserve [0,1] range.
    # This is important for compatibility with attack methods (e.g., PGD, MIA).

    if transform_train is None:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    if transform_test is None:
        transform_test = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.CIFAR10(
        root=path, train=True, transform=transform_train, download=True
    )
    test_set = datasets.CIFAR10(
        root=path, train=False, transform=transform_test, download=True
    )

    return AmuletDataset(
        train_set=train_set, test_set=test_set, num_features=32 * 32, num_classes=10
    )


def load_cifar100(
    path: str | Path = Path("./data/cifar100"),
    transform_train: transforms.Compose | None = None,
    transform_test: transforms.Compose | None = None,
) -> AmuletDataset:
    """
    Loads the CIFAR100 dataset from PyTorch after applying standard transformations.

    Args:
        path: str or Path object, default = './data/cifar100'
            String or Path object indicating where to store the dataset.
        transform_train: torchvision.transforms.Compose, default = transforms.Compose(
                                                    [
                                                        transforms.RandomCrop(32, padding=4),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                                        transforms.RandomErasing(p=0.2),
                                                        transforms.ToTensor(),
                                                    ]
                                                )
            Image transformations to apply to the training images.
        transform_test: torchvision.transforms.Compose, default = transforms.Compose(
                                                    [
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
                testing PyTorch models.
    """

    # Note: We do not normalize the inputs by default to preserve [0,1] range.
    # This is important for compatibility with attack methods (e.g., PGD, MIA).

    if transform_train is None:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.2),
            ]
        )

    if transform_test is None:
        transform_test = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.CIFAR100(
        root=path, train=True, transform=transform_train, download=True
    )
    test_set = datasets.CIFAR100(
        root=path, train=False, transform=transform_test, download=True
    )

    return AmuletDataset(
        train_set=train_set,
        test_set=test_set,
        num_features=32 * 32,
        num_classes=100,
    )


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
                testing PyTorch models.
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

    return AmuletDataset(
        train_set=train_set, test_set=test_set, num_features=28 * 28, num_classes=10
    )


def load_mnist(
    path: str | Path = Path("./data/mnist"),
    transform_train: transforms.Compose | None = None,
    transform_test: transforms.Compose | None = None,
) -> AmuletDataset:
    """
    Loads the MNIST dataset from PyTorch after applying standard transformations.

    Args:
        path: str or Path object, default = './data/mnist'
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
                testing PyTorch models.
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

    train_set = datasets.MNIST(
        root=path, train=True, transform=transform_train, download=True
    )
    test_set = datasets.MNIST(
        root=path, train=False, transform=transform_test, download=True
    )

    return AmuletDataset(
        train_set=train_set, test_set=test_set, num_features=28 * 28, num_classes=10
    )


def load_celeba(
    path: str | Path = Path("./data/celeba"),
    random_seed: int = 0,
    test_size: float = 0.5,
    target_attribute: str = "Smiling",
) -> AmuletDataset:
    """
    Loads the CelebA released by https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html.
    Link to download the dataset: https://drive.google.com/file/d/1KTaJraB9Koa4h5EVTJQ3y2Dig_vgE5MZ/view?usp=sharing
    Separates the sensitive attributes from the training and testing data.

    Args:
        path: str or Path object
            String or Path object indicating where to store the dataset.
        random_seed: int
            Determines random number generation for dataset shuffling. Pass an int
            for reproducible output across multiple function calls.
        test_size: float
            Proportion of data used for testing.
        target_attribute: str
            Which attribute to use as target. Options: ['Smiling', 'Wavy_Hair', 'Attractive', 'Young']
    Returns:
        Object (:class:`~amulet.datasets.Data`), with the following attributes:
            train_set: :class:`~torch.utils.data.VisionDataset`
                A dataset of images and labels used to build a DataLoader for
                training PyTorch models.
            test_set: :class:`~torch.utils.data.VisionDataset`
                A dataset of images and labels used to build a DataLoader for
                testing PyTorch models.
            x_train: :class:`~np.ndarray`
                Features for the train data.
            x_test: :class:`~np.ndarray`
                Features for the test data.
            y_train: :class:`~np.ndarray`
                Labels for the train data.
            y_test: :class:`~np.ndarray`
                Labels for the test data.
            z_train: :class:`~np.ndarray`
                Sensitive attribute labels for the train data.
            z_test: :class:`~np.ndarray`
                Sensitive attribute labels for the test data.
    """
    # TODO: Write code to download the dataset from Google Drive.

    if isinstance(path, str):
        path = Path(path)

    df = pd.read_csv(
        path / "celeba.csv", na_values="NA", index_col=None, sep=",", header=0
    )
    df["pixels"] = df["pixels"].apply(lambda x: np.array(x.split(), dtype="float32"))
    df["pixels"] = df["pixels"].apply(lambda x: x / 255)
    df["pixels"] = df["pixels"].apply(lambda x: np.reshape(x, (3, 48, 48)))

    images = df["pixels"].to_frame()

    images_np = np.stack(images["pixels"].to_list())
    sensitive_attributes: list[str] = ["Male"]
    attributes = df[[target_attribute] + sensitive_attributes]

    images_train, images_test, attributes_train, attributes_test = train_test_split(
        images_np,
        attributes,
        test_size=test_size,
        stratify=attributes,
        random_state=random_seed,
    )

    # For type setting, the train_test_split function messes with the data type
    attributes_train = pd.DataFrame(attributes_train)
    attributes_test = pd.DataFrame(attributes_test)

    target_train = attributes_train[target_attribute].to_numpy()
    target_test = attributes_test[target_attribute].to_numpy()
    sensitive_train = attributes_train[sensitive_attributes].to_numpy()
    sensitive_test = attributes_test[sensitive_attributes].to_numpy()

    train_set = TensorDataset(
        torch.from_numpy(images_train).type(torch.float32),
        torch.from_numpy(target_train).type(torch.long),
    )
    test_set = TensorDataset(
        torch.from_numpy(images_test).type(torch.float32),
        torch.from_numpy(target_test).type(torch.long),
    )

    return AmuletDataset(
        train_set=train_set,
        test_set=test_set,
        num_features=48 * 48,
        num_classes=2,
        x_train=np.array(images_train),
        x_test=np.array(images_test),
        y_train=target_train,
        y_test=target_test,
        z_train=sensitive_train,
        z_test=sensitive_test,
    )
