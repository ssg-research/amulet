"""
Contains methods to load reference datasets for 
computer vision applications.
"""

from pathlib import Path
from typing import Optional, Union

import torchvision.transforms as transforms
from torchvision import datasets
from sklearn.utils import Bunch

def load_cifar10(path: Optional[Union[str, Path]] = Path('./data/cifar10')) -> Bunch:
    """
    Loads the CIFAR10 dataset from PyTorch after applying standard transformations. 

    Args:
        path: str or Path object, default = './data/CIFAR10'
            String or Path object indicating where to store the dataset.

    Returns:
        Dictionary-like object (:class:`~sklearn.utils.Bunch`), with the following attributes:
            Dictionary-like object, with the following attributes:

            train_set: :class:`~torch.utils.data.TensorDataset`
                A dataset of images and labels used to build a DataLoader for
                training PyTorch models.

            test_set: :class:`~torch.utils.data.TensorDataset`
                A dataset of images and labels used to build a DataLoader for
                test PyTorch models.
    """
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)])
    transform_test = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize(mean, std)])

    train_set = datasets.CIFAR10(root=path, train=True, transform=transform_train, download=True)
    test_set = datasets.CIFAR10(root=path, train=False, transform=transform_test, download=True)

    return Bunch(
        train_set=train_set,
        test_set=test_set
    )

def load_fmnist(path: Optional[Union[str, Path]] = Path('./data/fmnist')) -> Bunch:
    """
    Loads the FashionMNIST dataset from PyTorch after applying standard transformations. 

    Args:
        path: str or Path object, default = './data/CIFAR10'
            String or Path object indicating where to store the dataset.

    Returns:
        Dictionary-like object (:class:`~sklearn.utils.Bunch`), with the following attributes:
            train_set: :class:`~torch.utils.data.TensorDataset`
                A dataset of images and labels used to build a DataLoader for
                training PyTorch models.

            test_set: :class:`~torch.utils.data.TensorDataset`
                A dataset of images and labels used to build a DataLoader for
                test PyTorch models.
    """
    transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(15),
                    transforms.RandomCrop([28, 28]),
                    transforms.ToTensor()
                ])
    train_set = datasets.FashionMNIST(root=path, train=True, transform=transform, download=True)
    test_set = datasets.FashionMNIST(root=path, train=False, transform=transform, download=True)

    return Bunch(
        train_set=train_set,
        test_set=test_set
    )
