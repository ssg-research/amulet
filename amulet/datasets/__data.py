"""Data Classes Implementation"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from dataclasses import dataclass
from torchvision.io import read_image, ImageReadMode


@dataclass
class AmuletDataset:
    """
    Wrapper class to return datasets.

    Attributes:
        train_set: :class:`~torch.utils.data.Dataset`
            Train data, usually as a PyTorch TensorDataset or VisionDataset.
        test_set: :class:`~torch.utils.data.Dataset`
            Test data, usually as a PyTorch TensorDataset or VisionDataset.
        x_train: :class:`~np.ndarray` or None
            Train features.
        x_test: :class:`~np.ndarray` or None
            Test features.
        y_train: :class:`~np.ndarray` or None
            Train labels.
        y_test: :class:`~np.ndarray` or None
            Test labels.
        z_train: :class:`~np.ndarray` or None
            Sensitive attributes (for datasets that have them) for train data.
        z_test: :class:`~np.ndarray` or None
            Sensitive attributes (for datasets that have them) for test data.
    """

    train_set: Dataset
    test_set: Dataset
    num_features: int
    num_classes: int
    x_train: np.ndarray | None = None
    x_test: np.ndarray | None = None
    y_train: np.ndarray | None = None
    y_test: np.ndarray | None = None
    z_train: np.ndarray | None = None
    z_test: np.ndarray | None = None


class CustomImageDataset(Dataset):
    """
    PyTorch dataset class to read a custom image dataset.

    Attributes:
        labels_file: str or Path object.
            CSV file containing the labels where each row is (img_filename, label).
        img_dir: str or Path object.
            Directory containing the images.
        transform: :class:`~torch.utils.data.Dataset`
            Transformation to apply to the images.
    """

    def __init__(
        self,
        labels_file: Path | str,
        img_dir: Path | str,
        transform: transforms.Compose | None = None,
    ):
        if isinstance(img_dir, str):
            img_dir = Path(img_dir)

        self.img_labels = pd.read_csv(labels_file)

        if self.img_labels.shape[1] != 2:
            raise Exception("Labels file should have 2 columns.")

        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        img_path = self.img_dir / self.img_labels.iloc[idx, 0]
        image = read_image(img_path, mode=ImageReadMode.RGB)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image).type(torch.float)

        return image, label
