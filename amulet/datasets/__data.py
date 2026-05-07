"""Data Classes Implementation"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image


@dataclass
class AmuletDataset:
    """
    Wrapper for datasets used throughout amulet pipelines.

    Attributes:
        train_set: Train data, usually a PyTorch TensorDataset or VisionDataset.
        test_set: Test data, usually a PyTorch TensorDataset or VisionDataset.
        num_features: Number of input features.
        num_classes: Number of output classes.
        modality: Describes the tensor shape seen by models. "image" for multi-dimensional
            (C, H, W) samples; "tabular" for 1-D feature vectors. LFW is "tabular" because
            its images are flattened in the loader.
        sensitive_columns: Column names of z_train/z_test, in order. None if the
            dataset has no sensitive attributes.
        x_train: Train features as a numpy array, or None.
        x_test: Test features as a numpy array, or None.
        y_train: Train labels, or None.
        y_test: Test labels, or None.
        z_train: Sensitive attributes for train data, or None.
        z_test: Sensitive attributes for test data, or None.
    """

    train_set: Dataset
    test_set: Dataset
    num_features: int
    num_classes: int
    modality: Literal["image", "tabular"]
    sensitive_columns: list[str] | None = None
    x_train: np.ndarray | None = None
    x_test: np.ndarray | None = None
    y_train: np.ndarray | None = None
    y_test: np.ndarray | None = None
    z_train: np.ndarray | None = None
    z_test: np.ndarray | None = None


class CustomImageDataset(Dataset):
    """
    PyTorch Dataset for a custom image directory with a CSV label file.

    Attributes:
        labels_file: CSV file where each row is (img_filename, label).
        img_dir: Directory containing the images.
        transform: Transformation to apply to the images.
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
