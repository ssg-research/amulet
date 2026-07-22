"""Data Classes Implementation"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, TensorDataset
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
            (C, H, W) samples; "tabular" for 1-D feature vectors; "text" for token-id
            vectors, where each sample is a padded `input_ids` tensor and the raw
            strings are retained on the `TextTensorDataset` (see that class). LFW is
            "tabular" because its images are flattened in the loader.
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
    modality: Literal["image", "tabular", "text"]
    sensitive_columns: list[str] | None = None
    x_train: np.ndarray | None = None
    x_test: np.ndarray | None = None
    y_train: np.ndarray | None = None
    y_test: np.ndarray | None = None
    z_train: np.ndarray | None = None
    z_test: np.ndarray | None = None


class TextTensorDataset(TensorDataset):
    """TensorDataset of `(input_ids, label)` that also carries the raw strings.

    This is the single artifact that flows attack -> target -> defense for the text
    modality. Because it subclasses `torch.utils.data.TensorDataset` over
    `(input_ids, labels)`, it is a `TensorDataset`: the
    `PoisoningAttack.poison_* -> TensorDataset` return type, every
    `for (data, target) in loader` consumer, and the single-tensor `DPSGD`
    training loop all work unchanged and never see the extra state. Only the two
    extra attributes below carry the string view an input-purification defense
    (ONION) needs.

    A `DataLoader` collates only the tensors; a consumer that needs the strings
    reads `loader.dataset.texts` (dataset order), which is why a defense must
    purify the dataset before the (optionally shuffled) target loader is built.

    Attributes:
        texts: The raw (possibly poisoned) strings in dataset order, one per row.
        tokenizer_name: Name of the tokenizer that produced `input_ids`, so a
            defense can re-tokenize after purifying `texts`.
    """

    def __init__(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        texts: list[str],
        tokenizer_name: str,
    ):
        """Build the dataset, checking that all inputs share the same row count.

        Raises:
            ValueError: If texts and input_ids have different numbers of rows.
        """
        if len(texts) != input_ids.shape[0]:
            raise ValueError(
                f"texts ({len(texts)}) and input_ids ({input_ids.shape[0]}) must have "
                "the same number of rows."
            )
        super().__init__(input_ids, labels)
        self.texts = texts
        self.tokenizer_name = tokenizer_name


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
