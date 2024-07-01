"""Data Class Implementation"""

from torch.utils.data import Dataset
import numpy as np
from dataclasses import dataclass


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
    x_train: np.ndarray | None = None
    x_test: np.ndarray | None = None
    y_train: np.ndarray | None = None
    y_test: np.ndarray | None = None
    z_train: np.ndarray | None = None
    z_test: np.ndarray | None = None
