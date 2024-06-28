"""Data Class Implementation"""

from torch.utils.data import Dataset
import numpy as np


class Data:
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

    def __init__(
        self,
        train_set: Dataset,
        test_set: Dataset,
        x_train: np.ndarray | None = None,
        x_test: np.ndarray | None = None,
        y_train: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
        z_train: np.ndarray | None = None,
        z_test: np.ndarray | None = None,
    ):
        self.train_set = train_set
        self.test_set = test_set
        if x_train is not None:
            self.x_train = x_train
        if x_test is not None:
            self.x_test = x_test
        if y_train is not None:
            self.y_train = y_train
        if y_test is not None:
            self.y_test = y_test
        if z_train is not None:
            self.z_train = z_train
        if z_test is not None:
            self.z_test = z_test
