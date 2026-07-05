import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from amulet.datasets.__data import AmuletDataset


@pytest.fixture
def datasets() -> tuple[TensorDataset, TensorDataset]:
    """Train/test TensorDataset pair sized 10 and 5 over a single feature."""
    return (
        TensorDataset(torch.randn(10, 1), torch.zeros(10)),
        TensorDataset(torch.randn(5, 1), torch.zeros(5)),
    )


def test_amulet_dataset_init(datasets):
    train, test = datasets

    data = AmuletDataset(
        train_set=train,
        test_set=test,
        num_features=1,
        num_classes=2,
        modality="tabular",
    )

    assert data.train_set == train
    assert data.test_set == test
    assert data.num_features == 1
    assert data.num_classes == 2
    assert data.modality == "tabular"
    assert data.sensitive_columns is None
    assert data.x_train is None
    assert data.z_test is None


def test_amulet_dataset_with_optionals(datasets):
    train, test = datasets
    x_train = np.array([[1], [2]])
    z_train = np.array([0, 1])

    data = AmuletDataset(
        train_set=train,
        test_set=test,
        num_features=1,
        num_classes=2,
        modality="tabular",
        sensitive_columns=["attr"],
        x_train=x_train,
        z_train=z_train,
    )

    assert data.x_train is not None
    assert data.z_train is not None
    assert np.array_equal(data.x_train, x_train)
    assert np.array_equal(data.z_train, z_train)
    assert data.x_test is None
    assert data.sensitive_columns == ["attr"]
