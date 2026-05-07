import torch
from torch.utils.data import TensorDataset

from amulet.utils.__pipeline import stratified_split


def test_stratified_split_union_equality():
    # Arrange
    x = torch.randn(20, 1)
    y = torch.tensor([0] * 10 + [1] * 10)
    dataset = TensorDataset(x, y)
    split_ratio = 0.5

    # Act
    split1, split2 = stratified_split(dataset, split_ratio, seed=42)

    # Assert
    assert len(split1) + len(split2) == len(dataset)
    assert len(split1) == 10
    assert len(split2) == 10

    # Check that indices are disjoint and cover the whole range
    indices1 = set(split1.indices)
    indices2 = set(split2.indices)
    assert indices1.isdisjoint(indices2)
    assert indices1.union(indices2) == set(range(20))


def test_stratified_split_proportions():
    # Arrange: 100 samples, 80 class 0, 20 class 1
    y = torch.tensor([0] * 80 + [1] * 20)
    x = torch.randn(100, 1)
    dataset = TensorDataset(x, y)
    split_ratio = 0.25  # 25 samples in split1, 75 in split2

    # Act
    split1, _ = stratified_split(dataset, split_ratio, seed=42)

    # Assert
    # In split1 (size 25), we expect 25% of each class: 80*0.25=20 (class 0), 20*0.25=5 (class 1)
    y1 = torch.tensor([split1[i][1] for i in range(len(split1))])
    assert (y1 == 0).sum().item() == 20
    assert (y1 == 1).sum().item() == 5


def test_stratified_split_determinism():
    # Arrange
    x = torch.randn(20, 1)
    y = torch.tensor([0] * 10 + [1] * 10)
    dataset = TensorDataset(x, y)

    # Act
    s1_a, _ = stratified_split(dataset, 0.5, seed=42)
    s1_b, _ = stratified_split(dataset, 0.5, seed=42)
    s1_c, _ = stratified_split(dataset, 0.5, seed=123)

    # Assert
    assert s1_a.indices == s1_b.indices
    assert s1_a.indices != s1_c.indices
