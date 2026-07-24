"""Tests for the pipeline helpers in amulet/utils/__pipeline.py:
create_dir, initialize_model, load_data, and stratified_split."""

from collections.abc import Sized
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, TensorDataset

from amulet.datasets import AmuletDataset
from amulet.models import VGG, AmuletModel, LinearNet, ResNet, SimpleCNN
from amulet.utils.__pipeline import (
    create_dir,
    initialize_model,
    load_data,
    stratified_split,
)

# ---------------------------------------------------------------------------
# create_dir
# ---------------------------------------------------------------------------


def test_create_dir_new(tmp_path):
    target = tmp_path / "new_dir"
    assert not target.exists()

    resolved = create_dir(target)

    assert resolved == target.resolve()
    assert target.exists()
    assert target.is_dir()


def test_create_dir_exists(tmp_path):
    target = tmp_path / "existing_dir"
    target.mkdir()
    assert target.exists()

    resolved = create_dir(target)

    assert resolved == target.resolve()
    assert target.exists()


def test_create_dir_str_path(tmp_path):
    target_str = str(tmp_path / "str_dir")

    resolved = create_dir(target_str)

    assert isinstance(resolved, Path)
    assert resolved.exists()
    assert str(resolved) == str(Path(target_str).resolve())


# ---------------------------------------------------------------------------
# initialize_model
# ---------------------------------------------------------------------------


def _input_for(arch: str, num_features: int) -> torch.Tensor:
    """Return a forward-compatible input tensor for each architecture."""
    if arch == "linearnet":
        return torch.randn(1, num_features)
    if arch == "cnn":
        # SimpleCNN is hard-coded to 28x28 single-channel input.
        return torch.randn(1, 1, 28, 28)
    # VGG and ResNet: 3-channel 32x32 (VGG has 5 MaxPools, ResNet uses replace_first).
    return torch.randn(1, 3, 32, 32)


@pytest.mark.parametrize(
    "arch, expected_class",
    [
        ("vgg", VGG),
        ("resnet", ResNet),
        ("linearnet", LinearNet),
        ("cnn", SimpleCNN),
    ],
)
@pytest.mark.parametrize("capacity", ["m1", "m2", "m3", "m4"])
def test_initialize_model_variants(arch, expected_class, capacity):
    num_features = 10
    num_classes = 2

    model = initialize_model(arch, capacity, num_features, num_classes)

    assert isinstance(model, expected_class)
    assert isinstance(model, AmuletModel)

    # eval() bypasses batch-norm's "batch size > 1" training-mode check
    # so the forward pass works with a single-sample probe input.
    model.eval()
    x = _input_for(arch, num_features)
    with torch.no_grad():
        out = model(x)
        hidden = model.get_hidden(x)

    assert out.shape == (1, num_classes)
    assert isinstance(hidden, torch.Tensor)
    assert hidden.shape[0] == 1


def test_initialize_model_invalid_arch():
    with pytest.raises(ValueError, match="Incorrect model architecture"):
        initialize_model("invalid", "m1", 10, 2)


def test_initialize_model_invalid_capacity():
    with pytest.raises(KeyError, match="not found in model_conf"):
        initialize_model("vgg", "invalid", 10, 2)


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------

# Row i of every split is the constant vector [i, i, i], so a record's identity
# survives subsetting. That is what makes misalignment between the NumPy views
# and the Dataset detectable: if the two are subsampled independently, the
# tensor at position i stops matching x_*[i] and the assertions below fail.
_FEATURES = 3


def _indexed_split(count: int, offset: int) -> tuple[np.ndarray, ...]:
    """Build (x, y, z) whose row i is identifiable by its own value."""
    x = np.repeat(np.arange(offset, offset + count, dtype=np.float32), _FEATURES)
    x = x.reshape(count, _FEATURES)
    y = (np.arange(count) % 2).astype(np.int64)
    z = (np.arange(count) % 3).reshape(-1, 1).astype(np.int64)
    return x, y, z


def _tensor_set(x: np.ndarray, y: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(x).type(torch.float), torch.from_numpy(y).type(torch.long)
    )


def _tabular_dataset(n_train: int = 100, n_test: int = 60) -> AmuletDataset:
    """A census-shaped dataset: both a Dataset and the NumPy views, aligned."""
    x_train, y_train, z_train = _indexed_split(n_train, 0)
    # Offset the test split so a train row can never be mistaken for a test row.
    x_test, y_test, z_test = _indexed_split(n_test, 1000)
    return AmuletDataset(
        train_set=_tensor_set(x_train, y_train),
        test_set=_tensor_set(x_test, y_test),
        num_features=_FEATURES,
        num_classes=2,
        modality="tabular",
        sensitive_columns=["dummy"],
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        z_train=z_train,
        z_test=z_test,
    )


def _vision_dataset(n_train: int = 100, n_test: int = 60) -> AmuletDataset:
    """A CIFAR-shaped dataset: Datasets only, no NumPy views (x_train is None)."""
    x_train, y_train, _ = _indexed_split(n_train, 0)
    x_test, y_test, _ = _indexed_split(n_test, 1000)
    return AmuletDataset(
        train_set=_tensor_set(x_train, y_train),
        test_set=_tensor_set(x_test, y_test),
        num_features=_FEATURES,
        num_classes=2,
        modality="image",
    )


@pytest.fixture
def stub_census(monkeypatch):
    """Serve a synthetic census in place of the real download."""

    def factory(n_train: int = 100, n_test: int = 60) -> AmuletDataset:
        data = _tabular_dataset(n_train, n_test)
        monkeypatch.setattr("amulet.utils.__pipeline.load_census", lambda *a, **k: data)
        return data

    return factory


@pytest.fixture
def stub_cifar(monkeypatch):
    """Serve a synthetic CIFAR-shaped dataset carrying no NumPy views."""

    def factory(n_train: int = 100, n_test: int = 60) -> AmuletDataset:
        data = _vision_dataset(n_train, n_test)
        monkeypatch.setattr(
            "amulet.utils.__pipeline.load_cifar10", lambda *a, **k: data
        )
        return data

    return factory


def _size(dataset: Dataset) -> int:
    """Length of a split. `Dataset` does not declare `__len__`, so this casts."""
    return len(cast("Sized", dataset))


def _dataset_ids(dataset: Dataset) -> list[int]:
    """Recover each record's identity from the Dataset, in dataset order."""
    indexable = cast("TensorDataset", dataset)
    return [int(indexable[i][0][0].item()) for i in range(_size(dataset))]


def _array_ids(array: np.ndarray | None) -> list[int]:
    """Recover each record's identity from a NumPy view, in array order."""
    assert array is not None, "the census stand-in always carries its NumPy views"
    return [int(value) for value in array[:, 0]]


def _labels(dataset: Dataset) -> list[int]:
    """Recover the label the Dataset yields for each record, in dataset order."""
    indexable = cast("TensorDataset", dataset)
    return [int(indexable[i][1].item()) for i in range(_size(dataset))]


def test_load_data_defaults_keep_both_splits_whole(tmp_path, stub_census):
    stub_census(n_train=100, n_test=60)

    data = load_data(tmp_path, "census")

    assert _size(data.train_set) == 100
    assert _size(data.test_set) == 60
    assert len(_array_ids(data.x_train)) == 100
    assert len(_array_ids(data.x_test)) == 60


def test_load_data_test_size_shrinks_the_test_split(tmp_path, stub_census):
    stub_census(n_train=100, n_test=60)

    data = load_data(tmp_path, "census", test_size=0.25)

    assert _size(data.test_set) == 15
    assert len(_array_ids(data.x_test)) == 15
    assert data.y_test is not None and data.y_test.shape[0] == 15
    assert data.z_test is not None and data.z_test.shape[0] == 15


def test_load_data_test_size_leaves_the_train_split_whole(tmp_path, stub_census):
    stub_census(n_train=100, n_test=60)

    data = load_data(tmp_path, "census", test_size=0.25)

    assert _size(data.train_set) == 100
    assert len(_array_ids(data.x_train)) == 100


def test_load_data_training_size_leaves_the_test_split_whole(tmp_path, stub_census):
    """The knobs are independent: this is the regression the artifact tripped on."""
    stub_census(n_train=100, n_test=60)

    data = load_data(tmp_path, "census", training_size=0.25)

    assert _size(data.test_set) == 60
    assert len(_array_ids(data.x_test)) == 60


@pytest.mark.parametrize("training_size, test_size", [(0.25, 0.5), (0.5, 0.25)])
def test_load_data_subsets_both_splits_together(
    tmp_path, stub_census, training_size, test_size
):
    stub_census(n_train=100, n_test=60)

    data = load_data(tmp_path, "census", training_size, test_size=test_size)

    assert _size(data.train_set) == int(training_size * 100)
    assert _size(data.test_set) == int(test_size * 60)


def test_load_data_keeps_train_views_index_aligned(tmp_path, stub_census):
    """train_set[i] must be the same record as x_train[i] after subsetting."""
    stub_census(n_train=100, n_test=60)

    data = load_data(tmp_path, "census", training_size=0.25)

    assert _dataset_ids(data.train_set) == _array_ids(data.x_train)


def test_load_data_keeps_test_views_index_aligned(tmp_path, stub_census):
    """test_set[i] must be the same record as x_test[i] after subsetting."""
    stub_census(n_train=100, n_test=60)

    data = load_data(tmp_path, "census", test_size=0.25)

    assert _dataset_ids(data.test_set) == _array_ids(data.x_test)


def test_load_data_keeps_labels_aligned_with_features(tmp_path, stub_census):
    """The label the Dataset yields must match y_test for the same record."""
    stub_census(n_train=100, n_test=60)

    data = load_data(tmp_path, "census", test_size=0.25)

    assert data.y_test is not None
    assert _labels(data.test_set) == [int(value) for value in data.y_test]


def test_load_data_subset_is_deterministic_in_exp_id(tmp_path, stub_census):
    stub_census(n_train=100, n_test=60)
    first = load_data(tmp_path, "census", 0.25, test_size=0.25, exp_id=7)
    stub_census(n_train=100, n_test=60)
    second = load_data(tmp_path, "census", 0.25, test_size=0.25, exp_id=7)

    assert _array_ids(first.x_test) == _array_ids(second.x_test)
    assert _array_ids(first.x_train) == _array_ids(second.x_train)


def test_load_data_subset_varies_with_exp_id(tmp_path, stub_census):
    stub_census(n_train=100, n_test=60)
    first = load_data(tmp_path, "census", test_size=0.25, exp_id=0)
    stub_census(n_train=100, n_test=60)
    second = load_data(tmp_path, "census", test_size=0.25, exp_id=1)

    assert _array_ids(first.x_test) != _array_ids(second.x_test)


def test_load_data_subsets_datasets_lacking_numpy_views(tmp_path, stub_cifar):
    """CIFAR-shaped loaders carry no arrays; the Datasets must still shrink."""
    stub_cifar(n_train=100, n_test=60)

    data = load_data(tmp_path, "cifar10", 0.25, test_size=0.5)

    assert _size(data.train_set) == 25
    assert _size(data.test_set) == 30
    assert data.x_train is None


def test_load_data_rejects_a_fraction_that_empties_a_split(tmp_path, stub_census):
    """A silently empty test split would make every downstream metric a NaN."""
    stub_census(n_train=100, n_test=60)

    with pytest.raises(ValueError, match="test_size"):
        _ = load_data(tmp_path, "census", test_size=0.001)


# ---------------------------------------------------------------------------
# stratified_split
# ---------------------------------------------------------------------------


def test_stratified_split_union_equality():
    x = torch.randn(20, 1)
    y = torch.tensor([0] * 10 + [1] * 10)
    dataset = TensorDataset(x, y)
    split_ratio = 0.5

    split1, split2 = stratified_split(dataset, split_ratio, seed=42)

    assert len(split1) + len(split2) == len(dataset)
    assert len(split1) == 10
    assert len(split2) == 10

    # Check that indices are disjoint and cover the whole range
    indices1 = set(split1.indices)
    indices2 = set(split2.indices)
    assert indices1.isdisjoint(indices2)
    assert indices1.union(indices2) == set(range(20))


def test_stratified_split_proportions():
    # 100 samples, 80 class 0, 20 class 1
    y = torch.tensor([0] * 80 + [1] * 20)
    x = torch.randn(100, 1)
    dataset = TensorDataset(x, y)
    split_ratio = 0.25  # 25 samples in split1, 75 in split2

    split1, _ = stratified_split(dataset, split_ratio, seed=42)

    # In split1 (size 25), we expect 25% of each class: 80*0.25=20 (class 0), 20*0.25=5 (class 1)
    y1 = torch.tensor([split1[i][1] for i in range(len(split1))])
    assert (y1 == 0).sum().item() == 20
    assert (y1 == 1).sum().item() == 5


def test_stratified_split_determinism():
    x = torch.randn(20, 1)
    y = torch.tensor([0] * 10 + [1] * 10)
    dataset = TensorDataset(x, y)

    s1_a, _ = stratified_split(dataset, 0.5, seed=42)
    s1_b, _ = stratified_split(dataset, 0.5, seed=42)
    s1_c, _ = stratified_split(dataset, 0.5, seed=123)

    assert s1_a.indices == s1_b.indices
    assert s1_a.indices != s1_c.indices
