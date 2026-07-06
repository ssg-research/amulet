"""Tests for the pipeline helpers in amulet/utils/__pipeline.py:
create_dir, initialize_model, and stratified_split."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from amulet.models import VGG, AmuletModel, LinearNet, ResNet, SimpleCNN
from amulet.utils.__pipeline import create_dir, initialize_model, stratified_split

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
