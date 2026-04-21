import pytest
import torch

from amulet.models import VGG, AmuletModel, LinearNet, ResNet, SimpleCNN
from amulet.utils.__pipeline import initialize_model


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
