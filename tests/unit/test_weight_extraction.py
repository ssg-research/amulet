import pytest
import torch
import torch.nn as nn

from amulet.utils.__weight_extraction import get_layer_parameters


def test_extract_linear_with_bias():
    model = nn.Linear(4, 8)

    params = get_layer_parameters(model)

    assert len(params) == 1
    # [out_features, in_features + 1] -> [8, 5]
    assert params[0].shape == (8, 5)
    # Check bias is in the last column
    assert torch.equal(params[0][:, -1], model.bias.data)


def test_extract_linear_no_bias():
    model = nn.Linear(4, 8, bias=False)

    params = get_layer_parameters(model)

    assert len(params) == 1
    # [out_features, in_features] -> [8, 4]
    assert params[0].shape == (8, 4)


def test_extract_conv2d_with_bias():
    # [out=2, in=3, k=3, k=3]
    model = nn.Conv2d(3, 2, kernel_size=3)

    params = get_layer_parameters(model)

    assert len(params) == 1
    # [out, in*k*k + 1] -> [2, 3*3*3 + 1] -> [2, 28]
    assert params[0].shape == (2, 28)


def test_extract_mixed_and_ignore_layers():
    class MixedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3)
            self.bn = nn.BatchNorm2d(16)
            self.relu = nn.ReLU()
            self.fc = nn.Linear(16, 10)

        def forward(self, x):
            return self.fc(self.relu(self.bn(self.conv(x))))

    model = MixedModel()

    params = get_layer_parameters(model)

    assert len(params) == 2  # Only Conv and Linear
    assert params[0].shape == (16, 3 * 3 * 3 + 1)
    assert params[1].shape == (10, 16 + 1)


def test_extract_dataparallel_unwrap():
    inner_model = nn.Linear(4, 2)
    model = nn.DataParallel(inner_model)

    params = get_layer_parameters(model)

    assert len(params) == 1
    assert params[0].shape == (2, 5)
    # Ensure it's the same data
    assert torch.equal(params[0][:, :4], inner_model.weight.data.cpu())


@pytest.mark.parametrize("in_f, out_f", [(4, 8), (10, 2)])
def test_extract_linear_parameterized(in_f, out_f):
    model = nn.Linear(in_f, out_f)

    params = get_layer_parameters(model)

    assert params[0].shape == (out_f, in_f + 1)


def test_extract_empty_model():
    # No Linear or Conv layers
    model = nn.Sequential(nn.ReLU(), nn.Sigmoid())

    params = get_layer_parameters(model)

    assert isinstance(params, list)
    assert len(params) == 0
