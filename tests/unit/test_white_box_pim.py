"""Unit tests for WhiteBoxPIM that need no model training.

Training smokes (prepare_model_populations + attack) live in
tests/integration/test_whitebox_pim.py; this file holds the guard checks and
the _get_layer_parameters formatting contract, none of which train anything.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from amulet.distribution_inference.attacks.white_box_pim import (
    WhiteBoxPIM,
    _get_layer_parameters,  # type: ignore[reportPrivateUsage]
)


def _make_attack(tmp_path) -> WhiteBoxPIM:
    """WhiteBoxPIM on tiny synthetic arrays; cheap because nothing trains."""
    rng = np.random.default_rng(0)
    n, num_features = 20, 4
    x = rng.standard_normal((n, num_features)).astype(np.float32)
    y = rng.integers(0, 2, size=n)
    z = rng.integers(0, 2, size=(n, 2))

    return WhiteBoxPIM(
        x_train=x,
        y_train=y,
        z_train=z,
        x_test=x,
        y_test=y,
        z_test=z,
        sensitive_columns=["race", "sex"],
        filter_column="sex",
        ratio1=0.1,
        ratio2=0.9,
        model_arch="linearnet",
        model_capacity="m1",
        num_features=num_features,
        num_classes=2,
        num_models=1,
        epochs=1,
        batch_size=8,
        device="cpu",
        models_dir=tmp_path,
        dataset="synthetic",
    )


def test_whitebox_pim_attack_requires_prepare_first(tmp_path):
    """Calling attack() before prepare_model_populations() must raise RuntimeError."""
    attack = _make_attack(tmp_path)

    with pytest.raises(RuntimeError, match="prepare_model_populations"):
        attack.attack()


# ---------------------------------------------------------------------------
# _get_layer_parameters — the PIM input-formatting contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("in_f, out_f", [(4, 8), (10, 2)])
def test_extract_linear_with_bias(in_f, out_f):
    model = nn.Linear(in_f, out_f)

    params = _get_layer_parameters(model)

    assert len(params) == 1
    # [out_features, in_features + 1]; bias occupies the last column
    assert params[0].shape == (out_f, in_f + 1)
    assert torch.equal(params[0][:, -1], model.bias.data)


def test_extract_linear_no_bias():
    model = nn.Linear(4, 8, bias=False)

    params = _get_layer_parameters(model)

    assert len(params) == 1
    assert params[0].shape == (8, 4)


def test_extract_conv2d_with_bias():
    # [out=2, in=3, k=3, k=3]
    model = nn.Conv2d(3, 2, kernel_size=3)

    params = _get_layer_parameters(model)

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

    params = _get_layer_parameters(model)

    assert len(params) == 2  # Only Conv and Linear
    assert params[0].shape == (16, 3 * 3 * 3 + 1)
    assert params[1].shape == (10, 16 + 1)


def test_extract_dataparallel_unwrap():
    inner_model = nn.Linear(4, 2)
    model = nn.DataParallel(inner_model)

    params = _get_layer_parameters(model)

    assert len(params) == 1
    assert params[0].shape == (2, 5)
    # Ensure it's the same data
    assert torch.equal(params[0][:, :4], inner_model.weight.data.cpu())


def test_extract_all_outputs_are_2d():
    """The PIM meta-classifier consumes [N_neurons, dim] tensors; every
    extracted layer must be 2-D regardless of architecture."""
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU(), nn.Linear(4, 2))

    params = _get_layer_parameters(model)

    assert all(p.ndim == 2 for p in params)


def test_extract_no_extractable_layers_raises():
    """A model with no Linear/Conv2d layers would otherwise surface as an
    opaque failure when the meta-classifier is built over an empty shape list."""
    model = nn.Sequential(nn.ReLU(), nn.Sigmoid())

    with pytest.raises(ValueError, match="no Linear or Conv2d layers"):
        _ = _get_layer_parameters(model)
