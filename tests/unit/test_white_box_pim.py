"""Unit tests for WhiteBoxPIM that need no model training.

Training smokes (prepare_model_populations + attack) live in
tests/integration/test_whitebox_pim.py; this file holds the guard checks,
which fire before any meta-classifier training.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from amulet.distribution_inference.attacks.white_box_pim import WhiteBoxPIM


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


class _NonMatrixLinear(nn.Module):
    """Module whose extracted layer parameters are 3-D.

    get_layer_parameters flattens Conv2d kernels to [out, in*k*k] and appends
    biases as an extra column, so every well-formed Linear/Conv2d model yields
    2-D layer tensors. Overwriting a bias-free Linear's weight with a 3-D
    tensor pushes a malformed tensor through the real extraction path, which
    is what the attack() shape guard defends against.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 2, bias=False)
        self.lin.weight = nn.Parameter(torch.zeros(2, 4, 1))


def test_whitebox_pim_attack_rejects_non_2d_layer_parameters(tmp_path):
    """attack() raises the documented ValueError when a population model's
    extracted layer parameters are not 2-D. The guard fires while collecting
    layer shapes, before the meta-classifier exists, so an assigned adversary
    population is the cheapest real trigger path."""
    attack = _make_attack(tmp_path)
    attack.models_adv_1 = [_NonMatrixLinear()]

    with pytest.raises(ValueError, match="Expected 2-D layer parameter"):
        attack.attack()
