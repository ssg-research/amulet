"""Unit tests for WhiteBoxPIM that need no model training.

Training smokes (prepare_model_populations + attack) live in
tests/integration/test_whitebox_pim.py; this file holds the precondition
check, which never gets past construction.
"""

import numpy as np
import pytest

from amulet.distribution_inference.attacks.white_box_pim import WhiteBoxPIM


def test_whitebox_pim_attack_requires_prepare_first(tmp_path):
    """Calling attack() before prepare_model_populations() must raise RuntimeError."""
    rng = np.random.default_rng(0)
    n, num_features = 20, 4
    x = rng.standard_normal((n, num_features)).astype(np.float32)
    y = rng.integers(0, 2, size=n)
    z = rng.integers(0, 2, size=(n, 2))

    attack = WhiteBoxPIM(
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

    with pytest.raises(RuntimeError, match="prepare_model_populations"):
        attack.attack()
