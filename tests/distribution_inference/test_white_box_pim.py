"""Tests for WhiteBoxPIM and its _get_layer_parameters helper.

Guard checks and the parameter-formatting contract train nothing and run in
the fast tier; the lifecycle smokes and the reproducibility test train tiny
model populations and carry @pytest.mark.integration.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from amulet.distribution_inference.attacks.white_box_pim import (
    WhiteBoxPIM,
    _get_layer_parameters,  # type: ignore[reportPrivateUsage]
)


@pytest.fixture
def whitebox_attack_factory(tmp_path, synthetic_data_factory):
    """Factory fixture returning a WhiteBoxPIM with tiny defaults; accepts overrides."""

    def _make(seed: int = 0, **overrides: object) -> WhiteBoxPIM:
        x_train, y_train, z_train, x_test, y_test, z_test = synthetic_data_factory(
            seed=seed
        )
        defaults: dict[str, object] = {
            "x_train": x_train,
            "y_train": y_train,
            "z_train": z_train,
            "x_test": x_test,
            "y_test": y_test,
            "z_test": z_test,
            "sensitive_columns": ["race", "sex"],
            "filter_column": "sex",
            "ratio1": 0.1,
            "ratio2": 0.9,
            "model_arch": "linearnet",
            "model_capacity": "m1",
            "num_features": 4,
            "num_classes": 2,
            "num_models": 1,
            "epochs": 1,
            "batch_size": 16,
            "device": "cpu",
            "models_dir": tmp_path,
            "dataset": "synthetic",
            "train_subsample": 50,
            "test_subsample": 25,
            "meta_epochs": 2,
            "lr": 1e-2,
        }
        defaults.update(overrides)
        return WhiteBoxPIM(**defaults)  # type: ignore[arg-type]

    return _make


# ---------------------------------------------------------------------------
# Guard checks — no training
# ---------------------------------------------------------------------------


def test_whitebox_pim_attack_requires_prepare_first(whitebox_attack_factory):
    """Calling attack() before prepare_model_populations() must raise RuntimeError."""
    attack = whitebox_attack_factory()

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


# ---------------------------------------------------------------------------
# Lifecycle smokes — train tiny model populations
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_whitebox_pim_smoke(whitebox_attack_factory):
    """Full lifecycle: construct → prepare_model_populations → attack."""
    attack = whitebox_attack_factory(
        seed=42, exp_id=0, filter_value=1, drop_values=None
    )

    attack.prepare_model_populations()
    results = attack.attack()

    assert "predictions" in results
    assert "ground_truth" in results


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_whitebox_pim_output_lengths_match(whitebox_attack_factory):
    """predictions and ground_truth must have the same length."""
    attack = whitebox_attack_factory(seed=1)
    attack.prepare_model_populations()
    results = attack.attack()

    assert len(results["predictions"]) == len(results["ground_truth"])


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_whitebox_pim_nonempty_outputs(whitebox_attack_factory):
    """Both output arrays must contain at least one element."""
    attack = whitebox_attack_factory(seed=2)
    attack.prepare_model_populations()
    results = attack.attack()

    assert len(results["predictions"]) > 0
    assert len(results["ground_truth"]) > 0


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_whitebox_pim_predictions_in_unit_interval(whitebox_attack_factory):
    """All prediction scores must lie in [0, 1] (sigmoid output)."""
    attack = whitebox_attack_factory(seed=3)
    attack.prepare_model_populations()
    results = attack.attack()

    preds = results["predictions"]
    assert (preds >= 0.0).all(), "Some predictions are below 0"
    assert (preds <= 1.0).all(), "Some predictions are above 1"


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_whitebox_pim_ground_truth_binary(whitebox_attack_factory):
    """ground_truth must contain only 0s and 1s (dist-1 vs dist-2 victims)."""
    attack = whitebox_attack_factory(seed=4)
    attack.prepare_model_populations()
    results = attack.attack()

    gt = results["ground_truth"]
    unique_vals = set(np.unique(gt).tolist())
    assert unique_vals.issubset({0, 1}), (
        f"Unexpected ground_truth values: {unique_vals}"
    )


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_whitebox_pim_models_dir_created(tmp_path, whitebox_attack_factory):
    """prepare_model_populations() must create models_dir if it does not exist."""
    nested_dir = tmp_path / "new" / "nested" / "dir"
    attack = whitebox_attack_factory(seed=6, models_dir=nested_dir)

    attack.prepare_model_populations()

    assert nested_dir.is_dir()


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_whitebox_pim_checkpoint_reuse(tmp_path, whitebox_attack_factory):
    """Running prepare_model_populations() twice reuses checkpoints (no retraining)."""
    # First run — trains and saves checkpoints
    attack1 = whitebox_attack_factory(seed=7, exp_id=99)
    attack1.prepare_model_populations()

    # Checkpoints should now exist on disk
    checkpoints = list(tmp_path.glob("*.pth"))
    assert len(checkpoints) > 0, "No checkpoint files were written"

    # Second run — should load from disk without errors
    attack2 = whitebox_attack_factory(seed=7, exp_id=99)
    attack2.prepare_model_populations()
    results = attack2.attack()

    assert "predictions" in results
    assert "ground_truth" in results


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_whitebox_pim_reproducible(tmp_path, whitebox_attack_factory, seed_everything):
    """Two full runs — each into its own empty checkpoint dir so the second
    cannot load the first's population models — yield identical predictions.

    Population training (initialize_model + Adam) draws from the global torch
    RNG, so seed_everything before each run makes the retrained populations
    identical; random_seed fixes the PIM meta-classifier's own training.
    """

    def run(subdir: str) -> np.ndarray:
        seed_everything(5)
        attack = whitebox_attack_factory(
            seed=0, random_seed=123, models_dir=tmp_path / subdir
        )
        attack.prepare_model_populations()
        return attack.attack()["predictions"]

    np.testing.assert_array_equal(run("run_a"), run("run_b"))
