"""Integration smoke tests for WhiteBoxPIM with the redesigned all-in-one API."""

import numpy as np
import pytest

from amulet.distribution_inference.attacks.white_box_pim import WhiteBoxPIM


@pytest.fixture
def synthetic_data_factory():
    """Factory fixture returning (x_train, y_train, z_train, x_test, y_test, z_test).

    Labels are binary (0/1). Sensitive attribute matrix has 2 columns
    ('race', 'sex') of random binary integers, giving the attack something
    to filter on.
    """

    def _make(
        n_train: int = 600,
        n_test: int = 200,
        num_features: int = 4,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        x_train = rng.standard_normal((n_train, num_features)).astype(np.float32)
        y_train = rng.integers(0, 2, size=n_train)
        z_train = rng.integers(0, 2, size=(n_train, 2))
        x_test = rng.standard_normal((n_test, num_features)).astype(np.float32)
        y_test = rng.integers(0, 2, size=n_test)
        z_test = rng.integers(0, 2, size=(n_test, 2))
        return x_train, y_train, z_train, x_test, y_test, z_test

    return _make


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
