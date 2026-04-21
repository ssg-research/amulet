"""Integration smoke tests for WhiteBoxPIM with the redesigned all-in-one API."""

import numpy as np
import pytest

from amulet.distribution_inference.attacks.white_box_pim import WhiteBoxPIM

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_synthetic_data(
    n_train: int = 600,
    n_test: int = 200,
    num_features: int = 4,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_train, y_train, z_train, x_test, y_test, z_test).

    Labels are binary (0/1). Sensitive attribute matrix has 2 columns
    ('race', 'sex') of random binary integers, giving the attack something
    to filter on.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    x_train = rng.standard_normal((n_train, num_features)).astype(np.float32)
    y_train = rng.integers(0, 2, size=n_train)
    z_train = rng.integers(0, 2, size=(n_train, 2))

    x_test = rng.standard_normal((n_test, num_features)).astype(np.float32)
    y_test = rng.integers(0, 2, size=n_test)
    z_test = rng.integers(0, 2, size=(n_test, 2))

    return x_train, y_train, z_train, x_test, y_test, z_test


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_whitebox_pim_smoke(tmp_path):
    """Full lifecycle: construct → prepare_model_populations → attack."""
    # Arrange
    rng = np.random.default_rng(42)
    x_train, y_train, z_train, x_test, y_test, z_test = _make_synthetic_data(
        n_train=600, n_test=200, rng=rng
    )

    attack = WhiteBoxPIM(
        x_train=x_train,
        y_train=y_train,
        z_train=z_train,
        x_test=x_test,
        y_test=y_test,
        z_test=z_test,
        sensitive_columns=["race", "sex"],
        filter_column="sex",
        ratio1=0.1,
        ratio2=0.9,
        model_arch="linearnet",
        model_capacity="m1",
        num_features=4,
        num_classes=2,
        num_models=1,
        epochs=1,
        batch_size=16,
        device="cpu",
        models_dir=tmp_path,
        dataset="synthetic",
        exp_id=0,
        filter_value=1,
        drop_values=None,
        train_subsample=50,
        test_subsample=25,
        meta_epochs=2,
        lr=1e-2,
    )

    # Act
    attack.prepare_model_populations()
    results = attack.attack()

    # Assert — required keys are present
    assert "predictions" in results
    assert "ground_truth" in results


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_whitebox_pim_output_lengths_match(tmp_path):
    """predictions and ground_truth must have the same length."""
    rng = np.random.default_rng(1)
    x_train, y_train, z_train, x_test, y_test, z_test = _make_synthetic_data(rng=rng)

    attack = WhiteBoxPIM(
        x_train=x_train,
        y_train=y_train,
        z_train=z_train,
        x_test=x_test,
        y_test=y_test,
        z_test=z_test,
        sensitive_columns=["race", "sex"],
        filter_column="sex",
        ratio1=0.1,
        ratio2=0.9,
        model_arch="linearnet",
        model_capacity="m1",
        num_features=4,
        num_classes=2,
        num_models=1,
        epochs=1,
        batch_size=16,
        device="cpu",
        models_dir=tmp_path,
        dataset="synthetic",
        train_subsample=50,
        test_subsample=25,
        meta_epochs=2,
        lr=1e-2,
    )

    attack.prepare_model_populations()
    results = attack.attack()

    # Act + Assert
    assert len(results["predictions"]) == len(results["ground_truth"])


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_whitebox_pim_nonempty_outputs(tmp_path):
    """Both output arrays must contain at least one element."""
    rng = np.random.default_rng(2)
    x_train, y_train, z_train, x_test, y_test, z_test = _make_synthetic_data(rng=rng)

    attack = WhiteBoxPIM(
        x_train=x_train,
        y_train=y_train,
        z_train=z_train,
        x_test=x_test,
        y_test=y_test,
        z_test=z_test,
        sensitive_columns=["race", "sex"],
        filter_column="sex",
        ratio1=0.1,
        ratio2=0.9,
        model_arch="linearnet",
        model_capacity="m1",
        num_features=4,
        num_classes=2,
        num_models=1,
        epochs=1,
        batch_size=16,
        device="cpu",
        models_dir=tmp_path,
        dataset="synthetic",
        train_subsample=50,
        test_subsample=25,
        meta_epochs=2,
        lr=1e-2,
    )

    attack.prepare_model_populations()
    results = attack.attack()

    assert len(results["predictions"]) > 0
    assert len(results["ground_truth"]) > 0


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_whitebox_pim_predictions_in_unit_interval(tmp_path):
    """All prediction scores must lie in [0, 1] (sigmoid output)."""
    rng = np.random.default_rng(3)
    x_train, y_train, z_train, x_test, y_test, z_test = _make_synthetic_data(rng=rng)

    attack = WhiteBoxPIM(
        x_train=x_train,
        y_train=y_train,
        z_train=z_train,
        x_test=x_test,
        y_test=y_test,
        z_test=z_test,
        sensitive_columns=["race", "sex"],
        filter_column="sex",
        ratio1=0.1,
        ratio2=0.9,
        model_arch="linearnet",
        model_capacity="m1",
        num_features=4,
        num_classes=2,
        num_models=1,
        epochs=1,
        batch_size=16,
        device="cpu",
        models_dir=tmp_path,
        dataset="synthetic",
        train_subsample=50,
        test_subsample=25,
        meta_epochs=2,
        lr=1e-2,
    )

    attack.prepare_model_populations()
    results = attack.attack()

    preds = results["predictions"]
    assert (preds >= 0.0).all(), "Some predictions are below 0"
    assert (preds <= 1.0).all(), "Some predictions are above 1"


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_whitebox_pim_ground_truth_binary(tmp_path):
    """ground_truth must contain only 0s and 1s (dist-1 vs dist-2 victims)."""
    rng = np.random.default_rng(4)
    x_train, y_train, z_train, x_test, y_test, z_test = _make_synthetic_data(rng=rng)

    attack = WhiteBoxPIM(
        x_train=x_train,
        y_train=y_train,
        z_train=z_train,
        x_test=x_test,
        y_test=y_test,
        z_test=z_test,
        sensitive_columns=["race", "sex"],
        filter_column="sex",
        ratio1=0.1,
        ratio2=0.9,
        model_arch="linearnet",
        model_capacity="m1",
        num_features=4,
        num_classes=2,
        num_models=1,
        epochs=1,
        batch_size=16,
        device="cpu",
        models_dir=tmp_path,
        dataset="synthetic",
        train_subsample=50,
        test_subsample=25,
        meta_epochs=2,
        lr=1e-2,
    )

    attack.prepare_model_populations()
    results = attack.attack()

    gt = results["ground_truth"]
    unique_vals = set(np.unique(gt).tolist())
    assert unique_vals.issubset({0, 1}), (
        f"Unexpected ground_truth values: {unique_vals}"
    )


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_whitebox_pim_attack_requires_prepare_first(tmp_path):
    """Calling attack() before prepare_model_populations() must raise RuntimeError."""
    rng = np.random.default_rng(5)
    x_train, y_train, z_train, x_test, y_test, z_test = _make_synthetic_data(rng=rng)

    attack = WhiteBoxPIM(
        x_train=x_train,
        y_train=y_train,
        z_train=z_train,
        x_test=x_test,
        y_test=y_test,
        z_test=z_test,
        sensitive_columns=["race", "sex"],
        filter_column="sex",
        ratio1=0.1,
        ratio2=0.9,
        model_arch="linearnet",
        model_capacity="m1",
        num_features=4,
        num_classes=2,
        num_models=1,
        epochs=1,
        batch_size=16,
        device="cpu",
        models_dir=tmp_path,
        dataset="synthetic",
        train_subsample=50,
        test_subsample=25,
        meta_epochs=2,
        lr=1e-2,
    )

    # Act + Assert — must raise before populations are prepared
    with pytest.raises(RuntimeError, match="prepare_model_populations"):
        attack.attack()


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_whitebox_pim_models_dir_created(tmp_path):
    """prepare_model_populations() must create models_dir if it does not exist."""
    rng = np.random.default_rng(6)
    x_train, y_train, z_train, x_test, y_test, z_test = _make_synthetic_data(rng=rng)

    nested_dir = tmp_path / "new" / "nested" / "dir"

    attack = WhiteBoxPIM(
        x_train=x_train,
        y_train=y_train,
        z_train=z_train,
        x_test=x_test,
        y_test=y_test,
        z_test=z_test,
        sensitive_columns=["race", "sex"],
        filter_column="sex",
        ratio1=0.1,
        ratio2=0.9,
        model_arch="linearnet",
        model_capacity="m1",
        num_features=4,
        num_classes=2,
        num_models=1,
        epochs=1,
        batch_size=16,
        device="cpu",
        models_dir=nested_dir,
        dataset="synthetic",
        train_subsample=50,
        test_subsample=25,
        meta_epochs=2,
        lr=1e-2,
    )

    attack.prepare_model_populations()

    assert nested_dir.is_dir()


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_whitebox_pim_checkpoint_reuse(tmp_path):
    """Running prepare_model_populations() twice reuses checkpoints (no retraining)."""
    rng = np.random.default_rng(7)
    x_train, y_train, z_train, x_test, y_test, z_test = _make_synthetic_data(rng=rng)

    common_kwargs: dict[str, object] = {
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
        "exp_id": 99,
        "train_subsample": 50,
        "test_subsample": 25,
        "meta_epochs": 2,
        "lr": 1e-2,
    }

    # First run — trains and saves checkpoints
    attack1 = WhiteBoxPIM(**common_kwargs)  # type: ignore[arg-type]
    attack1.prepare_model_populations()

    # Checkpoints should now exist on disk
    checkpoints = list(tmp_path.glob("*.pth"))
    assert len(checkpoints) > 0, "No checkpoint files were written"

    # Second run — should load from disk without errors
    attack2 = WhiteBoxPIM(**common_kwargs)  # type: ignore[arg-type]
    attack2.prepare_model_populations()
    results = attack2.attack()

    assert "predictions" in results
    assert "ground_truth" in results
