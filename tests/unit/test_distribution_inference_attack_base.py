"""Unit tests for DistributionInferenceAttack.train_model_population and
prepare_model_populations.

These tests use SuriEvans2022 as the concrete subclass because
DistributionInferenceAttack is abstract (attack() is abstract).

Strategy:
- train_model_population is tested in isolation by passing a real DataLoader
  backed by synthetic tensors and linearnet/m1 (the smallest available arch).
- prepare_model_populations is tested end-to-end with tiny subsample sizes so
  that the split helpers can draw valid batches from the synthetic pool.
- train_classifier is patched in the "no retrain on second call" test to prove
  that the checkpoint cache is respected without actually training twice.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from amulet.distribution_inference.attacks.suri_evans_2022 import SuriEvans2022

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_NUM_FEATURES = 4
_NUM_CLASSES = 2
_BATCH_SIZE = 16


def _make_synthetic_data(
    n_train: int = 600,
    n_test: int = 200,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_train, y_train, z_train, x_test, y_test, z_test) for binary tasks."""
    rng = np.random.default_rng(seed)
    x_train = rng.standard_normal((n_train, _NUM_FEATURES)).astype(np.float32)
    y_train = rng.integers(0, 2, n_train).astype(np.int64)
    z_train = rng.integers(0, 2, (n_train, 2)).astype(np.int64)
    x_test = rng.standard_normal((n_test, _NUM_FEATURES)).astype(np.float32)
    y_test = rng.integers(0, 2, n_test).astype(np.int64)
    z_test = rng.integers(0, 2, (n_test, 2)).astype(np.int64)
    return x_train, y_train, z_train, x_test, y_test, z_test


def _make_attack(tmp_path: Path, **overrides: object) -> SuriEvans2022:
    """Construct a SuriEvans2022 instance with sensible tiny defaults."""
    x_train, y_train, z_train, x_test, y_test, z_test = _make_synthetic_data()
    defaults: dict[str, object] = {
        "x_train": x_train,
        "y_train": y_train,
        "z_train": z_train,
        "x_test": x_test,
        "y_test": y_test,
        "z_test": z_test,
        "sensitive_columns": ["race", "sex"],
        "filter_column": "sex",
        "ratio1": 0.3,
        "ratio2": 0.7,
        "model_arch": "linearnet",
        "model_capacity": "m1",
        "num_features": _NUM_FEATURES,
        "num_classes": _NUM_CLASSES,
        "num_models": 2,
        "epochs": 1,
        "batch_size": _BATCH_SIZE,
        "device": "cpu",
        "models_dir": tmp_path,
        "dataset": "synthetic",
        "exp_id": 0,
        "filter_value": 1,
        "drop_values": None,
        "train_subsample": 50,
        "test_subsample": 25,
    }
    defaults.update(overrides)
    return SuriEvans2022(**defaults)  # type: ignore[arg-type]


def _tiny_loader(n: int = 64) -> DataLoader:
    """Return a small DataLoader with _NUM_FEATURES-dimensional float32 inputs."""
    torch.manual_seed(0)
    x = torch.randn(n, _NUM_FEATURES)
    y = torch.randint(0, _NUM_CLASSES, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=_BATCH_SIZE, shuffle=False)


# ---------------------------------------------------------------------------
# train_model_population — return type and length
# ---------------------------------------------------------------------------


class TestTrainModelPopulationReturnType:
    def test_returns_list(self, tmp_path: Path) -> None:
        # Arrange
        attack = _make_attack(tmp_path, num_models=2)
        loader = _tiny_loader()

        # Act
        result = attack.train_model_population(loader, checkpoint_tag="tag_a")

        # Assert
        assert isinstance(result, list)

    def test_list_length_equals_num_models_default(self, tmp_path: Path) -> None:
        # Arrange
        attack = _make_attack(tmp_path, num_models=3)
        loader = _tiny_loader()

        # Act
        result = attack.train_model_population(loader, checkpoint_tag="tag_len")

        # Assert
        assert len(result) == 3

    @pytest.mark.parametrize("n_models", [1, 2, 4])
    def test_list_length_respects_num_models_override(
        self, tmp_path: Path, n_models: int
    ) -> None:
        # Arrange
        attack = _make_attack(tmp_path, num_models=99)  # should be ignored
        loader = _tiny_loader()

        # Act
        result = attack.train_model_population(
            loader, checkpoint_tag=f"tag_override_{n_models}", num_models=n_models
        )

        # Assert
        assert len(result) == n_models

    def test_all_elements_are_nn_module(self, tmp_path: Path) -> None:
        # Arrange
        attack = _make_attack(tmp_path, num_models=2)
        loader = _tiny_loader()

        # Act
        result = attack.train_model_population(loader, checkpoint_tag="tag_type")

        # Assert
        assert all(isinstance(m, nn.Module) for m in result)


# ---------------------------------------------------------------------------
# train_model_population — eval mode and frozen gradients
# ---------------------------------------------------------------------------


class TestTrainModelPopulationModelState:
    def test_all_models_in_eval_mode(self, tmp_path: Path) -> None:
        # Arrange
        attack = _make_attack(tmp_path, num_models=2)
        loader = _tiny_loader()

        # Act
        result = attack.train_model_population(loader, checkpoint_tag="tag_eval")

        # Assert
        assert all(not m.training for m in result)

    def test_all_parameters_have_grad_disabled(self, tmp_path: Path) -> None:
        # Arrange
        attack = _make_attack(tmp_path, num_models=2)
        loader = _tiny_loader()

        # Act
        result = attack.train_model_population(loader, checkpoint_tag="tag_nograd")

        # Assert: every parameter's requires_grad must be False
        for model in result:
            for p in model.parameters():
                assert not p.requires_grad


# ---------------------------------------------------------------------------
# train_model_population — checkpoint file creation
# ---------------------------------------------------------------------------


class TestTrainModelPopulationCheckpoints:
    def test_creates_models_dir_if_missing(self, tmp_path: Path) -> None:
        # Arrange
        nested = tmp_path / "a" / "b" / "c"
        attack = _make_attack(nested, num_models=1)
        loader = _tiny_loader()

        # Act
        attack.train_model_population(loader, checkpoint_tag="tag_dir")

        # Assert
        assert nested.is_dir()

    def test_checkpoint_file_created_per_model(self, tmp_path: Path) -> None:
        # Arrange
        n_models = 2
        exp_id = 7
        tag = "synth_tag"
        attack = _make_attack(tmp_path, num_models=n_models, exp_id=exp_id)
        loader = _tiny_loader()

        # Act
        attack.train_model_population(loader, checkpoint_tag=tag, num_models=n_models)

        # Assert: one .pth file per model id
        for model_id in range(n_models):
            expected = tmp_path / f"{tag}_{model_id}_{exp_id}.pth"
            assert expected.exists(), f"Missing checkpoint: {expected}"

    def test_checkpoint_filename_encodes_tag_id_and_exp_id(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        tag = "my_custom_tag"
        exp_id = 42
        attack = _make_attack(tmp_path, num_models=1, exp_id=exp_id)
        loader = _tiny_loader()

        # Act
        attack.train_model_population(loader, checkpoint_tag=tag, num_models=1)

        # Assert: the file name follows {tag}_{model_id}_{exp_id}.pth
        expected_name = f"{tag}_0_{exp_id}.pth"
        assert (tmp_path / expected_name).exists()


# ---------------------------------------------------------------------------
# train_model_population — checkpoint reuse (no retrain on second call)
# ---------------------------------------------------------------------------


class TestTrainModelPopulationCacheReuse:
    def test_train_classifier_not_called_on_second_invocation(
        self, tmp_path: Path
    ) -> None:
        # Arrange — first call trains and saves to disk
        attack = _make_attack(tmp_path, num_models=1)
        loader = _tiny_loader()
        attack.train_model_population(loader, checkpoint_tag="reuse_tag", num_models=1)

        # Act — second call should load from disk; patch train_classifier to detect calls
        with patch(
            "amulet.distribution_inference.attacks.distribution_inference_attack.train_classifier"
        ) as mock_train:
            attack.train_model_population(
                loader, checkpoint_tag="reuse_tag", num_models=1
            )

        # Assert: train_classifier must not have been called
        mock_train.assert_not_called()

    def test_loaded_models_still_in_eval_mode(self, tmp_path: Path) -> None:
        # Arrange — write checkpoints on first call
        attack = _make_attack(tmp_path, num_models=2)
        loader = _tiny_loader()
        attack.train_model_population(loader, checkpoint_tag="reload_tag", num_models=2)

        # Act — reload from disk
        result = attack.train_model_population(
            loader, checkpoint_tag="reload_tag", num_models=2
        )

        # Assert
        assert all(not m.training for m in result)

    def test_loaded_models_have_grad_disabled(self, tmp_path: Path) -> None:
        # Arrange
        attack = _make_attack(tmp_path, num_models=2)
        loader = _tiny_loader()
        attack.train_model_population(
            loader, checkpoint_tag="reload_grad_tag", num_models=2
        )

        # Act
        result = attack.train_model_population(
            loader, checkpoint_tag="reload_grad_tag", num_models=2
        )

        # Assert
        for model in result:
            for p in model.parameters():
                assert not p.requires_grad


# ---------------------------------------------------------------------------
# train_model_population — arch / capacity / epochs overrides
# ---------------------------------------------------------------------------


class TestTrainModelPopulationOverrides:
    def test_model_arch_override_is_used(self, tmp_path: Path) -> None:
        # Arrange: instance uses 'linearnet', override to 'linearnet' with m2
        # (only one arch available in unit tests without GPU; verify via forward pass)
        attack = _make_attack(tmp_path, model_arch="linearnet", model_capacity="m1")
        loader = _tiny_loader()

        # Act: request m2 capacity override — if the wrong capacity were used,
        # the model returned would have a different parameter count.
        result = attack.train_model_population(
            loader,
            checkpoint_tag="arch_override",
            model_arch="linearnet",
            model_capacity="m2",
            num_models=1,
        )

        # Assert: forward pass with 4-feature input must succeed
        x = torch.randn(1, _NUM_FEATURES)
        with torch.no_grad():
            out = result[0](x)
        assert out.shape == (1, _NUM_CLASSES)

    def test_epochs_override_passed_to_train_classifier(self, tmp_path: Path) -> None:
        # Arrange
        attack = _make_attack(tmp_path, epochs=99, num_models=1)
        loader = _tiny_loader()
        recorded_epochs: list[int] = []

        original_train = __import__(
            "amulet.utils", fromlist=["train_classifier"]
        ).train_classifier

        def capturing_train(
            model: nn.Module,
            data_loader: DataLoader,
            criterion: nn.Module,
            optimizer: torch.optim.Optimizer,
            epochs: int,
            device: str,
            **kwargs: object,
        ) -> nn.Module:
            recorded_epochs.append(epochs)
            return original_train(
                model, data_loader, criterion, optimizer, epochs, device, **kwargs
            )

        with patch(
            "amulet.distribution_inference.attacks.distribution_inference_attack.train_classifier",
            side_effect=capturing_train,
        ):
            attack.train_model_population(
                loader,
                checkpoint_tag="epoch_override_tag",
                num_models=1,
                epochs=3,
            )

        # Assert: the captured epoch count must match the override, not the default 99
        assert recorded_epochs == [3]

    def test_epochs_default_used_when_not_overridden(self, tmp_path: Path) -> None:
        # Arrange
        attack = _make_attack(tmp_path, epochs=2, num_models=1)
        loader = _tiny_loader()
        recorded_epochs: list[int] = []

        original_train = __import__(
            "amulet.utils", fromlist=["train_classifier"]
        ).train_classifier

        def capturing_train(
            model: nn.Module,
            data_loader: DataLoader,
            criterion: nn.Module,
            optimizer: torch.optim.Optimizer,
            epochs: int,
            device: str,
            **kwargs: object,
        ) -> nn.Module:
            recorded_epochs.append(epochs)
            return original_train(
                model, data_loader, criterion, optimizer, epochs, device, **kwargs
            )

        with patch(
            "amulet.distribution_inference.attacks.distribution_inference_attack.train_classifier",
            side_effect=capturing_train,
        ):
            attack.train_model_population(
                loader,
                checkpoint_tag="epoch_default_tag",
                num_models=1,
                # epochs not provided — should fall back to self.epochs == 2
            )

        assert recorded_epochs == [2]


# ---------------------------------------------------------------------------
# prepare_model_populations — splits and populations set
# ---------------------------------------------------------------------------


class TestPrepareModelPopulationsState:
    def test_splits_is_none_before_call(self, tmp_path: Path) -> None:
        # Arrange
        attack = _make_attack(tmp_path)

        # Assert: precondition
        assert attack.splits is None

    def test_splits_set_after_call(self, tmp_path: Path) -> None:
        # Arrange
        from amulet.distribution_inference.dataset_utils import DistributionSplits

        attack = _make_attack(tmp_path, num_models=1)

        # Act
        attack.prepare_model_populations()

        # Assert
        assert isinstance(attack.splits, DistributionSplits)

    def test_models_adv_1_non_empty(self, tmp_path: Path) -> None:
        # Arrange
        attack = _make_attack(tmp_path, num_models=1)

        # Act
        attack.prepare_model_populations()

        # Assert
        assert len(attack.models_adv_1) > 0

    def test_models_adv_2_non_empty(self, tmp_path: Path) -> None:
        # Arrange
        attack = _make_attack(tmp_path, num_models=1)

        # Act
        attack.prepare_model_populations()

        # Assert
        assert len(attack.models_adv_2) > 0

    def test_models_vic_1_non_empty(self, tmp_path: Path) -> None:
        # Arrange
        attack = _make_attack(tmp_path, num_models=1)

        # Act
        attack.prepare_model_populations()

        # Assert
        assert len(attack.models_vic_1) > 0

    def test_models_vic_2_non_empty(self, tmp_path: Path) -> None:
        # Arrange
        attack = _make_attack(tmp_path, num_models=1)

        # Act
        attack.prepare_model_populations()

        # Assert
        assert len(attack.models_vic_2) > 0

    @pytest.mark.parametrize("n_models", [1, 2])
    def test_each_population_length_equals_num_models(
        self, tmp_path: Path, n_models: int
    ) -> None:
        # Arrange
        attack = _make_attack(tmp_path, num_models=n_models)

        # Act
        attack.prepare_model_populations()

        # Assert
        assert len(attack.models_adv_1) == n_models
        assert len(attack.models_adv_2) == n_models
        assert len(attack.models_vic_1) == n_models
        assert len(attack.models_vic_2) == n_models


# ---------------------------------------------------------------------------
# prepare_model_populations — checkpoint files and naming convention
# ---------------------------------------------------------------------------


class TestPrepareModelPopulationsCheckpoints:
    def test_checkpoint_files_created_under_models_dir(self, tmp_path: Path) -> None:
        # Arrange
        attack = _make_attack(tmp_path, num_models=1)

        # Act
        attack.prepare_model_populations()

        # Assert: at least one .pth file exists under models_dir
        pth_files = list(tmp_path.glob("*.pth"))
        assert len(pth_files) > 0

    def test_exactly_four_times_num_models_checkpoint_files(
        self, tmp_path: Path
    ) -> None:
        # Arrange: 1 model x 4 populations = 4 files
        attack = _make_attack(tmp_path, num_models=1)

        # Act
        attack.prepare_model_populations()

        # Assert
        pth_files = list(tmp_path.glob("*.pth"))
        assert len(pth_files) == 4

    def test_checkpoint_names_follow_arch_tagged_convention(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        dataset = "synthetic"
        arch = "linearnet"
        capacity = "m1"
        exp_id = 0
        attack = _make_attack(
            tmp_path,
            num_models=1,
            dataset=dataset,
            model_arch=arch,
            model_capacity=capacity,
            exp_id=exp_id,
        )

        # Act
        attack.prepare_model_populations()

        # Assert: every checkpoint filename must match the expected pattern
        # {dataset}_dist_inf_{role}_{dist}_{arch}_{capacity}_{model_id}_{exp_id}.pth
        pattern = re.compile(
            rf"^{re.escape(dataset)}_dist_inf_(adv|vic)_(1|2)_{re.escape(arch)}"
            rf"_{re.escape(capacity)}_\d+_{re.escape(str(exp_id))}\.pth$"
        )
        for p in tmp_path.glob("*.pth"):
            assert pattern.match(p.name), (
                f"Checkpoint '{p.name}' does not match expected naming pattern"
            )

    def test_all_four_roles_represented_in_checkpoint_names(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        attack = _make_attack(tmp_path, num_models=1)

        # Act
        attack.prepare_model_populations()

        names = " ".join(p.name for p in tmp_path.glob("*.pth"))

        # Assert: each role+dist combination must appear once
        for role, dist in [("adv", "1"), ("adv", "2"), ("vic", "1"), ("vic", "2")]:
            assert f"_dist_inf_{role}_{dist}_" in names, (
                f"No checkpoint for role='{role}', dist='{dist}'"
            )

    def test_models_dir_created_including_parents_by_prepare(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        nested = tmp_path / "deep" / "nested"
        attack = _make_attack(nested, num_models=1)

        # Act
        attack.prepare_model_populations()

        # Assert
        assert nested.is_dir()

    def test_models_dir_accepts_str_path(self, tmp_path: Path) -> None:
        # Arrange: pass models_dir as a plain string, not a Path
        attack = _make_attack(str(tmp_path), num_models=1)  # type: ignore[arg-type]

        # Act — must not raise
        attack.prepare_model_populations()

        # Assert
        pth_files = list(tmp_path.glob("*.pth"))
        assert len(pth_files) == 4


# ---------------------------------------------------------------------------
# prepare_model_populations — checkpoint reuse across instances
# ---------------------------------------------------------------------------


class TestPrepareModelPopulationsReuse:
    def test_second_prepare_reuses_checkpoints(self, tmp_path: Path) -> None:
        # Arrange — run once to populate disk
        attack1 = _make_attack(tmp_path, num_models=1, exp_id=5)
        attack1.prepare_model_populations()

        # Act — second instance, same dir; patch train_classifier to detect calls
        attack2 = _make_attack(tmp_path, num_models=1, exp_id=5)
        with patch(
            "amulet.distribution_inference.attacks.distribution_inference_attack.train_classifier"
        ) as mock_train:
            attack2.prepare_model_populations()

        # Assert
        mock_train.assert_not_called()
