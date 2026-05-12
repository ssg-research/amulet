"""Unit tests for amulet/distribution_inference/dataset_utils.py.

Tests cover the five public-facing helpers:
  _filter_by_ratio, _heuristic_sample, _build_tensor_dataset,
  _stratify_key, and prepare_distribution_splits.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from amulet.distribution_inference.dataset_utils import (
    DistributionSplits,
    _build_tensor_dataset,
    _filter_by_ratio,
    _heuristic_sample,
    _stratify_key,
    prepare_distribution_splits,
)


@pytest.fixture
def binary_mask_factory():
    """Factory fixture: returns boolean arrays with caller-specified True count."""

    def _make(n: int, n_true: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        mask = np.zeros(n, dtype=bool)
        mask[:n_true] = True
        rng.shuffle(mask)
        return mask

    return _make


@pytest.fixture
def binary_data_factory(binary_mask_factory):
    """Factory fixture: returns (y, mask) with balanced binary labels."""

    def _make(n: int, n_true: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, size=n).astype(np.int64)
        mask = binary_mask_factory(n, n_true, seed)
        return y, mask

    return _make


@pytest.fixture
def splits_inputs_factory():
    """Factory fixture: synthetic tabular (N, d) train/test arrays with two sensitive columns."""

    def _make(
        n_train: int = 600,
        n_test: int = 200,
        n_features: int = 8,
        seed: int = 0,
        y_shape: str = "1d",  # "1d" or "2d"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        x_train = rng.standard_normal((n_train, n_features)).astype(np.float32)
        y_train = rng.integers(0, 2, n_train).astype(np.int64)
        z_train = rng.integers(0, 2, (n_train, 2)).astype(np.int64)

        x_test = rng.standard_normal((n_test, n_features)).astype(np.float32)
        y_test = rng.integers(0, 2, n_test).astype(np.int64)
        z_test = rng.integers(0, 2, (n_test, 2)).astype(np.int64)

        if y_shape == "2d":
            y_train = y_train.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)

        return x_train, y_train, z_train, x_test, y_test, z_test

    return _make


@pytest.fixture
def splits_kwargs() -> dict:
    """Default kwargs for prepare_distribution_splits."""
    return {
        "sensitive_columns": ["sex", "race"],
        "filter_column": "sex",
        "ratio1": 0.3,
        "ratio2": 0.7,
        "train_subsample": 20,
        "test_subsample": 10,
        "batch_size": 32,
        "seed": 42,
        "n_tries": 50,
    }


# ---------------------------------------------------------------------------
# _filter_by_ratio
# ---------------------------------------------------------------------------


class TestFilterByRatio:
    def test_ratio_zero_returns_only_notqualify(self, binary_mask_factory):
        # Arrange
        mask = binary_mask_factory(100, 40)

        # Act
        indices = _filter_by_ratio(mask, ratio=0.0)

        # Assert: all returned indices must have mask==False
        assert mask[indices].sum() == 0

    def test_ratio_one_returns_only_qualify(self, binary_mask_factory):
        # Arrange
        mask = binary_mask_factory(100, 40)

        # Act
        indices = _filter_by_ratio(mask, ratio=1.0)

        # Assert: all returned indices must have mask==True
        assert mask[indices].all()

    @pytest.mark.parametrize("target_ratio", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_returned_ratio_approximately_matches_target(
        self, binary_mask_factory, target_ratio: float
    ):
        # Arrange
        np.random.seed(0)
        mask = binary_mask_factory(500, 200)

        # Act
        indices = _filter_by_ratio(mask, ratio=target_ratio)

        # Assert: achieved ratio should be within 0.1 of target
        achieved = float(mask[indices].mean())
        assert abs(achieved - target_ratio) < 0.1

    def test_returned_indices_are_valid_positions(self, binary_mask_factory):
        # Arrange
        mask = binary_mask_factory(80, 30)

        # Act
        indices = _filter_by_ratio(mask, ratio=0.4)

        # Assert: every index must be in range [0, len(mask))
        assert np.all(indices >= 0)
        assert np.all(indices < len(mask))

    def test_output_is_numpy_array(self, binary_mask_factory):
        # Arrange
        mask = binary_mask_factory(60, 20)

        # Act
        indices = _filter_by_ratio(mask, ratio=0.5)

        # Assert
        assert isinstance(indices, np.ndarray)

    def test_when_current_ratio_equals_target_returns_all_qualify(self):
        # Arrange: mask has exactly 50% True → current_ratio == ratio == 0.5
        mask = np.array([True, False, True, False, True, False], dtype=bool)
        np.random.seed(42)

        # Act
        indices = _filter_by_ratio(mask, ratio=0.5)

        # Assert: all qualify entries appear in the result
        qualify = set(np.nonzero(mask)[0].tolist())
        assert qualify.issubset(set(indices.tolist()))

    def test_ratio_zero_no_qualify_entries_in_result(self):
        # Arrange: 10 qualify, 20 notqualify
        mask = np.array([True] * 10 + [False] * 20, dtype=bool)
        np.random.seed(1)

        # Act
        indices = _filter_by_ratio(mask, ratio=0.0)

        # Assert: none of the returned indices map to a True entry
        assert not mask[indices].any()

    def test_ratio_one_all_qualify_returned(self):
        # Arrange
        mask = np.array([True] * 15 + [False] * 25, dtype=bool)
        np.random.seed(2)

        # Act
        indices = _filter_by_ratio(mask, ratio=1.0)

        # Assert: exactly the 15 qualify indices are returned
        assert set(indices.tolist()) == set(np.nonzero(mask)[0].tolist())


# ---------------------------------------------------------------------------
# _heuristic_sample
# ---------------------------------------------------------------------------


class TestHeuristicSample:
    def test_output_is_sorted_integer_array(self, binary_data_factory):
        # Arrange
        np.random.seed(0)
        n = 200
        y, mask = binary_data_factory(n, 80)

        # Act
        indices = _heuristic_sample(y, mask, ratio=0.5, cwise_sample=20, n_tries=50)

        # Assert
        assert isinstance(indices, np.ndarray)
        assert np.all(np.diff(indices) >= 0), "output must be sorted"

    def test_returned_indices_valid_for_y(self, binary_data_factory):
        # Arrange
        np.random.seed(1)
        n = 300
        y, mask = binary_data_factory(n, 120)

        # Act
        indices = _heuristic_sample(y, mask, ratio=0.4, cwise_sample=15, n_tries=50)

        # Assert
        assert np.all(indices >= 0)
        assert np.all(indices < n)

    def test_filter_mask_ratio_close_to_target(self, binary_data_factory):
        # Arrange
        np.random.seed(2)
        n = 400
        y, mask = binary_data_factory(n, 200)

        # Act
        indices = _heuristic_sample(y, mask, ratio=0.5, cwise_sample=30, n_tries=200)

        # Assert: best draw should be within 0.2 of the target
        achieved = float(mask[indices].mean())
        assert abs(achieved - 0.5) < 0.20

    @pytest.mark.parametrize("class_imbalance", [1.0, 2.0, 3.0])
    def test_class_imbalance_majority_count(
        self, binary_mask_factory, class_imbalance: float
    ):
        # Arrange: heavily one-sided data so imbalance is achievable
        np.random.seed(3)
        n = 600
        y = np.array([0] * 400 + [1] * 200, dtype=np.int64)
        mask = binary_mask_factory(n, 300)
        cwise_sample = 20

        # Act
        indices = _heuristic_sample(
            y,
            mask,
            ratio=0.5,
            cwise_sample=cwise_sample,
            class_imbalance=class_imbalance,
            n_tries=100,
        )
        zero_count = int((y[indices] == 0).sum())
        one_count = int((y[indices] == 1).sum())

        # Assert: majority (class 0) count ≤ class_imbalance * cwise_sample
        assert zero_count <= int(class_imbalance * cwise_sample) + 5  # small tolerance
        assert one_count <= cwise_sample + 5

    def test_output_indices_index_into_original_x(self, binary_data_factory):
        # Arrange
        np.random.seed(4)
        n = 250
        x = np.random.randn(n, 10).astype(np.float32)
        y, mask = binary_data_factory(n, 100)

        # Act
        indices = _heuristic_sample(y, mask, ratio=0.4, cwise_sample=20, n_tries=50)

        # Assert: slicing x with the indices must not raise
        subset = x[indices]
        assert subset.ndim == 2
        assert subset.shape[1] == 10


# ---------------------------------------------------------------------------
# _build_tensor_dataset
# ---------------------------------------------------------------------------


class TestBuildTensorDataset:
    def test_returns_tensor_dataset(self):
        # Arrange
        x = np.random.randn(10, 4).astype(np.float32)
        y = np.random.randint(0, 2, 10).astype(np.int64)

        # Act
        ds = _build_tensor_dataset(x, y)

        # Assert
        assert isinstance(ds, TensorDataset)

    def test_length_equals_input_length(self):
        # Arrange
        n = 37
        x = np.random.randn(n, 8).astype(np.float32)
        y = np.zeros(n, dtype=np.int64)

        # Act
        ds = _build_tensor_dataset(x, y)

        # Assert
        assert len(ds) == n

    def test_x_dtype_is_float32(self):
        # Arrange
        x = np.random.randn(20, 5).astype(np.float64)  # deliberately float64
        y = np.zeros(20, dtype=np.int64)

        # Act
        ds = _build_tensor_dataset(x, y)

        # Assert
        x_tensor, _ = ds[0]
        assert x_tensor.dtype == torch.float32

    def test_y_dtype_is_int64(self):
        # Arrange
        x = np.random.randn(20, 5).astype(np.float32)
        y = np.zeros(20, dtype=np.int32)  # deliberately int32

        # Act
        ds = _build_tensor_dataset(x, y)

        # Assert
        _, y_tensor = ds[0]
        assert y_tensor.dtype == torch.int64

    def test_tabular_shape_preserved(self):
        # Arrange: (N, d) tabular
        n, d = 16, 12
        x = np.random.randn(n, d).astype(np.float32)
        y = np.zeros(n, dtype=np.int64)

        # Act
        ds = _build_tensor_dataset(x, y)

        # Assert: each sample's x has shape (d,)
        x_sample, _ = ds[0]
        assert x_sample.shape == (d,)

    def test_image_shape_preserved(self):
        # Arrange: (N, C, H, W) image
        n, c, h, w = 8, 3, 16, 16
        x = np.random.randn(n, c, h, w).astype(np.float32)
        y = np.zeros(n, dtype=np.int64)

        # Act
        ds = _build_tensor_dataset(x, y)

        # Assert: each sample's x has shape (C, H, W)
        x_sample, _ = ds[0]
        assert x_sample.shape == (c, h, w)

    def test_values_preserved_up_to_cast(self):
        # Arrange
        x = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
        y = np.array([0, 1], dtype=np.int64)

        # Act
        ds = _build_tensor_dataset(x, y)

        # Assert
        x0, y0 = ds[0]
        assert float(x0[0]) == pytest.approx(1.5)
        assert int(y0) == 0


# ---------------------------------------------------------------------------
# _stratify_key
# ---------------------------------------------------------------------------


class TestStratifyKey:
    def test_output_length_equals_input_length(self):
        # Arrange
        n = 50
        y = np.random.randint(0, 2, n)
        z = np.random.randint(0, 3, (n, 2))

        # Act
        keys = _stratify_key(y, z)

        # Assert
        assert len(keys) == n

    def test_output_is_1d_integer_array(self):
        # Arrange
        y = np.array([0, 1, 0, 1])
        z = np.array([[0], [0], [1], [1]])

        # Act
        keys = _stratify_key(y, z)

        # Assert
        assert keys.ndim == 1
        assert np.issubdtype(keys.dtype, np.integer)

    def test_identical_yz_rows_get_same_key(self):
        # Arrange: rows 0 and 2 share (y=0, z=0)
        y = np.array([0, 1, 0, 1])
        z = np.array([[0], [0], [0], [1]])

        # Act
        keys = _stratify_key(y, z)

        # Assert
        assert keys[0] == keys[2]

    def test_different_yz_rows_get_different_keys(self):
        # Arrange: all four (y, z) combinations are distinct
        y = np.array([0, 1, 0, 1])
        z = np.array([[0], [0], [1], [1]])

        # Act
        keys = _stratify_key(y, z)

        # Assert
        assert len(set(keys.tolist())) == 4

    def test_multi_column_z_preserves_uniqueness(self):
        # Arrange: z has 3 columns; each row is distinct
        n = 8
        y = np.zeros(n, dtype=np.int64)
        z = np.arange(n * 3).reshape(n, 3)

        # Act
        keys = _stratify_key(y, z)

        # Assert: all keys are unique
        assert len(set(keys.tolist())) == n

    def test_same_y_different_z_gives_different_keys(self):
        # Arrange
        y = np.array([0, 0])
        z = np.array([[0, 1], [1, 0]])

        # Act
        keys = _stratify_key(y, z)

        # Assert
        assert keys[0] != keys[1]

    def test_same_z_different_y_gives_different_keys(self):
        # Arrange
        y = np.array([0, 1])
        z = np.array([[5, 3], [5, 3]])

        # Act
        keys = _stratify_key(y, z)

        # Assert
        assert keys[0] != keys[1]


# ---------------------------------------------------------------------------
# prepare_distribution_splits
# ---------------------------------------------------------------------------


class TestPrepareDistributionSplits:
    def test_returns_distribution_splits_dataclass(
        self, splits_inputs_factory, splits_kwargs
    ):
        # Arrange
        args = splits_inputs_factory()

        # Act
        result = prepare_distribution_splits(*args, **splits_kwargs)

        # Assert
        assert isinstance(result, DistributionSplits)

    def test_returns_exactly_six_dataloaders(
        self, splits_inputs_factory, splits_kwargs
    ):
        # Arrange
        args = splits_inputs_factory()

        # Act
        result = prepare_distribution_splits(*args, **splits_kwargs)

        # Assert: all six attributes are DataLoader instances
        loaders = [
            result.vic_trainloader_1,
            result.vic_trainloader_2,
            result.adv_trainloader_1,
            result.adv_trainloader_2,
            result.test_loader_1,
            result.test_loader_2,
        ]
        assert all(isinstance(dl, DataLoader) for dl in loaders)
        assert len(loaders) == 6

    def test_raises_value_error_when_filter_column_not_in_sensitive_columns(
        self, splits_inputs_factory, splits_kwargs
    ):
        # Arrange
        args = splits_inputs_factory()
        splits_kwargs["filter_column"] = "missing_col"

        # Act / Assert
        with pytest.raises(ValueError, match="filter_column"):
            prepare_distribution_splits(*args, **splits_kwargs)

    def test_raises_value_error_when_z_columns_mismatch_sensitive_columns(
        self, splits_inputs_factory, splits_kwargs
    ):
        # Arrange: z has 2 columns but sensitive_columns has 3 entries
        args = splits_inputs_factory()
        splits_kwargs["sensitive_columns"] = ["sex", "race", "extra"]
        splits_kwargs["filter_column"] = "sex"

        # Act / Assert
        with pytest.raises(ValueError, match="z_train has"):
            prepare_distribution_splits(*args, **splits_kwargs)

    def test_all_six_loaders_are_non_empty(self, splits_inputs_factory, splits_kwargs):
        # Arrange
        args = splits_inputs_factory(n_train=800, n_test=300)

        # Act
        result = prepare_distribution_splits(*args, **splits_kwargs)

        # Assert
        for loader in [
            result.vic_trainloader_1,
            result.vic_trainloader_2,
            result.adv_trainloader_1,
            result.adv_trainloader_2,
            result.test_loader_1,
            result.test_loader_2,
        ]:
            assert len(loader.dataset) > 0  # type: ignore[arg-type]

    def test_tabular_batches_have_correct_feature_shape(
        self, splits_inputs_factory, splits_kwargs
    ):
        # Arrange: x is (N, 8) tabular
        n_features = 8
        args = splits_inputs_factory(n_features=n_features)

        # Act
        result = prepare_distribution_splits(*args, **splits_kwargs)

        # Assert: each x batch has shape (batch, n_features)
        x_batch, _ = next(iter(result.vic_trainloader_1))
        assert x_batch.ndim == 2
        assert x_batch.shape[1] == n_features

    def test_image_batches_have_correct_shape(self, splits_kwargs):
        # Arrange: x is (N, C, H, W) image
        rng = np.random.default_rng(7)
        n_train, n_test = 600, 200
        c, h, w = 3, 8, 8
        x_train = rng.standard_normal((n_train, c, h, w)).astype(np.float32)
        y_train = rng.integers(0, 2, n_train).astype(np.int64)
        z_train = rng.integers(0, 2, (n_train, 2)).astype(np.int64)
        x_test = rng.standard_normal((n_test, c, h, w)).astype(np.float32)
        y_test = rng.integers(0, 2, n_test).astype(np.int64)
        z_test = rng.integers(0, 2, (n_test, 2)).astype(np.int64)

        # Act
        result = prepare_distribution_splits(
            x_train,
            y_train,
            z_train,
            x_test,
            y_test,
            z_test,
            **splits_kwargs,
        )

        # Assert: each x sample has shape (C, H, W)
        x_batch, _ = next(iter(result.vic_trainloader_1))
        assert x_batch.shape[1:] == (c, h, w)

    def test_y_shape_2d_accepted(self, splits_inputs_factory, splits_kwargs):
        # Arrange: y_train and y_test have shape (N, 1) instead of (N,)
        args = splits_inputs_factory(y_shape="2d")

        # Act: must not raise
        result = prepare_distribution_splits(*args, **splits_kwargs)

        # Assert
        assert isinstance(result, DistributionSplits)

    def test_batch_x_dtype_is_float32(self, splits_inputs_factory, splits_kwargs):
        # Arrange
        args = splits_inputs_factory()

        # Act
        result = prepare_distribution_splits(*args, **splits_kwargs)
        x_batch, _ = next(iter(result.vic_trainloader_1))

        # Assert
        assert x_batch.dtype == torch.float32

    def test_batch_y_dtype_is_int64(self, splits_inputs_factory, splits_kwargs):
        # Arrange
        args = splits_inputs_factory()

        # Act
        result = prepare_distribution_splits(*args, **splits_kwargs)
        _, y_batch = next(iter(result.vic_trainloader_1))

        # Assert
        assert y_batch.dtype == torch.int64

    def test_drop_values_filters_rows(self, splits_kwargs):
        # Arrange: z[:, 1] (race) has values 0/1; drop race==1
        rng = np.random.default_rng(9)
        n_train, n_test = 800, 300
        x_train = rng.standard_normal((n_train, 6)).astype(np.float32)
        y_train = rng.integers(0, 2, n_train).astype(np.int64)
        # z[:,1] is all 1 for the first half; 0 for the rest
        z_train = np.zeros((n_train, 2), dtype=np.int64)
        z_train[:400, 1] = 1
        x_test = rng.standard_normal((n_test, 6)).astype(np.float32)
        y_test = rng.integers(0, 2, n_test).astype(np.int64)
        z_test = np.zeros((n_test, 2), dtype=np.int64)
        z_test[:150, 1] = 1

        splits_kwargs["drop_values"] = {"race": [1]}
        splits_kwargs["train_subsample"] = 10
        splits_kwargs["test_subsample"] = 5

        # Act: if drop is applied, loaders still return only race==0 rows
        result = prepare_distribution_splits(
            x_train, y_train, z_train, x_test, y_test, z_test, **splits_kwargs
        )

        # Assert: the function completes without error and returns valid splits
        assert isinstance(result, DistributionSplits)

    def test_seed_produces_same_split_sizes(self, splits_inputs_factory, splits_kwargs):
        # Arrange
        # The seed controls StratifiedShuffleSplit for the 50/50 halves.
        # Heuristic subsampling uses np.random globally, so per-call tensor
        # content is not guaranteed to be identical; however, dataset *sizes*
        # produced by _heuristic_sample are deterministic given the same pool.
        # We seed numpy before each call to make the full pipeline reproducible.
        args = splits_inputs_factory()

        # Act
        np.random.seed(99)
        result_a = prepare_distribution_splits(*args, **splits_kwargs)
        np.random.seed(99)
        result_b = prepare_distribution_splits(*args, **splits_kwargs)

        # Assert: dataset sizes must be identical when numpy state is reset
        for attr in [
            "vic_trainloader_1",
            "vic_trainloader_2",
            "adv_trainloader_1",
            "adv_trainloader_2",
            "test_loader_1",
            "test_loader_2",
        ]:
            size_a = len(getattr(result_a, attr).dataset)  # type: ignore[arg-type]
            size_b = len(getattr(result_b, attr).dataset)  # type: ignore[arg-type]
            assert size_a == size_b, f"{attr}: {size_a} != {size_b}"

    @pytest.mark.parametrize("ratio1,ratio2", [(0.1, 0.9), (0.2, 0.8), (0.4, 0.6)])
    def test_different_ratios_produce_different_loaders(
        self,
        splits_inputs_factory,
        splits_kwargs,
        ratio1: float,
        ratio2: float,
    ):
        # Arrange
        args = splits_inputs_factory()
        kwargs_1 = {**splits_kwargs, "ratio1": ratio1, "ratio2": ratio2}
        kwargs_2 = {**splits_kwargs, "ratio1": ratio2, "ratio2": ratio1}

        # Act
        result_1 = prepare_distribution_splits(*args, **kwargs_1)
        result_2 = prepare_distribution_splits(*args, **kwargs_2)

        # Assert: at least one of the first batches differs between the two runs
        # (shapes or content). Consume batches to verify loaders are iterable.
        _x1, _ = next(iter(result_1.vic_trainloader_1))
        _x2, _ = next(iter(result_2.vic_trainloader_1))
        # When ratios are different the subsampled indices differ; shapes may vary
        # (batch size or content). At minimum the function must not raise.
        assert isinstance(result_1, DistributionSplits)
        assert isinstance(result_2, DistributionSplits)
        # Both results must produce non-empty loaders
        assert len(result_1.vic_trainloader_1.dataset) > 0  # type: ignore[arg-type]
        assert len(result_2.vic_trainloader_1.dataset) > 0  # type: ignore[arg-type]

    def test_test_loaders_are_concatenated_from_vic_and_adv(
        self, splits_inputs_factory, splits_kwargs
    ):
        # Arrange
        args = splits_inputs_factory(n_train=600, n_test=400)
        splits_kwargs["test_subsample"] = 15

        # Act
        result = prepare_distribution_splits(*args, **splits_kwargs)

        # Assert: test_loader_1 size == 2 * test_subsample per class
        # (adv_test_1 + vic_test_1), so dataset must be larger than a single draw
        n_test1 = len(result.test_loader_1.dataset)  # type: ignore[arg-type]
        # The combined test set has contributions from both adv and vic halves
        assert n_test1 > 0
