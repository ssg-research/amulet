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


def _loaders(result: DistributionSplits) -> list[DataLoader]:
    return [
        result.vic_trainloader_1,
        result.vic_trainloader_2,
        result.adv_trainloader_1,
        result.adv_trainloader_2,
        result.test_loader_1,
        result.test_loader_2,
    ]


def _all_x(loader: DataLoader) -> torch.Tensor:
    """Concatenate every feature batch a loader yields (empty tensor if none)."""
    batches = [batch[0] for batch in loader]
    return torch.cat(batches) if batches else torch.empty(0)


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
        mask = binary_mask_factory(100, 40)

        indices = _filter_by_ratio(mask, ratio=0.0)

        assert mask[indices].sum() == 0

    def test_ratio_one_returns_only_qualify(self, binary_mask_factory):
        mask = binary_mask_factory(100, 40)

        indices = _filter_by_ratio(mask, ratio=1.0)

        assert mask[indices].all()

    @pytest.mark.parametrize(
        "target_ratio, expected_n",
        [(0.1, 333), (0.3, 428), (0.5, 400), (0.7, 285), (0.9, 222)],
    )
    def test_returned_ratio_matches_target(
        self, binary_mask_factory, target_ratio: float, expected_n: int
    ):
        # 200 qualify / 300 notqualify. The pool size is a deterministic floor
        # of the counts (shuffling only reorders), so both the retained count and
        # the achieved ratio are exact. The old 0.1 tolerance was ~30-100x looser
        # than the true deviation and would not notice a swapped reference count
        # in the truncation formula (that shifts the ratio by ~0.03 and n by ~90).
        np.random.seed(0)
        mask = binary_mask_factory(500, 200)

        indices = _filter_by_ratio(mask, ratio=target_ratio)

        assert len(indices) == expected_n
        assert abs(float(mask[indices].mean()) - target_ratio) < 0.005

    def test_returned_indices_are_valid_positions(self, binary_mask_factory):
        mask = binary_mask_factory(80, 30)

        indices = _filter_by_ratio(mask, ratio=0.4)

        assert np.all(indices >= 0)
        assert np.all(indices < len(mask))

    def test_output_is_numpy_array(self, binary_mask_factory):
        mask = binary_mask_factory(60, 20)

        indices = _filter_by_ratio(mask, ratio=0.5)

        assert isinstance(indices, np.ndarray)

    def test_when_current_ratio_equals_target_returns_all_qualify(self):
        # mask has exactly 50% True → current_ratio == ratio == 0.5
        mask = np.array([True, False, True, False, True, False], dtype=bool)
        np.random.seed(42)

        indices = _filter_by_ratio(mask, ratio=0.5)

        qualify = set(np.nonzero(mask)[0].tolist())
        assert qualify.issubset(set(indices.tolist()))

    def test_ratio_zero_no_qualify_entries_in_result(self):
        # 10 qualify, 20 notqualify
        mask = np.array([True] * 10 + [False] * 20, dtype=bool)
        np.random.seed(1)

        indices = _filter_by_ratio(mask, ratio=0.0)

        assert not mask[indices].any()

    def test_ratio_one_all_qualify_returned(self):
        mask = np.array([True] * 15 + [False] * 25, dtype=bool)
        np.random.seed(2)

        indices = _filter_by_ratio(mask, ratio=1.0)

        assert set(indices.tolist()) == set(np.nonzero(mask)[0].tolist())


# ---------------------------------------------------------------------------
# _heuristic_sample
# ---------------------------------------------------------------------------


class TestHeuristicSample:
    def test_output_is_sorted_integer_array(self, binary_data_factory):
        np.random.seed(0)
        n = 200
        y, mask = binary_data_factory(n, 80)

        indices = _heuristic_sample(y, mask, ratio=0.5, cwise_sample=20, n_tries=50)

        assert isinstance(indices, np.ndarray)
        assert np.all(np.diff(indices) >= 0), "output must be sorted"

    def test_returned_indices_valid_for_y(self, binary_data_factory):
        np.random.seed(1)
        n = 300
        y, mask = binary_data_factory(n, 120)

        indices = _heuristic_sample(y, mask, ratio=0.4, cwise_sample=15, n_tries=50)

        assert np.all(indices >= 0)
        assert np.all(indices < n)

    def test_filter_mask_ratio_close_to_target(self, binary_data_factory):
        np.random.seed(2)
        n = 400
        y, mask = binary_data_factory(n, 200)

        indices = _heuristic_sample(y, mask, ratio=0.5, cwise_sample=30, n_tries=200)

        # best draw should be within 0.2 of the target
        achieved = float(mask[indices].mean())
        assert abs(achieved - 0.5) < 0.20

    @pytest.mark.parametrize("class_imbalance", [1.0, 2.0, 3.0])
    def test_class_imbalance_majority_count(
        self, binary_mask_factory, class_imbalance: float
    ):
        # heavily one-sided data so imbalance is achievable
        np.random.seed(3)
        n = 600
        y = np.array([0] * 400 + [1] * 200, dtype=np.int64)
        mask = binary_mask_factory(n, 300)
        cwise_sample = 20

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

        # majority (class 0) count ≤ class_imbalance * cwise_sample
        assert zero_count <= int(class_imbalance * cwise_sample) + 5  # small tolerance
        assert one_count <= cwise_sample + 5

    def test_output_indices_index_into_original_x(self, binary_data_factory):
        np.random.seed(4)
        n = 250
        x = np.random.randn(n, 10).astype(np.float32)
        y, mask = binary_data_factory(n, 100)

        indices = _heuristic_sample(y, mask, ratio=0.4, cwise_sample=20, n_tries=50)

        # slicing x with the indices must not raise
        subset = x[indices]
        assert subset.ndim == 2
        assert subset.shape[1] == 10


# ---------------------------------------------------------------------------
# _build_tensor_dataset
# ---------------------------------------------------------------------------


class TestBuildTensorDataset:
    def test_returns_tensor_dataset(self):
        x = np.random.randn(10, 4).astype(np.float32)
        y = np.random.randint(0, 2, 10).astype(np.int64)

        ds = _build_tensor_dataset(x, y)

        assert isinstance(ds, TensorDataset)

    def test_length_equals_input_length(self):
        n = 37
        x = np.random.randn(n, 8).astype(np.float32)
        y = np.zeros(n, dtype=np.int64)

        ds = _build_tensor_dataset(x, y)

        assert len(ds) == n

    def test_x_dtype_is_float32(self):
        x = np.random.randn(20, 5).astype(np.float64)  # deliberately float64
        y = np.zeros(20, dtype=np.int64)

        ds = _build_tensor_dataset(x, y)

        x_tensor, _ = ds[0]
        assert x_tensor.dtype == torch.float32

    def test_y_dtype_is_int64(self):
        x = np.random.randn(20, 5).astype(np.float32)
        y = np.zeros(20, dtype=np.int32)  # deliberately int32

        ds = _build_tensor_dataset(x, y)

        _, y_tensor = ds[0]
        assert y_tensor.dtype == torch.int64

    def test_tabular_shape_preserved(self):
        # (N, d) tabular
        n, d = 16, 12
        x = np.random.randn(n, d).astype(np.float32)
        y = np.zeros(n, dtype=np.int64)

        ds = _build_tensor_dataset(x, y)

        # each sample's x has shape (d,)
        x_sample, _ = ds[0]
        assert x_sample.shape == (d,)

    def test_image_shape_preserved(self):
        # (N, C, H, W) image
        n, c, h, w = 8, 3, 16, 16
        x = np.random.randn(n, c, h, w).astype(np.float32)
        y = np.zeros(n, dtype=np.int64)

        ds = _build_tensor_dataset(x, y)

        # each sample's x has shape (C, H, W)
        x_sample, _ = ds[0]
        assert x_sample.shape == (c, h, w)

    def test_values_preserved_up_to_cast(self):
        x = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
        y = np.array([0, 1], dtype=np.int64)

        ds = _build_tensor_dataset(x, y)

        x0, y0 = ds[0]
        assert float(x0[0]) == pytest.approx(1.5)
        assert int(y0) == 0


# ---------------------------------------------------------------------------
# _stratify_key
# ---------------------------------------------------------------------------


class TestStratifyKey:
    def test_output_length_equals_input_length(self):
        n = 50
        y = np.random.randint(0, 2, n)
        z = np.random.randint(0, 3, (n, 2))

        keys = _stratify_key(y, z)

        assert len(keys) == n

    def test_output_is_1d_integer_array(self):
        y = np.array([0, 1, 0, 1])
        z = np.array([[0], [0], [1], [1]])

        keys = _stratify_key(y, z)

        assert keys.ndim == 1
        assert np.issubdtype(keys.dtype, np.integer)

    def test_identical_yz_rows_get_same_key(self):
        # rows 0 and 2 share (y=0, z=0)
        y = np.array([0, 1, 0, 1])
        z = np.array([[0], [0], [0], [1]])

        keys = _stratify_key(y, z)

        assert keys[0] == keys[2]

    def test_different_yz_rows_get_different_keys(self):
        # all four (y, z) combinations are distinct
        y = np.array([0, 1, 0, 1])
        z = np.array([[0], [0], [1], [1]])

        keys = _stratify_key(y, z)

        assert len(set(keys.tolist())) == 4

    def test_multi_column_z_preserves_uniqueness(self):
        # z has 3 columns; each row is distinct
        n = 8
        y = np.zeros(n, dtype=np.int64)
        z = np.arange(n * 3).reshape(n, 3)

        keys = _stratify_key(y, z)

        # all keys are unique
        assert len(set(keys.tolist())) == n

    def test_same_y_different_z_gives_different_keys(self):
        y = np.array([0, 0])
        z = np.array([[0, 1], [1, 0]])

        keys = _stratify_key(y, z)

        assert keys[0] != keys[1]

    def test_same_z_different_y_gives_different_keys(self):
        y = np.array([0, 1])
        z = np.array([[5, 3], [5, 3]])

        keys = _stratify_key(y, z)

        assert keys[0] != keys[1]


# ---------------------------------------------------------------------------
# prepare_distribution_splits
# ---------------------------------------------------------------------------


class TestPrepareDistributionSplits:
    def test_returns_distribution_splits_dataclass(
        self, splits_inputs_factory, splits_kwargs
    ):
        args = splits_inputs_factory()

        result = prepare_distribution_splits(*args, **splits_kwargs)

        assert isinstance(result, DistributionSplits)

    def test_returns_exactly_six_dataloaders(
        self, splits_inputs_factory, splits_kwargs
    ):
        args = splits_inputs_factory()

        result = prepare_distribution_splits(*args, **splits_kwargs)

        # all six attributes are DataLoader instances
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
        args = splits_inputs_factory()
        splits_kwargs["filter_column"] = "missing_col"

        with pytest.raises(ValueError, match="filter_column"):
            prepare_distribution_splits(*args, **splits_kwargs)

    def test_raises_value_error_when_z_columns_mismatch_sensitive_columns(
        self, splits_inputs_factory, splits_kwargs
    ):
        # z has 2 columns but sensitive_columns has 3 entries
        args = splits_inputs_factory()
        splits_kwargs["sensitive_columns"] = ["sex", "race", "extra"]
        splits_kwargs["filter_column"] = "sex"

        with pytest.raises(ValueError, match="z_train has"):
            prepare_distribution_splits(*args, **splits_kwargs)

    def test_all_six_loaders_are_non_empty(self, splits_inputs_factory, splits_kwargs):
        args = splits_inputs_factory(n_train=800, n_test=300)

        result = prepare_distribution_splits(*args, **splits_kwargs)

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
        # x is (N, 8) tabular
        n_features = 8
        args = splits_inputs_factory(n_features=n_features)

        result = prepare_distribution_splits(*args, **splits_kwargs)

        # each x batch has shape (batch, n_features)
        x_batch, _ = next(iter(result.vic_trainloader_1))
        assert x_batch.ndim == 2
        assert x_batch.shape[1] == n_features

    def test_image_batches_have_correct_shape(self, splits_kwargs):
        # x is (N, C, H, W) image
        rng = np.random.default_rng(7)
        n_train, n_test = 600, 200
        c, h, w = 3, 8, 8
        x_train = rng.standard_normal((n_train, c, h, w)).astype(np.float32)
        y_train = rng.integers(0, 2, n_train).astype(np.int64)
        z_train = rng.integers(0, 2, (n_train, 2)).astype(np.int64)
        x_test = rng.standard_normal((n_test, c, h, w)).astype(np.float32)
        y_test = rng.integers(0, 2, n_test).astype(np.int64)
        z_test = rng.integers(0, 2, (n_test, 2)).astype(np.int64)

        result = prepare_distribution_splits(
            x_train,
            y_train,
            z_train,
            x_test,
            y_test,
            z_test,
            **splits_kwargs,
        )

        # each x sample has shape (C, H, W)
        x_batch, _ = next(iter(result.vic_trainloader_1))
        assert x_batch.shape[1:] == (c, h, w)

    def test_y_shape_2d_accepted(self, splits_inputs_factory, splits_kwargs):
        # y_train and y_test have shape (N, 1) instead of (N,)
        args = splits_inputs_factory(y_shape="2d")

        # must not raise
        result = prepare_distribution_splits(*args, **splits_kwargs)

        assert isinstance(result, DistributionSplits)

    def test_batch_x_dtype_is_float32(self, splits_inputs_factory, splits_kwargs):
        args = splits_inputs_factory()

        result = prepare_distribution_splits(*args, **splits_kwargs)
        x_batch, _ = next(iter(result.vic_trainloader_1))

        assert x_batch.dtype == torch.float32

    def test_batch_y_dtype_is_int64(self, splits_inputs_factory, splits_kwargs):
        args = splits_inputs_factory()

        result = prepare_distribution_splits(*args, **splits_kwargs)
        _, y_batch = next(iter(result.vic_trainloader_1))

        assert y_batch.dtype == torch.int64

    def test_drop_values_filters_rows(self, splits_kwargs):
        # Plant a sentinel in feature 0 of every race==1 row, then drop race==1.
        # Features flow through to the loaders untouched, so a working drop lets
        # no sentinel survive; a drop that is silently ignored leaks them (on this
        # data, 117 sentinels reach the loaders when the drop is skipped).
        rng = np.random.default_rng(9)

        def _make(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            z = np.zeros((n, 2), dtype=np.int64)
            z[:, 1] = np.arange(n) % 2  # race alternates 0/1 -> ~50/50
            z[:, 0] = rng.integers(0, 2, n)  # sex
            x = rng.standard_normal((n, 6)).astype(np.float32)
            x[z[:, 1] == 1, 0] = 999.0  # sentinel marks race==1
            y = rng.integers(0, 2, n).astype(np.int64)
            return x, y, z

        x_train, y_train, z_train = _make(800)
        x_test, y_test, z_test = _make(300)
        splits_kwargs["drop_values"] = {"race": [1]}
        splits_kwargs["train_subsample"] = 10
        splits_kwargs["test_subsample"] = 5

        result = prepare_distribution_splits(
            x_train, y_train, z_train, x_test, y_test, z_test, **splits_kwargs
        )

        leaked = sum(
            int((_all_x(loader)[:, 0] == 999.0).sum()) for loader in _loaders(result)
        )
        assert leaked == 0

    def test_seed_produces_same_split_sizes(self, splits_inputs_factory, splits_kwargs):
        # The seed controls StratifiedShuffleSplit for the 50/50 halves.
        # Heuristic subsampling uses np.random globally, so per-call tensor
        # content is not guaranteed to be identical; however, dataset *sizes*
        # produced by _heuristic_sample are deterministic given the same pool.
        # We seed numpy before each call to make the full pipeline reproducible.
        args = splits_inputs_factory()

        np.random.seed(99)
        result_a = prepare_distribution_splits(*args, **splits_kwargs)
        np.random.seed(99)
        result_b = prepare_distribution_splits(*args, **splits_kwargs)

        # dataset sizes must be identical when numpy state is reset
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
        args = splits_inputs_factory()
        # Reseed the global RNG identically before each call, so any difference
        # is attributable to the ratio rather than RNG drift between calls. With
        # the ratios swapped, vic_trainloader_1 (which uses ratio1) must sample a
        # different set of rows; a function that ignored the ratios would return
        # byte-identical data.
        np.random.seed(123)
        result_1 = prepare_distribution_splits(
            *args, **{**splits_kwargs, "ratio1": ratio1, "ratio2": ratio2}
        )
        np.random.seed(123)
        result_2 = prepare_distribution_splits(
            *args, **{**splits_kwargs, "ratio1": ratio2, "ratio2": ratio1}
        )

        x1 = _all_x(result_1.vic_trainloader_1)
        x2 = _all_x(result_2.vic_trainloader_1)
        assert not (x1.shape == x2.shape and torch.equal(x1, x2))

    @pytest.mark.parametrize("ratio1,ratio2", [(0.2, 0.8), (0.3, 0.7)])
    def test_train_loaders_achieve_target_filter_ratio(
        self, splits_kwargs, ratio1: float, ratio2: float
    ):
        # Encode the filter attribute (sex) into feature 0, so the fraction of
        # rows with feature0==1 in a loader equals its achieved filter ratio.
        # This pins the module's core contract end-to-end: the ratio1 loader hits
        # ~ratio1 and the ratio2 loader ~ratio2. A swapped/ignored ratio, wrong
        # filter column, or wrong filter value would miss the target by ~0.2+.
        rng = np.random.default_rng(1)
        n = 1200
        z = rng.integers(0, 2, (n, 2)).astype(np.int64)
        x = rng.standard_normal((n, 8)).astype(np.float32)
        x[:, 0] = z[:, 0]  # feature 0 mirrors the sex attribute
        y = rng.integers(0, 2, n).astype(np.int64)
        x_test, y_test, z_test = x[:400].copy(), y[:400].copy(), z[:400].copy()

        kwargs = {
            **splits_kwargs,
            "ratio1": ratio1,
            "ratio2": ratio2,
            "train_subsample": 40,
            "n_tries": 100,
        }
        result = prepare_distribution_splits(x, y, z, x_test, y_test, z_test, **kwargs)

        frac_1 = float((_all_x(result.vic_trainloader_1)[:, 0] == 1).float().mean())
        frac_2 = float((_all_x(result.vic_trainloader_2)[:, 0] == 1).float().mean())
        assert abs(frac_1 - ratio1) < 0.12
        assert abs(frac_2 - ratio2) < 0.12

    def test_test_loaders_are_concatenated_from_vic_and_adv(
        self, splits_inputs_factory, splits_kwargs
    ):
        args = splits_inputs_factory(n_train=600, n_test=400)
        splits_kwargs["test_subsample"] = 15

        result = prepare_distribution_splits(*args, **splits_kwargs)

        # Each test half (adv, vic) draws int(class_imbalance * test_subsample) +
        # test_subsample = 3*15 + 15 = 60 samples, so the concatenation of both
        # halves is exactly 120. A concat bug that dropped one half would halve
        # each combined test loader to 60.
        assert len(result.test_loader_1.dataset) == 120  # type: ignore[arg-type]
        assert len(result.test_loader_2.dataset) == 120  # type: ignore[arg-type]
