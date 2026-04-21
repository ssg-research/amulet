"""
Dataset preparation helpers for distribution inference attacks.

A distribution inference attack distinguishes two training distributions
that differ in the proportion of samples satisfying some filter (e.g.
``sex == 1``). Each side of the attack (victim and adversary) requires
training data sampled to the target proportion. These helpers perform the
ratio-preserving subsampling and return the six DataLoaders the attack
consumes.
"""

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset


@dataclass
class DistributionSplits:
    """DataLoaders produced by :func:`prepare_distribution_splits`."""

    vic_trainloader_1: DataLoader
    vic_trainloader_2: DataLoader
    adv_trainloader_1: DataLoader
    adv_trainloader_2: DataLoader
    test_loader_1: DataLoader
    test_loader_2: DataLoader


def _filter_by_ratio(mask: np.ndarray, ratio: float) -> np.ndarray:
    """Return indices such that mask[indices].mean() ≈ ratio."""
    qualify = np.nonzero(mask)[0]
    notqualify = np.nonzero(~mask)[0]
    current_ratio = len(qualify) / (len(qualify) + len(notqualify))

    if current_ratio <= ratio:
        np.random.shuffle(notqualify)
        if ratio < 1:
            nqi = notqualify[: int(((1 - ratio) * len(qualify)) / ratio)]
            return np.concatenate([qualify, nqi])
        return qualify

    np.random.shuffle(qualify)
    if ratio > 0:
        qi = qualify[: int((ratio * len(notqualify)) / (1 - ratio))]
        return np.concatenate([qi, notqualify])
    return notqualify


def _heuristic_sample(
    y: np.ndarray,
    filter_mask: np.ndarray,
    ratio: float,
    cwise_sample: int,
    class_imbalance: float = 2.0,
    n_tries: int = 1000,
) -> np.ndarray:
    """Return indices achieving the ratio closest to target over n_tries draws.

    Args:
        y: 1-D label array. Must be binary (0/1).
        filter_mask: Boolean array of the same length; True where the filter
            condition holds.
        ratio: Target fraction of True values in filter_mask after sampling.
        cwise_sample: Minority-class sample count per draw.
        class_imbalance: Majority / minority class count ratio.
        n_tries: Number of sampling attempts.

    Returns:
        Sorted integer index array selecting the best draw.
    """
    vals: list[float] = []
    idx_list: list[np.ndarray] = []

    for _ in range(n_tries):
        pool = _filter_by_ratio(filter_mask, ratio)

        zero_ids = pool[y[pool] == 0]
        one_ids = pool[y[pool] == 1]

        if class_imbalance >= 1:
            zero_ids = np.random.permutation(zero_ids)[
                : int(class_imbalance * cwise_sample)
            ]
            one_ids = np.random.permutation(one_ids)[:cwise_sample]
        else:
            zero_ids = np.random.permutation(zero_ids)[:cwise_sample]
            one_ids = np.random.permutation(one_ids)[
                : int(cwise_sample / class_imbalance)
            ]

        picked = np.sort(np.concatenate([zero_ids, one_ids]))
        vals.append(float(filter_mask[picked].mean()))
        idx_list.append(picked)

    diffs = np.abs(np.array(vals) - ratio)
    return idx_list[int(np.argmin(diffs))]


def _build_tensor_dataset(x: np.ndarray, y: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64)),
    )


def _stratify_key(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Encode each (y, z_row) combination as a unique integer for stratified splitting."""
    cols = np.column_stack([y.reshape(-1, 1), z])
    _, inverse = np.unique(cols, axis=0, return_inverse=True)
    return inverse


def prepare_distribution_splits(
    x_train: np.ndarray,
    y_train: np.ndarray,
    z_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    z_test: np.ndarray,
    sensitive_columns: list[str],
    filter_column: str,
    ratio1: float,
    ratio2: float,
    train_subsample: int,
    test_subsample: int,
    filter_value: int = 1,
    drop_values: dict[str, list[int]] | None = None,
    class_imbalance: float = 3.0,
    n_tries: int = 100,
    batch_size: int = 256,
    seed: int = 42,
) -> DistributionSplits:
    """
    Build the six DataLoaders a distribution inference attack consumes.

    The train/test arrays are each split 50/50 into victim and adversary halves
    (stratified on labels and every sensitive column). Each half is then
    subsampled twice, once for ``ratio1`` and once for ``ratio2``, to produce
    the four training loaders plus two combined test loaders.

    Works with any input shape: tabular ``(N, d)``, image ``(N, C, H, W)``, etc.
    Features and sensitive attributes are never merged, so the model receives
    only the original feature array.

    Args:
        x_train, y_train, z_train: Training features, labels (1-D), sensitive
            attributes (2-D, one column per attribute).
        x_test, y_test, z_test: Test features, labels, sensitive attributes.
        sensitive_columns: Column names for ``z_train``/``z_test``. Length must
            match ``z_train.shape[1]``.
        filter_column: Name of the sensitive column whose proportion is being
            inferred. Must be in ``sensitive_columns``.
        ratio1: Target proportion of rows satisfying ``filter_column == filter_value``
            for distribution 1.
        ratio2: Target proportion for distribution 2.
        train_subsample: Minority-class sample count per train draw.
        test_subsample: Minority-class sample count per test draw.
        filter_value: Value of ``filter_column`` that satisfies the filter.
        drop_values: Optional per-column list of values to drop before sampling,
            e.g. ``{"race": [2]}`` to exclude a multi-class label the binary
            filter cannot express.
        class_imbalance: Target ratio of majority-class to minority-class samples
            per draw.
        n_tries: Number of sampling attempts per draw.
        batch_size: Batch size for the returned DataLoaders.
        seed: Random seed for the 50/50 train/test split.
    """
    if filter_column not in sensitive_columns:
        raise ValueError(
            f"filter_column '{filter_column}' must be one of sensitive_columns "
            f"{sensitive_columns}"
        )
    if z_train.shape[1] != len(sensitive_columns):
        raise ValueError(
            f"z_train has {z_train.shape[1]} columns but sensitive_columns has "
            f"{len(sensitive_columns)}"
        )

    y_train = y_train.ravel()
    y_test = y_test.ravel()
    filter_col_idx = sensitive_columns.index(filter_column)

    if drop_values:
        keep_train = np.ones(len(x_train), dtype=bool)
        keep_test = np.ones(len(x_test), dtype=bool)
        for col, vals in drop_values.items():
            col_idx = sensitive_columns.index(col)
            for v in vals:
                keep_train &= z_train[:, col_idx] != v
                keep_test &= z_test[:, col_idx] != v
        x_train, y_train, z_train = (
            x_train[keep_train],
            y_train[keep_train],
            z_train[keep_train],
        )
        x_test, y_test, z_test = (
            x_test[keep_test],
            y_test[keep_test],
            z_test[keep_test],
        )

    def split_50_50(
        x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> tuple[
        tuple[np.ndarray, np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ]:
        strat = _stratify_key(y, z)
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        left_idx, right_idx = next(splitter.split(x, strat))
        return (x[left_idx], y[left_idx], z[left_idx]), (
            x[right_idx],
            y[right_idx],
            z[right_idx],
        )

    (x_vic, y_vic, z_vic), (x_adv, y_adv, z_adv) = split_50_50(
        x_train, y_train, z_train
    )
    (x_tv, y_tv, z_tv), (x_ta, y_ta, z_ta) = split_50_50(x_test, y_test, z_test)

    def draw(
        x: np.ndarray, y: np.ndarray, z: np.ndarray, ratio: float, subsample: int
    ) -> TensorDataset:
        idx = _heuristic_sample(
            y,
            z[:, filter_col_idx] == filter_value,
            ratio,
            subsample,
            class_imbalance,
            n_tries,
        )
        return _build_tensor_dataset(x[idx], y[idx])

    vic_train_1 = draw(x_vic, y_vic, z_vic, ratio1, train_subsample)
    vic_train_2 = draw(x_vic, y_vic, z_vic, ratio2, train_subsample)
    adv_train_1 = draw(x_adv, y_adv, z_adv, ratio1, train_subsample)
    adv_train_2 = draw(x_adv, y_adv, z_adv, ratio2, train_subsample)

    vic_test_1 = draw(x_tv, y_tv, z_tv, ratio1, test_subsample)
    vic_test_2 = draw(x_tv, y_tv, z_tv, ratio2, test_subsample)
    adv_test_1 = draw(x_ta, y_ta, z_ta, ratio1, test_subsample)
    adv_test_2 = draw(x_ta, y_ta, z_ta, ratio2, test_subsample)

    test_1: ConcatDataset = ConcatDataset([adv_test_1, vic_test_1])
    test_2: ConcatDataset = ConcatDataset([adv_test_2, vic_test_2])

    def make_loader(ds: TensorDataset | ConcatDataset) -> DataLoader:  # type: ignore[type-arg]
        return DataLoader(dataset=ds, batch_size=batch_size, shuffle=False)

    return DistributionSplits(
        vic_trainloader_1=make_loader(vic_train_1),
        vic_trainloader_2=make_loader(vic_train_2),
        adv_trainloader_1=make_loader(adv_train_1),
        adv_trainloader_2=make_loader(adv_train_2),
        test_loader_1=make_loader(test_1),
        test_loader_2=make_loader(test_2),
    )
