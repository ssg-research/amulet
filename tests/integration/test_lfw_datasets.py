"""
Integration and unit tests for load_lfw and its helpers in
amulet/datasets/__tabular_datasets.py.

Integration tests download real data from the network and disk, so they are
marked @pytest.mark.integration.  Unit tests for _lfw_read_attributes and
_lfw_attr_labels use only in-memory data and carry no special mark.

A single session-scoped tmp directory is used for all integration tests so
that the cold-start download (which is slow) happens once and is reused by
the cache-hit and changed-params cases.  After all tests complete the entire
tmp tree is deleted.  The real data/lfw/ directory at the repo root is never
touched.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from amulet.datasets.__tabular_datasets import (  # type: ignore[reportPrivateImportUsage]
    _lfw_attr_labels,
    _lfw_read_attributes,
    load_lfw,
)

# ---------------------------------------------------------------------------
# Session-scoped fixture: one tmp dir shared across all integration tests.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def lfw_tmp(tmp_path_factory: pytest.TempPathFactory):
    """Create a fresh temp directory once per session, then delete it after all tests."""
    tmp = tmp_path_factory.mktemp("lfw_integration", numbered=True)
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Integration tests — real downloads, real processing
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cold_start_creates_attributes_file(lfw_tmp: Path):
    """COLD START: lfw_attributes.txt must be downloaded on first call."""
    # nothing pre-exists in lfw_tmp
    attrs_path = lfw_tmp / "lfw_attributes.txt"

    load_lfw(path=lfw_tmp, target="age", attribute_1="race", attribute_2="gender")

    assert attrs_path.exists(), "lfw_attributes.txt must be downloaded"


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cold_start_creates_images_npz(lfw_tmp: Path):
    """COLD START: lfw_images.npz must be built on first call."""
    images_path = lfw_tmp / "lfw_images.npz"

    # no-op if already run; fixture is session-scoped
    load_lfw(path=lfw_tmp, target="age", attribute_1="race", attribute_2="gender")

    assert images_path.exists(), "lfw_images.npz must be built"


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cold_start_creates_processed_cache(lfw_tmp: Path):
    """COLD START: parameter-keyed processed cache .npz must exist after first call."""
    cache_path = lfw_tmp / "lfw_processed__target=age__attr1=race__attr2=gender.npz"

    load_lfw(path=lfw_tmp, target="age", attribute_1="race", attribute_2="gender")

    assert cache_path.exists(), "Processed cache .npz must be created"


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cold_start_returns_non_none_train_set(lfw_tmp: Path):
    """COLD START: train_set must be non-None."""
    ds = load_lfw(path=lfw_tmp, target="age", attribute_1="race", attribute_2="gender")

    assert ds.train_set is not None


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cold_start_returns_non_none_test_set(lfw_tmp: Path):
    """COLD START: test_set must be non-None."""
    ds = load_lfw(path=lfw_tmp, target="age", attribute_1="race", attribute_2="gender")

    assert ds.test_set is not None


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cold_start_returns_non_none_x_arrays(lfw_tmp: Path):
    """COLD START: x_train and x_test must be non-None."""
    ds = load_lfw(path=lfw_tmp, target="age", attribute_1="race", attribute_2="gender")

    assert ds.x_train is not None
    assert ds.x_test is not None


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cold_start_returns_non_none_y_arrays(lfw_tmp: Path):
    """COLD START: y_train and y_test must be non-None."""
    ds = load_lfw(path=lfw_tmp, target="age", attribute_1="race", attribute_2="gender")

    assert ds.y_train is not None
    assert ds.y_test is not None


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cold_start_returns_non_none_z_arrays(lfw_tmp: Path):
    """COLD START: z_train and z_test must be non-None."""
    ds = load_lfw(path=lfw_tmp, target="age", attribute_1="race", attribute_2="gender")

    assert ds.z_train is not None
    assert ds.z_test is not None


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cold_start_x_train_dtype_float32(lfw_tmp: Path):
    """COLD START: x_train must have dtype float32."""
    ds = load_lfw(path=lfw_tmp, target="age", attribute_1="race", attribute_2="gender")

    assert ds.x_train is not None
    assert ds.x_train.dtype == np.float32


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cold_start_y_train_dtype_int64(lfw_tmp: Path):
    """COLD START: y_train must have dtype int64 (long)."""
    ds = load_lfw(path=lfw_tmp, target="age", attribute_1="race", attribute_2="gender")

    assert ds.y_train is not None
    assert ds.y_train.dtype == np.int64


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cold_start_y_values_binary_for_age_target(lfw_tmp: Path):
    """COLD START (age target): all y values must be 0 or 1 only."""
    ds = load_lfw(path=lfw_tmp, target="age", attribute_1="race", attribute_2="gender")

    # age is binarized: Baby/Child/Youth -> 0, Middle Aged/Senior -> 1
    assert ds.y_train is not None
    assert ds.y_test is not None
    unique_train = set(ds.y_train.tolist())
    unique_test = set(ds.y_test.tolist())
    assert unique_train.issubset({0, 1}), f"Non-binary train labels: {unique_train}"
    assert unique_test.issubset({0, 1}), f"Non-binary test labels: {unique_test}"


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cold_start_num_classes_two_for_age_target(lfw_tmp: Path):
    """COLD START (age target): num_classes must be 2 after binary age encoding."""
    ds = load_lfw(path=lfw_tmp, target="age", attribute_1="race", attribute_2="gender")

    assert ds.num_classes == 2


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cold_start_z_train_has_two_columns(lfw_tmp: Path):
    """COLD START: z_train must have two columns (race and gender)."""
    ds = load_lfw(path=lfw_tmp, target="age", attribute_1="race", attribute_2="gender")

    assert ds.z_train is not None
    assert ds.z_train.ndim == 2
    assert ds.z_train.shape[1] == 2


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cold_start_train_test_size_split(lfw_tmp: Path):
    """COLD START: train+test sizes must reflect the default test_size=0.3 split."""
    ds = load_lfw(
        path=lfw_tmp,
        target="age",
        attribute_1="race",
        attribute_2="gender",
        test_size=0.3,
    )

    assert ds.x_train is not None
    assert ds.x_test is not None
    total = ds.x_train.shape[0] + ds.x_test.shape[0]
    test_ratio = ds.x_test.shape[0] / total
    # Allow ±2% tolerance due to integer rounding
    assert abs(test_ratio - 0.3) <= 0.02, f"Unexpected test ratio: {test_ratio:.3f}"


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cache_hit_x_train_same_shape(lfw_tmp: Path):
    """ROUTINE (cache hit): second call returns x_train with the same shape."""
    # cold-start already ran; call once to capture shape
    ds_first = load_lfw(
        path=lfw_tmp, target="age", attribute_1="race", attribute_2="gender"
    )

    # second call, same params
    ds_second = load_lfw(
        path=lfw_tmp, target="age", attribute_1="race", attribute_2="gender"
    )

    assert ds_first.x_train is not None and ds_second.x_train is not None
    assert ds_first.x_train.shape == ds_second.x_train.shape


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cache_hit_x_test_same_shape(lfw_tmp: Path):
    """ROUTINE (cache hit): second call returns x_test with the same shape."""
    ds_first = load_lfw(
        path=lfw_tmp, target="age", attribute_1="race", attribute_2="gender"
    )

    ds_second = load_lfw(
        path=lfw_tmp, target="age", attribute_1="race", attribute_2="gender"
    )

    assert ds_first.x_test is not None and ds_second.x_test is not None
    assert ds_first.x_test.shape == ds_second.x_test.shape


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cache_hit_no_new_files_created(lfw_tmp: Path):
    """ROUTINE (cache hit): second call must not create extra files in the directory."""
    # count files before second call
    cache_path = lfw_tmp / "lfw_processed__target=age__attr1=race__attr2=gender.npz"
    assert cache_path.exists(), "Precondition: cold-start test must have run first"
    files_before = set(lfw_tmp.iterdir())

    load_lfw(path=lfw_tmp, target="age", attribute_1="race", attribute_2="gender")

    files_after = set(lfw_tmp.iterdir())
    assert files_before == files_after, (
        f"New files created on cache hit: {files_after - files_before}"
    )


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_changed_params_creates_new_cache_file(lfw_tmp: Path):
    """CHANGED PARAMS: new parameter combination creates a new cache file."""
    new_cache = lfw_tmp / "lfw_processed__target=race__attr1=age__attr2=gender.npz"

    load_lfw(path=lfw_tmp, target="race", attribute_1="age", attribute_2="gender")

    assert new_cache.exists(), "New processed cache must be created for changed params"


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_changed_params_old_cache_still_exists(lfw_tmp: Path):
    """CHANGED PARAMS: original cache file must not be removed when params change."""
    # run changed-params call first (no-op if already done)
    load_lfw(path=lfw_tmp, target="race", attribute_1="age", attribute_2="gender")
    old_cache = lfw_tmp / "lfw_processed__target=age__attr1=race__attr2=gender.npz"

    assert old_cache.exists(), "Original cache file must survive a changed-params call"


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_changed_params_raw_files_not_redownloaded(lfw_tmp: Path):
    """CHANGED PARAMS: attributes and images files must already exist (no re-download)."""
    # both raw files must exist from cold start
    attrs_path = lfw_tmp / "lfw_attributes.txt"
    images_path = lfw_tmp / "lfw_images.npz"
    assert attrs_path.exists() and images_path.exists(), (
        "Precondition: raw files must exist from cold-start test"
    )

    attrs_mtime_before = attrs_path.stat().st_mtime
    images_mtime_before = images_path.stat().st_mtime

    load_lfw(path=lfw_tmp, target="race", attribute_1="age", attribute_2="gender")

    assert attrs_path.stat().st_mtime == attrs_mtime_before, (
        "lfw_attributes.txt must not be re-downloaded"
    )
    assert images_path.stat().st_mtime == images_mtime_before, (
        "lfw_images.npz must not be rebuilt"
    )


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_changed_params_race_target_y_values_are_valid_integers(lfw_tmp: Path):
    """CHANGED PARAMS (race target): y values must be non-negative integers."""
    ds = load_lfw(path=lfw_tmp, target="race", attribute_1="age", attribute_2="gender")

    assert ds.y_train is not None
    assert ds.y_test is not None
    assert np.all(ds.y_train >= 0), "Race labels must be non-negative"
    assert np.all(ds.y_test >= 0), "Race labels must be non-negative"
    assert ds.y_train.dtype in (np.int32, np.int64), (
        f"Unexpected dtype: {ds.y_train.dtype}"
    )


# ---------------------------------------------------------------------------
# Unit tests — _lfw_read_attributes
# ---------------------------------------------------------------------------


@pytest.fixture
def make_attributes_txt():
    """Factory fixture that builds a minimal in-memory LFW attributes file string.

    Returned callable signature: `(*, raw_format: bool) -> str`.
    """

    def _make(*, raw_format: bool) -> str:
        header_cols = "person\timagenum\tMale\tSmiling"
        row = "John_Doe\t1\t0.8\t-0.3"

        if raw_format:
            # Two-line header: description comment + "#\t"-prefixed column names
            return f"# LFW Attribute descriptions\n#\t{header_cols}\n{row}\n"
        else:
            # Pre-cleaned: single header line with no prefix
            return f"{header_cols}\n{row}\n"

    return _make


def test_lfw_read_attributes_raw_format_returns_correct_columns(
    tmp_path: Path, make_attributes_txt
):
    """_lfw_read_attributes: raw two-line header format must yield expected column names."""
    attrs_file = tmp_path / "lfw_attributes.txt"
    attrs_file.write_text(make_attributes_txt(raw_format=True), encoding="utf-8")

    df = _lfw_read_attributes(attrs_file)

    assert list(df.columns) == ["person", "imagenum", "Male", "Smiling"]


def test_lfw_read_attributes_precleaned_format_returns_correct_columns(
    tmp_path: Path, make_attributes_txt
):
    """_lfw_read_attributes: pre-cleaned single-header format must yield same column names."""
    attrs_file = tmp_path / "lfw_attributes.txt"
    attrs_file.write_text(make_attributes_txt(raw_format=False), encoding="utf-8")

    df = _lfw_read_attributes(attrs_file)

    assert list(df.columns) == ["person", "imagenum", "Male", "Smiling"]


def test_lfw_read_attributes_both_formats_same_dataframe(
    tmp_path: Path, make_attributes_txt
):
    """_lfw_read_attributes: raw and pre-cleaned formats must produce identical DataFrames."""
    raw_file = tmp_path / "raw.txt"
    clean_file = tmp_path / "clean.txt"
    raw_file.write_text(make_attributes_txt(raw_format=True), encoding="utf-8")
    clean_file.write_text(make_attributes_txt(raw_format=False), encoding="utf-8")

    df_raw = _lfw_read_attributes(raw_file)
    df_clean = _lfw_read_attributes(clean_file)

    pd.testing.assert_frame_equal(
        df_raw.reset_index(drop=True), df_clean.reset_index(drop=True)
    )


def test_lfw_read_attributes_row_count(tmp_path: Path):
    """_lfw_read_attributes: parsed DataFrame must have one data row per non-header line."""
    # add a second data row
    content = "# LFW Attribute descriptions\n#\tperson\timagenum\tMale\n"
    content += "Alice\t1\t0.5\n"
    content += "Bob\t2\t-0.5\n"
    attrs_file = tmp_path / "lfw_attributes.txt"
    attrs_file.write_text(content, encoding="utf-8")

    df = _lfw_read_attributes(attrs_file)

    assert len(df) == 2


# ---------------------------------------------------------------------------
# Unit tests — _lfw_attr_labels
# ---------------------------------------------------------------------------


@pytest.fixture
def make_male_df():
    """Factory fixture for a minimal attributes DataFrame with only a 'Male' column."""

    def _make(values: list[float]) -> pd.DataFrame:
        return pd.DataFrame({"Male": values})

    return _make


@pytest.fixture
def make_age_df():
    """Factory fixture for a minimal attributes DataFrame with age category columns."""

    def _make(rows: list[list[float]]) -> pd.DataFrame:
        cols = ["Baby", "Child", "Youth", "Middle Aged", "Senior"]
        return pd.DataFrame(rows, columns=cols)  # type: ignore[reportArgumentType]

    return _make


@pytest.mark.parametrize(
    "male_values",
    [
        [1.0, -1.0, 0.5, -0.5],
        [2.0, -2.0],
        [0.1, -0.1],
    ],
)
def test_lfw_attr_labels_gender_returns_binary_labels(
    male_values: list[float], make_male_df
):
    """_lfw_attr_labels: gender labels must be 0 or 1 only."""
    df = make_male_df(male_values)

    labels = _lfw_attr_labels(df, "gender")

    for label in labels.values():
        assert label in {0, 1}, f"Non-binary gender label: {label}"


@pytest.mark.parametrize(
    "male_values,expected",
    [
        ([1.0], {0: 1}),
        ([-1.0], {0: 0}),
        ([0.5, -0.5], {0: 1, 1: 0}),
    ],
)
def test_lfw_attr_labels_gender_correct_mapping(
    male_values: list[float], expected: dict[int, int], make_male_df
):
    """_lfw_attr_labels: positive Male score -> 1 (Male), negative -> 0 (Female)."""
    df = make_male_df(male_values)

    labels = _lfw_attr_labels(df, "gender")

    assert labels == expected


@pytest.mark.parametrize(
    "rows,expected_label",
    [
        ([[1.0, 0.0, 0.0, 0.0, 0.0]], 0),  # Baby -> 0
        ([[0.0, 1.0, 0.0, 0.0, 0.0]], 1),  # Child -> 1
        ([[0.0, 0.0, 1.0, 0.0, 0.0]], 2),  # Youth -> 2
        ([[0.0, 0.0, 0.0, 1.0, 0.0]], 3),  # Middle Aged -> 3
        ([[0.0, 0.0, 0.0, 0.0, 1.0]], 4),  # Senior -> 4
    ],
)
def test_lfw_attr_labels_age_argmax_label(
    rows: list[list[float]], expected_label: int, make_age_df
):
    """_lfw_attr_labels: age label must be the argmax index into the age columns."""
    df = make_age_df(rows)

    labels = _lfw_attr_labels(df, "age")

    assert labels[0] == expected_label


def test_lfw_attr_labels_age_labels_in_valid_range(make_age_df):
    """_lfw_attr_labels: all age labels must be integers in [0, 4]."""
    # multiple rows with varying dominant categories
    rows = [
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    df = make_age_df(rows)

    labels = _lfw_attr_labels(df, "age")

    for label in labels.values():
        assert 0 <= label <= 4, f"Age label out of range [0, 4]: {label}"


def test_lfw_attr_labels_unknown_attribute_raises_value_error():
    """_lfw_attr_labels: an unrecognized attribute string must raise ValueError."""
    df = pd.DataFrame({"Male": [1.0]})

    with pytest.raises(ValueError, match="Unknown LFW attribute"):
        _lfw_attr_labels(df, "nonexistent_attribute")


@pytest.mark.parametrize("bad_attr", ["hair_color", "beard", "GENDER", "Age", ""])
def test_lfw_attr_labels_various_unknown_attributes_raise_value_error(bad_attr: str):
    """_lfw_attr_labels: any unrecognized attribute string must raise ValueError."""
    df = pd.DataFrame({"Male": [1.0]})

    with pytest.raises(ValueError):
        _lfw_attr_labels(df, bad_attr)
