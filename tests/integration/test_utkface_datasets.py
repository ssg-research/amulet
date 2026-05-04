"""
Integration and unit tests for load_utkface and its helpers in
amulet/datasets/__image_datasets.py.

Integration tests download real data from the network and process ~24k images,
so they are marked @pytest.mark.integration with a 600 s timeout.  Unit tests
for _utkface_parse_labels and _utkface_build_processed_cache use only synthetic
in-memory or on-disk data.

A single session-scoped tmp directory is used for all integration tests so that
the expensive cold-start download + extraction happens once and is reused by
the cache-hit and changed-target cases.  After all tests complete the entire
tmp tree is deleted.  The real data/utkface/ directory at the repo root is never
touched.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from amulet.datasets.__image_datasets import (  # type: ignore[reportPrivateImportUsage]
    _utkface_build_processed_cache,
    _utkface_parse_labels,
    load_utkface,
)

# ---------------------------------------------------------------------------
# Session-scoped fixture: one tmp dir shared across all integration tests.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def utkface_tmp(tmp_path_factory: pytest.TempPathFactory) -> Path:  # type: ignore[misc]
    """Create a fresh temp directory once per session, then delete it after all tests."""
    tmp = tmp_path_factory.mktemp("utkface_integration", numbered=True)
    yield tmp  # type: ignore[misc]
    shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Integration tests — COLD START
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cold_start_creates_utk_images_dir(utkface_tmp: Path) -> None:
    """COLD START: UTKFace/ directory must exist after the first call."""
    # Arrange — nothing pre-exists in utkface_tmp
    imgs_dir = utkface_tmp / "UTKFace"

    # Act
    load_utkface(
        path=utkface_tmp, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    assert imgs_dir.exists() and imgs_dir.is_dir(), (
        "UTKFace/ directory must exist after extraction"
    )


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cold_start_creates_processed_cache(utkface_tmp: Path) -> None:
    """COLD START: parameter-keyed .npz cache must exist after the first call."""
    # Arrange
    cache_path = (
        utkface_tmp / "utkface_processed__target=age__attr1=gender__attr2=race.npz"
    )

    # Act
    load_utkface(
        path=utkface_tmp, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    assert cache_path.exists(), (
        "utkface_processed__target=age__attr1=gender__attr2=race.npz must be created"
    )


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cold_start_returns_non_none_x_train(utkface_tmp: Path) -> None:
    """COLD START: x_train must be non-None."""
    # Act
    ds = load_utkface(
        path=utkface_tmp, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    assert ds.x_train is not None


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cold_start_returns_non_none_x_test(utkface_tmp: Path) -> None:
    """COLD START: x_test must be non-None."""
    # Act
    ds = load_utkface(
        path=utkface_tmp, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    assert ds.x_test is not None


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cold_start_returns_non_none_y_train(utkface_tmp: Path) -> None:
    """COLD START: y_train must be non-None."""
    # Act
    ds = load_utkface(
        path=utkface_tmp, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    assert ds.y_train is not None


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cold_start_returns_non_none_y_test(utkface_tmp: Path) -> None:
    """COLD START: y_test must be non-None."""
    # Act
    ds = load_utkface(
        path=utkface_tmp, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    assert ds.y_test is not None


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cold_start_returns_non_none_z_train(utkface_tmp: Path) -> None:
    """COLD START: z_train must be non-None."""
    # Act
    ds = load_utkface(
        path=utkface_tmp, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    assert ds.z_train is not None


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cold_start_returns_non_none_z_test(utkface_tmp: Path) -> None:
    """COLD START: z_test must be non-None."""
    # Act
    ds = load_utkface(
        path=utkface_tmp, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    assert ds.z_test is not None


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cold_start_x_train_dtype_float32(utkface_tmp: Path) -> None:
    """COLD START: x_train must have dtype float32."""
    # Act
    ds = load_utkface(
        path=utkface_tmp, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    assert ds.x_train is not None
    assert ds.x_train.dtype == np.float32


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cold_start_x_train_values_in_unit_interval(utkface_tmp: Path) -> None:
    """COLD START: x_train pixel values must all be in [0, 1]."""
    # Act
    ds = load_utkface(
        path=utkface_tmp, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    assert ds.x_train is not None
    assert float(ds.x_train.min()) >= 0.0, "x_train minimum must be >= 0"
    assert float(ds.x_train.max()) <= 1.0, "x_train maximum must be <= 1"


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cold_start_x_train_shape_channels_height_width(utkface_tmp: Path) -> None:
    """COLD START: x_train must have shape (N, 3, 64, 64)."""
    # Act
    ds = load_utkface(
        path=utkface_tmp, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    assert ds.x_train is not None
    assert ds.x_train.ndim == 4, f"Expected 4D array, got {ds.x_train.ndim}D"
    assert ds.x_train.shape[1] == 3, f"Expected 3 channels, got {ds.x_train.shape[1]}"
    assert ds.x_train.shape[2] == 64, f"Expected height 64, got {ds.x_train.shape[2]}"
    assert ds.x_train.shape[3] == 64, f"Expected width 64, got {ds.x_train.shape[3]}"


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cold_start_y_train_dtype_int64(utkface_tmp: Path) -> None:
    """COLD START: y_train must have dtype int64."""
    # Act
    ds = load_utkface(
        path=utkface_tmp, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    assert ds.y_train is not None
    assert ds.y_train.dtype == np.int64


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cold_start_y_train_age_values_nonnegative(utkface_tmp: Path) -> None:
    """COLD START: raw age target values must all be integers >= 0."""
    # Act
    ds = load_utkface(
        path=utkface_tmp, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    assert ds.y_train is not None
    assert int(ds.y_train.min()) >= 0, (
        f"Age values must be >= 0; got min {ds.y_train.min()}"
    )


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cold_start_z_train_shape_is_n_by_2(utkface_tmp: Path) -> None:
    """COLD START: z_train must have shape (N_train, 2) for two sensitive attributes."""
    # Act
    ds = load_utkface(
        path=utkface_tmp, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    assert ds.x_train is not None and ds.z_train is not None
    assert ds.z_train.ndim == 2, f"Expected 2D z_train, got {ds.z_train.ndim}D"
    assert ds.z_train.shape[1] == 2, (
        f"Expected 2 sensitive attribute columns, got {ds.z_train.shape[1]}"
    )
    assert ds.z_train.shape[0] == ds.x_train.shape[0], (
        "z_train row count must match x_train row count"
    )


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cold_start_z_train_dtype_int64(utkface_tmp: Path) -> None:
    """COLD START: z_train must have dtype int64."""
    # Act
    ds = load_utkface(
        path=utkface_tmp, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    assert ds.z_train is not None
    assert ds.z_train.dtype == np.int64


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cold_start_num_classes_matches_unique_y_values(utkface_tmp: Path) -> None:
    """COLD START: num_classes must equal the number of unique y values (raw age)."""
    # Act
    ds = load_utkface(
        path=utkface_tmp, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    assert ds.y_train is not None and ds.y_test is not None
    y_all = np.concatenate([ds.y_train, ds.y_test])
    assert ds.num_classes == len(np.unique(y_all)), (
        f"num_classes={ds.num_classes} must equal unique y count {len(np.unique(y_all))}"
    )


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cold_start_train_plus_test_equals_total(utkface_tmp: Path) -> None:
    """COLD START: len(train_set) + len(test_set) must equal total sample count."""
    # Act
    ds = load_utkface(
        path=utkface_tmp, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    assert ds.x_train is not None and ds.x_test is not None
    total = ds.x_train.shape[0] + ds.x_test.shape[0]
    assert len(ds.train_set) + len(ds.test_set) == total  # type: ignore[reportArgumentType]


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cold_start_default_test_size_near_30_percent(utkface_tmp: Path) -> None:
    """COLD START: default test_size=0.3 must give test split within 1% of 30%."""
    # Act
    ds = load_utkface(
        path=utkface_tmp, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    assert ds.x_train is not None and ds.x_test is not None
    total = ds.x_train.shape[0] + ds.x_test.shape[0]
    test_ratio = ds.x_test.shape[0] / total
    assert abs(test_ratio - 0.3) <= 0.01, (
        f"Test ratio {test_ratio:.4f} deviates more than 1% from 30%"
    )


# ---------------------------------------------------------------------------
# Integration tests — AGE BINS
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_age_bins_y_train_values_in_three_bins(utkface_tmp: Path) -> None:
    """AGE BINS: y_train values must be in {0, 1, 2} when age_bins=[30, 60]."""
    # Act
    ds = load_utkface(
        path=utkface_tmp,
        target="age",
        attribute_1="gender",
        attribute_2="race",
        age_bins=[30, 60],
    )

    # Assert
    assert ds.y_train is not None
    unique = set(ds.y_train.tolist())
    assert unique.issubset({0, 1, 2}), (
        f"Binned age must be in {{0, 1, 2}}; got {unique}"
    )


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_age_bins_num_classes_is_three(utkface_tmp: Path) -> None:
    """AGE BINS: num_classes must equal 3 when age_bins=[30, 60]."""
    # Act
    ds = load_utkface(
        path=utkface_tmp,
        target="age",
        attribute_1="gender",
        attribute_2="race",
        age_bins=[30, 60],
    )

    # Assert
    assert ds.num_classes == 3, (
        f"Expected num_classes=3 for age_bins=[30, 60], got {ds.num_classes}"
    )


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_age_bins_no_new_cache_files_created(utkface_tmp: Path) -> None:
    """AGE BINS: binning is in-memory; no new .npz files must be created."""
    # Arrange — cold-start already ran; record files before
    files_before = set(utkface_tmp.iterdir())

    # Act
    load_utkface(
        path=utkface_tmp,
        target="age",
        attribute_1="gender",
        attribute_2="race",
        age_bins=[30, 60],
    )

    # Assert
    files_after = set(utkface_tmp.iterdir())
    assert files_before == files_after, (
        f"New files created during age-binned call: {files_after - files_before}"
    )


# ---------------------------------------------------------------------------
# Integration tests — ROUTINE (cache hit)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cache_hit_x_train_same_shape(utkface_tmp: Path) -> None:
    """ROUTINE (cache hit): second call returns x_train with the same shape."""
    # Arrange — cold-start already ran; capture shape from first call
    ds_first = load_utkface(
        path=utkface_tmp, target="age", attribute_1="gender", attribute_2="race"
    )

    # Act — second call, same params
    ds_second = load_utkface(
        path=utkface_tmp, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    assert ds_first.x_train is not None and ds_second.x_train is not None
    assert ds_first.x_train.shape == ds_second.x_train.shape


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cache_hit_x_test_same_shape(utkface_tmp: Path) -> None:
    """ROUTINE (cache hit): second call returns x_test with the same shape."""
    # Arrange
    ds_first = load_utkface(
        path=utkface_tmp, target="age", attribute_1="gender", attribute_2="race"
    )

    # Act
    ds_second = load_utkface(
        path=utkface_tmp, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    assert ds_first.x_test is not None and ds_second.x_test is not None
    assert ds_first.x_test.shape == ds_second.x_test.shape


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_cache_hit_no_new_files_created(utkface_tmp: Path) -> None:
    """ROUTINE (cache hit): second call with same params must not create extra files."""
    # Arrange — cold-start must have run; capture the file listing
    cache_path = (
        utkface_tmp / "utkface_processed__target=age__attr1=gender__attr2=race.npz"
    )
    assert cache_path.exists(), "Precondition: cold-start test must have run first"
    files_before = set(utkface_tmp.iterdir())

    # Act
    load_utkface(
        path=utkface_tmp, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    files_after = set(utkface_tmp.iterdir())
    assert files_before == files_after, (
        f"New files created on cache hit: {files_after - files_before}"
    )


# ---------------------------------------------------------------------------
# Integration tests — CHANGED TARGET
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_changed_target_creates_new_cache_file(utkface_tmp: Path) -> None:
    """CHANGED TARGET: new target creates a new .npz cache file."""
    # Arrange
    new_cache = (
        utkface_tmp / "utkface_processed__target=gender__attr1=age__attr2=race.npz"
    )

    # Act
    load_utkface(
        path=utkface_tmp, target="gender", attribute_1="age", attribute_2="race"
    )

    # Assert
    assert new_cache.exists(), (
        "utkface_processed__target=gender__attr1=age__attr2=race.npz must be created"
    )


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_changed_target_old_cache_still_exists(utkface_tmp: Path) -> None:
    """CHANGED TARGET: original age cache must survive when target changes."""
    # Arrange — run changed-target call (no-op if already done)
    load_utkface(
        path=utkface_tmp, target="gender", attribute_1="age", attribute_2="race"
    )
    old_cache = (
        utkface_tmp / "utkface_processed__target=age__attr1=gender__attr2=race.npz"
    )

    # Assert
    assert old_cache.exists(), "Original age cache must survive a changed-target call"


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_changed_target_utk_images_dir_mtime_unchanged(utkface_tmp: Path) -> None:
    """CHANGED TARGET: UTKFace/ mtime must be unchanged (no re-download/extraction)."""
    # Arrange — images dir must exist from cold start
    imgs_dir = utkface_tmp / "UTKFace"
    assert imgs_dir.exists(), "Precondition: UTKFace/ must exist from cold-start test"
    mtime_before = imgs_dir.stat().st_mtime

    # Act
    load_utkface(
        path=utkface_tmp, target="gender", attribute_1="age", attribute_2="race"
    )

    # Assert
    assert imgs_dir.stat().st_mtime == mtime_before, (
        "UTKFace/ must not be re-extracted on target change"
    )


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_changed_target_gender_y_values_binary(utkface_tmp: Path) -> None:
    """CHANGED TARGET (gender): y values for gender target must be 0 or 1 only."""
    # Act
    ds = load_utkface(
        path=utkface_tmp, target="gender", attribute_1="age", attribute_2="race"
    )

    # Assert
    assert ds.y_train is not None and ds.y_test is not None
    unique_train = set(ds.y_train.tolist())
    unique_test = set(ds.y_test.tolist())
    assert unique_train.issubset({0, 1}), (
        f"Gender train labels must be binary; got {unique_train}"
    )
    assert unique_test.issubset({0, 1}), (
        f"Gender test labels must be binary; got {unique_test}"
    )


# ---------------------------------------------------------------------------
# Unit tests — _utkface_parse_labels (no integration mark)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("25_0_2_20170116174525125", (25, 0, 2)),
        ("0_1_4_20170117134012721", (0, 1, 4)),
    ],
)
def test_parse_labels_valid_filenames(
    stem: str, expected: tuple[int, int, int]
) -> None:
    """_utkface_parse_labels: valid filenames return the correct (age, gender, race) tuple."""
    # Arrange
    img_path = Path(stem + ".jpg")

    # Act
    result = _utkface_parse_labels(img_path)

    # Assert
    assert result == expected, f"Expected {expected}, got {result}"


@pytest.mark.parametrize(
    "stem",
    [
        "25_0",  # too few parts (only 2)
        "abc_0_1_date",  # non-integer age
        "200_0_1_date",  # age out of range (> 116)
        "25_2_1_date",  # invalid gender (not 0 or 1)
        "25_0_5_date",  # invalid race (not in 0-4)
    ],
)
def test_parse_labels_malformed_returns_none(stem: str) -> None:
    """_utkface_parse_labels: malformed filenames must return None."""
    # Arrange
    img_path = Path(stem + ".jpg")

    # Act
    result = _utkface_parse_labels(img_path)

    # Assert
    assert result is None, f"Expected None for malformed stem {stem!r}, got {result}"


# ---------------------------------------------------------------------------
# Unit tests — _utkface_build_processed_cache (no integration mark)
# ---------------------------------------------------------------------------

_SYNTHETIC_FILENAMES = [
    "10_0_0_20170101000001.jpg",  # age=10, gender=0, race=0
    "25_1_1_20170101000002.jpg",  # age=25, gender=1, race=1
    "45_0_2_20170101000003.jpg",  # age=45, gender=0, race=2
    "60_1_3_20170101000004.jpg",  # age=60, gender=1, race=3
    "80_0_4_20170101000005.jpg",  # age=80, gender=0, race=4
]


@pytest.fixture
def make_fake_utkface():
    """Factory fixture that builds a synthetic UTKFace/ directory.

    Returned callable signature: `(tmp: Path, n: int = 5) -> imgs_dir`.
    Filenames follow the UTKFace naming convention so `_utkface_parse_labels`
    can parse them.
    """

    def _make(tmp: Path, n: int = 5) -> Path:
        imgs_dir = tmp / "UTKFace"
        imgs_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(42)
        for fname in _SYNTHETIC_FILENAMES[:n]:
            pixel_data = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
            img = Image.fromarray(pixel_data, mode="RGB")
            img.save(imgs_dir / fname, format="JPEG")

        return imgs_dir

    return _make


def test_build_processed_cache_npz_keys(tmp_path: Path, make_fake_utkface) -> None:
    """_utkface_build_processed_cache: .npz must contain keys imgs, y, z1, z2."""
    # Arrange
    imgs_dir = make_fake_utkface(tmp_path)
    cache_path = (
        tmp_path / "utkface_processed__target=age__attr1=gender__attr2=race.npz"
    )

    # Act
    _utkface_build_processed_cache(
        imgs_dir, cache_path, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    npz = np.load(cache_path)
    for key in ("imgs", "y", "z1", "z2"):
        assert key in npz, f"Expected key {key!r}; found keys: {list(npz.keys())}"
    npz.close()


def test_build_processed_cache_imgs_shape(tmp_path: Path, make_fake_utkface) -> None:
    """_utkface_build_processed_cache: imgs must have shape (5, 3, 64, 64)."""
    # Arrange
    n = 5
    imgs_dir = make_fake_utkface(tmp_path, n=n)
    cache_path = (
        tmp_path / "utkface_processed__target=age__attr1=gender__attr2=race.npz"
    )

    # Act
    _utkface_build_processed_cache(
        imgs_dir, cache_path, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    npz = np.load(cache_path)
    imgs = npz["imgs"]
    assert imgs.shape == (n, 3, 64, 64), (
        f"Expected imgs shape ({n}, 3, 64, 64), got {imgs.shape}"
    )
    npz.close()


def test_build_processed_cache_imgs_dtype_uint8(
    tmp_path: Path, make_fake_utkface
) -> None:
    """_utkface_build_processed_cache: imgs must have dtype uint8."""
    # Arrange
    imgs_dir = make_fake_utkface(tmp_path)
    cache_path = (
        tmp_path / "utkface_processed__target=age__attr1=gender__attr2=race.npz"
    )

    # Act
    _utkface_build_processed_cache(
        imgs_dir, cache_path, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    npz = np.load(cache_path)
    assert npz["imgs"].dtype == np.uint8, (
        f"Expected imgs dtype uint8, got {npz['imgs'].dtype}"
    )
    npz.close()


def test_build_processed_cache_y_shape(tmp_path: Path, make_fake_utkface) -> None:
    """_utkface_build_processed_cache: y must be a 1-D array of length 5."""
    # Arrange
    n = 5
    imgs_dir = make_fake_utkface(tmp_path, n=n)
    cache_path = (
        tmp_path / "utkface_processed__target=age__attr1=gender__attr2=race.npz"
    )

    # Act
    _utkface_build_processed_cache(
        imgs_dir, cache_path, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    npz = np.load(cache_path)
    y = npz["y"]
    assert y.shape == (n,), f"Expected y shape ({n},), got {y.shape}"
    npz.close()


def test_build_processed_cache_y_dtype_int64(tmp_path: Path, make_fake_utkface) -> None:
    """_utkface_build_processed_cache: y must have dtype int64."""
    # Arrange
    imgs_dir = make_fake_utkface(tmp_path)
    cache_path = (
        tmp_path / "utkface_processed__target=age__attr1=gender__attr2=race.npz"
    )

    # Act
    _utkface_build_processed_cache(
        imgs_dir, cache_path, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    npz = np.load(cache_path)
    assert npz["y"].dtype == np.int64, f"Expected y dtype int64, got {npz['y'].dtype}"
    npz.close()


def test_build_processed_cache_z1_shape(tmp_path: Path, make_fake_utkface) -> None:
    """_utkface_build_processed_cache: z1 must be a 1-D array of length 5."""
    # Arrange
    n = 5
    imgs_dir = make_fake_utkface(tmp_path, n=n)
    cache_path = (
        tmp_path / "utkface_processed__target=age__attr1=gender__attr2=race.npz"
    )

    # Act
    _utkface_build_processed_cache(
        imgs_dir, cache_path, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    npz = np.load(cache_path)
    z1 = npz["z1"]
    assert z1.shape == (n,), f"Expected z1 shape ({n},), got {z1.shape}"
    npz.close()


def test_build_processed_cache_z1_dtype_int64(
    tmp_path: Path, make_fake_utkface
) -> None:
    """_utkface_build_processed_cache: z1 must have dtype int64."""
    # Arrange
    imgs_dir = make_fake_utkface(tmp_path)
    cache_path = (
        tmp_path / "utkface_processed__target=age__attr1=gender__attr2=race.npz"
    )

    # Act
    _utkface_build_processed_cache(
        imgs_dir, cache_path, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    npz = np.load(cache_path)
    assert npz["z1"].dtype == np.int64, (
        f"Expected z1 dtype int64, got {npz['z1'].dtype}"
    )
    npz.close()


def test_build_processed_cache_z2_shape(tmp_path: Path, make_fake_utkface) -> None:
    """_utkface_build_processed_cache: z2 must be a 1-D array of length 5."""
    # Arrange
    n = 5
    imgs_dir = make_fake_utkface(tmp_path, n=n)
    cache_path = (
        tmp_path / "utkface_processed__target=age__attr1=gender__attr2=race.npz"
    )

    # Act
    _utkface_build_processed_cache(
        imgs_dir, cache_path, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    npz = np.load(cache_path)
    z2 = npz["z2"]
    assert z2.shape == (n,), f"Expected z2 shape ({n},), got {z2.shape}"
    npz.close()


def test_build_processed_cache_z2_dtype_int64(
    tmp_path: Path, make_fake_utkface
) -> None:
    """_utkface_build_processed_cache: z2 must have dtype int64."""
    # Arrange
    imgs_dir = make_fake_utkface(tmp_path)
    cache_path = (
        tmp_path / "utkface_processed__target=age__attr1=gender__attr2=race.npz"
    )

    # Act
    _utkface_build_processed_cache(
        imgs_dir, cache_path, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    npz = np.load(cache_path)
    assert npz["z2"].dtype == np.int64, (
        f"Expected z2 dtype int64, got {npz['z2'].dtype}"
    )
    npz.close()


def test_build_processed_cache_gender_values_binary(
    tmp_path: Path, make_fake_utkface
) -> None:
    """_utkface_build_processed_cache: gender (z1) values must be in {0, 1}."""
    # Arrange
    imgs_dir = make_fake_utkface(tmp_path)
    cache_path = (
        tmp_path / "utkface_processed__target=age__attr1=gender__attr2=race.npz"
    )

    # Act
    _utkface_build_processed_cache(
        imgs_dir, cache_path, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    npz = np.load(cache_path)
    unique = set(npz["z1"].tolist())
    assert unique.issubset({0, 1}), (
        f"Gender (z1) must contain only 0 and 1; got {unique}"
    )
    npz.close()


def test_build_processed_cache_race_values_in_range(
    tmp_path: Path, make_fake_utkface
) -> None:
    """_utkface_build_processed_cache: race (z2) values must be in {0, 1, 2, 3, 4}."""
    # Arrange
    imgs_dir = make_fake_utkface(tmp_path)
    cache_path = (
        tmp_path / "utkface_processed__target=age__attr1=gender__attr2=race.npz"
    )

    # Act
    _utkface_build_processed_cache(
        imgs_dir, cache_path, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    npz = np.load(cache_path)
    unique = set(npz["z2"].tolist())
    assert unique.issubset({0, 1, 2, 3, 4}), (
        f"Race (z2) must contain values in {{0,1,2,3,4}}; got {unique}"
    )
    npz.close()


def test_build_processed_cache_age_values_nonnegative(
    tmp_path: Path, make_fake_utkface
) -> None:
    """_utkface_build_processed_cache: age (y) values must all be >= 0."""
    # Arrange
    imgs_dir = make_fake_utkface(tmp_path)
    cache_path = (
        tmp_path / "utkface_processed__target=age__attr1=gender__attr2=race.npz"
    )

    # Act
    _utkface_build_processed_cache(
        imgs_dir, cache_path, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    npz = np.load(cache_path)
    y = npz["y"]
    assert int(y.min()) >= 0, f"Age (y) values must be >= 0; got min {y.min()}"
    npz.close()


def test_build_processed_cache_age_values_match_filenames(
    tmp_path: Path, make_fake_utkface
) -> None:
    """_utkface_build_processed_cache: parsed age values must match filenames exactly."""
    # Arrange — synthetic filenames encode ages [10, 25, 45, 60, 80] in sorted order
    expected_ages = sorted([10, 25, 45, 60, 80])
    imgs_dir = make_fake_utkface(tmp_path)
    cache_path = (
        tmp_path / "utkface_processed__target=age__attr1=gender__attr2=race.npz"
    )

    # Act
    _utkface_build_processed_cache(
        imgs_dir, cache_path, target="age", attribute_1="gender", attribute_2="race"
    )

    # Assert
    npz = np.load(cache_path)
    # glob sorts by filename so result order is deterministic
    actual_ages = sorted(npz["y"].tolist())
    assert actual_ages == expected_ages, (
        f"Expected ages {expected_ages}, got {actual_ages}"
    )
    npz.close()
