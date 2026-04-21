"""
Integration and unit tests for load_celeba and its helpers in
amulet/datasets/__image_datasets.py.

Integration tests download real data from the network and process ~202k images,
so they are marked @pytest.mark.integration with a 1200 s timeout.  Unit tests
for _celeba_build_processed_cache use only in-memory synthetic data.

A single session-scoped tmp directory is used for all integration tests so that
the expensive cold-start download + extraction happens once and is reused by
the cache-hit and changed-target cases.  After all tests complete the entire
tmp tree is deleted.  The real data/celeba/ directory at the repo root is never
touched.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from amulet.datasets.__image_datasets import (  # type: ignore[reportPrivateImportUsage]
    _celeba_build_processed_cache,
    load_celeba,
)

# ---------------------------------------------------------------------------
# Session-scoped fixture: one tmp dir shared across all integration tests.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def celeba_tmp(tmp_path_factory: pytest.TempPathFactory) -> Path:  # type: ignore[misc]
    """Create a fresh temp directory once per session, then delete it after all tests."""
    tmp = tmp_path_factory.mktemp("celeba_integration", numbered=True)
    yield tmp  # type: ignore[misc]
    shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Integration tests — COLD START
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(1200)
def test_cold_start_creates_attrs_file(celeba_tmp: Path) -> None:
    """COLD START: list_attr_celeba.txt must exist after the first call."""
    # Arrange — nothing pre-exists in celeba_tmp
    attrs_path = celeba_tmp / "list_attr_celeba.txt"

    # Act
    load_celeba(path=celeba_tmp, target_attribute="Smiling")

    # Assert
    assert attrs_path.exists(), "list_attr_celeba.txt must be downloaded"


@pytest.mark.integration
@pytest.mark.timeout(1200)
def test_cold_start_creates_images_dir(celeba_tmp: Path) -> None:
    """COLD START: img_align_celeba/ directory must exist after the first call."""
    # Arrange
    imgs_dir = celeba_tmp / "img_align_celeba"

    # Act
    load_celeba(path=celeba_tmp, target_attribute="Smiling")

    # Assert
    assert imgs_dir.exists() and imgs_dir.is_dir(), (
        "img_align_celeba/ directory must exist after extraction"
    )


@pytest.mark.integration
@pytest.mark.timeout(1200)
def test_cold_start_creates_processed_cache(celeba_tmp: Path) -> None:
    """COLD START: parameter-keyed .npz cache must exist after the first call."""
    # Arrange
    cache_path = celeba_tmp / "celeba_processed__target=Smiling.npz"

    # Act
    load_celeba(path=celeba_tmp, target_attribute="Smiling")

    # Assert
    assert cache_path.exists(), "celeba_processed__target=Smiling.npz must be created"


@pytest.mark.integration
@pytest.mark.timeout(1200)
def test_cold_start_returns_non_none_x_train(celeba_tmp: Path) -> None:
    """COLD START: x_train must be non-None."""
    # Act
    ds = load_celeba(path=celeba_tmp, target_attribute="Smiling")

    # Assert
    assert ds.x_train is not None


@pytest.mark.integration
@pytest.mark.timeout(1200)
def test_cold_start_returns_non_none_x_test(celeba_tmp: Path) -> None:
    """COLD START: x_test must be non-None."""
    # Act
    ds = load_celeba(path=celeba_tmp, target_attribute="Smiling")

    # Assert
    assert ds.x_test is not None


@pytest.mark.integration
@pytest.mark.timeout(1200)
def test_cold_start_returns_non_none_y_train(celeba_tmp: Path) -> None:
    """COLD START: y_train must be non-None."""
    # Act
    ds = load_celeba(path=celeba_tmp, target_attribute="Smiling")

    # Assert
    assert ds.y_train is not None


@pytest.mark.integration
@pytest.mark.timeout(1200)
def test_cold_start_returns_non_none_y_test(celeba_tmp: Path) -> None:
    """COLD START: y_test must be non-None."""
    # Act
    ds = load_celeba(path=celeba_tmp, target_attribute="Smiling")

    # Assert
    assert ds.y_test is not None


@pytest.mark.integration
@pytest.mark.timeout(1200)
def test_cold_start_returns_non_none_z_train(celeba_tmp: Path) -> None:
    """COLD START: z_train must be non-None."""
    # Act
    ds = load_celeba(path=celeba_tmp, target_attribute="Smiling")

    # Assert
    assert ds.z_train is not None


@pytest.mark.integration
@pytest.mark.timeout(1200)
def test_cold_start_returns_non_none_z_test(celeba_tmp: Path) -> None:
    """COLD START: z_test must be non-None."""
    # Act
    ds = load_celeba(path=celeba_tmp, target_attribute="Smiling")

    # Assert
    assert ds.z_test is not None


@pytest.mark.integration
@pytest.mark.timeout(1200)
def test_cold_start_x_train_dtype_float32(celeba_tmp: Path) -> None:
    """COLD START: x_train must have dtype float32."""
    # Act
    ds = load_celeba(path=celeba_tmp, target_attribute="Smiling")

    # Assert
    assert ds.x_train is not None
    assert ds.x_train.dtype == np.float32


@pytest.mark.integration
@pytest.mark.timeout(1200)
def test_cold_start_x_train_values_in_unit_interval(celeba_tmp: Path) -> None:
    """COLD START: x_train pixel values must all be in [0, 1]."""
    # Act
    ds = load_celeba(path=celeba_tmp, target_attribute="Smiling")

    # Assert
    assert ds.x_train is not None
    assert float(ds.x_train.min()) >= 0.0, "x_train minimum must be >= 0"
    assert float(ds.x_train.max()) <= 1.0, "x_train maximum must be <= 1"


@pytest.mark.integration
@pytest.mark.timeout(1200)
def test_cold_start_x_train_shape_channels_height_width(celeba_tmp: Path) -> None:
    """COLD START: x_train must have shape (N, 3, 64, 64)."""
    # Act
    ds = load_celeba(path=celeba_tmp, target_attribute="Smiling")

    # Assert
    assert ds.x_train is not None
    assert ds.x_train.ndim == 4, f"Expected 4D array, got {ds.x_train.ndim}D"
    assert ds.x_train.shape[1] == 3, f"Expected 3 channels, got {ds.x_train.shape[1]}"
    assert ds.x_train.shape[2] == 64, f"Expected height 64, got {ds.x_train.shape[2]}"
    assert ds.x_train.shape[3] == 64, f"Expected width 64, got {ds.x_train.shape[3]}"


@pytest.mark.integration
@pytest.mark.timeout(1200)
def test_cold_start_y_train_dtype_int64(celeba_tmp: Path) -> None:
    """COLD START: y_train must have dtype int64."""
    # Act
    ds = load_celeba(path=celeba_tmp, target_attribute="Smiling")

    # Assert
    assert ds.y_train is not None
    assert ds.y_train.dtype == np.int64


@pytest.mark.integration
@pytest.mark.timeout(1200)
def test_cold_start_y_train_values_binary(celeba_tmp: Path) -> None:
    """COLD START: all y_train values must be 0 or 1."""
    # Act
    ds = load_celeba(path=celeba_tmp, target_attribute="Smiling")

    # Assert
    assert ds.y_train is not None
    unique = set(ds.y_train.tolist())
    assert unique.issubset({0, 1}), f"Non-binary train labels: {unique}"


@pytest.mark.integration
@pytest.mark.timeout(1200)
def test_cold_start_z_train_shape_is_n_by_1(celeba_tmp: Path) -> None:
    """COLD START: z_train must have shape (N_train, 1)."""
    # Act
    ds = load_celeba(path=celeba_tmp, target_attribute="Smiling")

    # Assert
    assert ds.x_train is not None and ds.z_train is not None
    assert ds.z_train.ndim == 2, f"Expected 2D z_train, got {ds.z_train.ndim}D"
    assert ds.z_train.shape[1] == 1, (
        f"Expected 1 sensitive attribute column, got {ds.z_train.shape[1]}"
    )
    assert ds.z_train.shape[0] == ds.x_train.shape[0], (
        "z_train row count must match x_train row count"
    )


@pytest.mark.integration
@pytest.mark.timeout(1200)
def test_cold_start_z_train_dtype_int64(celeba_tmp: Path) -> None:
    """COLD START: z_train must have dtype int64."""
    # Act
    ds = load_celeba(path=celeba_tmp, target_attribute="Smiling")

    # Assert
    assert ds.z_train is not None
    assert ds.z_train.dtype == np.int64


@pytest.mark.integration
@pytest.mark.timeout(1200)
def test_cold_start_z_train_values_binary(celeba_tmp: Path) -> None:
    """COLD START: all z_train values must be 0 or 1."""
    # Act
    ds = load_celeba(path=celeba_tmp, target_attribute="Smiling")

    # Assert
    assert ds.z_train is not None
    unique = set(ds.z_train.flatten().tolist())
    assert unique.issubset({0, 1}), f"Non-binary z_train values: {unique}"


@pytest.mark.integration
@pytest.mark.timeout(1200)
def test_cold_start_num_classes_is_two(celeba_tmp: Path) -> None:
    """COLD START: num_classes must equal 2 (binary classification)."""
    # Act
    ds = load_celeba(path=celeba_tmp, target_attribute="Smiling")

    # Assert
    assert ds.num_classes == 2


@pytest.mark.integration
@pytest.mark.timeout(1200)
def test_cold_start_train_plus_test_equals_total(celeba_tmp: Path) -> None:
    """COLD START: len(train_set) + len(test_set) must equal total sample count."""
    # Act
    ds = load_celeba(path=celeba_tmp, target_attribute="Smiling")

    # Assert
    assert ds.x_train is not None and ds.x_test is not None
    total = ds.x_train.shape[0] + ds.x_test.shape[0]
    assert len(ds.train_set) + len(ds.test_set) == total  # type: ignore[reportArgumentType]


@pytest.mark.integration
@pytest.mark.timeout(1200)
def test_cold_start_default_test_size_near_50_percent(celeba_tmp: Path) -> None:
    """COLD START: default test_size=0.5 must give each split within 1% of 50%."""
    # Act
    ds = load_celeba(path=celeba_tmp, target_attribute="Smiling")

    # Assert
    assert ds.x_train is not None and ds.x_test is not None
    total = ds.x_train.shape[0] + ds.x_test.shape[0]
    test_ratio = ds.x_test.shape[0] / total
    assert abs(test_ratio - 0.5) <= 0.01, (
        f"Test ratio {test_ratio:.4f} deviates more than 1% from 50%"
    )


# ---------------------------------------------------------------------------
# Integration tests — ROUTINE (cache hit)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(1200)
def test_cache_hit_x_train_same_shape(celeba_tmp: Path) -> None:
    """ROUTINE (cache hit): second call returns x_train with the same shape."""
    # Arrange — cold-start already ran; capture shape from first call
    ds_first = load_celeba(path=celeba_tmp, target_attribute="Smiling")

    # Act — second call, same params
    ds_second = load_celeba(path=celeba_tmp, target_attribute="Smiling")

    # Assert
    assert ds_first.x_train is not None and ds_second.x_train is not None
    assert ds_first.x_train.shape == ds_second.x_train.shape


@pytest.mark.integration
@pytest.mark.timeout(1200)
def test_cache_hit_x_test_same_shape(celeba_tmp: Path) -> None:
    """ROUTINE (cache hit): second call returns x_test with the same shape."""
    # Arrange
    ds_first = load_celeba(path=celeba_tmp, target_attribute="Smiling")

    # Act
    ds_second = load_celeba(path=celeba_tmp, target_attribute="Smiling")

    # Assert
    assert ds_first.x_test is not None and ds_second.x_test is not None
    assert ds_first.x_test.shape == ds_second.x_test.shape


@pytest.mark.integration
@pytest.mark.timeout(1200)
def test_cache_hit_no_new_files_created(celeba_tmp: Path) -> None:
    """ROUTINE (cache hit): second call must not create extra files in the directory."""
    # Arrange — cold-start must have run; capture the file listing
    cache_path = celeba_tmp / "celeba_processed__target=Smiling.npz"
    assert cache_path.exists(), "Precondition: cold-start test must have run first"
    files_before = set(celeba_tmp.iterdir())

    # Act
    load_celeba(path=celeba_tmp, target_attribute="Smiling")

    # Assert
    files_after = set(celeba_tmp.iterdir())
    assert files_before == files_after, (
        f"New files created on cache hit: {files_after - files_before}"
    )


# ---------------------------------------------------------------------------
# Integration tests — CHANGED TARGET
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(1200)
def test_changed_target_creates_new_cache_file(celeba_tmp: Path) -> None:
    """CHANGED TARGET: new target_attribute creates a new .npz cache file."""
    # Arrange
    new_cache = celeba_tmp / "celeba_processed__target=Young.npz"

    # Act
    load_celeba(path=celeba_tmp, target_attribute="Young")

    # Assert
    assert new_cache.exists(), (
        "celeba_processed__target=Young.npz must be created for new target"
    )


@pytest.mark.integration
@pytest.mark.timeout(1200)
def test_changed_target_old_cache_still_exists(celeba_tmp: Path) -> None:
    """CHANGED TARGET: original Smiling cache must not be removed when target changes."""
    # Arrange — run changed-target call first (no-op if already done)
    load_celeba(path=celeba_tmp, target_attribute="Young")
    old_cache = celeba_tmp / "celeba_processed__target=Smiling.npz"

    # Assert
    assert old_cache.exists(), (
        "Original Smiling cache must survive a changed-target call"
    )


@pytest.mark.integration
@pytest.mark.timeout(1200)
def test_changed_target_attrs_file_not_redownloaded(celeba_tmp: Path) -> None:
    """CHANGED TARGET: list_attr_celeba.txt mtime must be unchanged after target change."""
    # Arrange — raw file must exist from cold start
    attrs_path = celeba_tmp / "list_attr_celeba.txt"
    assert attrs_path.exists(), (
        "Precondition: attrs file must exist from cold-start test"
    )
    mtime_before = attrs_path.stat().st_mtime

    # Act
    load_celeba(path=celeba_tmp, target_attribute="Young")

    # Assert
    assert attrs_path.stat().st_mtime == mtime_before, (
        "list_attr_celeba.txt must not be re-downloaded on target change"
    )


@pytest.mark.integration
@pytest.mark.timeout(1200)
def test_changed_target_imgs_dir_not_reextracted(celeba_tmp: Path) -> None:
    """CHANGED TARGET: img_align_celeba/ mtime must be unchanged after target change."""
    # Arrange — images dir must exist from cold start
    imgs_dir = celeba_tmp / "img_align_celeba"
    assert imgs_dir.exists(), "Precondition: imgs dir must exist from cold-start test"
    mtime_before = imgs_dir.stat().st_mtime

    # Act
    load_celeba(path=celeba_tmp, target_attribute="Young")

    # Assert
    assert imgs_dir.stat().st_mtime == mtime_before, (
        "img_align_celeba/ must not be re-extracted on target change"
    )


@pytest.mark.integration
@pytest.mark.timeout(1200)
def test_changed_target_young_y_values_binary(celeba_tmp: Path) -> None:
    """CHANGED TARGET (Young): y values must be 0 or 1 only."""
    # Act
    ds = load_celeba(path=celeba_tmp, target_attribute="Young")

    # Assert
    assert ds.y_train is not None and ds.y_test is not None
    unique_train = set(ds.y_train.tolist())
    unique_test = set(ds.y_test.tolist())
    assert unique_train.issubset({0, 1}), f"Non-binary train labels: {unique_train}"
    assert unique_test.issubset({0, 1}), f"Non-binary test labels: {unique_test}"


# ---------------------------------------------------------------------------
# Unit tests — _celeba_build_processed_cache (no integration mark, no real images)
# ---------------------------------------------------------------------------


def _make_fake_celeba(tmp: Path, n: int = 3) -> tuple[Path, Path]:
    """Build a minimal synthetic CelebA-style directory structure.

    Creates:
      tmp/list_attr_celeba.txt  — CelebA-format attribute file with n rows
      tmp/img_align_celeba/     — n small random RGB JPEG images

    Returns the paths to the attribute file and images directory.
    """
    imgs_dir = tmp / "img_align_celeba"
    imgs_dir.mkdir(parents=True, exist_ok=True)

    # CelebA list_attr_celeba.txt format:
    #   Line 1: number of images
    #   Line 2: space-separated attribute names
    #   Lines 3+: filename followed by attribute values (-1 or 1)
    attr_names = "Smiling Male"
    rng = np.random.default_rng(0)

    rows: list[str] = []
    for i in range(1, n + 1):
        fname = f"{i:06d}.jpg"
        # Alternate between +1 and -1 to cover both label values
        smiling = 1 if i % 2 == 0 else -1
        male = 1 if i % 3 == 0 else -1
        rows.append(f"{fname} {smiling} {male}")

        # Save a tiny 10x10 random RGB JPEG
        pixel_data = rng.integers(0, 256, (10, 10, 3), dtype=np.uint8)
        img = Image.fromarray(pixel_data, mode="RGB")
        img.save(imgs_dir / fname, format="JPEG")

    attrs_content = f"{n}\n{attr_names}\n" + "\n".join(rows) + "\n"
    attrs_path = tmp / "list_attr_celeba.txt"
    attrs_path.write_text(attrs_content, encoding="utf-8")

    return attrs_path, imgs_dir


def test_build_processed_cache_creates_npz_file(tmp_path: Path) -> None:
    """_celeba_build_processed_cache: output .npz file must be created on disk."""
    # Arrange
    attrs_path, imgs_dir = _make_fake_celeba(tmp_path)
    cache_path = tmp_path / "celeba_processed__target=Smiling.npz"

    # Act
    _celeba_build_processed_cache(attrs_path, imgs_dir, cache_path, "Smiling")

    # Assert
    assert cache_path.exists(), ".npz cache file must be created"


def test_build_processed_cache_npz_has_imgs_key(tmp_path: Path) -> None:
    """_celeba_build_processed_cache: .npz must contain the 'imgs' key."""
    # Arrange
    attrs_path, imgs_dir = _make_fake_celeba(tmp_path)
    cache_path = tmp_path / "celeba_processed__target=Smiling.npz"

    # Act
    _celeba_build_processed_cache(attrs_path, imgs_dir, cache_path, "Smiling")

    # Assert
    npz = np.load(cache_path)
    assert "imgs" in npz, f"Expected 'imgs' key; found keys: {list(npz.keys())}"
    npz.close()


def test_build_processed_cache_npz_has_y_key(tmp_path: Path) -> None:
    """_celeba_build_processed_cache: .npz must contain the 'y' key."""
    # Arrange
    attrs_path, imgs_dir = _make_fake_celeba(tmp_path)
    cache_path = tmp_path / "celeba_processed__target=Smiling.npz"

    # Act
    _celeba_build_processed_cache(attrs_path, imgs_dir, cache_path, "Smiling")

    # Assert
    npz = np.load(cache_path)
    assert "y" in npz, f"Expected 'y' key; found keys: {list(npz.keys())}"
    npz.close()


def test_build_processed_cache_npz_has_z_key(tmp_path: Path) -> None:
    """_celeba_build_processed_cache: .npz must contain the 'z' key."""
    # Arrange
    attrs_path, imgs_dir = _make_fake_celeba(tmp_path)
    cache_path = tmp_path / "celeba_processed__target=Smiling.npz"

    # Act
    _celeba_build_processed_cache(attrs_path, imgs_dir, cache_path, "Smiling")

    # Assert
    npz = np.load(cache_path)
    assert "z" in npz, f"Expected 'z' key; found keys: {list(npz.keys())}"
    npz.close()


def test_build_processed_cache_imgs_shape(tmp_path: Path) -> None:
    """_celeba_build_processed_cache: imgs must have shape (N, 3, 64, 64)."""
    # Arrange
    n = 3
    attrs_path, imgs_dir = _make_fake_celeba(tmp_path, n=n)
    cache_path = tmp_path / "celeba_processed__target=Smiling.npz"

    # Act
    _celeba_build_processed_cache(attrs_path, imgs_dir, cache_path, "Smiling")

    # Assert
    npz = np.load(cache_path)
    imgs = npz["imgs"]
    assert imgs.shape == (n, 3, 64, 64), (
        f"Expected imgs shape ({n}, 3, 64, 64), got {imgs.shape}"
    )
    npz.close()


def test_build_processed_cache_imgs_dtype_uint8(tmp_path: Path) -> None:
    """_celeba_build_processed_cache: imgs must have dtype uint8."""
    # Arrange
    attrs_path, imgs_dir = _make_fake_celeba(tmp_path)
    cache_path = tmp_path / "celeba_processed__target=Smiling.npz"

    # Act
    _celeba_build_processed_cache(attrs_path, imgs_dir, cache_path, "Smiling")

    # Assert
    npz = np.load(cache_path)
    assert npz["imgs"].dtype == np.uint8, (
        f"Expected imgs dtype uint8, got {npz['imgs'].dtype}"
    )
    npz.close()


def test_build_processed_cache_y_shape(tmp_path: Path) -> None:
    """_celeba_build_processed_cache: y must be a 1-D array of length N."""
    # Arrange
    n = 3
    attrs_path, imgs_dir = _make_fake_celeba(tmp_path, n=n)
    cache_path = tmp_path / "celeba_processed__target=Smiling.npz"

    # Act
    _celeba_build_processed_cache(attrs_path, imgs_dir, cache_path, "Smiling")

    # Assert
    npz = np.load(cache_path)
    y = npz["y"]
    assert y.shape == (n,), f"Expected y shape ({n},), got {y.shape}"
    npz.close()


def test_build_processed_cache_y_dtype_int64(tmp_path: Path) -> None:
    """_celeba_build_processed_cache: y must have dtype int64."""
    # Arrange
    attrs_path, imgs_dir = _make_fake_celeba(tmp_path)
    cache_path = tmp_path / "celeba_processed__target=Smiling.npz"

    # Act
    _celeba_build_processed_cache(attrs_path, imgs_dir, cache_path, "Smiling")

    # Assert
    npz = np.load(cache_path)
    assert npz["y"].dtype == np.int64, f"Expected y dtype int64, got {npz['y'].dtype}"
    npz.close()


def test_build_processed_cache_y_values_binary(tmp_path: Path) -> None:
    """_celeba_build_processed_cache: all y values must be 0 or 1 (not -1)."""
    # Arrange
    attrs_path, imgs_dir = _make_fake_celeba(tmp_path)
    cache_path = tmp_path / "celeba_processed__target=Smiling.npz"

    # Act
    _celeba_build_processed_cache(attrs_path, imgs_dir, cache_path, "Smiling")

    # Assert
    npz = np.load(cache_path)
    y = npz["y"]
    unique = set(y.tolist())
    assert unique.issubset({0, 1}), (
        f"y must contain only 0 and 1 (not -1); got {unique}"
    )
    npz.close()


def test_build_processed_cache_z_shape(tmp_path: Path) -> None:
    """_celeba_build_processed_cache: z must be a 1-D array of length N."""
    # Arrange
    n = 3
    attrs_path, imgs_dir = _make_fake_celeba(tmp_path, n=n)
    cache_path = tmp_path / "celeba_processed__target=Smiling.npz"

    # Act
    _celeba_build_processed_cache(attrs_path, imgs_dir, cache_path, "Smiling")

    # Assert
    npz = np.load(cache_path)
    z = npz["z"]
    assert z.shape == (n,), f"Expected z shape ({n},), got {z.shape}"
    npz.close()


def test_build_processed_cache_z_dtype_int64(tmp_path: Path) -> None:
    """_celeba_build_processed_cache: z must have dtype int64."""
    # Arrange
    attrs_path, imgs_dir = _make_fake_celeba(tmp_path)
    cache_path = tmp_path / "celeba_processed__target=Smiling.npz"

    # Act
    _celeba_build_processed_cache(attrs_path, imgs_dir, cache_path, "Smiling")

    # Assert
    npz = np.load(cache_path)
    assert npz["z"].dtype == np.int64, f"Expected z dtype int64, got {npz['z'].dtype}"
    npz.close()


def test_build_processed_cache_z_values_binary(tmp_path: Path) -> None:
    """_celeba_build_processed_cache: all z values must be 0 or 1 (not -1)."""
    # Arrange
    attrs_path, imgs_dir = _make_fake_celeba(tmp_path)
    cache_path = tmp_path / "celeba_processed__target=Smiling.npz"

    # Act
    _celeba_build_processed_cache(attrs_path, imgs_dir, cache_path, "Smiling")

    # Assert
    npz = np.load(cache_path)
    z = npz["z"]
    unique = set(z.tolist())
    assert unique.issubset({0, 1}), (
        f"z must contain only 0 and 1 (not -1); got {unique}"
    )
    npz.close()


def test_build_processed_cache_minus_one_converted_to_zero(tmp_path: Path) -> None:
    """_celeba_build_processed_cache: raw attribute value -1 must map to label 0."""
    # Arrange — build synthetic data where first image has Smiling=-1 (odd index)
    n = 3
    attrs_path, imgs_dir = _make_fake_celeba(tmp_path, n=n)
    cache_path = tmp_path / "celeba_processed__target=Smiling.npz"

    # Verify our synthetic helper encodes Smiling=-1 for image 1 (odd) and +1 for image 2 (even)
    # Per _make_fake_celeba: smiling = 1 if i % 2 == 0 else -1 (i is 1-based)
    # i=1 -> -1 -> expected 0; i=2 -> +1 -> expected 1; i=3 -> -1 -> expected 0
    expected_y = [0, 1, 0]

    # Act
    _celeba_build_processed_cache(attrs_path, imgs_dir, cache_path, "Smiling")

    # Assert
    npz = np.load(cache_path)
    y = npz["y"].tolist()
    assert y == expected_y, f"Expected y={expected_y}, got y={y}"
    npz.close()
