"""Fast tests for load_utkface and its helpers in amulet/datasets/__image_datasets.py.

The network boundary (gdown) is mocked: "downloading" plants a synthetic
UTKFace tarball, so cold-start orchestration, cache-hit, changed-target, and
age-binning control flow run in milliseconds against the real extraction,
label-parsing, and cache-building code. The real download is exercised by the
slow-tier smoke in test_downloads.py.
"""

import io
import tarfile
from pathlib import Path

import numpy as np
import pytest

from amulet.datasets.__image_datasets import (  # type: ignore[reportPrivateImportUsage]
    _utkface_build_processed_cache,
    _utkface_parse_labels,
    load_utkface,
)

# UTKFace filenames encode age_gender_race_timestamp; ages span the [30, 60)
# bin edges used in the age-binning test so all three bins are populated.
_SYNTHETIC_FILENAMES = [
    "10_0_0_20170101000001.jpg",
    "25_1_1_20170101000002.jpg",
    "45_0_2_20170101000003.jpg",
    "60_1_3_20170101000004.jpg",
    "80_0_4_20170101000005.jpg",
]
_AGES = [10, 25, 45, 60, 80]


@pytest.fixture
def make_fake_utkface(tmp_path: Path, make_jpeg_bytes):
    """Plant a raw UTKFace/ images directory directly on disk."""

    def _make() -> Path:
        imgs_dir = tmp_path / "UTKFace"
        imgs_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42)
        for fname in _SYNTHETIC_FILENAMES:
            _ = (imgs_dir / fname).write_bytes(make_jpeg_bytes(rng, 32, 32))
        return imgs_dir

    return _make


@pytest.fixture
def mock_gdown(mocker, make_jpeg_bytes):
    """Patch gdown.download so 'downloading' writes a synthetic tarball instead."""

    def _download(id: str, output: str, quiet: bool = False) -> None:
        out = Path(output)
        assert out.name == "UTKFace.tar.gz", f"unexpected gdown target: {out}"
        rng = np.random.default_rng(42)
        with tarfile.open(out, "w:gz") as tf:
            for fname in _SYNTHETIC_FILENAMES:
                data = make_jpeg_bytes(rng, 32, 32)
                info = tarfile.TarInfo(name=f"UTKFace/{fname}")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))

    return mocker.patch("gdown.download", side_effect=_download)


def test_cold_start_downloads_raw_and_builds_cache(tmp_path: Path, mock_gdown) -> None:
    load_utkface(path=tmp_path)

    assert mock_gdown.call_count == 1
    assert (tmp_path / "UTKFace").is_dir()
    assert (
        tmp_path / "utkface_processed__target=age__attr1=gender__attr2=race.npz"
    ).exists()


def test_cold_start_returns_well_formed_bundle(tmp_path: Path, mock_gdown) -> None:
    data = load_utkface(path=tmp_path)

    assert data.x_train is not None and data.x_test is not None
    assert data.y_train is not None and data.y_test is not None
    assert data.z_train is not None
    n_train, n_test = len(data.y_train), len(data.y_test)
    assert n_train + n_test == len(_SYNTHETIC_FILENAMES)
    assert data.x_train.dtype == np.float32
    assert data.x_train.shape[1:] == (3, 64, 64)
    assert data.x_train.min() >= 0.0 and data.x_train.max() <= 1.0
    assert data.y_train.dtype == np.int64
    assert set(np.concatenate([data.y_train, data.y_test])) <= set(_AGES)
    assert data.z_train.shape == (n_train, 2)
    assert data.z_train.dtype == np.int64
    assert data.num_features == 4096  # _UTKFACE_IMG_SIZE**2 (64*64)


def test_cache_hit_skips_download_and_reproduces_split(
    tmp_path: Path, mock_gdown
) -> None:
    first = load_utkface(path=tmp_path)
    mock_gdown.reset_mock()

    second = load_utkface(path=tmp_path)

    mock_gdown.assert_not_called()
    assert first.y_train is not None and second.y_train is not None
    np.testing.assert_array_equal(first.y_train, second.y_train)


def test_changed_target_rebuilds_cache_without_redownload(
    tmp_path: Path, mock_gdown
) -> None:
    load_utkface(path=tmp_path)
    mock_gdown.reset_mock()

    data = load_utkface(path=tmp_path, target="gender", attribute_1="age")

    mock_gdown.assert_not_called()
    assert (
        tmp_path / "utkface_processed__target=gender__attr1=age__attr2=race.npz"
    ).exists()
    assert (
        tmp_path / "utkface_processed__target=age__attr1=gender__attr2=race.npz"
    ).exists()
    assert data.y_train is not None and data.y_test is not None
    assert set(np.concatenate([data.y_train, data.y_test])) <= {0, 1}


def test_age_bins_discretize_target_without_new_cache(
    tmp_path: Path, mock_gdown
) -> None:
    load_utkface(path=tmp_path)
    npz_before = sorted(tmp_path.glob("*.npz"))

    data = load_utkface(path=tmp_path, age_bins=[30, 60])

    # Binning happens after the cache load, so no new cache key appears.
    assert sorted(tmp_path.glob("*.npz")) == npz_before
    assert data.y_train is not None and data.y_test is not None
    y_all = np.concatenate([data.y_train, data.y_test])
    # np.digitize([10, 25, 45, 60, 80], [30, 60]) with the loader's default
    # right=False: pinning the full sorted multiset (not just the set of
    # distinct bins) catches a right=False/right=True flip at the age=60
    # edge, which changes counts per bin but not the set of bins present.
    assert sorted(y_all.tolist()) == [0, 0, 1, 2, 2]
    assert data.num_classes == 3


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
    assert _utkface_parse_labels(Path(stem + ".jpg")) == expected


@pytest.mark.parametrize(
    "stem",
    [
        "25_0",  # too few parts
        "abc_0_1_date",  # non-integer age
        "200_0_1_date",  # age out of range
        "25_2_1_date",  # invalid gender
        "25_0_5_date",  # invalid race
    ],
)
def test_parse_labels_malformed_returns_none(stem: str) -> None:
    assert _utkface_parse_labels(Path(stem + ".jpg")) is None


def test_build_processed_cache_writes_well_formed_arrays(
    tmp_path: Path, make_fake_utkface
) -> None:
    imgs_dir = make_fake_utkface()
    cache_path = tmp_path / "cache.npz"
    n = len(_SYNTHETIC_FILENAMES)

    _utkface_build_processed_cache(imgs_dir, cache_path, "age", "gender", "race")

    npz = np.load(cache_path)
    assert set(npz.files) == {"imgs", "y", "z1", "z2"}
    assert npz["imgs"].shape == (n, 3, 64, 64)
    assert npz["imgs"].dtype == np.uint8
    for key in ("y", "z1", "z2"):
        assert npz[key].shape == (n,) and npz[key].dtype == np.int64
    assert set(npz["z1"]) <= {0, 1}
    assert set(npz["z2"]) <= set(range(5))


def test_build_processed_cache_labels_match_filenames(
    tmp_path: Path, make_fake_utkface
) -> None:
    imgs_dir = make_fake_utkface()
    cache_path = tmp_path / "cache.npz"

    _utkface_build_processed_cache(imgs_dir, cache_path, "age", "gender", "race")

    npz = np.load(cache_path)
    np.testing.assert_array_equal(npz["y"], np.array(_AGES))
